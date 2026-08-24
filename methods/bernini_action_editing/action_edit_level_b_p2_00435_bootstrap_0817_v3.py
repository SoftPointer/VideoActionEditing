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
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import FunctionType, ModuleType
from typing import Any, Mapping, NoReturn, Sequence, Tuple


METHOD = "bernini-action-edit-level-b-p2-00435-bootstrap-0817-v3"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
TAG = "fresh-world8-level-b-p2-00435-v3"
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
    "380b433d4be8c349bb79c8eb3914442136e153c2dccd4cb57ff25db9f7688a16"
)
LEVEL_B_RENDERER_SHA256 = (
    "8e34d976481ed81e3b8b285253878f0c02bbfbe177ea608aa51b0f4b594bf1c6"
)
LEVEL_B_RENDERER_SIZE = 404000
STATIC_PREFLIGHT_STDOUT_SHA256 = "2fb4d9d5b1e8e875025260a7287d0c00178e030ac79f1db5d8f2611fcf0618f1"
STATIC_PREFLIGHT_STDOUT_SIZE = 29205
STATIC_PREFLIGHT_PASS_TOKEN = "LEVEL_B_P2_00435_V3_CPU_STATIC_PREFLIGHT_OK"
STATIC_PREFLIGHT_DEVNULL_RPLUS_OPEN_COUNT = 1
STATIC_PREFLIGHT_DEVNULL_WRITE_OPEN_COUNT = 1
STATIC_PREFLIGHT_DEVNULL_OS_OPEN_COUNT = 1
STATIC_PREFLIGHT_DEVNULL_OS_WRITE_OPEN_COUNT = 1
STATIC_PREFLIGHT_BLOCKED_SOCKET_DELEGATION_COUNT = 1
STATIC_PREFLIGHT_STDLIB_SOCKET_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/socket.py"
)
STATIC_PREFLIGHT_STDLIB_SOCKET_SHA256 = (
    "7d4d4c66e6f4bcc961ab462c4f08002ca97def8713a4be1c7373bdbd970a5274"
)
STATIC_PREFLIGHT_STDLIB_SOCKET_SIZE = 37815
STATIC_PREFLIGHT_STDLIB_SOCKET_NLINK = 13
STATIC_PREFLIGHT_URLLIB3_CONNECTION_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/urllib3/util/connection.py"
)
STATIC_PREFLIGHT_URLLIB3_CONNECTION_SHA256 = (
    "2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74"
)
STATIC_PREFLIGHT_URLLIB3_CONNECTION_SIZE = 4444
STATIC_PREFLIGHT_URLLIB3_CONNECTION_NLINK = 1
STATIC_PREFLIGHT_TORCH_JIT_INSTANTIATOR_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/torch/distributed/nn/jit/instantiator.py"
)
STATIC_PREFLIGHT_TORCH_JIT_INSTANTIATOR_SHA256 = (
    "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
)
STATIC_PREFLIGHT_TORCH_JIT_INSTANTIATOR_SIZE = 5510
STATIC_PREFLIGHT_TORCH_REMOTE_MODULE_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/torch/distributed/nn/api/remote_module.py"
)
STATIC_PREFLIGHT_TORCH_REMOTE_MODULE_SHA256 = (
    "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1"
)
STATIC_PREFLIGHT_TORCH_REMOTE_MODULE_SIZE = 31251
STATIC_PREFLIGHT_TORCH_JIT_SENTINEL = (
    "/nonexistent/bernini-level-b-static-preflight-torch-jit"
)
STATIC_PREFLIGHT_NUMPY_CORE_INIT_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/numpy/core/__init__.py"
)
STATIC_PREFLIGHT_NUMPY_CORE_INIT_SHA256 = (
    "08db0ef806f8cb03365b3dc06ea58e1f78a0d6ae419e8f4fb1432b0aff87352e"
)
STATIC_PREFLIGHT_NUMPY_CORE_INIT_SIZE = 5780
STATIC_PREFLIGHT_VEOMNI_LOGGING_PATH = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11/"
    "veomni/utils/logging.py"
)
STATIC_PREFLIGHT_VEOMNI_LOGGING_SHA256 = (
    "91a613a68a5a32b239900bd72cfdf5d172996fec37bf67a69b0cefa699c9fc5a"
)
STATIC_PREFLIGHT_VEOMNI_LOGGING_SIZE = 5246
STATIC_PREFLIGHT_BOTOCORE_SIX_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/botocore/vendored/six.py"
)
STATIC_PREFLIGHT_BOTOCORE_SIX_SHA256 = (
    "4ce39f422ee71467ccac8bed76beb05f8c321c7f0ceda9279ae2dfa3670106b3"
)
STATIC_PREFLIGHT_BOTOCORE_SIX_SIZE = 34549
STATIC_PREFLIGHT_SIX_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/six.py"
)
STATIC_PREFLIGHT_SIX_SHA256 = (
    "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df"
)
STATIC_PREFLIGHT_SIX_SIZE = 34703

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
    / "00435ad621c44fac_p2_seed2026080821_v3.mp4"
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
        "STATIC_PREFLIGHT_STDOUT_SHA256": STATIC_PREFLIGHT_STDOUT_SHA256,
    }
    bad = sorted(
        name
        for name, value in pending.items()
        if not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
    )
    if (
        bad
        or type(LEVEL_B_RENDERER_SIZE) is not int
        or LEVEL_B_RENDERER_SIZE <= 0
        or type(STATIC_PREFLIGHT_STDOUT_SIZE) is not int
        or STATIC_PREFLIGHT_STDOUT_SIZE <= 0
    ):
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


def _install_static_preflight_read_only_guard() -> Mapping[str, Any]:
    """Fail closed on mutation, process creation, or network during preflight."""

    write_open_flags = (
        os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    )
    devnull_path = Path("/dev/null")
    devnull_info = devnull_path.lstat()
    devnull_rplus_flags = os.O_RDWR | os.O_CLOEXEC
    devnull_write_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC
    if (
        devnull_path.resolve(strict=True) != devnull_path
        or not stat.S_ISCHR(devnull_info.st_mode)
        or devnull_info.st_mode != 0o20666
        or stat.S_IMODE(devnull_info.st_mode) != 0o666
        or devnull_info.st_nlink != 1
        or devnull_info.st_rdev != 259
        or os.major(devnull_info.st_rdev) != 1
        or os.minor(devnull_info.st_rdev) != 3
        or devnull_rplus_flags != 524290
        or devnull_write_flags != 524865
    ):
        fail("CPU static preflight /dev/null identity differs")

    def stable_source(
        path: Path,
        *,
        expected_sha256: str,
        expected_size: int,
        expected_nlink: int = 1,
        label: str,
    ) -> Mapping[str, Any]:
        source = _plain_file(path, nlink=expected_nlink)
        before = source.lstat()
        raw = source.read_bytes()
        after = source.lstat()
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_nlink != expected_nlink
            or after.st_nlink != expected_nlink
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(raw) != expected_size
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            fail(f"CPU static preflight {label} source differs")
        return {
            "path": str(source),
            "sha256": expected_sha256,
            "size": expected_size,
            "nlink": expected_nlink,
        }

    socket_source = stable_source(
        STATIC_PREFLIGHT_STDLIB_SOCKET_PATH,
        expected_sha256=STATIC_PREFLIGHT_STDLIB_SOCKET_SHA256,
        expected_size=STATIC_PREFLIGHT_STDLIB_SOCKET_SIZE,
        expected_nlink=STATIC_PREFLIGHT_STDLIB_SOCKET_NLINK,
        label="stdlib socket",
    )
    urllib3_source = stable_source(
        STATIC_PREFLIGHT_URLLIB3_CONNECTION_PATH,
        expected_sha256=STATIC_PREFLIGHT_URLLIB3_CONNECTION_SHA256,
        expected_size=STATIC_PREFLIGHT_URLLIB3_CONNECTION_SIZE,
        expected_nlink=STATIC_PREFLIGHT_URLLIB3_CONNECTION_NLINK,
        label="urllib3 connection",
    )
    guard_state = {
        "sealed": False,
        "devnull_rplus_open_count": 0,
        "devnull_write_open_count": 0,
        "devnull_os_open_count": 0,
        "devnull_os_write_open_count": 0,
        "blocked_socket_probe_delegation_count": 0,
        "socket_source": socket_source,
        "urllib3_connection_source": urllib3_source,
        "devnull": {
            "path": "/dev/null",
            "st_mode": int(devnull_info.st_mode),
            "mode": int(stat.S_IMODE(devnull_info.st_mode)),
            "nlink": int(devnull_info.st_nlink),
            "st_rdev": int(devnull_info.st_rdev),
            "major": int(os.major(devnull_info.st_rdev)),
            "minor": int(os.minor(devnull_info.st_rdev)),
            "character_device": True,
        },
    }
    mutation_events = frozenset(
        {
            "os.chdir",
            "os.chmod",
            "os.chown",
            "os.link",
            "os.mkdir",
            "os.mknod",
            "os.putenv",
            "os.remove",
            "os.removexattr",
            "os.rename",
            "os.rmdir",
            "os.setxattr",
            "os.symlink",
            "os.truncate",
            "os.unsetenv",
            "os.utime",
            "shutil.copyfile",
            "shutil.copymode",
            "shutil.copystat",
            "shutil.make_archive",
            "shutil.move",
            "shutil.rmtree",
        }
    )
    process_events = frozenset(
        {
            "os.exec",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.spawn",
            "os.system",
            "pty.spawn",
            "subprocess.Popen",
        }
    )
    network_events = frozenset(
        {
            "socket.connect",
            "socket.connect_ex",
            "socket.__new__",
            "socket.bind",
            "socket.getaddrinfo",
            "socket.gethostbyaddr",
            "socket.gethostbyname",
            "socket.getnameinfo",
            "socket.listen",
            "socket.sendmsg",
            "socket.sendto",
        }
    )

    def is_expected_blocked_socket_probe(arguments: Tuple[Any, ...]) -> bool:
        if (
            guard_state["sealed"] is not False
            or guard_state["blocked_socket_probe_delegation_count"] != 0
            or len(arguments) != 4
            or any(type(value) is not int for value in arguments[1:])
            or arguments[1:] != (10, 1, 0)
        ):
            return False
        socket_module = sys.modules.get("socket")
        urllib3_module = sys.modules.get("urllib3.util.connection")
        socket_value = arguments[0]
        socket_class = getattr(socket_module, "socket", None)
        socket_init = getattr(socket_class, "__init__", None)
        has_ipv6 = getattr(urllib3_module, "_has_ipv6", None)
        try:
            socket_frame = sys._getframe(2)
            urllib3_frame = socket_frame.f_back
            module_frame = urllib3_frame.f_back if urllib3_frame is not None else None
            socket_state = (
                socket_value.fileno(),
                socket_value.family,
                socket_value.type,
                socket_value.proto,
            )
        except Exception:
            return False
        return bool(
            socket_module is not None
            and urllib3_module is not None
            and type(socket_value) is socket_class
            and socket_state == (-1, 0, 0, 0)
            and type(socket_init) is FunctionType
            and socket_frame.f_code is socket_init.__code__
            and socket_frame.f_lineno == 233
            and socket_init.__module__ == "socket"
            and socket_init.__qualname__ == "socket.__init__"
            and socket_init.__code__.co_firstlineno == 221
            and tuple(socket_init.__code__.co_freevars) == ()
            and socket_frame.f_globals is vars(socket_module)
            and socket_frame.f_code.co_filename == str(STATIC_PREFLIGHT_STDLIB_SOCKET_PATH)
            and type(has_ipv6) is FunctionType
            and urllib3_frame is not None
            and urllib3_frame.f_code is has_ipv6.__code__
            and urllib3_frame.f_lineno == 126
            and has_ipv6.__module__ == "urllib3.util.connection"
            and has_ipv6.__qualname__ == "_has_ipv6"
            and has_ipv6.__code__.co_firstlineno == 114
            and tuple(has_ipv6.__code__.co_freevars) == ()
            and urllib3_frame.f_globals is vars(urllib3_module)
            and urllib3_frame.f_code.co_filename
            == str(STATIC_PREFLIGHT_URLLIB3_CONNECTION_PATH)
            and module_frame is not None
            and module_frame.f_globals is vars(urllib3_module)
            and module_frame.f_code.co_name == "<module>"
            and getattr(module_frame.f_code, "co_qualname", "<module>") == "<module>"
            and module_frame.f_code.co_firstlineno == 1
            and tuple(module_frame.f_code.co_freevars) == ()
            and module_frame.f_lineno == 137
            and module_frame.f_code.co_filename
            == str(STATIC_PREFLIGHT_URLLIB3_CONNECTION_PATH)
        )

    def audit(event: str, args: Tuple[Any, ...]) -> None:
        if event == "open":
            path = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            write_requested = (
                isinstance(mode, str)
                and any(token in mode for token in ("w", "a", "x", "+"))
            ) or (isinstance(flags, int) and bool(flags & write_open_flags))
            if write_requested:
                allowed_key = (path, mode, flags)
                if guard_state["sealed"] is not False:
                    fail("CPU static preflight write-open occurred after guard seal")
                if allowed_key == ("/dev/null", "r+", devnull_rplus_flags):
                    guard_state["devnull_rplus_open_count"] += 1
                    if (
                        guard_state["devnull_rplus_open_count"]
                        > STATIC_PREFLIGHT_DEVNULL_RPLUS_OPEN_COUNT
                    ):
                        fail("CPU static preflight /dev/null r+ count differs")
                    return
                if allowed_key == ("/dev/null", "w", devnull_write_flags):
                    guard_state["devnull_write_open_count"] += 1
                    if (
                        guard_state["devnull_write_open_count"]
                        > STATIC_PREFLIGHT_DEVNULL_WRITE_OPEN_COUNT
                    ):
                        fail("CPU static preflight /dev/null write count differs")
                    return
                if allowed_key == ("/dev/null", None, devnull_rplus_flags):
                    guard_state["devnull_os_open_count"] += 1
                    if (
                        guard_state["devnull_os_open_count"]
                        > STATIC_PREFLIGHT_DEVNULL_OS_OPEN_COUNT
                    ):
                        fail("CPU static preflight /dev/null os.open count differs")
                    return
                if allowed_key == ("/dev/null", None, devnull_write_flags):
                    guard_state["devnull_os_write_open_count"] += 1
                    if (
                        guard_state["devnull_os_write_open_count"]
                        > STATIC_PREFLIGHT_DEVNULL_OS_WRITE_OPEN_COUNT
                    ):
                        fail("CPU static preflight /dev/null os.write count differs")
                    return
                fail("CPU static preflight attempted a persistent filesystem write")
        if event in mutation_events:
            fail("CPU static preflight attempted a filesystem mutation")
        if event in process_events:
            fail("CPU static preflight attempted to spawn or replace a process")
        if event == "socket.__new__" and is_expected_blocked_socket_probe(args):
            # Registration order is deliberate: this outer hook authenticates
            # the sole import-time capability probe, then delegates it to the
            # already-authenticated renderer hook registered later.  That hook
            # must raise before socket initialization and bind the blocked
            # attempt in its receipt; every other network event is rejected
            # here without delegation.
            guard_state["blocked_socket_probe_delegation_count"] += 1
            return
        if event in network_events:
            fail("CPU static preflight attempted network access")

    sys.addaudithook(audit)
    return guard_state


def _call_static_preflight_owner_without_stdout(function: Any) -> Any:
    """Run authenticated imports with a byte-capable, zero-output stdout sink."""

    previous_stdout = sys.stdout
    captured_bytes = io.BytesIO()
    captured_stdout = io.TextIOWrapper(
        captured_bytes,
        encoding="utf-8",
        errors="strict",
        newline="\n",
        write_through=True,
    )
    stdout_tampered = False
    try:
        sys.stdout = captured_stdout
        result = function()
        stdout_tampered = sys.stdout is not captured_stdout
    finally:
        if sys.stdout is not captured_stdout:
            stdout_tampered = True
        sys.stdout = previous_stdout
    try:
        captured_stdout.flush()
        captured = captured_bytes.getvalue()
    except (ValueError, OSError) as error:
        raise BootstrapError("CPU static preflight vendor stdout sink changed") from error
    if stdout_tampered or sys.stdout is not previous_stdout or captured:
        fail("CPU static preflight vendor import wrote or replaced stdout")
    return result


def static_preflight() -> Mapping[str, Any]:
    """Authenticate the sealed CPU call graph without weights, CUDA, or writes."""

    _require_finalized()
    required_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OPENBLAS_MAIN_FREE": "1",
        "GOTOBLAS_MAIN_FREE": "1",
        "VEOMNI_VERBOSITY": "ERROR",
        "HOME": "/nonexistent/bernini-level-b-p2-00435-v3",
        "XDG_CACHE_HOME": "/nonexistent/bernini-level-b-p2-00435-v3/cache",
        "HF_HOME": "/nonexistent/bernini-level-b-p2-00435-v3/huggingface",
        "TRANSFORMERS_CACHE": "/nonexistent/bernini-level-b-p2-00435-v3/transformers",
        "TMPDIR": "/nonexistent/bernini-level-b-p2-00435-v3/tmp",
    }
    if any(os.environ.get(key) != value for key, value in required_environment.items()):
        fail("CPU static preflight environment differs")
    if (
        sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.no_user_site != 1
        or not sys.dont_write_bytecode
        or any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("torch", "diffusers", "transformers", "bernini", "veomni")
        )
    ):
        fail("CPU static preflight process isolation differs")

    guard_state = _install_static_preflight_read_only_guard()
    _verify_nonsemantic_authorities()

    def authenticated_owner_call() -> Tuple[ModuleType, Mapping[str, Any]]:
        level_b_module = _authenticate_level_b_module()
        verified_runtime = level_b_module.authenticate_level_b_runtime_release(
            LEVEL_B_MANIFEST
        )
        receipt = level_b_module.run_level_b_cpu_static_runtime_preflight(
            verified_runtime=verified_runtime
        )
        return level_b_module, receipt

    level_b, renderer_receipt = _call_static_preflight_owner_without_stdout(
        authenticated_owner_call
    )
    required_true = (
        "complete",
        "exact_five_release_authenticated",
        "all_vendor_source_pins_rehashed",
        "vae_loader_wrappers_authenticated",
        "vae_apply_forward_hook_and_annotations_authenticated",
        "tokenizer_factory_and_bound_call_authenticated",
        "renderer_and_diffusion_no_grad_layers_authenticated",
        "shared_step_signature_authenticated",
        "network_probe_blocked_before_socket_creation",
        "blas_import_environment_preseeded_before_vendor_imports",
        "veomni_logging_environment_preseeded_before_vendor_imports",
        "python_bytecode_writes_disabled_during_import",
        "pinned_bernini_and_veomni_roots_scoped_and_restored",
    )
    required_false = (
        "cuda_initialized_before",
        "cuda_initialized_after",
        "weights_loaded",
        "model_constructors_called",
        "product_output_writes",
        "persistent_filesystem_writes",
        "subprocesses_spawned",
        "network_accessed",
        "blas_import_environment_mutations_allowed",
        "preexisting_bernini_or_veomni_modules_accepted",
    )
    process_guard = (
        renderer_receipt.get("process_audit_guard")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    tempfile_scope = (
        renderer_receipt.get("tempfile_import_probe_suppression")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    jit_suppression = (
        renderer_receipt.get("torch_jit_temporary_directory_suppression")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    remote_suppression = (
        renderer_receipt.get("torch_remote_module_template_suppression")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    numpy_environment_seal = (
        renderer_receipt.get("numpy_blas_import_environment_seal")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    veomni_logging_seal = (
        renderer_receipt.get("veomni_stdout_logging_suppression")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    scoped_module_closure = (
        renderer_receipt.get("scoped_module_source_closure")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    six_importer_scope = (
        renderer_receipt.get("six_meta_path_importer_scope")
        if isinstance(renderer_receipt, Mapping)
        else None
    )
    scoped_closure_authority = tuple(
        getattr(level_b, "PINNED_CPU_STATIC_SCOPED_MODULE_CLOSURE", ())
    )
    if (
        len(scoped_closure_authority) != 52
        or any(type(row) is not tuple or len(row) != 5 for row in scoped_closure_authority)
    ):
        fail("CPU static preflight scoped module authority differs")
    expected_scoped_rows = [
        {
            "module": module_name,
            "prefix": prefix,
            "relative_path": relative,
            "source_path": str(
                (BERNINI_ROOT if prefix == "bernini" else VEOMNI_ROOT) / relative
            ),
            "sha256": expected_sha,
            "size": expected_size,
        }
        for module_name, prefix, relative, expected_sha, expected_size
        in scoped_closure_authority
    ]
    expected_scoped_digest = hashlib.sha256(
        canonical_json_bytes(expected_scoped_rows)
    ).hexdigest()
    expected_six_rows = [
        {
            "live_index_without_owned_finder": 4,
            "module": "botocore.vendored.six",
            "type_module": "botocore.vendored.six",
            "type_qualname": "_SixMetaPathImporter",
            "module_export_identity_verified": True,
            "module_file_and_spec_origin_exact": True,
            "source_path": str(STATIC_PREFLIGHT_BOTOCORE_SIX_PATH),
            "source_sha256": STATIC_PREFLIGHT_BOTOCORE_SIX_SHA256,
            "source_size": STATIC_PREFLIGHT_BOTOCORE_SIX_SIZE,
        },
        {
            "live_index_without_owned_finder": 5,
            "module": "six",
            "type_module": "six",
            "type_qualname": "_SixMetaPathImporter",
            "module_export_identity_verified": True,
            "module_file_and_spec_origin_exact": True,
            "source_path": str(STATIC_PREFLIGHT_SIX_PATH),
            "source_sha256": STATIC_PREFLIGHT_SIX_SHA256,
            "source_size": STATIC_PREFLIGHT_SIX_SIZE,
        },
    ]
    expected_six_digest = hashlib.sha256(
        canonical_json_bytes(expected_six_rows)
    ).hexdigest()
    if (
        not isinstance(renderer_receipt, Mapping)
        or renderer_receipt.get("schema_version")
        != level_b.CPU_STATIC_PREFLIGHT_SCHEMA
        or renderer_receipt.get("authority") != AUTHORITY
        or any(renderer_receipt.get(key) is not True for key in required_true)
        or any(renderer_receipt.get(key) is not False for key in required_false)
        or scoped_module_closure
        != {
            "exact_module_count": 52,
            "exact_bernini_module_count": 13,
            "exact_veomni_module_count": 39,
            "rows": expected_scoped_rows,
            "digest": expected_scoped_digest,
            "module_file_and_spec_origin_exact": True,
            "no_process_specific_repr_or_object_address_recorded": True,
        }
        or six_importer_scope
        != {
            "snapshot_count": 4,
            "owned_finder_live_index": 0,
            "addition_count": 2,
            "deletion_count": 0,
            "original_snapshot_identity_order_preserved": True,
            "additions": expected_six_rows,
            "digest": expected_six_digest,
            "exact_append_order_verified": True,
            "exact_reverse_removal_verified": True,
            "restored_exactly": True,
            "repr_or_object_address_in_receipt": False,
        }
        or renderer_receipt.get("accelerator_visibility_environment")
        != {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        }
        or renderer_receipt.get("blas_import_environment")
        != {
            "OPENBLAS_MAIN_FREE": "1",
            "GOTOBLAS_MAIN_FREE": "1",
        }
        or renderer_receipt.get("veomni_logging_environment")
        != {"VEOMNI_VERBOSITY": "ERROR"}
        or not isinstance(process_guard, Mapping)
        or process_guard.get("devnull_path") != "/dev/null"
        or process_guard.get("devnull_rdev") != 259
        or process_guard.get("devnull_mode") != 0o666
        or process_guard.get("devnull_major") != 1
        or process_guard.get("devnull_minor") != 3
        or process_guard.get("devnull_open_counts")
        != {
            "python-open-r+": 1,
            "python-open-w": 1,
            "os-open-O_RDWR": 1,
            "os-open-O_WRONLY|O_CREAT|O_TRUNC": 1,
        }
        or process_guard.get("persistent_filesystem_writes") is not False
        or process_guard.get("model_weight_reads") is not False
        or process_guard.get("subprocesses_spawned") is not False
        or process_guard.get("blocked_network_probe_count") != 1
        or process_guard.get("socket_objects_created") is not False
        or process_guard.get("network_accessed") is not False
        or tempfile_scope
        != {
            "guard_installed_before_tempfile_import": True,
            "prior_tempfile_tempdir_was_none": True,
            "sentinel_path": "/nonexistent",
            "sentinel_lstat_was_enoent": True,
            "scoped_tempdir_restored_by_identity": True,
        }
        or jit_suppression
        != {
            "source_path": str(STATIC_PREFLIGHT_TORCH_JIT_INSTANTIATOR_PATH),
            "source_sha256": STATIC_PREFLIGHT_TORCH_JIT_INSTANTIATOR_SHA256,
            "source_size": STATIC_PREFLIGHT_TORCH_JIT_INSTANTIATOR_SIZE,
            "exact_meta_path_finder_calls": 1,
            "exact_loader_create_module_calls": 1,
            "exact_loader_exec_module_calls": 1,
            "exact_remote_finder_calls": 1,
            "exact_remote_loader_create_module_calls": 1,
            "exact_remote_loader_exec_module_calls": 1,
            "meta_path_restored_exactly": True,
            "path_importer_cache_restored_exactly": True,
            "jit_parent_importer_cache_was_missing_then_added": True,
            "remote_parent_importer_cache_was_added": True,
            "module_loader_and_spec_loader_restored": True,
            "constructor_call_line": 21,
            "constructor_calls_suppressed": 1,
            "atexit_registration_line": 23,
            "atexit_registrations_suppressed": 1,
            "atexit_interception_only_during_exact_module_exec": True,
            "cleanup_calls_observed": 0,
            "fixed_sentinel_name": STATIC_PREFLIGHT_TORCH_JIT_SENTINEL,
            "fixed_sentinel_lstat_was_enoent": True,
            "sys_path_append_removed": True,
            "temporary_directory_class_restored_by_identity": True,
            "atexit_register_restored_by_identity": True,
            "real_directory_created": False,
            "real_atexit_handler_registered": False,
        }
        or remote_suppression
        != {
            "source_path": str(STATIC_PREFLIGHT_TORCH_REMOTE_MODULE_PATH),
            "source_sha256": STATIC_PREFLIGHT_TORCH_REMOTE_MODULE_SHA256,
            "source_size": STATIC_PREFLIGHT_TORCH_REMOTE_MODULE_SIZE,
            "module_call_line": 30,
            "instantiator_factory_firstlineno": 143,
            "factory_calls_suppressed": 1,
            "module_global_is_exact_opaque_sentinel": True,
            "factory_restored_by_identity": True,
            "template_source_written": False,
            "generated_template_imported": False,
        }
        or numpy_environment_seal
        != {
            "source_path": str(STATIC_PREFLIGHT_NUMPY_CORE_INIT_PATH),
            "source_sha256": STATIC_PREFLIGHT_NUMPY_CORE_INIT_SHA256,
            "source_size": STATIC_PREFLIGHT_NUMPY_CORE_INIT_SIZE,
            "required_environment": {
                "OPENBLAS_MAIN_FREE": "1",
                "GOTOBLAS_MAIN_FREE": "1",
            },
            "preseeded_before_vendor_imports": True,
            "putenv_and_unsetenv_audit_events_allowed": False,
            "process_environment_mutations_observed": 0,
        }
        or veomni_logging_seal
        != {
            "source_path": str(STATIC_PREFLIGHT_VEOMNI_LOGGING_PATH),
            "source_sha256": STATIC_PREFLIGHT_VEOMNI_LOGGING_SHA256,
            "source_size": STATIC_PREFLIGHT_VEOMNI_LOGGING_SIZE,
            "required_environment": {"VEOMNI_VERBOSITY": "ERROR"},
            "preseeded_before_vendor_imports": True,
            "putenv_and_unsetenv_audit_events_allowed": False,
            "timestamped_info_output_enabled": False,
        }
        or _SHA256.fullmatch(str(renderer_receipt.get("contract_digest"))) is None
        or renderer_receipt.get("preflight_digest")
        != hashlib.sha256(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in renderer_receipt.items()
                    if key != "preflight_digest"
                }
            )
        ).hexdigest()
    ):
        fail("CPU static renderer preflight receipt differs")
    torch_module = sys.modules.get("torch")
    if (
        torch_module is None
        or bool(torch_module.cuda.is_initialized())
        or not sys.dont_write_bytecode
    ):
        fail("CPU static preflight initialized CUDA or enabled bytecode writes")
    if (
        guard_state.get("devnull_rplus_open_count")
        != STATIC_PREFLIGHT_DEVNULL_RPLUS_OPEN_COUNT
        or guard_state.get("devnull_write_open_count")
        != STATIC_PREFLIGHT_DEVNULL_WRITE_OPEN_COUNT
        or guard_state.get("devnull_os_open_count")
        != STATIC_PREFLIGHT_DEVNULL_OS_OPEN_COUNT
        or guard_state.get("devnull_os_write_open_count")
        != STATIC_PREFLIGHT_DEVNULL_OS_WRITE_OPEN_COUNT
        or guard_state.get("blocked_socket_probe_delegation_count")
        != STATIC_PREFLIGHT_BLOCKED_SOCKET_DELEGATION_COUNT
    ):
        fail("CPU static preflight exact capability-probe count differs")
    guard_state["sealed"] = True
    devnull_receipt = {
        **dict(guard_state["devnull"]),
        "allowed_open_events": [
            {
                "mode": "r+",
                "flags": 524290,
                "exact_count": STATIC_PREFLIGHT_DEVNULL_RPLUS_OPEN_COUNT,
            },
            {
                "mode": "w",
                "flags": 524865,
                "exact_count": STATIC_PREFLIGHT_DEVNULL_WRITE_OPEN_COUNT,
            },
            {
                "mode": None,
                "flags": 524290,
                "exact_count": STATIC_PREFLIGHT_DEVNULL_OS_OPEN_COUNT,
            },
            {
                "mode": None,
                "flags": 524865,
                "exact_count": STATIC_PREFLIGHT_DEVNULL_OS_WRITE_OPEN_COUNT,
            },
        ],
        "discard_only_device": True,
        "persistent_filesystem_writes": False,
    }
    blocked_socket_receipt = {
        "outer_delegation_count": int(
            guard_state["blocked_socket_probe_delegation_count"]
        ),
        "owner_blocked_attempt_count": int(
            process_guard["blocked_network_probe_count"]
        ),
        "event": "socket.__new__",
        "arguments_tail": [10, 1, 0],
        "socket_preinit_state": [-1, 0, 0, 0],
        "stdlib_socket_source": dict(guard_state["socket_source"]),
        "socket_init_first_line": 221,
        "socket_init_event_line": 233,
        "urllib3_connection_source": dict(
            guard_state["urllib3_connection_source"]
        ),
        "urllib3_has_ipv6_first_line": 114,
        "urllib3_has_ipv6_event_line": 126,
        "urllib3_module_event_line": 137,
        "socket_objects_created": False,
        "network_accessed": False,
    }

    unsigned = {
        "schema_version": "bernini-action-edit-level-b-p2-cpu-static-preflight-v3",
        "method": METHOD,
        "authority": AUTHORITY,
        "tag": TAG,
        "pass_token": STATIC_PREFLIGHT_PASS_TOKEN,
        "sealed_release_manifest_sha256": LEVEL_B_MANIFEST_SHA256,
        "renderer_sha256": LEVEL_B_RENDERER_SHA256,
        "renderer_preflight": dict(renderer_receipt),
        "blas_import_environment": {
            "OPENBLAS_MAIN_FREE": required_environment["OPENBLAS_MAIN_FREE"],
            "GOTOBLAS_MAIN_FREE": required_environment["GOTOBLAS_MAIN_FREE"],
        },
        "filesystem_mutation_audit_guard_installed": True,
        "process_creation_audit_guard_installed": True,
        "network_audit_guard_installed": True,
        "vendor_stdout_captured_in_memory": True,
        "vendor_stdout_bytes_observed": 0,
        "vendor_stdout_identity_restored": True,
        "ephemeral_devnull_type_probe": devnull_receipt,
        "blocked_network_capability_probe": blocked_socket_receipt,
        "persistent_filesystem_writes": False,
        "caller_semantic_inputs_present": False,
        "output_files_written": False,
        "weights_loaded": False,
        "cuda_initialized": False,
        "network_accessed": False,
    }
    result = {
        **unsigned,
        "preflight_digest": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    raw = canonical_json_bytes(result)
    if (
        len(raw) != STATIC_PREFLIGHT_STDOUT_SIZE
        or hashlib.sha256(raw).hexdigest() != STATIC_PREFLIGHT_STDOUT_SHA256
    ):
        fail("CPU static preflight canonical stdout differs from frozen authority")
    return result


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
        "schema_version": "bernini-action-edit-level-b-p2-product-validation-v3",
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
    if values == ["static-preflight"]:
        sys.stdout.buffer.write(canonical_json_bytes(static_preflight()))
        return 0
    fail(
        "bootstrap accepts exactly one fixed subcommand: "
        "run, validate-product, or static-preflight"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as error:
        print(f"Level-B P2 bootstrap refused: {error}", file=sys.stderr, flush=True)
        raise SystemExit(94)
