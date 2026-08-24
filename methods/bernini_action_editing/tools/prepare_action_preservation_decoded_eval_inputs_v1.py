#!/usr/bin/env python3
"""Prepare, but never execute, the two-stage preservation-v2 eval inputs.

This helper is deliberately outside the frozen exact14 release.  It resolves
and replays the intended inputs for these operations:

* ``check`` replays the completed r5 training tree and the static source/model
  authorities without reading loss for selection;
* ``phase-a`` would publish one create-only deployment request whose runtime
  outputs are still absent; and
* ``phase-b`` is explicitly unavailable for exact14.

Both publication phases are currently stopped by an unconditional release
gate: exact14 does not prove the physical model bytes actually reopened by
the production decoder.  The gate has no command-line override and runs
before phase-A creates any path.  A successor release must close that P0 and
receive new independently audited pins before this helper can be superseded.

It does not execute the detached controller, torchrun, srun, inference,
training, upload, retry, promotion, or a model/evaluator.  In particular, a
file SHA printed by this program is only a preparation result: the detached
controller must still receive a separately reviewed literal SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "bernini-action-preservation-decoded-eval-deployment-request-v2"
DEPLOYMENT_RECEIPT_SCHEMA = (
    "bernini-action-preservation-decoded-eval-deployment-receipt-v2"
)
SOURCE_RUNTIME_SCHEMA = "bernini-action-preservation-source-runtime-spec-v2"
TRAINING_COMPLETE_SCHEMA = "bernini-action-preservation-v2-training-complete-v3"
TRAINING_AUDIT_SCHEMA_FIELDS = frozenset(
    {
        "training_audit_go",
        "arm_count",
        "checkpoint_count",
        "checkpoint_steps",
        "route_scopes",
        "initialization_digest_by_scope",
        "checkpoint_zero_adapter_sha256_by_scope",
        "adapter_config_sha256_by_scope",
        "receipt_rows",
        "decoded_evaluation_complete",
        "scientific_promotion_authorized",
    }
)
ACTION_REVIEW_CONTRACT_SCHEMA = (
    "bernini-action-preservation-action-review-contract-v1"
)
ACTION_INTERVAL_SEMANTICS = (
    "preregistered_evaluation_acceptance_windows_not_observed_ground_truth"
)
CONTROLLER_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-controller-authority-v1"
)

# P0: exact14 binds a 23-row checkpoint manifest and a symbolic tree digest,
# but the production decoder/infer path does not recompute those rows (or
# consume a physical-authority receipt) when Transformers/Diffusers reopen the
# model files.  There is intentionally no CLI override for this gate.  A
# successor release must enforce the consumed model bytes before this helper
# may publish either phase.
MODEL_CONSUMPTION_AUTHORITY_ENFORCED_BY_PRODUCTION = False
MODEL_CONSUMPTION_AUTHORITY_P0 = (
    "P0: exact14 does not enforce same-run physical authority for every model "
    "file reopened by Bernini/Transformers/Diffusers; the pinned 23-row model "
    "manifest is not replayed by the production decoder, so decoded evaluation "
    "input publication is blocked pending a superseding production release"
)

RELEASE_GENERATION = "preservation-v2-decoded-eval-exact14-r1"
EVAL_ARCHIVE_SHA256 = (
    "60987b0ec72c3ea04cdb83e4afb031c59fe60ffc0138c412685a510bbdfd9f58"
)
EVAL_MANIFEST_SHA256 = (
    "644da39cfa9300003be6d900838fd21e2541159402b893800c8ee387e7755443"
)
EVAL_MANIFEST_DIGEST = (
    "79948953ad3f7a0825a81e8250233b429eecd6577d2a065118460154da569186"
)
EVAL_CONTENT_REVISION = "4c06ef3f6e1fc665d2b3917a249c24b7e184bb6f"
EVAL_ENVELOPE_SHA256 = (
    "fbc18b9dc5fa7f26f35a02801ed982b41884340995078e528b18152101d0ea93"
)
EVAL_ENVELOPE_DIGEST = (
    "a4175d64146c37ec86ca38b54db0673d198107df70e97dde44599adf09f9f961"
)
DETACHED_CONTROLLER_SHA256 = (
    "99003b6b942d286c44adda9a66a0c8f99d2fb7dfed72b7eb95134937c7386cf8"
)
VERIFIED_RUNTIME_SHA256 = (
    "4fb1b7bf474951cfe4d8fa9a083ab0b1caf17e5b981fa73faa156547945812ac"
)
INFER_LORA_SHA256 = (
    "3dd890e60d4427fefd8a9619fc8b918210b88670737a470fd53dae5538e5cead"
)
TRAIN_LORA_SHA256 = (
    "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e"
)
DECODER_ADAPTER_SHA256 = (
    "d73d1f0a5aae8dbf9359a552bd24f9806250a14a5460031eec5883f3f20fdd45"
)

R5_SOURCE_REVISION = "ab699d9bf7e1645289907e397f129e51c535779c"
R5_SOURCE_ARCHIVE_SHA256 = (
    "c302c3af2f144f6bf68a841e8cbe98545f4738b416c619cc3e6a708dc583929e"
)
R5_RELEASE_MANIFEST_SHA256 = (
    "9754bbe1a39f936ccbc6a96c3a91acbe7a07e19d78f72cf3877a50a3dbba56ab"
)
R5_CONTROLLER_SHA256 = (
    "d522fa711014a5ca5b671448ce24afab14e3dbf63fd9df45b0112745a01dd995"
)
R5_DEPLOYMENT_ENVELOPE_SHA256 = (
    "472601440735a2963c484ecb930c94851296b46ac11632e961d2e4de25e92923"
)
SOURCE_MANIFEST_SHA256 = (
    "62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8"
)
SOURCE_MANIFEST_DIGEST = (
    "2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503"
)
MODEL_RELEASE_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
INFERENCE_CONFIG_SHA256 = (
    "4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496"
)

ROOT_PYTHON_SHA256 = (
    "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
)
FROZEN_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
TORCHRUN_SHA256 = (
    "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c"
)
FFPROBE_SHA256 = (
    "d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd"
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)

EVAL_RELEASE_MEMBER_ROOT = "methods/bernini_action_editing"
EVAL_RELEASE_MEMBERS = (
    "infer_lora.py",
    "train_lora.py",
    "self_generated_action_preservation_v2.py",
    "action_preservation_gate_v1.py",
    "action_preservation_decoded_eval_plan_v1.py",
    "action_preservation_decoded_eval_bridge_v1.py",
    "action_preservation_decoded_eval_decoder_adapter_v1.py",
    "action_preservation_decoded_eval_executor_v1.py",
    "action_preservation_decoded_eval_launcher_v1.py",
    "action_preservation_decoded_eval_aggregate_v1.py",
    "action_preservation_loop_controller_v1.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "action_preservation_decoded_eval_verified_release_v1.py",
)
EVAL_EXECUTABLE_MEMBER = (
    "action_preservation_decoded_eval_decoder_adapter_v1.py"
)

CAPTURED_FILE_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size",
        "mode",
        "device",
        "inode",
        "uid",
        "gid",
        "nlink",
        "rdev",
        "blocks",
        "mtime_ns",
        "ctime_ns",
    }
)
CAPTURED_DIRECTORY_FIELDS = frozenset(CAPTURED_FILE_FIELDS - {"sha256"})

REMOTE_BASE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
R5_EXPERIMENT_ROOT = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_r5"
)
R5_RELEASE_ROOT = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_release_"
    "ab699d9b_checkpointreadme1"
)
SOURCE_MANIFEST_PATH = (
    REMOTE_BASE / "action_quotient_job140846_v4/source_only/manifest.json"
)
EVAL_DEPLOYMENT_BUNDLE = REMOTE_BASE / (
    "action_preservation_v2_decoded_eval_deployment_4c06ef3f_99003b6b_r1"
)
EVAL_ARTIFACTS_ROOT = EVAL_DEPLOYMENT_BUNDLE / "exact14-release"
DETACHED_CONTROLLER_PATH = (
    EVAL_DEPLOYMENT_BUNDLE
    / "action_preservation_decoded_eval_deployment_controller_v1.py"
)
VERIFIED_RUNTIME_SOURCE_PATH = (
    REMOTE_BASE
    / "action_preservation_v2_decoded_eval_preverify_4c06ef3f_r1"
    / "methods/bernini_action_editing/"
    "action_preservation_decoded_eval_verified_release_v1.py"
)

ROOT_PYTHON_PATH = Path("/usr/bin/python3.10")
FROZEN_PYTHON_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
SITE_PACKAGES_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
)
TORCHRUN_PATH = SITE_PACKAGES_PATH / "torch/distributed/run.py"
FFPROBE_PATH = Path("/usr/bin/ffprobe")
BERNINI_ROOT = (
    REMOTE_BASE / "motive_action_repr_auto/vendor/Bernini-2d2b4591"
)
VEOMNI_ROOT = (
    REMOTE_BASE.parent
    / "VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
)
MODEL_ROOT = (
    REMOTE_BASE.parent
    / "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
MODEL_RELEASE_MANIFEST_PATH = (
    REMOTE_BASE
    / "bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/runtime/"
    "source_ea900d5/methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
INFERENCE_CONFIG_PATH = (
    BERNINI_ROOT / "configs/bernini_renderer_wan21_1p3b/config.json"
)

ARMS = (
    "v2_onset_all",
    "v2_noop020_all",
    "v2_func010_all",
    "v2_func025_all",
    "v2_func050_all",
    "v2_onset_cross_qo",
    "v2_func010_cross_qo",
    "v2_func025_cross_qo",
)
CHECKPOINT_STEPS = (0, 5, 10, 20)
FITTED_IIDS = (
    "7b88a1ca1f804f41",
    "841b5e0080a1441d",
    "a35b590961d24694",
    "a66e6818e4144928",
)

SOURCE_VIDEO_BASE = (
    REMOTE_BASE
    / "goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples"
)
SOURCE_RECEIPT_BASE = (
    REMOTE_BASE
    / "bernini_pair_v5_t2v_core4_v2_20260808_17cc2c7/runs/core4_bank_v2"
)

# These are preregistered evaluation acceptance windows.  They are not
# measurements inferred from generated media and must never be described as
# observed action timing.
EVALUATION_ONSET_FRAME_MIN = 4
EVALUATION_ONSET_FRAME_MAX = 20
EVALUATION_TERMINAL_HOLD_START_FRAME_MIN = 65

SOURCE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "iid": "7b88a1ca1f804f41",
        "source_video_sha256": (
            "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
        ),
        "source_receipt_sha256": (
            "28583d21996bdebc634240c1b53dbe384eba37f4bbff851db1c146f362f04a46"
        ),
        "instruction": (
            "A static camera shows a happy single grey French bulldog in a black "
            "harness, standing in a large pile of vibrant yellow autumn leaves "
            "in a sunlit park, facing the camera with its tongue out while its "
            "chest and the nearby leaves move subtly. The main French bulldog "
            "bends its hind legs, lowers its hips to the ground, settles into a "
            "stable sit facing the camera, and holds that seated pose. The shot "
            "stays continuous, the illumination remains stable, and the final "
            "frame is temporally coherent."
        ),
        "instruction_sha256": (
            "5c8defcffe8413cd556e90c5e17345eb21b51bc787d62d6cde0698cd303cd736"
        ),
        "action_order_description": (
            "The French bulldog lowers from standing into a stable sit facing "
            "the camera, then holds the seated pose through the end."
        ),
        "seed": 2026081801,
    },
    {
        "iid": "841b5e0080a1441d",
        "source_video_sha256": (
            "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a"
        ),
        "source_receipt_sha256": (
            "a4d58c4813f706abe5e4e9cc8d6607dd108642d10d726365283c551c8d389cd8"
        ),
        "instruction": (
            "A fixed camera shows a single black-and-tan German shepherd standing "
            "on a leash in a grassy field, looking upward toward the viewer, with "
            "clumps of shed fur on the ground and subtle ear and breeze movement. "
            "The main German shepherd bends its hind legs, lowers its hips to the "
            "ground, settles into a stable sit facing the camera, and holds that "
            "seated pose. The shot stays continuous, the illumination remains "
            "stable, and the final frame is temporally coherent."
        ),
        "instruction_sha256": (
            "739e74f72516a94df75fc60f32f96662599eefc5f21afd2becd16d437f87157f"
        ),
        "action_order_description": (
            "The German shepherd lowers from standing into a stable sit facing "
            "the camera, then holds the seated pose through the end."
        ),
        "seed": 2026081802,
    },
    {
        "iid": "a35b590961d24694",
        "source_video_sha256": (
            "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed"
        ),
        "source_receipt_sha256": (
            "b15629d0d12489cf9f6ae09f1eedcc2757bcd195739a3b82e51bb2df3b22d000"
        ),
        "instruction": (
            "A static camera shows a single young woman with long brown hair "
            "kneeling on one knee and looking at the camera against a solid warm "
            "yellow background, wearing a black jacket, pink top, cream trousers, "
            "and breathing gently. The main person shifts weight onto both feet, "
            "rises smoothly from kneeling, straightens both legs and the torso, "
            "and holds a stable upright standing pose. The shot stays continuous, "
            "the illumination remains stable, and the final frame is temporally "
            "coherent."
        ),
        "instruction_sha256": (
            "472f6bf16e90c4fa8a6538a9c34f02946ef70278fe5d7916c425f76a6581826b"
        ),
        "action_order_description": (
            "The woman transfers weight from kneeling onto both feet, rises until "
            "her legs and torso are upright, then holds the standing pose through "
            "the end."
        ),
        "seed": 2026081803,
    },
    {
        "iid": "a66e6818e4144928",
        "source_video_sha256": (
            "0fdc54d89250f355d2170a4d6f6aac0867abf592afb849668a8e2879a6617147"
        ),
        "source_receipt_sha256": (
            "c5d9f1cdd235643d2fa2c1efffb77ddec8a93d20f439885c96659f5c9ea56ef4"
        ),
        "instruction": (
            "A static camera shows a single young woman in a low crouch on a city "
            "sidewalk beside a modern dark-glass building with horizontal metal "
            "bars, looking back over her shoulder, with tied dark hair, a black "
            "sports bra, grey trousers, and a blue-green plaid shirt in harsh "
            "sunlight and deep shadow. The main person presses through both feet, "
            "rises smoothly from the low crouch, straightens the legs and torso, "
            "faces the viewer, and holds upright. The shot stays continuous, the "
            "illumination remains stable, and the final frame is temporally "
            "coherent."
        ),
        "instruction_sha256": (
            "f162aa0276e89b485e5216ce825f4f65e198e91e73aed3a6ca44c6e24f4856f7"
        ),
        "action_order_description": (
            "The woman rises from the low crouch, straightens her legs and torso, "
            "turns to face the viewer, then holds upright through the end."
        ),
        "seed": 2026081804,
    },
)

_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvalInputPreparationError(RuntimeError):
    """A prerequisite or create-only publication contract differs."""


def fail(message: str) -> None:
    raise EvalInputPreparationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise EvalInputPreparationError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-1")
    return value


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        fail(f"{label} field closure differs")
    return dict(value)


def _absolute(value: str | Path, *, label: str, must_exist: bool = False) -> Path:
    text = str(value)
    path = Path(text)
    if (
        not path.is_absolute()
        or text == os.path.sep
        or os.path.normpath(text) != text
    ):
        fail(f"{label} must be a normalized absolute non-root path")
    if must_exist:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise EvalInputPreparationError(f"{label} is unavailable: {error}") from error
        if resolved != path:
            fail(f"{label} is not canonical")
    return path


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_rdev,
        value.st_size,
        getattr(value, "st_blocks", 0),
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    blocks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(blocks)
        blocks.append(block)


def stable_file(
    value: str | Path,
    *,
    label: str,
    expected_sha256: str,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    path = _absolute(value, label=label, must_exist=True)
    expected = _sha256(expected_sha256, label=f"{label} expected SHA")
    if not hasattr(os, "O_NOFOLLOW"):
        fail(f"{label} no-follow capture is unavailable")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(first).hexdigest()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or first != second
        or len(first) != before.st_size
        or digest != expected
        or (
            expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode
        )
    ):
        fail(f"{label} stable physical identity or bytes differ")
    return first, {
        "path": str(path),
        "sha256": digest,
        "size": len(first),
        "mode": stat.S_IMODE(before.st_mode),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "uid": int(before.st_uid),
        "gid": int(before.st_gid),
        "nlink": int(before.st_nlink),
        "rdev": int(before.st_rdev),
        "blocks": int(getattr(before, "st_blocks", 0)),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
    }


def _json(raw: bytes, *, label: str, canonical_newline: bool = True) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                fail(f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise EvalInputPreparationError(f"cannot decode {label}") from error
    if type(value) is not dict:
        fail(f"{label} root is not an object")
    if canonical_newline and raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} is not canonical newline JSON")
    return value


def _verify_object_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    claimed = _sha256(value.get(field), label=f"{label} digest")
    unsigned = dict(value)
    unsigned.pop(field)
    if object_sha256(unsigned) != claimed:
        fail(f"{label} digest differs")
    return claimed


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = _absolute(value, label=label, must_exist=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        fail(f"{label} is not a plain directory")
    return path


def _directory_binding(value: str | Path, *, label: str) -> dict[str, Any]:
    path = _absolute(value, label=label, must_exist=True)
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        fail(f"{label} no-follow directory capture is unavailable")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        middle = os.fstat(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
    ):
        fail(f"{label} stable physical identity differs")
    return {
        "path": str(path),
        "size": int(before.st_size),
        "mode": stat.S_IMODE(before.st_mode),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "uid": int(before.st_uid),
        "gid": int(before.st_gid),
        "nlink": int(before.st_nlink),
        "rdev": int(before.st_rdev),
        "blocks": int(getattr(before, "st_blocks", 0)),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
    }


def _captured_shape(
    value: Any,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    row = _closed(value, fields, label=label)
    _absolute(row["path"], label=f"{label} path")
    if "sha256" in fields:
        _sha256(row["sha256"], label=f"{label} SHA")
    for field in fields - {"path", "sha256"}:
        if type(row[field]) is not int or row[field] < 0:
            fail(f"{label} {field} differs")
    if row["mode"] & ~0o7777:
        fail(f"{label} mode differs")
    if "sha256" in fields and (row["size"] <= 0 or row["nlink"] != 1):
        fail(f"{label} regular-file topology differs")
    return row


def replay_captured_file(
    value: Any,
    *,
    label: str,
    expected_path: str | Path | None = None,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> dict[str, Any]:
    row = _captured_shape(value, CAPTURED_FILE_FIELDS, label=label)
    if expected_path is not None and row["path"] != str(expected_path):
        fail(f"{label} path differs from its external pin")
    if expected_sha256 is not None and row["sha256"] != _sha256(
        expected_sha256, label=f"{label} external SHA"
    ):
        fail(f"{label} SHA differs from its external pin")
    if expected_mode is not None and row["mode"] != expected_mode:
        fail(f"{label} mode differs from its external pin")
    _, observed = stable_file(
        row["path"],
        label=label,
        expected_sha256=row["sha256"],
        expected_mode=row["mode"],
    )
    if observed != row:
        fail(f"{label} full physical identity differs")
    return row


def replay_captured_directory(
    value: Any,
    *,
    label: str,
    expected_path: str | Path | None = None,
) -> dict[str, Any]:
    row = _captured_shape(value, CAPTURED_DIRECTORY_FIELDS, label=label)
    if expected_path is not None and row["path"] != str(expected_path):
        fail(f"{label} path differs from its external pin")
    observed = _directory_binding(row["path"], label=label)
    if observed != row:
        fail(f"{label} full physical identity differs")
    return row


def _pair(binding: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": str(binding["path"]),
        "sha256": _sha256(binding["sha256"], label="file binding SHA"),
    }


def write_create_only(path_value: str | Path, raw: bytes, *, label: str) -> dict[str, Any]:
    path = _absolute(path_value, label=f"{label} path")
    if os.path.lexists(path):
        fail(f"{label} path is not fresh")
    parent = _plain_directory(path.parent, label=f"{label} parent")
    if parent != path.parent:
        fail(f"{label} parent differs")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        fail(f"{label} no-follow creation is unavailable")
    flags |= os.O_NOFOLLOW
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(path.name, flags, 0o444, dir_fd=parent_fd)
        try:
            offset = 0
            while offset < len(raw):
                count = os.write(descriptor, raw[offset:])
                if count <= 0:
                    fail(f"{label} write made no progress")
                offset += count
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o444
                or _identity(before) != _identity(middle)
                or _identity(before) != _identity(after)
                or _identity(before) != _identity(named)
                or first != raw
                or second != raw
            ):
                fail(f"{label} same-FD replay differs")
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return stable_file(
        path,
        label=label,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_mode=0o444,
    )[1]


_COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "seed",
        "cache_sha256",
        "source_archive_sha256",
        "source_revision",
        "source_data_manifest_sha256",
        "source_data_manifest_digest",
        "release_manifest_sha256",
        "controller_sha256",
        "deployment_envelope_sha256",
        "cache_audit_sha256",
        "training_audit_sha256",
        "cache_receipt_sha256",
        "retained_tree_digest",
        "retained_tree_file_count",
        "retained_tree_stable_double_read_before_commit",
        "retained_tree_held_fd_identity_replay",
        "optimizer_updates_per_arm",
        "arm_count",
        "decoded_evaluation_complete",
        "scientific_promotion_authorized",
        "parent_allocations_cancelled",
        "automatic_retry",
        "completion_digest",
    }
)
_AUDIT_ROW_FIELDS = frozenset(
    {
        "arm",
        "step",
        "receipt_sha256",
        "adapter_sha256",
        "adapter_config_sha256",
        "optimizer_sha256",
        # These fields remain part of the upstream audit closure.  Their
        # numeric values are intentionally never inspected, compared, sorted,
        # selected, returned, or copied by this helper.
        "loss",
        "preclip_gradient_norm",
    }
)


def _load_completion(expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _plain_directory(R5_EXPERIMENT_ROOT, label="r5 experiment root")
    raw, binding = stable_file(
        root / "TRAINING_COMPLETE.json",
        label="r5 TRAINING_COMPLETE",
        expected_sha256=_sha256(expected_sha256, label="TRAINING_COMPLETE literal SHA"),
        expected_mode=0o444,
    )
    value = _closed(
        _json(raw, label="r5 TRAINING_COMPLETE"),
        _COMPLETION_FIELDS,
        label="r5 TRAINING_COMPLETE",
    )
    if (
        value["schema_version"] != TRAINING_COMPLETE_SCHEMA
        or value["seed"] != 20260818
        or value["source_archive_sha256"] != R5_SOURCE_ARCHIVE_SHA256
        or value["source_revision"] != R5_SOURCE_REVISION
        or value["source_data_manifest_sha256"] != SOURCE_MANIFEST_SHA256
        or value["source_data_manifest_digest"] != SOURCE_MANIFEST_DIGEST
        or value["release_manifest_sha256"] != R5_RELEASE_MANIFEST_SHA256
        or value["controller_sha256"] != R5_CONTROLLER_SHA256
        or value["deployment_envelope_sha256"]
        != R5_DEPLOYMENT_ENVELOPE_SHA256
        or value["optimizer_updates_per_arm"] != 20
        or value["arm_count"] != len(ARMS)
        or value["decoded_evaluation_complete"] is not False
        or value["scientific_promotion_authorized"] is not False
        or value["parent_allocations_cancelled"] is not False
        or value["automatic_retry"] is not False
        or value["retained_tree_stable_double_read_before_commit"] is not True
        or value["retained_tree_held_fd_identity_replay"] is not True
    ):
        fail("r5 TRAINING_COMPLETE authority differs")
    for field in (
        "cache_sha256",
        "cache_audit_sha256",
        "training_audit_sha256",
        "cache_receipt_sha256",
        "retained_tree_digest",
    ):
        _sha256(value[field], label=f"r5 completion {field}")
    _verify_object_digest(value, field="completion_digest", label="r5 completion")
    return value, binding


def _validate_checkpoint_receipt(value: Mapping[str, Any], *, arm: str, step: int) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    contract = value.get("training_contract")
    if (
        _sha256(claimed, label=f"checkpoint {arm}@{step} receipt digest")
        != object_sha256(unsigned)
        or type(contract) is not dict
        or contract.get("arm") != arm
        or value.get("global_step") != step
    ):
        fail(f"checkpoint receipt differs: {arm}@{step}")


def validate_training_authority(
    expected_completion_sha256: str,
) -> dict[str, Any]:
    """Replay the exact r5 completion/audit/checkpoint path matrix.

    The audit's loss-bearing bytes are necessarily inside its physical file,
    but no loss or gradient value is accessed or used by this function.
    """

    completion, completion_file = _load_completion(expected_completion_sha256)
    audit_path = R5_EXPERIMENT_ROOT / "logs/training-audit.json"
    audit_raw, audit_file = stable_file(
        audit_path,
        label="r5 training audit",
        expected_sha256=_sha256(
            completion["training_audit_sha256"], label="training audit SHA"
        ),
        expected_mode=0o444,
    )
    audit = _closed(
        _json(audit_raw, label="r5 training audit"),
        TRAINING_AUDIT_SCHEMA_FIELDS,
        label="r5 training audit",
    )
    if (
        audit["training_audit_go"] is not True
        or audit["arm_count"] != len(ARMS)
        or audit["checkpoint_count"] != len(ARMS) * len(CHECKPOINT_STEPS)
        or audit["checkpoint_steps"] != list(CHECKPOINT_STEPS)
        or audit["decoded_evaluation_complete"] is not False
        or audit["scientific_promotion_authorized"] is not False
    ):
        fail("r5 training audit authority differs")
    rows = audit["receipt_rows"]
    if type(rows) is not list or len(rows) != 32:
        fail("r5 training audit row count differs")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row_value in rows:
        row = _closed(row_value, _AUDIT_ROW_FIELDS, label="training audit row")
        key = (row["arm"], row["step"])
        if (
            key not in {(arm, step) for arm in ARMS for step in CHECKPOINT_STEPS}
            or key in by_key
        ):
            fail("training audit arm/step closure differs")
        for field in (
            "receipt_sha256",
            "adapter_sha256",
            "adapter_config_sha256",
            "optimizer_sha256",
        ):
            _sha256(row[field], label=f"training audit {field}")
        by_key[key] = row
    expected_keys = {(arm, step) for arm in ARMS for step in CHECKPOINT_STEPS}
    if set(by_key) != expected_keys:
        fail("training audit exact 32 checkpoint closure differs")

    checkpoint_paths: list[dict[str, Any]] = []
    for arm in ARMS:
        for step in CHECKPOINT_STEPS:
            row = by_key[(arm, step)]
            root = _plain_directory(
                R5_EXPERIMENT_ROOT
                / "runs"
                / arm
                / f"checkpoint-{step:08d}",
                label=f"checkpoint root {arm}@{step}",
            )
            receipt_raw, receipt_file = stable_file(
                root / "receipt.json",
                label=f"checkpoint receipt {arm}@{step}",
                expected_sha256=row["receipt_sha256"],
                expected_mode=0o444,
            )
            _validate_checkpoint_receipt(
                _json(receipt_raw, label=f"checkpoint receipt {arm}@{step}"),
                arm=arm,
                step=step,
            )
            _, adapter_file = stable_file(
                root / "adapter/adapter_model.safetensors",
                label=f"adapter model {arm}@{step}",
                expected_sha256=row["adapter_sha256"],
                expected_mode=0o444,
            )
            config_raw, config_file = stable_file(
                root / "adapter/adapter_config.json",
                label=f"adapter config {arm}@{step}",
                expected_sha256=row["adapter_config_sha256"],
                expected_mode=0o444,
            )
            _json(
                config_raw,
                label=f"adapter config {arm}@{step}",
                canonical_newline=False,
            )
            _, optimizer_file = stable_file(
                root / "optimizer.pt",
                label=f"optimizer state {arm}@{step}",
                expected_sha256=row["optimizer_sha256"],
                expected_mode=0o444,
            )
            checkpoint_paths.append(
                {
                    "arm": arm,
                    "step": step,
                    "root": str(root),
                    "receipt": _pair(receipt_file),
                    "adapter_model": _pair(adapter_file),
                    "adapter_config": _pair(config_file),
                    "optimizer": _pair(optimizer_file),
                }
            )
    return {
        "completion": completion,
        "completion_file": _pair(completion_file),
        "training_audit_file": _pair(audit_file),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_count": len(checkpoint_paths),
        "training_loss_read_or_used_for_selection": False,
    }


def _source_video_path(iid: str) -> Path:
    return SOURCE_VIDEO_BASE / iid / "samples" / iid / "source_video.mp4"


def _source_receipt_path(iid: str) -> Path:
    return (
        SOURCE_RECEIPT_BASE
        / f"pair5-t2v-core4-v2-{iid}-action"
        / "pair-v5-t2v-calibration-receipt.json"
    )


def action_review_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    description = str(source["action_order_description"])
    value: dict[str, Any] = {
        "schema_version": ACTION_REVIEW_CONTRACT_SCHEMA,
        "action_order_description": description,
        "action_order_description_sha256": text_sha256(description),
        "expected_onset_frame_min": EVALUATION_ONSET_FRAME_MIN,
        "expected_onset_frame_max": EVALUATION_ONSET_FRAME_MAX,
        "terminal_hold_start_frame_min": EVALUATION_TERMINAL_HOLD_START_FRAME_MIN,
        "terminal_hold_end_frame": 80,
        "full_video_frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
    }
    value["contract_digest"] = object_sha256(value)
    return value


def validate_static_source_authority() -> list[dict[str, Any]]:
    manifest_raw, _ = stable_file(
        SOURCE_MANIFEST_PATH,
        label="source-only manifest",
        expected_sha256=SOURCE_MANIFEST_SHA256,
        expected_mode=0o444,
    )
    source_manifest = _json(manifest_raw, label="source-only manifest")
    if (
        source_manifest.get("schema_version")
        != "bernini-self-generated-action-quotient-source-manifest-v1"
        or source_manifest.get("training_process_can_reach_historical_selected_target")
        is not False
        or source_manifest.get("self_generated_anchor_is_rv2v_supervision_target")
        is not False
    ):
        fail("source-only manifest authority differs")
    source_unsigned = dict(source_manifest)
    source_declared = source_unsigned.pop("manifest_digest", None)
    if (
        source_declared != SOURCE_MANIFEST_DIGEST
        or object_sha256(source_unsigned) != SOURCE_MANIFEST_DIGEST
    ):
        fail("source-only manifest digest differs")
    manifest_rows = source_manifest.get("rows")
    if type(manifest_rows) is not list or [row.get("iid") for row in manifest_rows] != list(
        FITTED_IIDS
    ):
        fail("source-only fitted IID order differs")
    source_manifest_by_iid = {row["iid"]: row for row in manifest_rows}

    result: list[dict[str, Any]] = []
    for source in SOURCE_ROWS:
        iid = source["iid"]
        instruction = source["instruction"]
        if (
            text_sha256(instruction) != source["instruction_sha256"]
            or source_manifest_by_iid[iid].get("instruction") != instruction
        ):
            fail(f"source instruction authority differs: {iid}")
        _, video_file = stable_file(
            _source_video_path(iid),
            label=f"source video {iid}",
            expected_sha256=source["source_video_sha256"],
            expected_mode=0o600,
        )
        receipt_raw, receipt_file = stable_file(
            _source_receipt_path(iid),
            label=f"source action receipt {iid}",
            expected_sha256=source["source_receipt_sha256"],
            expected_mode=0o400,
        )
        receipt = _json(
            receipt_raw,
            label=f"source action receipt {iid}",
            canonical_newline=False,
        )
        candidate = receipt.get("candidate")
        if (
            type(candidate) is not dict
            or candidate.get("candidate_id")
            != f"pair5-t2v-core4-v2-{iid}-action"
            or candidate.get("geometry_source_video") != str(_source_video_path(iid))
            or candidate.get("geometry_source_video_sha256")
            != source["source_video_sha256"]
            or candidate.get("full_t2v_caption") != instruction
            or candidate.get("full_t2v_caption_utf8_sha256")
            != source["instruction_sha256"]
            or candidate.get("semantic_branch") != "action"
        ):
            fail(f"source action receipt semantic authority differs: {iid}")
        result.append(
            {
                "iid": iid,
                "source_video_path": video_file["path"],
                "source_video_sha256": video_file["sha256"],
                "source_receipt_path": receipt_file["path"],
                "source_receipt_sha256": receipt_file["sha256"],
                "instruction": instruction,
                "instruction_sha256": source["instruction_sha256"],
                "action_review_contract": action_review_contract(source),
                "seed": source["seed"],
            }
        )
    return result


def _exact_directory_entries(path: Path, expected: set[str], *, label: str) -> None:
    observed = {entry.name for entry in os.scandir(_plain_directory(path, label=label))}
    if observed != expected:
        fail(f"{label} exact entry closure differs")


def validate_model_authority() -> dict[str, Any]:
    root = _plain_directory(MODEL_ROOT, label="model checkpoint root")
    raw, manifest_file = stable_file(
        MODEL_RELEASE_MANIFEST_PATH,
        label="model release SHA manifest",
        expected_sha256=MODEL_RELEASE_MANIFEST_SHA256,
        expected_mode=0o444,
    )
    try:
        text = raw.decode("ascii", "strict")
    except UnicodeDecodeError as error:
        raise EvalInputPreparationError("model release manifest is not ASCII") from error
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  \./([^\x00\r\n]+)", line)
        if match is None:
            fail("model release manifest row differs")
        digest, relative_text = match.groups()
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            fail("model release manifest path differs")
        rows.append((digest, relative_text))
    if len(rows) != 23 or len({relative for _, relative in rows}) != 23:
        fail("model release manifest exact 23-file closure differs")
    # The independently pinned manifest is the byte authority.  Avoid a
    # second 30+ GiB double-read here; native inference still loads the pinned
    # checkpoint, and the runtime separately pins CHECKPOINT_TREE_SHA256.
    for _, relative in rows:
        path = root / relative
        try:
            info = path.lstat()
        except OSError as error:
            raise EvalInputPreparationError(
                f"model release member is unavailable: {relative}: {error}"
            ) from error
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or path.resolve(strict=True) != path
        ):
            fail(f"model release member topology differs: {relative}")
    return {
        "root": str(root),
        "manifest": _pair(manifest_file),
        "entry_count": len(rows),
        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
        "manifest_entries_rehashed_by_this_helper": False,
    }


def require_production_model_consumption_authority() -> None:
    """Block publication until production enforces the bytes it consumes.

    The check is intentionally a compile-time release fact, not an operator
    flag.  Merely hashing the model earlier in this preparation process cannot
    protect later path-based reopens by the frozen exact14 inference stack.
    """

    if MODEL_CONSUMPTION_AUTHORITY_ENFORCED_BY_PRODUCTION is not True:
        fail(MODEL_CONSUMPTION_AUTHORITY_P0)


def validate_eval_release_artifacts() -> dict[str, Any]:
    root = _plain_directory(EVAL_ARTIFACTS_ROOT, label="exact14 artifact root")
    if stat.S_IMODE(root.lstat().st_mode) != 0o555:
        fail("exact14 artifact root is not sealed mode 0555")
    _exact_directory_entries(
        root,
        {"source.tar", "source.manifest.json", "deployment-envelope.json"},
        label="exact14 artifact root",
    )
    _, archive = stable_file(
        root / "source.tar",
        label="exact14 source archive",
        expected_sha256=EVAL_ARCHIVE_SHA256,
        expected_mode=0o444,
    )
    manifest_raw, manifest = stable_file(
        root / "source.manifest.json",
        label="exact14 source manifest",
        expected_sha256=EVAL_MANIFEST_SHA256,
        expected_mode=0o444,
    )
    manifest_value = _json(manifest_raw, label="exact14 source manifest")
    if (
        manifest_value.get("schema_version")
        != "bernini-action-preservation-decoded-eval-source-release-v2"
        or manifest_value.get("release_generation") != RELEASE_GENERATION
        or manifest_value.get("content_revision") != EVAL_CONTENT_REVISION
        or manifest_value.get("manifest_digest") != EVAL_MANIFEST_DIGEST
        or manifest_value.get("file_count") != 14
        or manifest_value.get("exact_member_closure") is not True
        or manifest_value.get("component_sha256", {}).get("infer_lora.py")
        != INFER_LORA_SHA256
        or manifest_value.get("component_sha256", {}).get("train_lora.py")
        != TRAIN_LORA_SHA256
        or manifest_value.get("component_sha256", {}).get(
            "action_preservation_decoded_eval_decoder_adapter_v1.py"
        )
        != DECODER_ADAPTER_SHA256
        or manifest_value.get("component_sha256", {}).get(
            "action_preservation_decoded_eval_verified_release_v1.py"
        )
        != VERIFIED_RUNTIME_SHA256
    ):
        fail("exact14 source manifest authority differs")
    _verify_object_digest(
        manifest_value, field="manifest_digest", label="exact14 source manifest"
    )
    envelope_raw, envelope = stable_file(
        root / "deployment-envelope.json",
        label="exact14 deployment envelope",
        expected_sha256=EVAL_ENVELOPE_SHA256,
        expected_mode=0o444,
    )
    envelope_value = _json(envelope_raw, label="exact14 deployment envelope")
    if (
        envelope_value.get("release_generation") != RELEASE_GENERATION
        or envelope_value.get("envelope_digest") != EVAL_ENVELOPE_DIGEST
        or envelope_value.get("source_archive", {}).get("sha256")
        != EVAL_ARCHIVE_SHA256
        or envelope_value.get("source_manifest", {}).get("sha256")
        != EVAL_MANIFEST_SHA256
        or envelope_value.get("create_only_deployment_required") is not True
        or envelope_value.get("fresh_materialized_root_required") is not True
        or envelope_value.get("automatic_scientific_promotion_authorized")
        is not False
    ):
        fail("exact14 deployment envelope authority differs")
    _verify_object_digest(
        envelope_value, field="envelope_digest", label="exact14 envelope"
    )
    _, controller = stable_file(
        DETACHED_CONTROLLER_PATH,
        label="detached eval controller",
        expected_sha256=DETACHED_CONTROLLER_SHA256,
        expected_mode=0o444,
    )
    _, verified_runtime = stable_file(
        VERIFIED_RUNTIME_SOURCE_PATH,
        label="detached exact14 verified runtime",
        expected_sha256=VERIFIED_RUNTIME_SHA256,
        expected_mode=0o444,
    )
    return {
        "archive": _pair(archive),
        "manifest": _pair(manifest),
        "envelope": _pair(envelope),
        "controller": _pair(controller),
        "verified_runtime_source": _pair(verified_runtime),
    }


def validate_static_runtime() -> dict[str, Any]:
    _, root_python = stable_file(
        ROOT_PYTHON_PATH,
        label="root Python",
        expected_sha256=ROOT_PYTHON_SHA256,
        expected_mode=0o755,
    )
    _, frozen_python = stable_file(
        FROZEN_PYTHON_PATH,
        label="frozen Python",
        expected_sha256=FROZEN_PYTHON_SHA256,
        expected_mode=0o755,
    )
    _, torchrun = stable_file(
        TORCHRUN_PATH,
        label="torch distributed run.py",
        expected_sha256=TORCHRUN_SHA256,
        expected_mode=0o644,
    )
    _, ffprobe = stable_file(
        FFPROBE_PATH,
        label="root ffprobe",
        expected_sha256=FFPROBE_SHA256,
        expected_mode=0o755,
    )
    _plain_directory(SITE_PACKAGES_PATH, label="runtime site-packages")
    _plain_directory(BERNINI_ROOT, label="Bernini root")
    _plain_directory(VEOMNI_ROOT, label="VeOmni root")
    model = validate_model_authority()
    _, inference_config = stable_file(
        INFERENCE_CONFIG_PATH,
        label="Bernini inference config",
        expected_sha256=INFERENCE_CONFIG_SHA256,
        expected_mode=0o444,
    )
    _, adapter_release_manifest = stable_file(
        R5_RELEASE_ROOT / "source.manifest.json",
        label="r5 adapter release manifest",
        expected_sha256=R5_RELEASE_MANIFEST_SHA256,
        expected_mode=0o444,
    )
    return {
        "root_python": _pair(root_python),
        "frozen_python": _pair(frozen_python),
        "torchrun": _pair(torchrun),
        "ffprobe": _pair(ffprobe),
        "model": model,
        "inference_config": _pair(inference_config),
        "adapter_release_manifest": _pair(adapter_release_manifest),
    }


def check_all(expected_completion_sha256: str) -> dict[str, Any]:
    training = validate_training_authority(expected_completion_sha256)
    sources = validate_static_source_authority()
    runtime = validate_static_runtime()
    release = validate_eval_release_artifacts()
    require_production_model_consumption_authority()
    return {
        "status": "READY_FOR_PHASE_A_INPUT_PUBLICATION",
        "training_complete": training["completion_file"],
        "training_audit": training["training_audit_file"],
        "checkpoint_count": training["checkpoint_count"],
        "checkpoint_roots": [row["root"] for row in training["checkpoint_paths"]],
        "fitted_iids": [row["iid"] for row in sources],
        "eval_seeds": [row["seed"] for row in sources],
        "action_interval_semantics": ACTION_INTERVAL_SEMANTICS,
        "model_release_manifest": runtime["model"]["manifest"],
        "exact14_manifest": release["manifest"],
        "training_loss_read_or_used_for_selection": False,
        "remote_launch_performed": False,
        "controller_executed": False,
        "automatic_retry": False,
        "scientific_promotion_authorized": False,
    }


def _fresh_path(value: str | Path, *, label: str) -> Path:
    path = _absolute(value, label=label)
    if os.path.lexists(path):
        fail(f"{label} is not fresh")
    return path


def _validate_output_topology(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths):
        fail("phase-A output paths are not distinct")
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left in right.parents or right in left.parents:
                fail("phase-A output path ancestry overlaps")


def build_deployment_request(
    *,
    expected_completion_sha256: str,
    work_root: str | Path,
    materialized_release_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    # Phase A is deliberately gated on the complete 32-checkpoint authority,
    # even though the detached controller itself only needs runtime inputs.
    check_all(expected_completion_sha256)
    work = _plain_directory(work_root, label="evaluation work root")
    request_path = _fresh_path(work / "deployment-request.json", label="deployment request")
    release_root = _fresh_path(
        materialized_release_root, label="fresh materialized exact14 root"
    )
    source_spec = _fresh_path(
        work / "source_runtime_spec.json", label="source runtime spec"
    )
    source_authority = _fresh_path(
        work / "source-spec-authority.json", label="source spec authority receipt"
    )
    controller_authority = _fresh_path(
        work / "controller-authority.json", label="controller authority receipt"
    )
    deployment_receipt = _fresh_path(
        work / "deployment-receipt.json", label="deployment receipt"
    )
    _validate_output_topology(
        [release_root, source_spec, source_authority, controller_authority, deployment_receipt]
    )
    runtime = validate_static_runtime()
    release = validate_eval_release_artifacts()
    value: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "controller": release["controller"],
        "root_python": runtime["root_python"],
        "frozen_python": runtime["frozen_python"],
        "site_packages_path": str(SITE_PACKAGES_PATH),
        "torchrun": runtime["torchrun"],
        "release_root": str(release_root),
        "archive": release["archive"],
        "manifest": release["manifest"],
        "manifest_digest": EVAL_MANIFEST_DIGEST,
        "content_revision": EVAL_CONTENT_REVISION,
        "envelope": release["envelope"],
        "envelope_digest": EVAL_ENVELOPE_DIGEST,
        "verified_runtime_source": release["verified_runtime_source"],
        "source_runtime_spec_path": str(source_spec),
        "source_spec_authority_receipt_path": str(source_authority),
        "controller_authority_receipt_path": str(controller_authority),
        "deployment_receipt_path": str(deployment_receipt),
        "automatic_retry": False,
        "network_allowed": False,
        "scientific_promotion_authorized": False,
    }
    value["request_digest"] = object_sha256(value)
    return value, request_path


def publish_deployment_request(
    *,
    expected_completion_sha256: str,
    work_root: str | Path,
    materialized_release_root: str | Path,
) -> dict[str, Any]:
    value, path = build_deployment_request(
        expected_completion_sha256=expected_completion_sha256,
        work_root=work_root,
        materialized_release_root=materialized_release_root,
    )
    binding = write_create_only(
        path,
        canonical_json_bytes(value) + b"\n",
        label="deployment request",
    )
    return {
        "status": "PHASE_A_REQUEST_PREPARED_NOT_EXECUTED",
        "deployment_request": _pair(binding),
        "request_digest": value["request_digest"],
        "independent_literal_sha_review_still_required": True,
        "controller_executed": False,
        "remote_launch_performed": False,
        "automatic_retry": False,
    }


def authority_status() -> dict[str, Any]:
    return {
        "status": "BLOCKED_MODEL_CONSUMPTION_AUTHORITY_P0",
        "release_generation": RELEASE_GENERATION,
        "reason": MODEL_CONSUMPTION_AUTHORITY_P0,
        "phase_a_publication_authorized": False,
        "phase_b_publication_authorized": False,
        "cli_override_available": False,
        "controller_executed": False,
        "remote_launch_performed": False,
        "training_loss_read_or_used_for_selection": False,
        "action_interval_semantics": ACTION_INTERVAL_SEMANTICS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("authority-status")
    check = commands.add_parser("check")
    check.add_argument("--training-complete-sha256", required=True)
    phase_a = commands.add_parser("phase-a")
    phase_a.add_argument("--training-complete-sha256", required=True)
    phase_a.add_argument("--work-root", required=True)
    phase_a.add_argument("--materialized-release-root", required=True)
    # Keep an explicit command that fails closed instead of inviting operators
    # to hand-author an exact14 source/runtime spec around this helper.
    commands.add_parser("phase-b")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "authority-status":
            result = authority_status()
            print(canonical_json_bytes(result).decode("utf-8"))
            return 2
        if args.command == "check":
            result = check_all(
                _sha256(
                    args.training_complete_sha256,
                    label="TRAINING_COMPLETE literal SHA",
                )
            )
        elif args.command == "phase-a":
            result = publish_deployment_request(
                expected_completion_sha256=_sha256(
                    args.training_complete_sha256,
                    label="TRAINING_COMPLETE literal SHA",
                ),
                work_root=args.work_root,
                materialized_release_root=args.materialized_release_root,
            )
        else:
            require_production_model_consumption_authority()
            fail("phase B requires a superseding audited release")
    except EvalInputPreparationError as error:
        result = {
            **authority_status(),
            "error": str(error),
        }
        print(canonical_json_bytes(result).decode("utf-8"))
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
