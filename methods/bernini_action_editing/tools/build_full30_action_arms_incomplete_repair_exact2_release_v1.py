#!/usr/bin/env python3
"""Build/audit the deterministic BOX-EXP-013 exact2 r6 release.

Eighteen members of the frozen BOX-EXP-011 fit-repair r2 resource stack are
carried byte-for-byte.  The rank wrapper and base/specialized resource are
r6-owned replacements, then four exact2 business/control members are added.
The launcher is deliberately detached so it can literally pin the final core
archive/manifest identities without a cryptographic self-reference.  This
builder also seals the detached launcher's path-independent SHA in a canonical
deployment envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from types import ModuleType
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = (
    "bernini-full30-action-arms-incomplete-repair-exact2-core-release-v6"
)
DEPLOYMENT_ENVELOPE_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-deployment-envelope-v4"
)
RELEASE_GENERATION = "r6"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
FROZEN_BASE_ARCHIVE_SHA256 = (
    "24456d098551f07e7bb634259dad95209258c4db0153aee305ae14ec28b05f43"
)
FROZEN_BASE_MANIFEST_SHA256 = (
    "fe7e6130c77de9b627428452b0bc3db3772689fc26efd79ee38cd0abee3c70d3"
)
FROZEN_BASE_MANIFEST_DIGEST = (
    "be432ed919c4b8f95932ed670d6004f777804b2a19d62f65134ad95d177c8e58"
)
FROZEN_BASE_REVISION = "603865eb19543b14222f85c6f6bceb19872717aa"
EXTERNAL_KEY_SHA256 = (
    "4c0864c7018b28b284a49d7134bce574e8d8fe47d5d795a71497b78fff446f8c"
)
EXTERNAL_REVIEW_RECEIPT_SHA256 = (
    "1b40da8dde07f348c2501adf3fd62fb528062053cde6e99c62f6d02e3ad8a4bc"
)
PROMPT_SHA256 = (
    "225d66cf0ad29fa7b7b51bf6177843629f2f8710d60b3278008495cbb049cde4"
)
PROMPT_BUNDLE_SHA256 = (
    "6abd07b6f952171879790f34d1e908e79472405dbc6c4ac87290529d8426c102"
)
PORTABLE_FFPROBE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
    "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
PORTABLE_FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
REVOKED_LIVE_LOG_SHA256 = (
    "5e42f66e61c0c3c3c33fd414a155832094fefec546525c126c9781ee137abb5e"
)
REVOKED_PORTABLE_R2_ARCHIVE_SHA256 = (
    "dc945cc658934050501d327d5f81afb97de93f43bd39a2e00472e515f502484c"
)
REVOKED_PORTABLE_R2_MANIFEST_SHA256 = (
    "5a9f069bfc1f1452e12f11d81fdf61a43ad8f5a716b2a166126cac54bfef6d83"
)
REVOKED_PORTABLE_R2_LAUNCHER_SHA256 = (
    "c22f8a16328e7f7660f85c665faeabc48be50fbac8e70f2eff14098d7c53d6ba"
)
REVOKED_TERMINAL_R3_ARCHIVE_SHA256 = (
    "ce1a5ff5cd8ed4458e2a704ea3850a08f57630d8d7f8b9139c02fdbe0701f5fb"
)
REVOKED_TERMINAL_R3_MANIFEST_SHA256 = (
    "e146c3c426980b9e299514036af58fa9675d43a3b466c1fe78076401d40b6f6c"
)
REVOKED_TERMINAL_R3_LAUNCHER_SHA256 = (
    "3917a978ec4d6c42da139349b3553e9f18606ad27e8c9dfa8f3ce9869d6c30de"
)
REVOKED_TERMINAL_R3_ENVELOPE_SHA256 = (
    "5412ed07234b7e60eeb68d46f8dbf9271b16a95c03837e56cd71ad5cf1f7b41a"
)
REVOKED_RESOURCE_REUSE_R4_ARCHIVE_SHA256 = (
    "29b2f5002363673b3957bfcc8135859bfac304434a590220919fcbc1051f0a23"
)
REVOKED_RESOURCE_REUSE_R4_MANIFEST_SHA256 = (
    "fafd3d250f4b489cc5be5451279957aa95fd9607528911c4e888a8e2f3984330"
)
REVOKED_RESOURCE_REUSE_R4_LAUNCHER_SHA256 = (
    "8524f21373476147f905c081339f6281ebfc7ebfc38cbbb641faa542d7b420d6"
)
REVOKED_RESOURCE_REUSE_R4_ENVELOPE_SHA256 = (
    "31c6c949403cf3816176ecbff8da95686a717f35eb8e5e93e9d3373c3ad77b24"
)
REVOKED_TERMINAL_R5_ARCHIVE_SHA256 = (
    "3db741464fa8e5bd258d4d0b7a3f90c5ee9c9eeb78695cab094dcea919fb94b5"
)
REVOKED_TERMINAL_R5_MANIFEST_SHA256 = (
    "b0129020fd4134e50e6840a2fdf8e61d0cd32f267bc2dadda8ba17e241d92208"
)
REVOKED_TERMINAL_R5_LAUNCHER_SHA256 = (
    "018ddaf9f4ab8c423dd8d081fd7303139d3256eb32c815f8c9d4a8494b4f2e6d"
)
REVOKED_TERMINAL_R5_ENVELOPE_SHA256 = (
    "517e31a381fb7e8fe626a1cc71829e59aea4cbe62febb4a68aeb8f39b49c3154"
)
REVOKED_TERMINAL_R5_LOG_SHA256 = (
    "6df1462415b72bbf966f8b41c125932e7fcd39353c58a8172121577c41f9285a"
)
REVOKED_TERMINAL_R5_RELEASE_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_generic_source_anchored_action_v1_20260814/releases/"
    "full30-action-arms-incomplete-exact2-r5-3b59480b-3db74146"
)
REVOKED_TERMINAL_R5_MATERIALIZATION_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_generic_source_anchored_action_v1_20260814/"
    "release_materializations/"
    "full30-action-arms-incomplete-exact2-r5-3b59480b-3db74146"
)
REVOKED_TERMINAL_R5_RUN_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
    "full30-action-arms-incomplete-exact2-r5-3b59480b-j136140-r1"
)
REVOKED_RESOURCE_REUSE_R4_RELEASE_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_generic_source_anchored_action_v1_20260814/releases/"
    "full30-action-arms-incomplete-exact2-r4-233bb4be-29b2f500"
)
REVOKED_RESOURCE_REUSE_R4_MATERIALIZATION_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_generic_source_anchored_action_v1_20260814/"
    "release_materializations/"
    "full30-action-arms-incomplete-exact2-r4-233bb4be-29b2f500"
)
REVOKED_RESOURCE_REUSE_R4_RUN_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
    "full30-action-arms-incomplete-exact2-r4-233bb4be-j136140-r1"
)
FROZEN_PYTHON_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
FROZEN_PYTHON_REALPATH = FROZEN_PYTHON_PATH
FROZEN_PYTHON_FILE_TYPE = "regular file"
FROZEN_PYTHON_MODE_OCTAL = "0755"
FROZEN_PYTHON_UID = 2012
FROZEN_PYTHON_SIZE = 31_490_256
FROZEN_PYTHON_LINK_COUNT = 1
FROZEN_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
REJECTED_PYTHON_SYMLINK_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python"
)

# (archive mode, byte SHA-256) from the sealed BOX-EXP-011 r2 manifest.
FROZEN_SOURCE_MEMBER_PINS: Mapping[str, tuple[int, str]] = {
    "full30_action_fit_repair_exact8_controller_v1.py": (
        0o444, "4b12d0b0c7bd514bf40a2fc1c4a12da3364498d30540a1979ba752e6e9629d84"
    ),
    "full30_action_fit_repair_exact8_generator_v1.py": (
        0o444, "276ba6521f412cd9d393d3352cd4829579577d4da3d61c98ea27c7725aeb822e"
    ),
    "full30_action_fit_repair_exact8_plan_v1.py": (
        0o444, "bf3cfdc590a75363d2daf42d1c474debe8992b20da2b9f07a9087617af507bbb"
    ),
    "infer_lora.py": (
        0o444, "babd6d63287723ccd14b2bbe43bd4550c30b4feaa794d17c66f5a5ddefe979fe"
    ),
    "infer_native_identity_generation_canary.py": (
        0o444, "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42"
    ),
    "infer_pair_v5_t2v_calibration_bank.py": (
        0o444, "f695dba0304d7b574fd15ab058ecc639d3dae594a14a174c1de6c38279eb5544"
    ),
    "infer_source_kv_carrier_oracle.py": (
        0o444, "fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604"
    ),
    "infer_source_value_residual_oracle.py": (
        0o444, "40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8"
    ),
    "pair_v5_t2v_calibration_bank_spec.py": (
        0o444, "c8a81c4a1ab57aa9422d3d0ccf5084bf9ccbc60df0f69fe8849d04e132e25288"
    ),
    "scripts/auh_full30_action_fit_repair_exact8_136140_world4_v1.sh": (
        0o555, "75061c96bf6b05dedb8687c0bf781c406608252e6134fd28c89098ae78643e6e"
    ),
    "scripts/auh_generic_action_data_prep_rank_exec_v1.sh": (
        0o555, "97a8ddf18b5066c2899d9a71a5f44110dc4226defc0ab332c08c31bae8dd0730"
    ),
    "source_kv_replay.py": (
        0o444, "45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3"
    ),
    "source_kv_route_batches.py": (
        0o444, "7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e"
    ),
    "source_value_residual.py": (
        0o444, "420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124"
    ),
    "tools/build_full30_action_fit_repair_exact8_release_v1.py": (
        0o444, "023cf111a32c7c11fed8baecce560a80436e934911352d450c544a6d3b3086d7"
    ),
    "tools/build_pair_v5_t2v_seed2_bank.py": (
        0o444, "37d417ef399268b7704957e9957e1b67030eaa6777f24851c94fee07ad904420"
    ),
    "tools/build_renderer_dataset.py": (
        0o444, "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
    ),
    "tools/materialize_vae.py": (
        0o444, "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
    ),
    "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py": (
        0o444, "aa2f5c01c9d231ad5340cbb572c1523546fa2e148143ee1b5bf04f53f005f017"
    ),
    "tools/reserve4_fixed_generation_sp4_v1.py": (
        0o444, "be722e4020040ba446f290f07378e870e2d3c1a4228ec997c3447770fcb53d5d"
    ),
    "train_lora.py": (
        0o444, "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5"
    ),
}

# These three members replace their r2 bytes.  The specialized member is
# deterministically derived from the exact r6 base resource below.
R6_REPLACEMENT_PINS: Mapping[str, tuple[int, str]] = {
    "scripts/auh_generic_action_data_prep_rank_exec_v1.sh": (
        0o555, "8d8a9ac1c96ba2d1c62ef3f205650b2f06a0398b5c60280b33043c11ca14be4f"
    ),
    "tools/reserve4_fixed_generation_sp4_v1.py": (
        0o444, "2fa752f284cbe96f869eb65595f1c2ca1a5c64185789282e0c5b3c429bc0e446"
    ),
    "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py": (
        0o444, "c0749bdb7694128fcf5deffb7503c46e3dbe2967adabec4aca4b5c800a5ed01b"
    ),
}
FROZEN_MEMBER_PINS: Mapping[str, tuple[int, str]] = {
    path: pin
    for path, pin in FROZEN_SOURCE_MEMBER_PINS.items()
    if path not in R6_REPLACEMENT_PINS
}

PLAN_MEMBER = "full30_action_arms_incomplete_repair_exact2_plan_v1.py"
GENERATOR_MEMBER = "full30_action_arms_incomplete_repair_exact2_generator_v1.py"
CONTROLLER_MEMBER = "full30_action_arms_incomplete_repair_exact2_controller_v1.py"
BUILDER_MEMBER = "tools/build_full30_action_arms_incomplete_repair_exact2_release_v1.py"
LAUNCHER_MEMBER = "scripts/auh_full30_action_arms_incomplete_repair_exact2_136140_world4_v1.sh"
NEW_FILES_AND_MODES: Mapping[str, int] = {
    PLAN_MEMBER: 0o444,
    GENERATOR_MEMBER: 0o444,
    CONTROLLER_MEMBER: 0o444,
    BUILDER_MEMBER: 0o444,
}
FILES_AND_MODES: Mapping[str, int] = {
    **{path: mode for path, (mode, _) in FROZEN_MEMBER_PINS.items()},
    **{path: mode for path, (mode, _) in R6_REPLACEMENT_PINS.items()},
    **NEW_FILES_AND_MODES,
}
COMPONENT_FILES: Mapping[str, str] = {
    "prompt_plan_sha256": PLAN_MEMBER,
    "generator_sha256": GENERATOR_MEMBER,
    "controller_sha256": CONTROLLER_MEMBER,
    "release_builder_sha256": BUILDER_MEMBER,
    "frozen_generator_sha256": "full30_action_fit_repair_exact8_generator_v1.py",
    "frozen_resource_sha256": (
        "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py"
    ),
    "rank_cache_wrapper_sha256": (
        "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
    ),
}
ENTRYPOINTS = (CONTROLLER_MEMBER,)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UNSAFE_PYTHON_ENVIRONMENT = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONSAFEPATH",
        "PYTHONUSERBASE",
    }
)
ROOT_BOOTSTRAP_PYTHON = Path("/usr/bin/python3.10")
ROOT_BOOTSTRAP_PYTHON_SHA256 = (
    "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
)
ROOT_BOOTSTRAP_PYTHON_SIZE = 5_937_800
FROZEN_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
FROZEN_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
FROZEN_PYTHON_SIZE = 31_490_256
UNSAFE_EXEC_ENVIRONMENT = frozenset(
    {
        *UNSAFE_PYTHON_ENVIRONMENT,
        "BASH_ENV",
        "ENV",
        "SHELLOPTS",
        "BASHOPTS",
        "CDPATH",
        "GLOBIGNORE",
        "GCONV_PATH",
        "LOCPATH",
    }
)


class ArmsIncompleteExact2ReleaseError(RuntimeError):
    """Raised before a drifted or ambiguous release can pass."""


def fail(message: str) -> NoReturn:
    raise ArmsIncompleteExact2ReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ArmsIncompleteExact2ReleaseError(
            "release is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail("release input must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ReleaseError("release input is unavailable") from error
    if resolved != path:
        fail("release input must be one canonical single-link plain file")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            fail("release input must be one canonical single-link plain file")

        def read_pass() -> bytes:
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)

        first = read_pass()
        middle = os.fstat(descriptor)
        second = read_pass()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_uid,
        item.st_gid,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_blocks,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    named = path.lstat()
    if (
        stable_fields(before)
        != stable_fields(middle)
        or stable_fields(before) != stable_fields(after)
        or stable_fields(before) != stable_fields(named)
        or first != second
        or len(first) != before.st_size
        or not first
    ):
        fail("release input changed while reading or is empty")
    return first


def _specialize_r6_resource(raw: bytes) -> bytes:
    source_sha = R6_REPLACEMENT_PINS[
        "tools/reserve4_fixed_generation_sp4_v1.py"
    ][1]
    specialized_sha = R6_REPLACEMENT_PINS[
        "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py"
    ][1]
    if (
        hashlib.sha256(raw).hexdigest() != source_sha
        or raw.count(b"136141") != 7
        or raw.count(b"136140") != 0
    ):
        fail("r6 resource specialization preimage differs")
    specialized = raw.replace(b"136141", b"136140")
    if (
        len(specialized) != len(raw)
        or hashlib.sha256(specialized).hexdigest() != specialized_sha
        or specialized.count(b"136141") != 0
        or specialized.count(b"136140") != 7
    ):
        fail("r6 resource specialization postimage differs")
    return specialized


def _frozen_base_payloads(root: Path) -> Mapping[str, bytes]:
    """Read the sealed r2 archive when present; never borrow drifted worktree bytes."""

    archive_path = root / "releases/full30_action_fit_repair_exact8_r1/source.tar"
    manifest_path = (
        root
        / "releases/full30_action_fit_repair_exact8_r1/source.manifest.json"
    )
    if not archive_path.exists() or not manifest_path.exists():
        fail("frozen BOX-EXP-011 release archive/manifest closure differs")
    archive_raw = _stable_plain_bytes(archive_path.resolve(strict=True))
    manifest_raw = _stable_plain_bytes(manifest_path.resolve(strict=True))
    if (
        hashlib.sha256(archive_raw).hexdigest() != FROZEN_BASE_ARCHIVE_SHA256
        or hashlib.sha256(manifest_raw).hexdigest()
        != FROZEN_BASE_MANIFEST_SHA256
    ):
        fail("frozen BOX-EXP-011 release identity differs")
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArmsIncompleteExact2ReleaseError(
            "frozen BOX-EXP-011 manifest is invalid"
        ) from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("manifest_digest") != FROZEN_BASE_MANIFEST_DIGEST
    ):
        fail("frozen BOX-EXP-011 manifest digest differs")
    result: dict[str, bytes] = {}
    prefix = f"{MEMBER_ROOT}/"
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.startswith(prefix):
                    fail("frozen BOX-EXP-011 archive member type/path differs")
                relative = member.name[len(prefix) :]
                if relative not in FROZEN_SOURCE_MEMBER_PINS or relative in result:
                    fail("frozen BOX-EXP-011 archive member closure differs")
                extracted = archive.extractfile(member)
                if extracted is None:
                    fail("frozen BOX-EXP-011 archive member is unreadable")
                raw = extracted.read()
                expected_mode, expected_sha = FROZEN_SOURCE_MEMBER_PINS[relative]
                if (
                    member.mode != expected_mode
                    or hashlib.sha256(raw).hexdigest() != expected_sha
                ):
                    fail(f"frozen BOX-EXP-011 archive member differs: {relative}")
                result[relative] = raw
    except (tarfile.TarError, OSError) as error:
        raise ArmsIncompleteExact2ReleaseError(
            "frozen BOX-EXP-011 archive is invalid"
        ) from error
    if set(result) != set(FROZEN_SOURCE_MEMBER_PINS):
        fail("frozen BOX-EXP-011 archive exact member closure differs")
    return result


def _authority() -> Mapping[str, Any]:
    return {
        "experiment_id": "BOX-EXP-013",
        "purpose": "repair only the two missing arms incomplete clips for R4",
        "scientific_target": "two complete same-seed action/before-terminal arms cells",
        "learning_target": "N/A; optimizer-free frozen data authoring",
        "numeric_target": {
            "new_incomplete_full81_pass": [2, 2],
            "external_action_full81_pass": [2, 2],
            "cross_run_same_gaussian_pass": [2, 2],
            "training_updates": 0,
        },
        "dataset": "fit_arms_incomplete_only_repair_exact2",
        "analysis_split": "fit",
        "formal_candidate_count": 2,
        "external_action_count": 2,
        "comparator_cell_count": 2,
        "formal_new_branch_order": ["incomplete", "incomplete"],
        "selected_cells": [
            ["00435ad621c44fac", 2026080821, "sp4-a"],
            ["00435ad621c44fac", 2026080921, "sp4-a"],
        ],
        "external_action_authority": {
            "source_experiment_id": "BOX-EXP-011",
            "sealed_key_file_sha256": EXTERNAL_KEY_SHA256,
            "blind_reviewer_receipt_file_sha256": EXTERNAL_REVIEW_RECEIPT_SHA256,
            "blind_complete_and_hold": [2, 2],
            "mp4_native_calibration_gaussian_physical_reopen_required": True,
        },
        "prompt_utf8_sha256": PROMPT_SHA256,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        "forbidden_prompt_tokens": ["hip", "hips", "hands-on-hips"],
        "forbidden_prompt_token_count": 0,
        "prompts_frozen_before_any_new_media": True,
        "num_inference_steps_per_clip": 40,
        "frame_count_per_clip": 81,
        "cross_run_same_seed_official_gaussian_required_per_cell": True,
        "completion_physically_recomputes_each_gaussian_proof": True,
        "generation_audit_binds_generated_mp4_path_sha_exact81_25fps": True,
        "post_generation_blind_review_input_create_only": True,
        "review_tool_creates_no_decision_or_verdict": True,
        "blind_copy_canonical_opaque_filename_and_reviewer_dir_required": True,
        "blind_copy_path_and_inode_distinct_from_generated_required": True,
        "blind_copy_bytes_sha_equal_to_generated_required": True,
        "manifest_key_review_topology_chain_replayed_at_completion": True,
        "portable_ffprobe": {
            "path": PORTABLE_FFPROBE_PATH,
            "file_sha256": PORTABLE_FFPROBE_SHA256,
            "caller_override_allowed": False,
            "compute_child_external_action_exact81_25fps_preflight_required": True,
            "review_and_completion_revalidation_required": True,
        },
        "compute_child_scratch_filesystem_type": {
            "source": (
                "signed child prepare for launcher-created exact /tmp outer; "
                "caller SLURM_TMPDIR is absent after parent env scrub and is not authority"
            ),
            "authority": "launcher_created_compute_child_tmp",
            "allowed_raw_values": ["ext2/ext3"],
            "required_mount_filesystem": "ext4",
            "required_mount_source": "/dev/mapper/vgroot-lvroot",
            "required_mount_major_minor": "253:0",
            "slurm_tmpdir_exported_or_fabricated": False,
            "caller_declaration_allowed": False,
            "prepare_compute_bind_attestation_retention_chain_required": True,
            "success_and_failure_scratch_retained_point_in_time": True,
            "scratch_reusable": False,
            "manual_cleanup_authorized_by_release": False,
            "future_persistence_guaranteed": False,
        },
        "frozen_python": {
            "path": FROZEN_PYTHON_PATH,
            "realpath": FROZEN_PYTHON_REALPATH,
            "file_type": FROZEN_PYTHON_FILE_TYPE,
            "mode_octal": FROZEN_PYTHON_MODE_OCTAL,
            "uid": FROZEN_PYTHON_UID,
            "size": FROZEN_PYTHON_SIZE,
            "link_count": FROZEN_PYTHON_LINK_COUNT,
            "file_sha256": FROZEN_PYTHON_SHA256,
            "rejected_symlink_alias": REJECTED_PYTHON_SYMLINK_PATH,
            "caller_override_allowed": False,
            "symlink_allowed": False,
            "physical_validation_before_run_root_or_srun_required": True,
        },
        "revoked_live_attempt": {
            "holder_step": "136140.14",
            "release_leaf": (
                "full30-action-arms-incomplete-exact2-rfinal-96c4095e-a59fb218"
            ),
            "run_leaf": (
                "full30-action-arms-incomplete-exact2-rfinal-96c4095e-j136140-r1"
            ),
            "terminal_log_file_sha256": REVOKED_LIVE_LOG_SHA256,
            "permanent_no_go": True,
            "generated_candidate_count": 0,
        },
        "revoked_static_portable_r2": {
            "archive_file_sha256": REVOKED_PORTABLE_R2_ARCHIVE_SHA256,
            "manifest_file_sha256": REVOKED_PORTABLE_R2_MANIFEST_SHA256,
            "launcher_file_sha256": REVOKED_PORTABLE_R2_LAUNCHER_SHA256,
            "permanent_no_go": True,
        },
        "revoked_terminal_physical_r3": {
            "archive_file_sha256": REVOKED_TERMINAL_R3_ARCHIVE_SHA256,
            "manifest_file_sha256": REVOKED_TERMINAL_R3_MANIFEST_SHA256,
            "detached_launcher_file_sha256": (
                REVOKED_TERMINAL_R3_LAUNCHER_SHA256
            ),
            "deployment_envelope_file_sha256": (
                REVOKED_TERMINAL_R3_ENVELOPE_SHA256
            ),
            "permanent_no_go": True,
        },
        "revoked_resource_reuse_r4": {
            "archive_file_sha256": REVOKED_RESOURCE_REUSE_R4_ARCHIVE_SHA256,
            "manifest_file_sha256": REVOKED_RESOURCE_REUSE_R4_MANIFEST_SHA256,
            "detached_launcher_file_sha256": (
                REVOKED_RESOURCE_REUSE_R4_LAUNCHER_SHA256
            ),
            "deployment_envelope_file_sha256": (
                REVOKED_RESOURCE_REUSE_R4_ENVELOPE_SHA256
            ),
            "release_root": REVOKED_RESOURCE_REUSE_R4_RELEASE_ROOT,
            "materialization_root": (
                REVOKED_RESOURCE_REUSE_R4_MATERIALIZATION_ROOT
            ),
            "requested_run_root": REVOKED_RESOURCE_REUSE_R4_RUN_ROOT,
            "launcher_invocation_count": 1,
            "numbered_child_count": 0,
            "run_root_created": False,
            "gpu_or_model_invocation_count": 0,
            "failure": "configured Python path was a symlink",
            "permanent_no_go": True,
        },
        "revoked_terminal_fail_close_r5": {
            "archive_file_sha256": REVOKED_TERMINAL_R5_ARCHIVE_SHA256,
            "manifest_file_sha256": REVOKED_TERMINAL_R5_MANIFEST_SHA256,
            "detached_launcher_file_sha256": REVOKED_TERMINAL_R5_LAUNCHER_SHA256,
            "deployment_envelope_file_sha256": REVOKED_TERMINAL_R5_ENVELOPE_SHA256,
            "release_root": REVOKED_TERMINAL_R5_RELEASE_ROOT,
            "materialization_root": REVOKED_TERMINAL_R5_MATERIALIZATION_ROOT,
            "requested_run_root": REVOKED_TERMINAL_R5_RUN_ROOT,
            "holder_step": "136140.15",
            "terminal_state": "FAILED",
            "terminal_exit_code": "1:0",
            "elapsed_seconds": 1,
            "terminal_log_file_sha256": REVOKED_TERMINAL_R5_LOG_SHA256,
            "formal_candidate_count": 0,
            "gpu_or_model_invocation_count": 0,
            "failure": "compute child SLURM_TMPDIR absent",
            "exact_roots_descendants_and_renamed_identity_copies_forbidden": True,
            "permanent_no_go": True,
        },
        "resource_module_loaded_once_and_exact_object_reused": True,
        "preloaded_or_replaced_resource_module_rejected": True,
        "terminal_preflight_monitor_job_step_exact_match_required": True,
        "terminal_gate_physical_rederivation_required_at_validation_and_completion": True,
        "completion_reopens_hashes_and_probes_each_reviewed_generated_mp4": True,
        "diagnostic_task_count": 0,
        "diagnostic_generation_allowed": False,
        "action_generation_allowed": False,
        "q_input_authorized": False,
        "a_min_input_authorized": False,
        "generated_media_can_enter_optimizer": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "training_authorized": False,
    }


def _topology() -> Mapping[str, Any]:
    return {
        "holder": {"job_id": 136140, "node": "auh7-1b-gpu-215"},
        "static_release_binds_live_child": False,
        "slurm_child_gpu_count": 8,
        "compute_world_size": 4,
        "parallelism": "dp1_sp4_one_model_replica_at_a_time",
        "formal_shard_count": 1,
        "formal_world4_model_invocation_count": 2,
        "compile_smoke_world4_model_invocation_count": 1,
        "total_required_native_model_invocation_count": 3,
        "diagnostic_model_invocation_count": 0,
        "all_required_model_invocations_strictly_serial": True,
        "physical_island_used": [0, 1, 2, 3],
        "fresh_run_root_required": True,
        "frozen_python_physical_validation_before_run_root_or_srun": True,
        "host_memory_request_gib": 60,
        "host_sampled_current_safe_ceiling_gib": 56,
        "host_cgroup_sample_interval_ns": 10_000_000,
        "host_cgroup_max_sample_gap_ns": 100_000_000,
        "t2v_rank_gpu_memory_limit_gib": 52,
        "per_rank_node_local_cache_wrapper": True,
        "scratch_authority": "launcher_created_compute_child_tmp",
        "slurm_tmpdir_exported_or_fabricated": False,
        "outer_exact_leaf_create_only": True,
        "inner_and_renderer_lock_controller_bound_before_export": True,
        "success_scratch_retained_at_child_terminal_seal": True,
        "failure_scratch_retained_when_identity_observable": True,
        "scratch_reusable": False,
        "physical_or_manual_scratch_cleanup_authorized": False,
        "future_scratch_persistence_guaranteed": False,
        "parent_child_tmp_or_proc_physical_replay_allowed": False,
        "compute_child_portable_ffprobe_preflight_before_monitor_smoke_model": True,
        "compute_child_portable_ffprobe_preflight_before_source_plan_replay": True,
        "compute_child_external_action_exact81_25fps_probe_count": 2,
        "compute_child_scratch_stat_f_receipt_before_monitor_smoke_model": True,
        "scratch_fstype_export_only_from_compute_preflight_receipt": True,
        "generation_audit_replays_rank_resource_scratch_receipt_chain": True,
        "core_archive_member_count": 25,
        "frozen_base_byte_exact_member_count": 18,
        "r6_owned_replacement_member_count": 3,
        "detached_launcher_excluded_from_core": True,
        "deployment_envelope_remote_exact_entry_count": 4,
        "terminal_gate_reopens_preflight_monitor_start_and_journal": True,
        "terminal_preflight_monitor_job_step_exact_match_required": True,
        "physical_safetensors_safe_open_required": True,
        "terminal_zero_oom_and_oom_kill_required": True,
        "parent_cancel_release_requeue_forbidden": True,
    }


def build_manifest(
    method_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir():
        fail("method root must be one canonical directory")
    payloads: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    frozen_payloads = _frozen_base_payloads(root)
    r6_resource = _stable_plain_bytes(
        root / "tools/reserve4_fixed_generation_sp4_v1.py"
    )
    replacement_payloads = {
        "scripts/auh_generic_action_data_prep_rank_exec_v1.sh": (
            _stable_plain_bytes(
                root / "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
            )
        ),
        "tools/reserve4_fixed_generation_sp4_v1.py": r6_resource,
        "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py": (
            _specialize_r6_resource(r6_resource)
        ),
    }
    for relative, raw in replacement_payloads.items():
        if hashlib.sha256(raw).hexdigest() != R6_REPLACEMENT_PINS[relative][1]:
            fail(f"r6-owned replacement member drifted: {relative}")
    for relative in sorted(FILES_AND_MODES):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("release member path differs")
        raw = (
            replacement_payloads[relative]
            if relative in replacement_payloads
            else frozen_payloads[relative]
            if relative in FROZEN_MEMBER_PINS
            else _stable_plain_bytes(root / relative)
        )
        if relative in FROZEN_MEMBER_PINS and relative not in replacement_payloads:
            _, expected_sha = FROZEN_MEMBER_PINS[relative]
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                fail(f"frozen BOX-EXP-011 member drifted: {relative}")
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "mode": FILES_AND_MODES[relative],
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    row_by_path = {row["path"]: row for row in rows}
    component_pins = {
        label: row_by_path[relative]["sha256"]
        for label, relative in COMPONENT_FILES.items()
    }
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "component_pins": component_pins,
        "allowed_entrypoints": list(ENTRYPOINTS),
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(
            canonical_json_bytes(closure)
        ).hexdigest(),
        "exact_member_closure": True,
        "release_scope": (
            "BOX-EXP-013-arms-incomplete-only-exact2-r6-"
            "launcher-created-child-scratch-retention"
        ),
        "frozen_base": {
            "source_experiment_id": "BOX-EXP-011",
            "archive_sha256": FROZEN_BASE_ARCHIVE_SHA256,
            "manifest_sha256": FROZEN_BASE_MANIFEST_SHA256,
            "manifest_digest": FROZEN_BASE_MANIFEST_DIGEST,
            "method_revision": FROZEN_BASE_REVISION,
            "source_member_count": 21,
            "byte_exact_carried_member_count": 18,
            "r6_owned_replacement_member_count": 3,
            "r6_owned_replacement_members": sorted(R6_REPLACEMENT_PINS),
            "all_members_byte_exact": False,
            "modified": True,
        },
        "topology": _topology(),
        "authority": _authority(),
    }
    manifest = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    validate_manifest(manifest)
    return manifest, payloads


def build_archive(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            raw = payloads[row["path"]]
            info = tarfile.TarInfo(f"{MEMBER_ROOT}/{row['path']}")
            info.size = len(raw)
            info.mode = row["mode"]
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(raw))
    result = stream.getvalue()
    verify_archive(result, manifest)
    return result


def deployment_envelope_value(
    *,
    manifest: Mapping[str, Any],
    archive_sha256: str,
    manifest_sha256: str,
    detached_launcher_sha256: str,
) -> Mapping[str, Any]:
    business_labels = (
        "prompt_plan_sha256",
        "generator_sha256",
        "controller_sha256",
        "release_builder_sha256",
    )
    launcher_basename = LAUNCHER_MEMBER.rsplit("/", 1)[-1]
    unsigned = {
        "schema_version": DEPLOYMENT_ENVELOPE_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "core_archive": {
            "basename": "source.tar",
            "file_sha256": archive_sha256,
            "expected_mode_octal": "0444",
        },
        "core_manifest": {
            "basename": "source.manifest.json",
            "file_sha256": manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
            "content_closure_sha1": manifest["content_closure_sha1"],
            "file_count": manifest["file_count"],
            "expected_mode_octal": "0444",
        },
        "owned_business_component_sha256": {
            label: manifest["component_pins"][label] for label in business_labels
        },
        "detached_launcher": {
            "basename": launcher_basename,
            "file_sha256": detached_launcher_sha256,
            "expected_mode_octal": "0555",
            "path_independent_identity": True,
            "excluded_from_core_archive_and_manifest": True,
        },
        "deployment_envelope": {
            "basename": "deployment-envelope.json",
            "expected_mode_octal": "0444",
        },
        "remote_release_exact_entries": [
            "source.tar",
            "source.manifest.json",
            launcher_basename,
            "deployment-envelope.json",
        ],
        "remote_release_exact_entry_count": 4,
        "remote_release_extra_entries_allowed": False,
        "remote_release_entries_canonical_regular_single_link": True,
        "create_only_deployment_required": True,
        "static_envelope_binds_live_child": False,
    }
    return {**unsigned, "envelope_digest": object_sha256(unsigned)}


def validate_deployment_envelope(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    archive_sha256: str,
    manifest_sha256: str,
    detached_launcher_sha256: str,
) -> Mapping[str, Any]:
    expected = deployment_envelope_value(
        manifest=manifest,
        archive_sha256=archive_sha256,
        manifest_sha256=manifest_sha256,
        detached_launcher_sha256=detached_launcher_sha256,
    )
    if type(value) is not dict or value != expected:
        fail("deployment envelope identity/closure differs")
    return value


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            expected_names = [
                f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]
            ]
            if [member.name for member in members] != expected_names:
                fail("archive exact member order differs")
            for member, row in zip(members, manifest["files"]):
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or member.mode != row["mode"]
                    or member.size != row["size"]
                ):
                    fail(f"archive member metadata differs: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    fail(f"archive member cannot be read: {member.name}")
                if hashlib.sha256(handle.read()).hexdigest() != row["sha256"]:
                    fail(f"archive member content differs: {member.name}")
    except (tarfile.TarError, OSError) as error:
        raise ArmsIncompleteExact2ReleaseError("archive is invalid") from error


def validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_top_level_fields = {
        "schema_version",
        "release_generation",
        "archive_format",
        "member_root",
        "file_count",
        "files",
        "component_pins",
        "allowed_entrypoints",
        "revision_kind",
        "content_closure_sha1",
        "exact_member_closure",
        "release_scope",
        "frozen_base",
        "topology",
        "authority",
        "manifest_digest",
    }
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest", None)
    if (
        type(value) is not dict
        or set(value) != expected_top_level_fields
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("exact_member_closure") is not True
        or value.get("allowed_entrypoints") != list(ENTRYPOINTS)
        or value.get("revision_kind") != "content-closure-sha1"
        or value.get("release_scope")
        != (
            "BOX-EXP-013-arms-incomplete-only-exact2-r6-"
            "launcher-created-child-scratch-retention"
        )
        or not isinstance(declared, str)
        or SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        fail("release manifest schema/digest differs")
    rows = value.get("files")
    if (
        not isinstance(rows, list)
        or len(rows) != len(FILES_AND_MODES)
        or value.get("file_count") != len(rows)
        or [row.get("path") for row in rows] != sorted(FILES_AND_MODES)
    ):
        fail("release manifest file closure differs")
    expected_closure_sha1 = hashlib.sha1(
        canonical_json_bytes({"member_root": MEMBER_ROOT, "files": rows})
    ).hexdigest()
    if value.get("content_closure_sha1") != expected_closure_sha1:
        fail("release manifest content closure differs")
    seen: set[str] = set()
    for row in rows:
        path = row.get("path")
        if (
            type(row) is not dict
            or set(row) != {"path", "mode", "size", "sha256"}
            or not isinstance(path, str)
            or path in seen
            or row.get("mode") != FILES_AND_MODES.get(path)
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or SHA256_RE.fullmatch(str(row.get("sha256"))) is None
        ):
            fail("release manifest file row differs")
        seen.add(path)
    row_by_path = {row["path"]: row for row in rows}
    for path, (_, expected_sha) in FROZEN_MEMBER_PINS.items():
        if path in R6_REPLACEMENT_PINS:
            continue
        if row_by_path[path]["sha256"] != expected_sha:
            fail(f"frozen member manifest pin differs: {path}")
    for path, (_, expected_sha) in R6_REPLACEMENT_PINS.items():
        if row_by_path[path]["sha256"] != expected_sha:
            fail(f"r6-owned replacement manifest pin differs: {path}")
    expected_components = {
        label: row_by_path[path]["sha256"]
        for label, path in COMPONENT_FILES.items()
    }
    frozen = value.get("frozen_base", {})
    if (
        value.get("component_pins") != expected_components
        or type(frozen) is not dict
        or set(frozen)
        != {
            "source_experiment_id",
            "archive_sha256",
            "manifest_sha256",
            "manifest_digest",
            "method_revision",
            "source_member_count",
            "byte_exact_carried_member_count",
            "r6_owned_replacement_member_count",
            "r6_owned_replacement_members",
            "all_members_byte_exact",
            "modified",
        }
        or frozen.get("source_experiment_id") != "BOX-EXP-011"
        or frozen.get("archive_sha256") != FROZEN_BASE_ARCHIVE_SHA256
        or frozen.get("manifest_sha256") != FROZEN_BASE_MANIFEST_SHA256
        or frozen.get("manifest_digest") != FROZEN_BASE_MANIFEST_DIGEST
        or frozen.get("method_revision") != FROZEN_BASE_REVISION
        or frozen.get("source_member_count") != 21
        or frozen.get("byte_exact_carried_member_count") != 18
        or frozen.get("r6_owned_replacement_member_count") != 3
        or frozen.get("r6_owned_replacement_members")
        != sorted(R6_REPLACEMENT_PINS)
        or frozen.get("all_members_byte_exact") is not False
        or frozen.get("modified") is not True
        or value.get("authority") != _authority()
        or value.get("topology") != _topology()
    ):
        fail("release frozen-base/authority/topology differs")
    return value


def _write_create_only(path: Path, raw: bytes, *, mode: int) -> None:
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        fail("release output must be a fresh absolute path")
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                fail("release output write made no progress")
            offset += written
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(sealed.st_mode)
            or stat.S_IMODE(sealed.st_mode) != mode
            or sealed.st_nlink != 1
            or sealed.st_size != len(raw)
        ):
            fail("release output opened-file identity differs")
        os.close(descriptor)
        descriptor = None
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def build(
    method_root: Path,
    archive: Path,
    manifest_path: Path,
    detached_launcher: Path,
    deployment_envelope_path: Path,
) -> Mapping[str, Any]:
    launcher_basename = LAUNCHER_MEMBER.rsplit("/", 1)[-1]
    deployment_root = archive.parent
    expected_paths = {
        "source.tar": archive,
        "source.manifest.json": manifest_path,
        launcher_basename: detached_launcher,
        "deployment-envelope.json": deployment_envelope_path,
    }
    try:
        root_resolved = deployment_root.resolve(strict=True)
        launcher_metadata = detached_launcher.stat()
    except OSError as error:
        raise ArmsIncompleteExact2ReleaseError(
            "fresh deployment root/detached launcher is unavailable"
        ) from error
    if (
        not deployment_root.is_absolute()
        or deployment_root.is_symlink()
        or root_resolved != deployment_root
        or not deployment_root.is_dir()
        or any(
            path.parent != deployment_root or path.name != basename
            for basename, path in expected_paths.items()
        )
        or {entry.name for entry in deployment_root.iterdir()}
        != {launcher_basename}
        or detached_launcher.is_symlink()
        or detached_launcher.resolve(strict=True) != detached_launcher
        or not stat.S_ISREG(launcher_metadata.st_mode)
        or launcher_metadata.st_nlink != 1
        or stat.S_IMODE(launcher_metadata.st_mode) != 0o555
    ):
        fail("build requires a fresh exact-four root containing only mode-0555 launcher")
    manifest, payloads = build_manifest(method_root)
    archive_raw = build_archive(manifest, payloads)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    archive_sha = hashlib.sha256(archive_raw).hexdigest()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    launcher_sha = hashlib.sha256(
        _stable_plain_bytes(detached_launcher)
    ).hexdigest()
    envelope = deployment_envelope_value(
        manifest=manifest,
        archive_sha256=archive_sha,
        manifest_sha256=manifest_sha,
        detached_launcher_sha256=launcher_sha,
    )
    envelope_raw = canonical_json_bytes(envelope) + b"\n"
    _write_create_only(archive, archive_raw, mode=0o444)
    try:
        _write_create_only(manifest_path, manifest_raw, mode=0o444)
        _write_create_only(
            deployment_envelope_path, envelope_raw, mode=0o444
        )
    except Exception:
        archive.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        deployment_envelope_path.unlink(missing_ok=True)
        raise
    envelope_sha = hashlib.sha256(envelope_raw).hexdigest()
    audit_deployment(
        archive=archive,
        expected_archive_sha256=archive_sha,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        detached_launcher=detached_launcher,
        expected_detached_launcher_sha256=launcher_sha,
        deployment_envelope_path=deployment_envelope_path,
        expected_deployment_envelope_sha256=envelope_sha,
    )
    return {
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "detached_launcher": str(detached_launcher),
        "detached_launcher_sha256": launcher_sha,
        "deployment_envelope": str(deployment_envelope_path),
        "deployment_envelope_sha256": envelope_sha,
        "deployment_envelope_digest": envelope["envelope_digest"],
        "file_count": manifest["file_count"],
    }


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def audit(
    archive: Path, expected_archive_sha256: str, manifest_path: Path,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    archive_raw = _stable_plain_bytes(archive)
    manifest_raw = _stable_plain_bytes(manifest_path)
    if hashlib.sha256(archive_raw).hexdigest() != expected_archive_sha256:
        fail("release archive SHA-256 differs")
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        fail("release manifest SHA-256 differs")
    try:
        manifest = json.loads(
            manifest_raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArmsIncompleteExact2ReleaseError(
            "release manifest is invalid JSON"
        ) from error
    if canonical_json_bytes(manifest) + b"\n" != manifest_raw:
        fail("release manifest bytes are not canonical")
    validate_manifest(manifest)
    verify_archive(archive_raw, manifest)
    return manifest


def audit_deployment(
    *,
    archive: Path,
    expected_archive_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    detached_launcher: Path,
    expected_detached_launcher_sha256: str,
    deployment_envelope_path: Path,
    expected_deployment_envelope_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_paths = {
        "source.tar": archive,
        "source.manifest.json": manifest_path,
        LAUNCHER_MEMBER.rsplit("/", 1)[-1]: detached_launcher,
        "deployment-envelope.json": deployment_envelope_path,
    }
    deployment_root = archive.parent
    try:
        root_resolved = deployment_root.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ReleaseError(
            "deployment root is unavailable"
        ) from error
    if (
        not deployment_root.is_absolute()
        or deployment_root.is_symlink()
        or root_resolved != deployment_root
        or not deployment_root.is_dir()
        or any(
            path.parent != deployment_root or path.name != basename
            for basename, path in expected_paths.items()
        )
        or {entry.name for entry in deployment_root.iterdir()}
        != set(expected_paths)
    ):
        fail("deployment root exact-four canonical topology differs")
    expected_modes = {
        "source.tar": 0o444,
        "source.manifest.json": 0o444,
        LAUNCHER_MEMBER.rsplit("/", 1)[-1]: 0o555,
        "deployment-envelope.json": 0o444,
    }
    for basename, path in expected_paths.items():
        try:
            metadata = path.stat()
        except OSError as error:
            raise ArmsIncompleteExact2ReleaseError(
                f"deployment entry is unavailable: {basename}"
            ) from error
        if (
            path.is_symlink()
            or path.resolve(strict=True) != path
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != expected_modes[basename]
        ):
            fail(f"deployment entry identity/mode differs: {basename}")
    manifest = audit(
        archive,
        expected_archive_sha256,
        manifest_path,
        expected_manifest_sha256,
    )
    launcher_raw = _stable_plain_bytes(detached_launcher)
    launcher_sha = hashlib.sha256(launcher_raw).hexdigest()
    if launcher_sha != expected_detached_launcher_sha256:
        fail("detached launcher SHA-256 differs")
    envelope_raw = _stable_plain_bytes(deployment_envelope_path)
    if (
        hashlib.sha256(envelope_raw).hexdigest()
        != expected_deployment_envelope_sha256
    ):
        fail("deployment envelope SHA-256 differs")
    try:
        envelope = json.loads(
            envelope_raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArmsIncompleteExact2ReleaseError(
            "deployment envelope is invalid JSON"
        ) from error
    if canonical_json_bytes(envelope) + b"\n" != envelope_raw:
        fail("deployment envelope bytes are not canonical")
    validate_deployment_envelope(
        envelope,
        manifest=manifest,
        archive_sha256=expected_archive_sha256,
        manifest_sha256=expected_manifest_sha256,
        detached_launcher_sha256=launcher_sha,
    )
    return manifest, envelope


def _verified_materialization_payloads(
    *,
    method_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_runner_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    """Capture the exact release closure before any release module executes."""

    if any(name in os.environ for name in UNSAFE_PYTHON_ENVIRONMENT):
        fail("caller-controlled Python environment presence is forbidden")
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        fail("verified release runner requires python -I -S")
    if (
        SHA256_RE.fullmatch(expected_manifest_sha256) is None
        or SHA256_RE.fullmatch(expected_runner_sha256) is None
    ):
        fail("verified release runner SHA declaration differs")
    manifest_raw = _stable_plain_bytes(manifest_path)
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        fail("verified release runner manifest SHA differs")
    try:
        manifest = json.loads(
            manifest_raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArmsIncompleteExact2ReleaseError(
            "verified release runner manifest JSON differs"
        ) from error
    if canonical_json_bytes(manifest) + b"\n" != manifest_raw:
        fail("verified release runner manifest is not canonical")
    validate_manifest(manifest)
    if not method_root.is_absolute() or method_root.is_symlink():
        fail("verified release runner method root differs")
    try:
        resolved_root = method_root.resolve(strict=True)
        root_before = method_root.lstat()
    except OSError as error:
        raise ArmsIncompleteExact2ReleaseError(
            "verified release runner method root is unavailable"
        ) from error
    if (
        resolved_root != method_root
        or not stat.S_ISDIR(root_before.st_mode)
        or root_before.st_nlink < 2
    ):
        fail("verified release runner method root identity differs")
    rows = manifest["files"]
    expected_files = {row["path"] for row in rows}
    expected_directories = {"."}
    for relative in expected_files:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("verified release runner member path differs")
        parent = pure.parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    observed_files: set[str] = set()
    observed_directories = {"."}
    for current, directories, files in os.walk(method_root, topdown=True):
        current_path = Path(current)
        relative_current = current_path.relative_to(method_root)
        current_key = (
            "." if relative_current == Path(".") else relative_current.as_posix()
        )
        try:
            current_metadata = current_path.lstat()
        except OSError as error:
            raise ArmsIncompleteExact2ReleaseError(
                "verified release runner directory is unavailable"
            ) from error
        if (
            current_path.is_symlink()
            or not stat.S_ISDIR(current_metadata.st_mode)
            or current_metadata.st_nlink < 2
            or stat.S_IMODE(current_metadata.st_mode) & 0o022
        ):
            fail("verified release runner directory topology/mode differs")
        directories.sort()
        files.sort()
        for name in directories:
            child = current_path / name
            child_relative = child.relative_to(method_root).as_posix()
            if child.is_symlink():
                fail("verified release runner directory symlink is forbidden")
            observed_directories.add(child_relative)
        for name in files:
            observed_files.add((current_path / name).relative_to(method_root).as_posix())
        if current_key not in expected_directories:
            fail("verified release runner found an unexpected directory")
    if observed_files != expected_files or observed_directories != expected_directories:
        fail("verified release runner exact materialization closure differs")

    payloads: dict[str, bytes] = {}
    row_by_path = {row["path"]: row for row in rows}
    for relative in sorted(expected_files):
        row = row_by_path[relative]
        member = method_root / relative
        raw = _stable_plain_bytes(member)
        metadata = member.lstat()
        if (
            metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != row["mode"]
            or len(raw) != row["size"]
            or hashlib.sha256(raw).hexdigest() != row["sha256"]
        ):
            fail(f"verified release runner member identity differs: {relative}")
        payloads[relative] = raw
    try:
        root_after = method_root.lstat()
    except OSError as error:
        raise ArmsIncompleteExact2ReleaseError(
            "verified release runner method root disappeared"
        ) from error
    stable_root_fields = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_uid,
        item.st_gid,
        item.st_mode,
        item.st_nlink,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if stable_root_fields(root_before) != stable_root_fields(root_after):
        fail("verified release runner method root changed during capture")
    runner_row = row_by_path.get(BUILDER_MEMBER)
    if (
        type(runner_row) is not dict
        or runner_row.get("sha256") != expected_runner_sha256
        or hashlib.sha256(payloads[BUILDER_MEMBER]).hexdigest()
        != expected_runner_sha256
    ):
        fail("verified release runner bootstrap/member SHA differs")
    return manifest, payloads


class _VerifiedReleaseBytesLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, path: Path, raw: bytes) -> None:
        self.fullname = fullname
        self.path = path
        self.raw = raw

    def create_module(self, spec: Any) -> Optional[ModuleType]:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self.path)
        module.__package__ = ""
        exec(
            compile(self.raw, str(self.path), "exec", dont_inherit=True),
            module.__dict__,
        )


class _VerifiedReleaseFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        *,
        method_root: Path,
        modules: Mapping[str, tuple[Path, bytes]],
    ) -> None:
        self.method_root = method_root
        self.tools_root = method_root / "tools"
        self.modules = modules

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> Any:
        # Released modules sometimes insert their own directory into sys.path.
        # Remove both release roots before delegating any non-release import so
        # PathFinder can never reopen a swapped or newly added local file.
        forbidden = {str(self.method_root), str(self.tools_root), ""}
        sys.path[:] = [entry for entry in sys.path if entry not in forbidden]
        item = self.modules.get(fullname)
        if item is None:
            return None
        source, raw = item
        loader = _VerifiedReleaseBytesLoader(fullname, source, raw)
        return importlib.machinery.ModuleSpec(
            fullname, loader, origin=str(source), is_package=False
        )


def verified_run_module(
    *,
    method_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_runner_sha256: str,
    target: str,
    target_arguments: Sequence[str],
) -> int:
    """Execute a target and all release-local imports from captured bytes."""

    manifest, payloads = _verified_materialization_payloads(
        method_root=method_root,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_runner_sha256=expected_runner_sha256,
    )
    target_pure = PurePosixPath(target)
    if (
        target_pure.is_absolute()
        or ".." in target_pure.parts
        or target_pure.as_posix() != target
        or target not in payloads
        or target_pure.suffix != ".py"
    ):
        fail("verified release runner target differs")
    module_rows: dict[str, tuple[Path, bytes]] = {}
    for relative, raw in payloads.items():
        pure = PurePosixPath(relative)
        if pure.suffix != ".py":
            continue
        module_name = pure.stem
        if module_name in module_rows:
            fail("verified release runner module basename is ambiguous")
        module_rows[module_name] = (method_root / relative, raw)
    if any(name in sys.modules for name in module_rows):
        fail("release-local module was imported before verified closure")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / version / "site-packages"
    try:
        site_packages_resolved = site_packages.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2ReleaseError(
            "isolated interpreter site-packages is unavailable"
        ) from error
    if site_packages_resolved != site_packages or not site_packages.is_dir():
        fail("isolated interpreter site-packages identity differs")
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry not in {"", str(method_root), str(method_root / "tools")}
    ]
    if str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))
    finder = _VerifiedReleaseFinder(method_root=method_root, modules=module_rows)
    sys.meta_path.insert(0, finder)
    sys.dont_write_bytecode = True
    target_path = method_root / target
    arguments = list(target_arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    sys.argv = [str(target_path), *arguments]
    globals_value: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(target_path),
        "__package__": None,
        "__loader__": _VerifiedReleaseBytesLoader(
            "__main__", target_path, payloads[target]
        ),
        "__spec__": None,
        "__builtins__": __builtins__,
    }
    exec(
        compile(payloads[target], str(target_path), "exec", dont_inherit=True),
        globals_value,
    )
    return 0


def _stable_executable_fd(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_uid: int,
    expected_gid: int,
) -> int:
    """Open, verify, and retain one exact executable inode for fd execve."""

    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or SHA256_RE.fullmatch(expected_sha256) is None
    ):
        fail("held-fd executable authority differs")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )

    def fields(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_uid,
            value.st_gid,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_blocks,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def hash_pass() -> tuple[int, str]:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return size, digest.hexdigest()
            digest.update(chunk)
            size += len(chunk)

    try:
        opened = os.fstat(descriptor)
        first_size, first_sha256 = hash_pass()
        middle = os.fstat(descriptor)
        second_size, second_sha256 = hash_pass()
        after = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o755
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or fields(opened)
            != fields(middle)
            or fields(opened) != fields(after)
            or fields(opened) != fields(named)
            or first_size != expected_size
            or second_size != expected_size
            or first_sha256 != expected_sha256
            or second_sha256 != expected_sha256
        ):
            fail("held-fd executable physical identity/SHA differs")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def held_fd_exec_frozen_python(
    target_arguments: Sequence[str], *, start_gate_stdin: bool
) -> NoReturn:
    """Use root-owned Python only to fexecve the exact frozen Python inode."""

    if (
        sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or sys.flags.dont_write_bytecode != 1
        or not target_arguments
        or target_arguments[0] != "--"
        or len(target_arguments) == 1
    ):
        fail("held-fd Python bootstrap flags/arguments differ")
    try:
        running_path = os.readlink("/proc/self/exe")
        running_identity = os.stat("/proc/self/exe")
    except OSError as error:
        raise ArmsIncompleteExact2ReleaseError(
            "root-owned bootstrap executable identity is unavailable"
        ) from error
    if running_path != str(ROOT_BOOTSTRAP_PYTHON):
        fail("running executable is not the root-owned bootstrap Python")
    root_descriptor = _stable_executable_fd(
        ROOT_BOOTSTRAP_PYTHON,
        expected_sha256=ROOT_BOOTSTRAP_PYTHON_SHA256,
        expected_size=ROOT_BOOTSTRAP_PYTHON_SIZE,
        expected_uid=0,
        expected_gid=0,
    )
    try:
        root_identity = os.fstat(root_descriptor)
        if (root_identity.st_dev, root_identity.st_ino) != (
            running_identity.st_dev,
            running_identity.st_ino,
        ):
            fail("running root-owned bootstrap inode differs")
    finally:
        os.close(root_descriptor)
    for name in os.environ:
        if name in UNSAFE_EXEC_ENVIRONMENT or name.startswith(
            ("LD_", "DYLD_", "GLIBC_", "MALLOC_", "BASH_FUNC_")
        ):
            fail("unsafe loader/shell/Python environment is present")
    target_descriptor = _stable_executable_fd(
        FROZEN_PYTHON,
        expected_sha256=FROZEN_PYTHON_SHA256,
        expected_size=FROZEN_PYTHON_SIZE,
        expected_uid=2012,
        expected_gid=2000,
    )
    if os.execve not in os.supports_fd:
        os.close(target_descriptor)
        fail("held-fd execve is unavailable")
    if start_gate_stdin:
        try:
            gate_metadata = os.fstat(0)
            gate_token = bytearray()
            while len(gate_token) < 4:
                chunk = os.read(0, 4 - len(gate_token))
                if not chunk:
                    break
                gate_token.extend(chunk)
            gate_trailer = os.read(0, 1)
        except OSError as error:
            os.close(target_descriptor)
            raise ArmsIncompleteExact2ReleaseError(
                "anonymous candidate start gate read failed"
            ) from error
        if (
            not stat.S_ISFIFO(gate_metadata.st_mode)
            or bytes(gate_token) != b"go\n"
            or gate_trailer != b""
        ):
            os.close(target_descriptor)
            fail("anonymous candidate start gate token differs")
    environment = dict(os.environ)
    environment["PATH"] = "/usr/bin:/bin"
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    try:
        os.execve(
            target_descriptor,
            [str(FROZEN_PYTHON), *target_arguments[1:]],
            environment,
        )
    finally:
        os.close(target_descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--method-root", required=True)
    create.add_argument("--archive", required=True)
    create.add_argument("--manifest", required=True)
    create.add_argument("--detached-launcher", required=True)
    create.add_argument("--deployment-envelope", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--archive", required=True)
    check.add_argument("--expected-archive-sha256", required=True)
    check.add_argument("--manifest", required=True)
    check.add_argument("--expected-manifest-sha256", required=True)
    check.add_argument("--detached-launcher", required=True)
    check.add_argument("--expected-detached-launcher-sha256", required=True)
    check.add_argument("--deployment-envelope", required=True)
    check.add_argument("--expected-deployment-envelope-sha256", required=True)
    verified = commands.add_parser("verified-run-module")
    verified.add_argument("--method-root", required=True)
    verified.add_argument("--manifest", required=True)
    verified.add_argument("--expected-manifest-sha256", required=True)
    verified.add_argument("--expected-runner-sha256", required=True)
    verified.add_argument("--target", required=True)
    verified.add_argument("target_arguments", nargs=argparse.REMAINDER)
    held_exec = commands.add_parser("held-fd-exec-frozen-python")
    held_exec.add_argument("--start-gate-stdin", action="store_true")
    held_exec.add_argument("target_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        value = build(
            Path(args.method_root),
            Path(args.archive),
            Path(args.manifest),
            Path(args.detached_launcher),
            Path(args.deployment_envelope),
        )
    elif args.command == "audit":
        manifest, envelope = audit_deployment(
            archive=Path(args.archive),
            expected_archive_sha256=args.expected_archive_sha256,
            manifest_path=Path(args.manifest),
            expected_manifest_sha256=args.expected_manifest_sha256,
            detached_launcher=Path(args.detached_launcher),
            expected_detached_launcher_sha256=(
                args.expected_detached_launcher_sha256
            ),
            deployment_envelope_path=Path(args.deployment_envelope),
            expected_deployment_envelope_sha256=(
                args.expected_deployment_envelope_sha256
            ),
        )
        value = {
            "archive": args.archive,
            "archive_sha256": args.expected_archive_sha256,
            "manifest": args.manifest,
            "manifest_sha256": args.expected_manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
            "content_closure_sha1": manifest["content_closure_sha1"],
            "detached_launcher": args.detached_launcher,
            "detached_launcher_sha256": (
                args.expected_detached_launcher_sha256
            ),
            "deployment_envelope": args.deployment_envelope,
            "deployment_envelope_sha256": (
                args.expected_deployment_envelope_sha256
            ),
            "deployment_envelope_digest": envelope["envelope_digest"],
            "file_count": manifest["file_count"],
        }
    elif args.command == "verified-run-module":
        return verified_run_module(
            method_root=Path(args.method_root),
            manifest_path=Path(args.manifest),
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_runner_sha256=args.expected_runner_sha256,
            target=args.target,
            target_arguments=args.target_arguments,
        )
    else:
        held_fd_exec_frozen_python(
            args.target_arguments,
            start_gate_stdin=args.start_gate_stdin,
        )
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArmsIncompleteExact2ReleaseError",
    "FILES_AND_MODES",
    "FROZEN_MEMBER_PINS",
    "FROZEN_SOURCE_MEMBER_PINS",
    "LAUNCHER_MEMBER",
    "MEMBER_ROOT",
    "PLAN_MEMBER",
    "R6_REPLACEMENT_PINS",
    "audit",
    "audit_deployment",
    "build",
    "build_archive",
    "build_manifest",
    "deployment_envelope_value",
    "held_fd_exec_frozen_python",
    "validate_deployment_envelope",
    "validate_manifest",
    "verified_run_module",
    "verify_archive",
]
