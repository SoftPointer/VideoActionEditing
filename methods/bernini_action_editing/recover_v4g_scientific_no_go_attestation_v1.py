"""Create a sibling attestation for the burned v4G scientific NO-GO run.

This recovery program never repairs or chmods the original run.  It first
revalidates the complete burned exact26 tree, the five sealed inner receipts,
all ten checkpoints, the detached release/controller/Python authorities, the
controller logs, and the failed seal-child accounting row.  Only after every
read-only check succeeds may it create one fresh sibling directory containing
one immutable JSON attestation.

The program is intentionally disabled until its detached recovery release and
controller are audited.  The original controller remains incomplete; recovery
does not rewrite that historical fact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "v4g-scientific-no-go-sibling-recovery-attestation-v1"
RESULT_SCHEMA = "v4g-scientific-no-go-sibling-recovery-result-v4"
RELEASE_SEALED = True

BASE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g_20260821"
)
ORIGINAL_RUN_ROOT = BASE / "runs/exact5_parallel_38b2cbec_v1"
RECOVERY_ROOT = BASE / "runs/exact5_parallel_38b2cbec_v1_recovery_v1"
RELEASE_ROOT = BASE / "release_38b2cbec_v1"
RECOVERY_RELEASE_ROOT = BASE / "recovery_release_v1"
RECOVERY_CONTROLLER_PATH = BASE / (
    "controllers/auh_v4g_scientific_no_go_sibling_recovery_v1.sh"
)
RECOVERY_MANIFEST_NAME = "release-manifest-v4g-recovery.json"
RECOVERY_RUNTIME_RELATIVE_PATH = (
    "methods/bernini_action_editing/"
    "recover_v4g_scientific_no_go_attestation_v1.py"
)
RECOVERY_TESTS_RELATIVE_PATH = (
    "methods/bernini_action_editing/tests/"
    "test_recover_v4g_scientific_no_go_attestation_v1.py"
)
TRUSTED_PATH_ANCHOR = Path("/vast/users")
CONTROLLER_PATH = BASE / (
    "controllers/auh_v4g_vjepa2_role_directed_teacher_margin_"
    "exact5_parallel_v1_5bfba26c.sh"
)
OUTER_STDOUT_PATH = BASE / "controller-launch.stdout"
OUTER_STDERR_PATH = BASE / "controller-launch.stderr"
PYTHON_PATH = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
PYTHON_SITE_PACKAGES = (
    PYTHON_PATH.parent.parent / "lib/python3.12/site-packages"
)
SACCT_PATH = Path("/usr/bin/sacct")
SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"

RELEASE_TREE_SHA256 = (
    "e4e158e064ceb181345673c86a8fb275436ddd25edc74a3e7ef8f1c31d4f16ff"
)
RELEASE_MANIFEST_SHA256 = (
    "d1e0c42904057e14d47c87e746c32db375ba6ee6b006f813ea91cdb2daae4882"
)
RELEASE_MANIFEST_DIGEST = (
    "91adc268b8b86f083c36aa043e5c6f10956e720b03a3b5ebf1022b3775ac46a4"
)
CONTROLLER_SHA256 = (
    "5bfba26c65930c6929522fee1145d2ea83cca3ec2ab54cf42a6202e5df0816f8"
)
PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
RUNTIME_SHA256 = (
    "38b2cbecaf022e203ccf09e6808661013f4f23dee0d02ffa1756e24d0c167cf9"
)
TESTS_SHA256 = (
    "7fe6b42208f77171f99d44d5a9fc9eae58c3bb2d4663ca016e9b154a4d3c4996"
)
AUTHORITY_SNAPSHOT = (
    "e4e158e064ceb181345673c86a8fb275436ddd25edc74a3e7ef8f1c31d4f16ff:"
    "b93140a654dde52270897e96607aceeed0614f7a54041dc183cf2b949e1c4d3e:"
    "5bfba26c65930c6929522fee1145d2ea83cca3ec2ab54cf42a6202e5df0816f8:"
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
INPUT_SNAPSHOT_SHA256 = (
    "b93140a654dde52270897e96607aceeed0614f7a54041dc183cf2b949e1c4d3e"
)
EXACT26_MANIFEST_SHA256 = (
    "14bf42749c97b20934a2a088a560fa23ed2b1e37555262e9d4c7f2f368e74265"
)
PARENT_STABLE_SIGNATURE_SHA256 = (
    "0540ac21631bc948db012c77003c99d0de32cb1f769ffe38e8ab8b8e380cac76"
)
LAUNCH_PLAN_SHA256 = (
    "0a4dae7cc1049c44857e6186928363220bb9f57b92e0c36ecae61d4c64d25356"
)
OUTER_STDOUT = {
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "size_bytes": 0,
    "mode_octal": "0600",
    "nlink": 1,
    "device": 48,
    "inode": 18025654629908591336,
}
OUTER_STDERR = {
    "sha256": "a48e4f13f9b85360c5306e274d59cf672549da8adb01aaa452670eeb9807590a",
    "size_bytes": 4426,
    "mode_octal": "0600",
    "nlink": 1,
    "device": 48,
    "inode": 4709107117863586105,
}
INNER_RECEIPT_SHA256 = (
    "45753aef53fc1a69b46154990b36196be5c2d2e0b9aaf8e60c285b59c9f00cfc",
    "6f501749811e2c57a806bb9cebdd895cfbed285cf64e0ac41eafb484e3042380",
    "96b3638388a21752f5a7ed4a4225d83589df0f0487b948d1a5bc34286cb867b3",
    "5b714ce73c5178f9043e837374487b69e162613f98587194a846b0b9276f8249",
    "affeef9ec5023437a379f00ac424d117a29db796cc3afeb9f086cc83eab1c2fb",
)
INNER_RECEIPT_DIGEST = (
    "3c435b13d4d8f08abc448b4cef8e33c0dc89abb13ae1c2811ae6350c3cab1585",
    "f0df0f452d27a2f25e144e62fd7b4c78762edec434a1b930b2c2c27dcf400607",
    "d5dc0481a90dd32789d0d2d39527c5af3c5418e2ee2296838e52065988c4344f",
    "f3be5838eee6f530efe47b2da28d07ddd88d13d0089aaf0993fee6f82d714d61",
    "fc8a512a63db2f5c390fdd12e391491d884fce079dda20d512dae747ea23d829",
)

AUTHORITY_FILES = (
    (
        Path(
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "semantic_anchor_vjepa2_contextual_frontier_v4c_20260820/"
            "runs/extract_720033ac_v4/features/feature_extraction_receipt.json"
        ),
        "895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a",
    ),
    (
        Path(
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/semantic_anchor_linear_"
            "frontier_v4a_20260820/runs/exact5_e7e755a4_v1/receipt.json"
        ),
        "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2",
    ),
    (
        Path(
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/semantic_anchor_vjepa2_"
            "contextual_frontier_v4c_20260820/runs/frontier_d286c23b_v2/"
            "receipt.json"
        ),
        "8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9",
    ),
    (
        Path(
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/semantic_anchor_vjepa2_"
            "nonlinear_codec_v4d_20260820/runs/exact5_20934925_v2/receipt.json"
        ),
        "53910bcb71ce02a193bd47e44c3a97de0ee24f431576db64a763637447720b6f",
    ),
)

RUN_BINDING = {
    "extractor_implementation_sha256": (
        "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc"
    ),
    "frozen_v4f_runtime_path": str(
        RELEASE_ROOT / "methods/bernini_action_editing/"
        "semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py"
    ),
    "frozen_v4f_runtime_sha256": (
        "97cd77e64a4dfaf3036e6c50a5b85060fd616f87371e5d967e69db1170466d74"
    ),
    "implementation_path": str(
        RELEASE_ROOT / "methods/bernini_action_editing/"
        "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g.py"
    ),
    "implementation_sha256": RUNTIME_SHA256,
    "v4a_implementation_sha256": (
        "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973"
    ),
    "v4c_implementation_sha256": (
        "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef"
    ),
    "v4d_implementation_sha256": (
        "20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc"
    ),
    "v4e_burned_implementation_sha256": (
        "4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a"
    ),
}

EXPECTED_DIRS = {"fold0", "fold1", "fold2", "fold3", "fold4", "logs"}
EXPECTED_FILES = {
    "launch-plan.json",
    *{
        f"fold{fold}/{name}"
        for fold in range(5)
        for name in ("preselection.pt", "fixed1200.pt", "inner.json")
    },
    *{
        f"logs/train-fold{fold}.{stream}"
        for fold in range(5)
        for stream in ("stdout", "stderr")
    },
}
EXPECTED_FILE_COUNT = 26

EXPECTED_QUALIFICATION_SCOPE = {
    "action_representation_qualified": False,
    "aggregate_gate_evaluated": False,
    "exposed_five_view_codec_development_gate": None,
    "full644_refit_authorized": False,
    "generation_qualified": False,
    "identity_disentanglement_qualified": False,
    "identity_preservation_qualified": False,
    "inference_authorized": False,
    "inner_fold_local_gate_passed": False,
    "latent_metric_qualified": False,
    "prior_generation_qualified": False,
    "prior_qualified": False,
    "renderer_qualified": False,
    "unseen_hostile_transform_gate": False,
    "unseen_hostile_transform_gate_evaluated": False,
    "vae_necessary": None,
    "video_editing_qualified": False,
    "video_model_training_performed": False,
    "web_evaluation_authorized": False,
}

EXPECTED_ATTESTATION_QUALIFICATION_SCOPE = {
    "known_exposed_development_gate": None,
    "known_exposed_development_gate_evaluated": False,
    "unseen_hostile_transform_gate": False,
    "unseen_hostile_transform_gate_evaluated": False,
    "latent_metric_qualified": False,
    "action_representation_qualified": False,
    "identity_disentanglement_qualified": False,
    "identity_preservation_qualified": False,
    "prior_qualified": False,
    "prior_generation_qualified": False,
    "generation_qualified": False,
    "renderer_qualified": False,
    "video_editing_qualified": False,
    "inference_authorized": False,
    "web_evaluation_authorized": False,
    "full644_refit_authorized": False,
    "video_model_training_performed": False,
    "html_or_video_generated": False,
    "vae_necessary": None,
}

ATTESTATION_KEYS = {
    "schema_version", "authority", "original_run_root", "recovery_root",
    "original_run_mutated_by_recovery", "original_run_postverified_unchanged",
    "original_controller_complete", "original_controller_exit_nonzero",
    "scientifically_verified_all_inner_no_go", "global_inner_barrier_created",
    "evaluate_fold_executed", "aggregate_executed",
    "all_fold_oof_semantic_tensor_read_count",
    "all_fold_oof_semantic_tensor_materialized_count",
    "recovery_ledger_reconstructed_from_burned_exact26",
    "burned_exact26_file_count", "burned_exact26_manifest_sha256",
    "burned_exact26_manifest", "burned_parent_stable_signature_sha256",
    "original_run_root_binding",
    "old_controller_identity_schema_bug", "source_authority", "launch_plan",
    "original_controller_logs", "failed_seal_child_accounting", "folds",
    "all_qualification_claims_false", "qualification_scope",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _object_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_release_guard() -> None:
    if RELEASE_SEALED is not True:
        raise RuntimeError("v4G recovery release is not sealed; intentional NO-GO")


def _json_no_duplicates(raw: bytes) -> Any:
    def hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    if _canonical_bytes(value) + b"\n" != raw:
        raise RuntimeError("JSON is not canonical one-line ASCII plus newline")
    return value


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_size, stat.S_IMODE(info.st_mode),
        info.st_nlink, info.st_mtime_ns, info.st_ctime_ns,
    )


def _read_regular(
    path: Path, *, mode: int | None = None, nlink: int | None = None,
    capture: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError(f"path is not canonical authority: {path}") from error
    if (
        not path.is_absolute() or path.is_symlink()
        or str(path) != str(resolved)
    ):
        raise RuntimeError(f"path is not canonical absolute/non-symlink: {path}")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"path is not a regular O_NOFOLLOW authority: {path}")
    if mode is not None and stat.S_IMODE(before.st_mode) != mode:
        raise RuntimeError(f"file mode differs: {path}")
    if nlink is not None and before.st_nlink != nlink:
        raise RuntimeError(f"file nlink differs: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    if len({_identity(value) for value in (before, opened, closed, after)}) != 1:
        raise RuntimeError(f"single-FD file identity changed: {path}")
    binding = {
        "path": str(path.resolve(strict=True)),
        "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
        "nlink": before.st_nlink,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }
    return binding, (b"".join(chunks) if chunks is not None else None)


def _read_directory_binding(path: Path, *, mode: int) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError(f"directory is not canonical authority: {path}") from error
    if (
        not path.is_absolute() or path.is_symlink()
        or str(path) != str(resolved)
        or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
    ):
        raise RuntimeError(f"directory is not canonical absolute authority: {path}")
    before = path.lstat()
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        opened = os.fstat(descriptor)
        members = sorted(os.listdir(descriptor))
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        len({_identity(item) for item in (before, opened, closed, after)}) != 1
        or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
    ):
        raise RuntimeError(f"directory same-FD identity/mode differs: {path}")
    return {
        "path": str(resolved), "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns, "members": members,
        "single_fd_pre_post_identity_and_membership_exact": True,
    }


def _require_binding(binding: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if binding.get(key) != value:
            raise RuntimeError(f"authority binding differs for {key}")


def _scan_original(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if (
        not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or stat.S_IMODE(root.lstat().st_mode) != 0o700
    ):
        raise RuntimeError("original run root authority differs")
    found_dirs: set[str] = set()
    found_files: set[str] = set()
    stack = [root]
    while stack:
        parent = stack.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise RuntimeError("original run contains a symlink")
                if stat.S_ISDIR(info.st_mode):
                    if stat.S_IMODE(info.st_mode) != 0o700:
                        raise RuntimeError("original run directory mode differs")
                    found_dirs.add(relative)
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    found_files.add(relative)
                else:
                    raise RuntimeError("original run contains a special entry")
    if found_dirs != EXPECTED_DIRS or found_files != EXPECTED_FILES:
        raise RuntimeError("original run exact directory/file set differs")
    rows: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    for relative in sorted(found_files):
        binding, _ = _read_regular(root / relative, mode=0o444, nlink=1)
        bindings[relative] = binding
        rows.append({
            "path": relative,
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
            "mode_octal": binding["mode_octal"],
            "nlink": binding["nlink"],
            "device": binding["device"],
            "inode": binding["inode"],
        })
    stable_directories = [
        {"path": relative, "mode_octal": "0700"}
        for relative in sorted(EXPECTED_DIRS)
    ]
    stable_files = [{
        key: row[key] for key in (
            "path", "sha256", "size_bytes", "mode_octal", "nlink",
        )
    } for row in rows]
    if (
        len(rows) != EXPECTED_FILE_COUNT
        or _object_sha(rows) != EXACT26_MANIFEST_SHA256
        or _object_sha({
            "directories": stable_directories, "files": stable_files,
        }) != PARENT_STABLE_SIGNATURE_SHA256
    ):
        raise RuntimeError("burned exact26 manifest differs")
    return rows, bindings


def _verify_release(root: Path) -> dict[str, Any]:
    if (
        not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or stat.S_IMODE(root.lstat().st_mode) != 0o555
    ):
        raise RuntimeError("v4G release root differs")
    manifest_path = root / "release-manifest-v4g.json"
    manifest_binding, manifest_raw = _read_regular(
        manifest_path, mode=0o444, nlink=1, capture=True,
    )
    if manifest_raw is None or manifest_binding["sha256"] != RELEASE_MANIFEST_SHA256:
        raise RuntimeError("v4G release manifest raw SHA differs")
    manifest = _json_no_duplicates(manifest_raw)
    unsigned = dict(manifest)
    manifest_digest = unsigned.pop("manifest_digest", None)
    payload = manifest.get("payload")
    if (
        manifest_digest != RELEASE_MANIFEST_DIGEST
        or manifest_digest != _object_sha(unsigned)
        or manifest.get("status") != "V4G_DETACHED_RELEASE_MANIFEST_SEALED"
        or manifest.get("payload_count") != 10
        or type(payload) is not list or len(payload) != 10
    ):
        raise RuntimeError("v4G release manifest semantics differ")
    expected_paths = {"release-manifest-v4g.json"}
    declared: dict[str, str] = {}
    for row in payload:
        if type(row) is not dict or set(row) != {"relative_path", "role", "sha256"}:
            raise RuntimeError("v4G release payload row differs")
        relative = row["relative_path"]
        if type(relative) is not str or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError("v4G release payload path differs")
        if relative in declared:
            raise RuntimeError("v4G release payload path duplicates")
        declared[relative] = row["sha256"]
        expected_paths.add(relative)
    found_files: set[str] = set()
    found_dirs: set[str] = set()
    stack = [root]
    while stack:
        parent = stack.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise RuntimeError("v4G release contains symlink")
                if stat.S_ISDIR(info.st_mode):
                    if stat.S_IMODE(info.st_mode) != 0o555:
                        raise RuntimeError("v4G release directory mode differs")
                    found_dirs.add(relative)
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    found_files.add(relative)
                else:
                    raise RuntimeError("v4G release contains special entry")
    if found_files != expected_paths or len(found_dirs) != 3:
        raise RuntimeError("v4G release exact11 tree membership differs")
    tree_rows: list[dict[str, Any]] = []
    for relative in sorted(found_files):
        binding, _ = _read_regular(root / relative, mode=0o444, nlink=1)
        if relative in declared and binding["sha256"] != declared[relative]:
            raise RuntimeError("v4G release payload SHA differs")
        tree_rows.append({
            "path": relative, "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        })
    if _object_sha(tree_rows) != RELEASE_TREE_SHA256:
        raise RuntimeError("v4G release exact11 tree digest differs")
    if declared.get(
        "methods/bernini_action_editing/"
        "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g.py"
    ) != RUNTIME_SHA256 or declared.get(
        "methods/bernini_action_editing/tests/"
        "test_semantic_anchor_vjepa2_role_directed_teacher_margin_v4g.py"
    ) != TESTS_SHA256:
        raise RuntimeError("v4G release runtime/tests pins differ")
    return {
        "root": str(root), "tree_sha256": RELEASE_TREE_SHA256,
        "manifest": manifest_binding, "manifest_digest": manifest_digest,
        "file_count": 11, "directory_count": 3,
    }


def _verify_input_snapshot(
    receipt_bindings_and_raw: Sequence[tuple[dict[str, Any], bytes]],
) -> dict[str, Any]:
    if len(receipt_bindings_and_raw) != 4:
        raise RuntimeError("input receipt binding count differs")
    labels = ("feature", "v4a", "v4c", "v4d")
    rows = []
    for label, (binding, _) in zip(labels, receipt_bindings_and_raw):
        rows.append({
            "label": label, "path": binding["path"],
            "sha256": binding["sha256"], "size_bytes": binding["size_bytes"],
        })
    feature_raw = receipt_bindings_and_raw[0][1]

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise RuntimeError("feature receipt duplicate JSON key")
            result[key] = item
        return result

    feature = json.loads(
        feature_raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
    )
    shards = feature.get("shards") if type(feature) is dict else None
    if type(shards) is not list or len(shards) != 6:
        raise RuntimeError("feature receipt exact6 shard ledger differs")
    shard_bindings = []
    for index, shard in enumerate(shards):
        if (
            type(shard) is not dict or shard.get("index") != index
            or type(shard.get("path")) is not str
            or type(shard.get("sha256")) is not str
        ):
            raise RuntimeError("feature receipt shard order/schema differs")
        path = Path(shard["path"])
        binding, _ = _read_regular(path, mode=0o444, nlink=1)
        if binding["sha256"] != shard["sha256"]:
            raise RuntimeError("feature shard frozen SHA differs")
        shard_bindings.append(binding)
        rows.append({
            "label": f"feature-shard-{index}", "path": binding["path"],
            "sha256": binding["sha256"], "size_bytes": binding["size_bytes"],
        })
    digest = _object_sha(rows)
    if digest != INPUT_SNAPSHOT_SHA256:
        raise RuntimeError("input authority snapshot differs")
    return {
        "sha256": digest, "ordered_rows": rows,
        "feature_shard_bindings": shard_bindings,
        "exact_receipt_count": 4, "exact_feature_shard_count": 6,
        "all_ten_files_single_fd_reverified": True,
    }


def _verify_recovery_release_and_controller() -> dict[str, Any]:
    root = RECOVERY_RELEASE_ROOT
    root_binding = _read_directory_binding(root, mode=0o555)
    expected_files = {
        RECOVERY_RUNTIME_RELATIVE_PATH,
        RECOVERY_TESTS_RELATIVE_PATH,
        RECOVERY_MANIFEST_NAME,
    }
    expected_dirs = {"methods", "methods/bernini_action_editing",
                     "methods/bernini_action_editing/tests"}
    found_files: set[str] = set()
    found_dirs: set[str] = set()
    stack = [root]
    while stack:
        parent = stack.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise RuntimeError("recovery release contains symlink")
                if stat.S_ISDIR(info.st_mode):
                    if stat.S_IMODE(info.st_mode) != 0o555:
                        raise RuntimeError("recovery release directory mode differs")
                    found_dirs.add(relative)
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    found_files.add(relative)
                else:
                    raise RuntimeError("recovery release contains special entry")
    if found_files != expected_files or found_dirs != expected_dirs:
        raise RuntimeError("recovery release exact3 tree membership differs")

    bindings: dict[str, dict[str, Any]] = {}
    tree_rows = []
    manifest_raw: bytes | None = None
    for relative in sorted(found_files):
        binding, raw = _read_regular(
            root / relative, mode=0o444, nlink=1,
            capture=relative == RECOVERY_MANIFEST_NAME,
        )
        bindings[relative] = binding
        tree_rows.append({
            "path": relative, "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        })
        if relative == RECOVERY_MANIFEST_NAME:
            manifest_raw = raw
    if manifest_raw is None:
        raise RuntimeError("recovery release manifest read differs")
    manifest = _json_no_duplicates(manifest_raw)
    unsigned = dict(manifest)
    manifest_digest = unsigned.pop("manifest_digest", None)
    payload = manifest.get("payload")
    if (
        set(manifest) != {
            "schema_version", "status", "payload", "payload_count",
            "manifest_digest", "manifest_target_relative_path",
            "release_tree_contract", "authority_graph",
        }
        or manifest.get("schema_version")
            != "v4g-scientific-no-go-recovery-detached-release-manifest-v1"
        or manifest.get("status") != "SEALED"
        or manifest_digest != _object_sha(unsigned)
        or manifest.get("payload_count") != 2
        or type(payload) is not list or len(payload) != 2
        or manifest.get("manifest_target_relative_path") != RECOVERY_MANIFEST_NAME
        or manifest.get("release_tree_contract") != {
            "exact_file_count_including_manifest": 3,
            "exact_directory_count_below_root": 3,
            "all_files_mode_0444_nlink1": True,
            "all_directories_mode_0555": True,
        }
        or manifest.get("authority_graph") != {
            "sha256_graph_is_directed_and_acyclic": True,
            "manifest_pins_runtime_and_tests": True,
            "runtime_pins_controller_or_manifest": False,
            "detached_controller_is_outside_release_tree": True,
        }
    ):
        raise RuntimeError("recovery release manifest semantics differ")
    expected_payload = [
        {
            "relative_path": RECOVERY_RUNTIME_RELATIVE_PATH,
            "role": "recovery_runtime",
            "sha256": bindings[RECOVERY_RUNTIME_RELATIVE_PATH]["sha256"],
        },
        {
            "relative_path": RECOVERY_TESTS_RELATIVE_PATH,
            "role": "recovery_runtime_tests",
            "sha256": bindings[RECOVERY_TESTS_RELATIVE_PATH]["sha256"],
        },
    ]
    if payload != expected_payload:
        raise RuntimeError("recovery release payload pins differ")
    executed = Path(__file__).resolve(strict=True)
    if executed != root / RECOVERY_RUNTIME_RELATIVE_PATH:
        raise RuntimeError("executed recovery runtime is outside sealed release")
    controller, _ = _read_regular(
        RECOVERY_CONTROLLER_PATH, mode=0o555, nlink=1,
    )
    return {
        "release_root": str(root), "release_root_binding": root_binding,
        "release_tree_sha256": _object_sha(tree_rows), "tree_rows": tree_rows,
        "manifest": bindings[RECOVERY_MANIFEST_NAME],
        "manifest_digest": manifest_digest,
        "runtime": bindings[RECOVERY_RUNTIME_RELATIVE_PATH],
        "tests": bindings[RECOVERY_TESTS_RELATIVE_PATH],
        "controller": controller,
        "exact_file_count": 3, "exact_directory_count_below_root": 3,
        "one_way_sha256_dag_reverified": True,
        "controller_identity_recorded_not_runtime_reverse_pinned": True,
    }


def _parse_train_stdout(raw: bytes, fold: int) -> dict[str, Any]:
    lines = raw.decode("utf-8").splitlines()
    prefixes = (
        "V4G_GPU_GATE=", "V4G_TRAIN_RUNTIME=", "V4G_TRAIN_VERIFY=",
        "V4G_CACHE_CLEANED=",
    )
    if len(lines) != 4 or any(not line.startswith(prefix) for line, prefix in zip(lines, prefixes)):
        raise RuntimeError("train stdout exact4 structure differs")
    values = [json.loads(line[len(prefix):]) for line, prefix in zip(lines, prefixes)]
    gate, runtime, verify, cleanup = values
    if (
        gate.get("role") != f"train-fold{fold}"
        or gate.get("visible_gpu_count") != 1
        or gate.get("exact_one_uuid_join") is not True
        or gate.get("device_name") != "AMD Instinct MI210"
        or gate.get("torch") != "2.7.1+rocm6.3"
        or runtime.get("fold_index") != fold
        or runtime.get("inner_pass") is not False
        or runtime.get("inference_authorized") is not False
        or runtime.get("oof_semantic_tensor_read_count") != 0
        or runtime.get("inner_receipt_sha256") != INNER_RECEIPT_SHA256[fold]
        or runtime.get("inner_receipt_digest") != INNER_RECEIPT_DIGEST[fold]
        or verify.get("fold_index") != fold
        or verify.get("inner_pass") is not False
        or verify.get("oof_semantic_tensor_read_count") != 0
        or verify.get("inner_receipt_sha256") != INNER_RECEIPT_SHA256[fold]
        or verify.get("inner_receipt_digest") != INNER_RECEIPT_DIGEST[fold]
        or verify.get("distinct_checkpoint_inodes") is not True
        or verify.get("same_model_state_sha256") is not True
        or cleanup.get("role") != f"train-fold{fold}"
        or cleanup.get("absent_after_cleanup") is not True
    ):
        raise RuntimeError("train stdout semantic join differs")
    return {"gpu_gate": gate, "runtime": runtime, "verify": verify, "cleanup": cleanup}


def _verify_launch_plan(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    binding = bindings["launch-plan.json"]
    if binding["sha256"] != LAUNCH_PLAN_SHA256 or binding["size_bytes"] != 3298:
        raise RuntimeError("launch-plan frozen binding differs")
    _, raw = _read_regular(root / "launch-plan.json", mode=0o444, nlink=1, capture=True)
    if raw is None:
        raise RuntimeError("launch-plan read failed")
    value = _json_no_duplicates(raw)
    preflight = value.get("cpu_preflight", {})
    source = value.get("source_authority", {})
    if (
        value.get("authority_snapshot") != AUTHORITY_SNAPSHOT
        or value.get("controller_sha256") != CONTROLLER_SHA256
        or value.get("allfold_oof_exact0_on_train_inner_or_barrier_no_go") is not True
        or value.get("barrier_before_any_evaluate") is not True
        or value.get("official_controller_cli_caller_supplied_inner_barrier_or_fold_sha") is not False
        or value.get("train_fold_oof_semantic_tensor_read_count") != 0
        or preflight.get("normal_tests_run") != 36
        or preflight.get("normal_tests_skipped") != 0
        or preflight.get("optimized_tests_run") != 36
        or preflight.get("optimized_tests_skipped") != 0
        or source.get("release_tree_sha256") != RELEASE_TREE_SHA256
        or source.get("runtime_sha256") != RUNTIME_SHA256
        or source.get("runtime_test_sha256") != TESTS_SHA256
        or source.get("python_sha256") != PYTHON_SHA256
    ):
        raise RuntimeError("launch-plan semantic authority differs")
    return value


def _query_failed_step() -> dict[str, Any]:
    sacct_binding, _ = _read_regular(SACCT_PATH, mode=0o755, nlink=1)
    if sacct_binding["sha256"] != SACCT_SHA256:
        raise RuntimeError("pinned sacct executable differs")
    command = [
        str(SACCT_PATH), "-n", "-P", "-j", "143811.94", "--format="
        "JobIDRaw,State,ExitCode,NodeList,AllocCPUS,ReqMem,Elapsed,Start,End",
    ]
    result = subprocess.run(
        command, check=True, text=True, capture_output=True, timeout=30,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    lines = [line for line in result.stdout.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError("seal-child sacct row cardinality differs")
    columns = lines[0].split("|")
    if len(columns) != 9:
        raise RuntimeError("seal-child sacct row schema differs")
    keys = (
        "job_id_raw", "state", "exit_code", "node", "alloc_cpus",
        "req_mem", "elapsed", "start", "end",
    )
    row = dict(zip(keys, columns))
    expected = {
        "job_id_raw": "143811.94", "state": "FAILED", "exit_code": "1:0",
        "node": "auh7-1b-gpu-306", "alloc_cpus": "8", "req_mem": "",
        "elapsed": "00:00:03", "start": "2026-08-21T09:43:49",
        "end": "2026-08-21T09:43:52",
    }
    if row != expected:
        raise RuntimeError("seal-child failed accounting authority differs")
    return {
        "record": row, "sacct_executable": sacct_binding,
        "query_columns": list(keys), "exact_row_replayed": True,
    }


def _verify_process_python() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    flags = sys.flags
    if (
        executable != PYTHON_PATH
        or tuple(sys.version_info[:3]) != (3, 12, 13)
        or flags.isolated != 1 or flags.no_site != 1
        or flags.ignore_environment != 1 or flags.safe_path is not True
        or flags.dont_write_bytecode != 1
    ):
        raise RuntimeError("recovery process is not the pinned Python")
    binding, _ = _read_regular(executable, mode=0o755, nlink=1)
    if binding["sha256"] != PYTHON_SHA256:
        raise RuntimeError("recovery process Python SHA differs")
    return binding


class _CapturedSourceLoader:
    """Execute one module from bytes already captured by a verified FD read."""

    def __init__(self, name: str, path: Path, raw: bytes) -> None:
        self.name = name
        self.path = path
        self.raw = raw

    def create_module(self, spec: Any) -> None:
        del spec
        return None

    def exec_module(self, module: Any) -> None:
        module.__file__ = str(self.path)
        code = compile(
            self.raw, str(self.path), "exec", dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        exec(code, module.__dict__)


class _CapturedSourceFinder:
    def __init__(
        self, sources: Mapping[str, tuple[Path, bytes]], prefix: str,
    ) -> None:
        self.sources = dict(sources)
        self.prefix = prefix
        self.loaders: dict[str, _CapturedSourceLoader] = {}

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None,
    ) -> Any:
        del path, target
        if fullname in self.sources:
            source_path, raw = self.sources[fullname]
            loader = _CapturedSourceLoader(fullname, source_path, raw)
            self.loaders[fullname] = loader
            return importlib.util.spec_from_loader(
                fullname, loader, origin=str(source_path), is_package=False,
            )
        if fullname.startswith(self.prefix):
            raise ImportError(f"unsealed frozen dependency requested: {fullname}")
        return None


def _require_no_preloaded_torch() -> None:
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("recovery parser Torch dependency was preloaded")


def _import_captured_source_graph(
    release_package: Path,
    expected_modules: Mapping[str, tuple[str, str]],
    target: str,
) -> Any:
    prefix = "methods.bernini_action_editing."
    protected_modules = {
        "methods", "methods.bernini_action_editing", *expected_modules,
    }
    already_loaded = sorted(name for name in protected_modules if name in sys.modules)
    if already_loaded:
        raise RuntimeError("frozen v4G dependency was preloaded")
    _require_no_preloaded_torch()
    if sys.meta_path != [
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    ]:
        raise RuntimeError("frozen v4G import machinery differs")

    captured: dict[str, tuple[Path, bytes]] = {}
    creation_bindings: dict[str, dict[str, Any]] = {}
    for name, (basename, expected_sha) in expected_modules.items():
        source_path = release_package / basename
        binding, raw = _read_regular(
            source_path, mode=0o444, nlink=1, capture=True,
        )
        if raw is None or binding["sha256"] != expected_sha:
            raise RuntimeError("frozen v4G captured source authority differs")
        captured[name] = (source_path, raw)
        creation_bindings[name] = binding

    finder = _CapturedSourceFinder(captured, "methods.")
    methods_package = types.ModuleType("methods")
    methods_spec = importlib.machinery.ModuleSpec(
        "methods", loader=None, is_package=True,
    )
    methods_spec.submodule_search_locations = [str(release_package.parent)]
    methods_package.__package__ = "methods"
    methods_package.__path__ = [str(release_package.parent)]
    methods_package.__spec__ = methods_spec
    action_package = types.ModuleType("methods.bernini_action_editing")
    action_spec = importlib.machinery.ModuleSpec(
        "methods.bernini_action_editing", loader=None, is_package=True,
    )
    action_spec.submodule_search_locations = [str(release_package)]
    action_package.__package__ = "methods.bernini_action_editing"
    action_package.__path__ = [str(release_package)]
    action_package.__spec__ = action_spec
    methods_package.bernini_action_editing = action_package

    old_meta_path = list(sys.meta_path)
    sys.modules["methods"] = methods_package
    sys.modules["methods.bernini_action_editing"] = action_package
    sys.meta_path = [finder, *old_meta_path]
    try:
        module = importlib.import_module(target)
        for name, (basename, expected_sha) in expected_modules.items():
            loaded = sys.modules.get(name)
            expected_path = release_package / basename
            spec = getattr(loaded, "__spec__", None)
            loader = finder.loaders.get(name)
            binding, _ = _read_regular(
                expected_path, mode=0o444, nlink=1,
            )
            if (
                loaded is None or Path(getattr(loaded, "__file__", "")) != expected_path
                or binding != creation_bindings[name]
                or binding["sha256"] != expected_sha
                or spec is None or Path(spec.origin) != expected_path
                or loader is None or spec.loader is not loader
                or loader.raw != captured[name][1]
            ):
                raise RuntimeError("imported frozen v4G executed-byte authority differs")
    except BaseException:
        for name in protected_modules:
            sys.modules.pop(name, None)
        # Torch namespace was exact-empty on entry.  A partially imported
        # dependency graph must not poison a same-process diagnostic retry.
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)
        raise
    finally:
        sys.meta_path = old_meta_path
    return module


def _import_frozen_v4g(release_root: Path) -> Any:
    prefix = "methods.bernini_action_editing."
    expected_modules = {
        prefix + "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g": (
            "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g.py",
            RUNTIME_SHA256,
        ),
        prefix + "semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy": (
            "semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py",
            RUN_BINDING["frozen_v4f_runtime_sha256"],
        ),
        prefix + "semantic_anchor_vjepa2_analytic_frontier_v4c": (
            "semantic_anchor_vjepa2_analytic_frontier_v4c.py",
            RUN_BINDING["v4c_implementation_sha256"],
        ),
        prefix + "semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d": (
            "semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py",
            RUN_BINDING["v4d_implementation_sha256"],
        ),
        prefix + "semantic_anchor_linear_frontier_v4_fast": (
            "semantic_anchor_linear_frontier_v4_fast.py",
            RUN_BINDING["v4a_implementation_sha256"],
        ),
        prefix + "extract_vjepa2_ordered_contextual_features_v4c": (
            "extract_vjepa2_ordered_contextual_features_v4c.py",
            RUN_BINDING["extractor_implementation_sha256"],
        ),
        prefix + "semantic_anchor_action_sequence_vae_v2": (
            "semantic_anchor_action_sequence_vae_v2.py",
            "46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca",
        ),
        prefix + "semantic_action_cvae_canary_v1": (
            "semantic_action_cvae_canary_v1.py",
            "74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233",
        ),
    }
    protected_modules = {
        "methods", "methods.bernini_action_editing", *expected_modules,
    }
    if any(name in sys.modules for name in protected_modules):
        raise RuntimeError("frozen v4G dependency was preloaded")
    if sys.meta_path != [
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    ]:
        raise RuntimeError("frozen v4G import machinery differs")
    release_package = release_root / "methods/bernini_action_editing"
    site_packages = PYTHON_SITE_PACKAGES.resolve(strict=True)
    if (
        str(release_package) != str(release_package.resolve(strict=True))
        or str(PYTHON_SITE_PACKAGES) != str(site_packages)
    ):
        raise RuntimeError("frozen v4G release package path is not canonical")
    target = prefix + "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g"
    old_sys_path = list(sys.path)
    sys.path = [str(site_packages), *old_sys_path]
    try:
        module = _import_captured_source_graph(
            release_package, expected_modules, target,
        )
    finally:
        sys.path = old_sys_path
    return module


def _verify_recovery_parser_torch(v4g: Any) -> dict[str, Any]:
    torch = getattr(v4g, "torch", None)
    raw_version = getattr(torch, "__version__", None)
    version = str(raw_version) if raw_version is not None else None
    version_namespace = getattr(torch, "version", None)
    raw_hip = getattr(version_namespace, "hip", None)
    hip = str(raw_hip) if raw_hip is not None else None
    if (
        version != "2.7.1+rocm6.3" or type(hip) is not str
        or not hip.startswith("6.3")
    ):
        raise RuntimeError("recovery parser Torch/ROCm runtime differs")
    return {
        "torch_version": version, "torch_hip_version": hip,
        "torch_version_exact_2_7_1_rocm6_3": True,
        "torch_hip_release_6_3": True,
    }


def _verify_recovery_parser_torch_origins(v4g: Any) -> dict[str, Any]:
    package_root = (PYTHON_SITE_PACKAGES.resolve(strict=True) / "torch").resolve(
        strict=True,
    )
    expected = {
        "torch": package_root / "__init__.py",
        "torch.nn": package_root / "nn/__init__.py",
        "torch.nn.functional": package_root / "nn/functional.py",
    }
    origins: dict[str, str] = {}
    for name, path in expected.items():
        loaded = sys.modules.get(name)
        raw_file = getattr(loaded, "__file__", None)
        spec = getattr(loaded, "__spec__", None)
        if loaded is None or type(raw_file) is not str or spec is None:
            raise RuntimeError("recovery parser Torch module set differs")
        origin = Path(raw_file).resolve(strict=True)
        if (
            origin != path.resolve(strict=True)
            or type(spec.origin) is not str
            or Path(spec.origin).resolve(strict=True) != origin
            or type(spec.loader) is not importlib.machinery.SourceFileLoader
        ):
            raise RuntimeError("recovery parser Torch module origin differs")
        origins[name] = str(origin)
    if (
        getattr(v4g, "torch", None) is not sys.modules["torch"]
        or getattr(v4g, "nn", None) is not sys.modules["torch.nn"]
        or getattr(v4g, "F", None) is not sys.modules["torch.nn.functional"]
    ):
        raise RuntimeError("recovery parser Torch module identity differs")
    if list(getattr(sys.modules["torch"], "__path__", ())) != [str(package_root)]:
        raise RuntimeError("recovery parser Torch package path differs")
    return {
        "torch_package_root": str(package_root),
        "torch_module_origins": origins,
        "v4g_torch_module_identities_exact": True,
        "torch_standard_source_loaders_and_package_path_exact": True,
    }


def _verify_receipts(
    root: Path, bindings: Mapping[str, Mapping[str, Any]], loader: Callable[..., Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    state_shas: set[str] = set()
    for fold in range(5):
        relative = f"fold{fold}/inner.json"
        if bindings[relative]["sha256"] != INNER_RECEIPT_SHA256[fold]:
            raise RuntimeError("inner receipt burned SHA differs")
        receipt, receipt_binding = loader(
            str(root / f"fold{fold}"), INNER_RECEIPT_SHA256[fold], RUN_BINDING,
        )
        candidate = receipt.get("fixed_candidate", {})
        gate = candidate.get("gate", {})
        scope = receipt.get("qualification_scope", {})
        pre = receipt.get("preselection_checkpoint_artifact", {})
        fixed = receipt.get("fixed1200_checkpoint_artifact", {})
        pair = receipt.get("preselection_fixed1200_checkpoint_pair_join", {})
        selective = receipt.get("selective_feature_materialization_before_global_barrier", {})
        training = receipt.get("training", {})
        pre_physical = pre.get("physical_identity", {})
        fixed_physical = fixed.get("physical_identity", {})
        pre_scan = bindings[f"fold{fold}/preselection.pt"]
        fixed_scan = bindings[f"fold{fold}/fixed1200.pt"]
        if (
            receipt.get("fold_index") != fold
            or receipt.get("receipt_digest") != INNER_RECEIPT_DIGEST[fold]
            or receipt_binding.get("receipt_digest") != INNER_RECEIPT_DIGEST[fold]
            or receipt_binding.get("file_sha256") != INNER_RECEIPT_SHA256[fold]
            or receipt.get("status") != "V4G_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD"
            or receipt.get("inner_pass") is not False
            or receipt.get("global_barrier_required_before_any_fold_oof") is not True
            or receipt.get("oof_semantic_tensor_materialized_count") != 0
            or receipt.get("oof_semantic_tensor_read_count_exact0") is not True
            or receipt.get("oof_used_for_training_checkpoint_or_inner_gate") is not False
            or candidate.get("inner_pass") is not False
            or candidate.get("pass") is not False
            or gate.get("complete_candidate_dependent_inner_gate") is not False
            or gate.get("five_view_fidelity_gate") is not False
            or gate.get("all_three_negative_full_gates") is not False
            or training.get("full_budget_steps_executed") != 1200
            or training.get("fixed_step") != 1200
            or training.get("early_stopped") is not False
            or scope != EXPECTED_QUALIFICATION_SCOPE
            or selective.get("stage1_oof_semantic_tensor_count") != 0
            or selective.get("stage2_oof_semantic_tensor_count") != 0
            or pre.get("file_sha256") != pre_scan["sha256"]
            or fixed.get("file_sha256") != fixed_scan["sha256"]
            or pre.get("size_bytes") != pre_scan["size_bytes"]
            or fixed.get("size_bytes") != fixed_scan["size_bytes"]
            or pre.get("mode_octal") != "0444" or fixed.get("mode_octal") != "0444"
            or pre.get("nlink") != 1 or fixed.get("nlink") != 1
            or pre_physical != {
                "device": pre_scan["device"], "inode": pre_scan["inode"],
                "size_bytes": pre_scan["size_bytes"],
            }
            or fixed_physical != {
                "device": fixed_scan["device"], "inode": fixed_scan["inode"],
                "size_bytes": fixed_scan["size_bytes"],
            }
            or pre_physical == fixed_physical
            or pair.get("distinct_device_inode_pair") is not True
            or pair.get("same_model_state_sha256") is not True
            or pre.get("model_state_sha256") != fixed.get("model_state_sha256")
        ):
            raise RuntimeError(f"fold{fold} scientific/checkpoint closure differs")
        state_sha = fixed["model_state_sha256"]
        if state_sha in state_shas:
            raise RuntimeError("cross-fold model states unexpectedly duplicate")
        state_shas.add(state_sha)
        summaries.append({
            "fold_index": fold,
            "inner_receipt_sha256": INNER_RECEIPT_SHA256[fold],
            "inner_receipt_digest": INNER_RECEIPT_DIGEST[fold],
            "inner_status": receipt["status"],
            "inner_pass": False,
            "oof_semantic_tensor_read_count": 0,
            "oof_semantic_tensor_materialized_count": 0,
            "model_fit_count": receipt["model_fit_original_count"],
            "inner_count": receipt["inner_validation_original_count"],
            "oof_count": receipt["oof_original_count"],
            "preselection_checkpoint": {
                "sha256": pre_scan["sha256"], "size_bytes": pre_scan["size_bytes"],
                "device": pre_scan["device"], "inode": pre_scan["inode"],
                "metadata_digest": pre["metadata_digest"],
            },
            "fixed1200_checkpoint": {
                "sha256": fixed_scan["sha256"], "size_bytes": fixed_scan["size_bytes"],
                "device": fixed_scan["device"], "inode": fixed_scan["inode"],
                "metadata_digest": fixed["metadata_digest"],
                "model_state_sha256": state_sha,
            },
            "three_field_runtime_physical_identity_projection_exact": True,
            "mode_and_nlink_verified_separately": True,
            "fidelity_gate": False,
            "all_three_negative_full_gates": False,
            "complete_gate": False,
        })
    if sum(row["oof_semantic_tensor_read_count"] for row in summaries) != 0:
        raise RuntimeError("all-fold OOF exact0 closure differs")
    return summaries


def _reverify_prepublication_authorities(value: Mapping[str, Any]) -> None:
    """Rejoin every mutable source authority immediately before publication."""
    source = value["source_authority"]
    if _verify_release(RELEASE_ROOT) != source["release"]:
        raise RuntimeError("historical release changed before recovery publication")
    if _verify_recovery_release_and_controller() != source["recovery"]:
        raise RuntimeError("recovery release changed before recovery publication")
    controller, _ = _read_regular(
        CONTROLLER_PATH, mode=0o555, nlink=1,
    )
    python, _ = _read_regular(PYTHON_PATH, mode=0o755, nlink=1)
    if (
        controller != source["controller"] or python != source["python"]
        or _verify_process_python() != source["process_python"]
    ):
        raise RuntimeError("controller/Python changed before recovery publication")
    target = (
        "methods.bernini_action_editing."
        "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g"
    )
    loaded_v4g = sys.modules.get(target)
    if (
        loaded_v4g is None
        or {
            **_verify_recovery_parser_torch(loaded_v4g),
            **_verify_recovery_parser_torch_origins(loaded_v4g),
        }
            != source["recovery_parser_torch"]
    ):
        raise RuntimeError("recovery parser Torch changed before publication")
    authorities: list[dict[str, Any]] = []
    authority_raw: list[tuple[dict[str, Any], bytes]] = []
    for path, _ in AUTHORITY_FILES:
        binding, raw = _read_regular(path, mode=0o444, nlink=1, capture=True)
        if raw is None:
            raise RuntimeError("input authority recapture differs")
        authorities.append(binding)
        authority_raw.append((binding, raw))
    if (
        authorities != source["input_receipts"]
        or _verify_input_snapshot(authority_raw) != source["input_snapshot"]
    ):
        raise RuntimeError("input authorities changed before recovery publication")
    stdout, _ = _read_regular(
        OUTER_STDOUT_PATH, mode=0o600, nlink=1,
    )
    stderr, _ = _read_regular(
        OUTER_STDERR_PATH, mode=0o600, nlink=1,
    )
    if (
        {"stdout": stdout, "stderr": stderr}
            != value["original_controller_logs"]
        or _query_failed_step() != value["failed_seal_child_accounting"]
    ):
        raise RuntimeError("failure authorities changed before recovery publication")


READ_BINDING_KEYS = {
    "path", "sha256", "size_bytes", "mode_octal", "nlink", "device",
    "inode", "mtime_ns", "ctime_ns",
    "single_fd_pre_post_identity_and_sha_exact",
}


def _require_recorded_binding(
    value: Any, *, path: Path | None = None, sha256: str | None = None,
    mode: str | None = None, nlink: int | None = None,
) -> None:
    if type(value) is not dict or set(value) != READ_BINDING_KEYS:
        raise RuntimeError("recorded authority binding schema differs")
    if (
        value.get("single_fd_pre_post_identity_and_sha_exact") is not True
        or type(value.get("sha256")) is not str or len(value["sha256"]) != 64
        or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0
        or type(value.get("device")) is not int
        or type(value.get("inode")) is not int
        or type(value.get("mtime_ns")) is not int
        or type(value.get("ctime_ns")) is not int
    ):
        raise RuntimeError("recorded authority binding values differ")
    if path is not None and value.get("path") != str(path):
        raise RuntimeError("recorded authority binding path differs")
    if sha256 is not None and value.get("sha256") != sha256:
        raise RuntimeError("recorded authority binding SHA differs")
    if mode is not None and value.get("mode_octal") != mode:
        raise RuntimeError("recorded authority binding mode differs")
    if nlink is not None and value.get("nlink") != nlink:
        raise RuntimeError("recorded authority binding nlink differs")


def _validate_publish_value(root: Path, value: Mapping[str, Any]) -> None:
    scope = value.get("qualification_scope")
    folds = value.get("folds")
    rows = value.get("burned_exact26_manifest")
    source = value.get("source_authority")
    release = source.get("release") if type(source) is dict else None
    recovery_source = source.get("recovery") if type(source) is dict else None
    parser_torch = (
        source.get("recovery_parser_torch") if type(source) is dict else None
    )
    input_snapshot = source.get("input_snapshot") if type(source) is dict else None
    original_root_binding = value.get("original_run_root_binding")
    launch = value.get("launch_plan")
    logs = value.get("original_controller_logs")
    accounting = value.get("failed_seal_child_accounting")
    bug = value.get("old_controller_identity_schema_bug")
    if (
        type(value) is not dict or set(value) != ATTESTATION_KEYS
        or "receipt_digest" in value
        or value.get("schema_version") != SCHEMA
        or value.get("authority")
            != "burned_known_transform_development_scientific_no_go_only"
        or value.get("original_run_root") != str(ORIGINAL_RUN_ROOT)
        or value.get("recovery_root") != str(root)
        or value.get("original_run_mutated_by_recovery") is not False
        or value.get("original_run_postverified_unchanged") is not True
        or value.get("original_controller_complete") is not False
        or value.get("original_controller_exit_nonzero") is not True
        or value.get("scientifically_verified_all_inner_no_go") is not True
        or value.get("global_inner_barrier_created") is not False
        or value.get("evaluate_fold_executed") is not False
        or value.get("aggregate_executed") is not False
        or value.get("all_fold_oof_semantic_tensor_read_count") != 0
        or value.get("all_fold_oof_semantic_tensor_materialized_count") != 0
        or value.get("recovery_ledger_reconstructed_from_burned_exact26") is not True
        or value.get("burned_exact26_file_count") != 26
        or value.get("burned_exact26_manifest_sha256")
            != EXACT26_MANIFEST_SHA256
        or type(rows) is not list or len(rows) != EXPECTED_FILE_COUNT
        or any(
            type(row) is not dict or set(row) != {
                "path", "sha256", "size_bytes", "mode_octal", "nlink",
                "device", "inode",
            }
            for row in rows
        )
        or _object_sha(rows) != EXACT26_MANIFEST_SHA256
        or [row["path"] for row in rows] != sorted(EXPECTED_FILES)
        or any(row["mode_octal"] != "0444" or row["nlink"] != 1 for row in rows)
        or value.get("burned_parent_stable_signature_sha256")
            != PARENT_STABLE_SIGNATURE_SHA256
        or type(folds) is not list or len(folds) != 5
        or [row.get("fold_index") for row in folds] != list(range(5))
        or any(row.get("inner_pass") is not False for row in folds)
        or any(row.get("oof_semantic_tensor_read_count") != 0 for row in folds)
        or any(row.get("oof_semantic_tensor_materialized_count") != 0 for row in folds)
        or any(row.get("inner_receipt_sha256") != INNER_RECEIPT_SHA256[index]
               for index, row in enumerate(folds))
        or any(row.get("inner_receipt_digest") != INNER_RECEIPT_DIGEST[index]
               for index, row in enumerate(folds))
        or any(row.get("inner_status")
               != "V4G_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD" for row in folds)
        or any(row.get("complete_gate") is not False for row in folds)
        or any(row.get("fidelity_gate") is not False for row in folds)
        or any(row.get("all_three_negative_full_gates") is not False for row in folds)
        or value.get("all_qualification_claims_false") is not True
        or scope != EXPECTED_ATTESTATION_QUALIFICATION_SCOPE
        or bug != {
            "runtime_receipt_identity_keys": ["device", "inode", "size_bytes"],
            "phase_binding_additional_keys": [
                "mode_octal", "nlink", "mtime_ns", "ctime_ns",
            ],
            "all_ten_checkpoint_three_field_projections_exact": True,
            "all_ten_checkpoint_full_object_equal": False,
            "mode_and_nlink_verified_independently": True,
            "bug_affected_scientific_values": False,
            "bug_only_blocked_final_controller_seal": True,
        }
        or type(source) is not dict or set(source) != {
            "authority_snapshot", "release", "controller", "python",
            "process_python", "input_receipts", "input_snapshot",
            "runtime_sha256", "tests_sha256", "recovery",
            "recovery_parser_torch",
        }
        or source.get("authority_snapshot") != AUTHORITY_SNAPSHOT
        or source.get("runtime_sha256") != RUNTIME_SHA256
        or source.get("tests_sha256") != TESTS_SHA256
        or type(parser_torch) is not dict or set(parser_torch) != {
            "torch_version", "torch_hip_version",
            "torch_version_exact_2_7_1_rocm6_3", "torch_hip_release_6_3",
            "torch_package_root", "torch_module_origins",
            "v4g_torch_module_identities_exact",
            "torch_standard_source_loaders_and_package_path_exact",
        }
        or parser_torch.get("torch_version") != "2.7.1+rocm6.3"
        or type(parser_torch.get("torch_hip_version")) is not str
        or not parser_torch["torch_hip_version"].startswith("6.3")
        or parser_torch.get("torch_version_exact_2_7_1_rocm6_3") is not True
        or parser_torch.get("torch_hip_release_6_3") is not True
        or parser_torch.get("torch_package_root")
            != str(PYTHON_SITE_PACKAGES / "torch")
        or parser_torch.get("torch_module_origins") != {
            "torch": str(PYTHON_SITE_PACKAGES / "torch/__init__.py"),
            "torch.nn": str(PYTHON_SITE_PACKAGES / "torch/nn/__init__.py"),
            "torch.nn.functional": str(
                PYTHON_SITE_PACKAGES / "torch/nn/functional.py"
            ),
        }
        or parser_torch.get("v4g_torch_module_identities_exact") is not True
        or parser_torch.get(
            "torch_standard_source_loaders_and_package_path_exact"
        ) is not True
        or type(release) is not dict or set(release) != {
            "root", "tree_sha256", "manifest", "manifest_digest",
            "file_count", "directory_count",
        }
        or release.get("root") != str(RELEASE_ROOT)
        or release.get("tree_sha256") != RELEASE_TREE_SHA256
        or release.get("manifest_digest") != RELEASE_MANIFEST_DIGEST
        or release.get("file_count") != 11 or release.get("directory_count") != 3
        or type(source.get("input_receipts")) is not list
        or len(source["input_receipts"]) != len(AUTHORITY_FILES)
        or type(input_snapshot) is not dict or set(input_snapshot) != {
            "sha256", "ordered_rows", "feature_shard_bindings",
            "exact_receipt_count", "exact_feature_shard_count",
            "all_ten_files_single_fd_reverified",
        }
        or input_snapshot.get("sha256") != INPUT_SNAPSHOT_SHA256
        or type(input_snapshot.get("ordered_rows")) is not list
        or len(input_snapshot["ordered_rows"]) != 10
        or _object_sha(input_snapshot["ordered_rows"]) != INPUT_SNAPSHOT_SHA256
        or type(input_snapshot.get("feature_shard_bindings")) is not list
        or len(input_snapshot["feature_shard_bindings"]) != 6
        or input_snapshot.get("exact_receipt_count") != 4
        or input_snapshot.get("exact_feature_shard_count") != 6
        or input_snapshot.get("all_ten_files_single_fd_reverified") is not True
        or type(recovery_source) is not dict or set(recovery_source) != {
            "release_root", "release_root_binding", "release_tree_sha256",
            "tree_rows", "manifest", "manifest_digest", "runtime", "tests", "controller",
            "exact_file_count", "exact_directory_count_below_root",
            "one_way_sha256_dag_reverified",
            "controller_identity_recorded_not_runtime_reverse_pinned",
        }
        or recovery_source.get("release_root") != str(RECOVERY_RELEASE_ROOT)
        or type(recovery_source.get("release_tree_sha256")) is not str
        or len(recovery_source["release_tree_sha256"]) != 64
        or type(recovery_source.get("tree_rows")) is not list
        or len(recovery_source["tree_rows"]) != 3
        or _object_sha(recovery_source["tree_rows"])
            != recovery_source["release_tree_sha256"]
        or type(recovery_source.get("manifest_digest")) is not str
        or len(recovery_source["manifest_digest"]) != 64
        or recovery_source.get("exact_file_count") != 3
        or recovery_source.get("exact_directory_count_below_root") != 3
        or recovery_source.get("one_way_sha256_dag_reverified") is not True
        or recovery_source.get(
            "controller_identity_recorded_not_runtime_reverse_pinned"
        ) is not True
        or type(original_root_binding) is not dict
        or set(original_root_binding) != {
            "path", "mode_octal", "nlink", "device", "inode", "mtime_ns",
            "ctime_ns", "members",
            "single_fd_pre_post_identity_and_membership_exact",
        }
        or original_root_binding.get("path") != str(ORIGINAL_RUN_ROOT)
        or original_root_binding.get("mode_octal") != "0700"
        or original_root_binding.get("members") != sorted(
            {"fold0", "fold1", "fold2", "fold3", "fold4", "logs",
             "launch-plan.json"}
        )
        or original_root_binding.get(
            "single_fd_pre_post_identity_and_membership_exact"
        ) is not True
        or type(launch) is not dict or set(launch) != {
            "binding", "schema_version", "normal_tests_run",
            "normal_tests_skipped", "optimized_tests_run", "optimized_tests_skipped",
        }
        or launch.get("normal_tests_run") != 36
        or launch.get("normal_tests_skipped") != 0
        or launch.get("optimized_tests_run") != 36
        or launch.get("optimized_tests_skipped") != 0
        or type(launch.get("schema_version")) is not str
        or type(logs) is not dict or set(logs) != {"stdout", "stderr"}
        or type(accounting) is not dict or set(accounting) != {
            "record", "sacct_executable", "query_columns", "exact_row_replayed",
        }
        or accounting.get("record") != {
            "job_id_raw": "143811.94", "state": "FAILED", "exit_code": "1:0",
            "node": "auh7-1b-gpu-306", "alloc_cpus": "8", "req_mem": "",
            "elapsed": "00:00:03", "start": "2026-08-21T09:43:49",
            "end": "2026-08-21T09:43:52",
        }
        or accounting.get("query_columns") != [
            "job_id_raw", "state", "exit_code", "node", "alloc_cpus",
            "req_mem", "elapsed", "start", "end",
        ]
        or accounting.get("exact_row_replayed") is not True
    ):
        raise RuntimeError("recovery attestation publish schema/claims differ")

    _require_recorded_binding(
        release["manifest"], path=RELEASE_ROOT / "release-manifest-v4g.json",
        sha256=RELEASE_MANIFEST_SHA256, mode="0444", nlink=1,
    )
    _require_recorded_binding(
        source["controller"], path=CONTROLLER_PATH, sha256=CONTROLLER_SHA256,
        mode="0555", nlink=1,
    )
    _require_recorded_binding(
        source["python"], path=PYTHON_PATH, sha256=PYTHON_SHA256,
        mode="0755", nlink=1,
    )
    if source["process_python"] != source["python"]:
        raise RuntimeError("recorded process Python binding differs")
    for binding, (path, sha256) in zip(source["input_receipts"], AUTHORITY_FILES):
        _require_recorded_binding(
            binding, path=path, sha256=sha256, mode="0444", nlink=1,
        )
    for index, binding in enumerate(input_snapshot["feature_shard_bindings"]):
        _require_recorded_binding(binding, mode="0444", nlink=1)
        row = input_snapshot["ordered_rows"][4 + index]
        if row != {
            "label": f"feature-shard-{index}", "path": binding["path"],
            "sha256": binding["sha256"], "size_bytes": binding["size_bytes"],
        }:
            raise RuntimeError("recorded feature shard snapshot differs")
    for index, binding in enumerate(source["input_receipts"]):
        row = input_snapshot["ordered_rows"][index]
        if row != {
            "label": ("feature", "v4a", "v4c", "v4d")[index],
            "path": binding["path"], "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }:
            raise RuntimeError("recorded input receipt snapshot differs")
    release_root_binding = recovery_source["release_root_binding"]
    if (
        type(release_root_binding) is not dict
        or set(release_root_binding) != {
            "path", "mode_octal", "nlink", "device", "inode", "mtime_ns",
            "ctime_ns", "members",
            "single_fd_pre_post_identity_and_membership_exact",
        }
        or release_root_binding.get("path") != str(RECOVERY_RELEASE_ROOT)
        or release_root_binding.get("mode_octal") != "0555"
        or release_root_binding.get("members") != [
            "methods", RECOVERY_MANIFEST_NAME,
        ]
        or release_root_binding.get(
            "single_fd_pre_post_identity_and_membership_exact"
        ) is not True
    ):
        raise RuntimeError("recorded recovery release root binding differs")
    _require_recorded_binding(
        recovery_source["manifest"],
        path=RECOVERY_RELEASE_ROOT / RECOVERY_MANIFEST_NAME,
        mode="0444", nlink=1,
    )
    _require_recorded_binding(
        recovery_source["runtime"],
        path=RECOVERY_RELEASE_ROOT / RECOVERY_RUNTIME_RELATIVE_PATH,
        mode="0444", nlink=1,
    )
    _require_recorded_binding(
        recovery_source["tests"],
        path=RECOVERY_RELEASE_ROOT / RECOVERY_TESTS_RELATIVE_PATH,
        mode="0444", nlink=1,
    )
    _require_recorded_binding(
        recovery_source["controller"], path=RECOVERY_CONTROLLER_PATH,
        mode="0555", nlink=1,
    )
    expected_recovery_tree_rows = [{
        "path": relative, "sha256": binding["sha256"],
        "size_bytes": binding["size_bytes"],
    } for relative, binding in sorted({
        RECOVERY_MANIFEST_NAME: recovery_source["manifest"],
        RECOVERY_RUNTIME_RELATIVE_PATH: recovery_source["runtime"],
        RECOVERY_TESTS_RELATIVE_PATH: recovery_source["tests"],
    }.items())]
    if recovery_source["tree_rows"] != expected_recovery_tree_rows:
        raise RuntimeError("recorded recovery release tree rows differ")
    _require_recorded_binding(
        launch["binding"], path=ORIGINAL_RUN_ROOT / "launch-plan.json",
        sha256=LAUNCH_PLAN_SHA256, mode="0444", nlink=1,
    )
    _require_recorded_binding(
        logs["stdout"], path=OUTER_STDOUT_PATH, sha256=OUTER_STDOUT["sha256"],
        mode="0600", nlink=1,
    )
    _require_recorded_binding(
        logs["stderr"], path=OUTER_STDERR_PATH, sha256=OUTER_STDERR["sha256"],
        mode="0600", nlink=1,
    )
    for key, expected in OUTER_STDOUT.items():
        if logs["stdout"].get(key) != expected:
            raise RuntimeError("recorded outer stdout binding differs")
    for key, expected in OUTER_STDERR.items():
        if logs["stderr"].get(key) != expected:
            raise RuntimeError("recorded outer stderr binding differs")
    _require_recorded_binding(
        accounting["sacct_executable"], path=SACCT_PATH,
        sha256=SACCT_SHA256, mode="0755", nlink=1,
    )

    row_by_path = {row["path"]: row for row in rows}
    expected_counts = ((400, 113, 131), (402, 115, 127), (401, 115, 128),
                       (403, 112, 129), (403, 112, 129))
    for fold, summary in enumerate(folds):
        if type(summary) is not dict or set(summary) != {
            "fold_index", "inner_receipt_sha256", "inner_receipt_digest",
            "inner_status", "inner_pass", "oof_semantic_tensor_read_count",
            "oof_semantic_tensor_materialized_count", "model_fit_count",
            "inner_count", "oof_count", "preselection_checkpoint",
            "fixed1200_checkpoint",
            "three_field_runtime_physical_identity_projection_exact",
            "mode_and_nlink_verified_separately", "fidelity_gate",
            "all_three_negative_full_gates", "complete_gate",
        }:
            raise RuntimeError("recorded fold summary schema differs")
        if (
            (summary["model_fit_count"], summary["inner_count"], summary["oof_count"])
                != expected_counts[fold]
            or summary["three_field_runtime_physical_identity_projection_exact"] is not True
            or summary["mode_and_nlink_verified_separately"] is not True
        ):
            raise RuntimeError("recorded fold summary values differ")
        for name, key in (("preselection.pt", "preselection_checkpoint"),
                          ("fixed1200.pt", "fixed1200_checkpoint")):
            checkpoint = summary[key]
            burned = row_by_path[f"fold{fold}/{name}"]
            required = {"sha256", "size_bytes", "device", "inode", "metadata_digest"}
            if name == "fixed1200.pt":
                required.add("model_state_sha256")
            if (
                type(checkpoint) is not dict or set(checkpoint) != required
                or checkpoint["sha256"] != burned["sha256"]
                or checkpoint["size_bytes"] != burned["size_bytes"]
                or checkpoint["device"] != burned["device"]
                or checkpoint["inode"] != burned["inode"]
                or type(checkpoint["metadata_digest"]) is not str
                or len(checkpoint["metadata_digest"]) != 64
            ):
                raise RuntimeError("recorded fold checkpoint binding differs")


def _require_directory_path_fd(path: Path, descriptor: int, *, mode: int) -> os.stat_result:
    try:
        resolved = path.resolve(strict=True)
        current = path.lstat()
        opened = os.fstat(descriptor)
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError("directory path/FD authority is unavailable") from error
    if (
        str(path) != str(resolved)
        or not stat.S_ISDIR(current.st_mode)
        or _identity(current) != _identity(opened)
        or stat.S_IMODE(current.st_mode) != mode
    ):
        raise RuntimeError("directory current-path/same-FD identity differs")
    return opened


def _require_child_directory_fd(
    parent_fd: int, name: str, descriptor: int, *, mode: int,
) -> os.stat_result:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(current.st_mode)
        or _identity(current) != _identity(opened)
        or stat.S_IMODE(current.st_mode) != mode
    ):
        raise RuntimeError("child directory current-path/same-FD identity differs")
    return opened


def _open_held_directory_chain(
    anchor: Path, target: Path,
) -> tuple[list[int], list[tuple[int, str, int]], int]:
    try:
        relative = target.relative_to(anchor)
    except ValueError as error:
        raise RuntimeError("recovery target is outside trusted path anchor") from error
    if (
        not anchor.is_absolute() or not target.is_absolute()
        or str(anchor) != str(anchor.resolve(strict=True))
        or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
    ):
        raise RuntimeError("trusted path anchor is not canonical")
    anchor_fd = os.open(
        anchor, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    descriptors = [anchor_fd]
    links: list[tuple[int, str, int]] = []
    try:
        _require_directory_path_fd(
            anchor, anchor_fd, mode=stat.S_IMODE(os.fstat(anchor_fd).st_mode),
        )
        current_fd = anchor_fd
        for part in relative.parts:
            if part in {"", ".", ".."}:
                raise RuntimeError("trusted path chain component differs")
            child_fd = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            descriptors.append(child_fd)
            links.append((current_fd, part, child_fd))
            _require_child_directory_fd(
                current_fd, part, child_fd,
                mode=stat.S_IMODE(os.fstat(child_fd).st_mode),
            )
            current_fd = child_fd
        return descriptors, links, current_fd
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _require_held_directory_chain(
    anchor: Path, descriptors: Sequence[int], links: Sequence[tuple[int, str, int]],
) -> None:
    if not descriptors:
        raise RuntimeError("trusted path descriptor chain differs")
    anchor_fd = descriptors[0]
    _require_directory_path_fd(
        anchor, anchor_fd, mode=stat.S_IMODE(os.fstat(anchor_fd).st_mode),
    )
    for parent_fd, name, child_fd in links:
        _require_child_directory_fd(
            parent_fd, name, child_fd,
            mode=stat.S_IMODE(os.fstat(child_fd).st_mode),
        )


def _require_attestation_file_at(
    directory_fd: int, file_fd: int, raw: bytes, creation: os.stat_result,
    *, mode: int,
) -> dict[str, Any]:
    name = "recovery-attestation.json"
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    opened = os.fstat(file_fd)
    os.lseek(file_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
    closed = os.fstat(file_fd)
    after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    identities = {_identity(item) for item in (creation, before, opened, closed, after)}
    if (
        len(identities) != 1 or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode or before.st_nlink != 1
        or b"".join(chunks) != raw
        or digest.hexdigest() != hashlib.sha256(raw).hexdigest()
    ):
        raise RuntimeError("recovery attestation current-path/same-FD binding differs")
    return {
        "sha256": digest.hexdigest(), "size_bytes": before.st_size,
        "mode_octal": f"{mode:04o}", "nlink": 1,
        "device": before.st_dev, "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
        "single_fd_pre_post_identity_and_sha_exact": True,
    }


def _require_recovery_root_binding_at(
    parent_fd: int, name: str, root_fd: int, path: Path, *, mode: int,
) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        path_info = path.lstat()
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError("recovery root current path is unavailable") from error
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(root_fd)
    members = sorted(os.listdir(root_fd))
    closed = os.fstat(root_fd)
    after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        str(path) != str(resolved)
        or stat.S_ISLNK(path_info.st_mode)
        or len({
            _identity(item)
            for item in (path_info, before, opened, closed, after)
        }) != 1
        or not stat.S_ISDIR(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
        or members != ["recovery-attestation.json"]
    ):
        raise RuntimeError("recovery root current-path/same-FD binding differs")
    return {
        "path": str(path), "mode_octal": f"{mode:04o}",
        "nlink": before.st_nlink, "device": before.st_dev,
        "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns, "members": members,
        "single_fd_pre_post_identity_and_membership_exact": True,
    }


def _publish_attestation_create_only(root: Path) -> dict[str, Any]:
    """Claim the final namespace directly and commit only by final modes.

    The final directory name and final file name are both create-only
    reservations.  A failure after either reservation deliberately leaves a
    non-accepted tombstone.  The namespace is never removed, renamed, linked,
    or retried by this recovery program.
    """
    _require_release_guard()
    if root != RECOVERY_ROOT:
        raise RuntimeError("official recovery path differs from frozen authority")
    value, rows_before, bindings_before = _collect_verified_attestation(
        ORIGINAL_RUN_ROOT, root,
    )
    _validate_publish_value(root, value)
    if (
        not root.is_absolute() or root.name in {"", ".", ".."}
        or str(root.parent) != str(root.parent.resolve(strict=True))
        or os.path.lexists(root)
        or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
    ):
        raise RuntimeError("recovery publish path/capabilities differ")
    sealed_value = dict(value)
    sealed_value["receipt_digest"] = _object_sha(sealed_value)
    raw = _canonical_bytes(sealed_value) + b"\n"
    if _json_no_duplicates(raw) != sealed_value:
        raise RuntimeError("recovery attestation canonical payload differs")

    parent = root.parent
    chain_descriptors, chain_links, parent_fd = _open_held_directory_chain(
        TRUSTED_PATH_ANCHOR, parent,
    )
    root_fd: int | None = None
    file_fd: int | None = None
    try:
        parent_before = parent.lstat()
        _require_held_directory_chain(
            TRUSTED_PATH_ANCHOR, chain_descriptors, chain_links,
        )
        parent_opened = _require_directory_path_fd(
            parent, parent_fd, mode=stat.S_IMODE(parent_before.st_mode),
        )
        if (
            not stat.S_ISDIR(parent_before.st_mode)
            or _identity(parent_before) != _identity(parent_opened)
        ):
            raise RuntimeError("recovery parent same-FD identity differs")

        # Atomic, NFS-safe final-name reservation.  EEXIST is terminal: this
        # recovery never mutates or retries an occupied official namespace.
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise RuntimeError("official recovery root is not fresh") from error
        root_fd = os.open(
            root.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        created_root = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(created_root.st_mode)
            or stat.S_IMODE(created_root.st_mode) != 0o700
        ):
            raise RuntimeError("recovery tombstone root creation differs")
        _require_child_directory_fd(
            parent_fd, root.name, root_fd, mode=0o700,
        )
        os.fsync(parent_fd)

        flags = (
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        try:
            # Mode 000 is deliberately non-authoritative.  If any later check
            # fails, the held 0700 root plus this file remain a rejected
            # tombstone and permanently burn this recovery namespace.
            file_fd = os.open(
                "recovery-attestation.json", flags, 0,
                dir_fd=root_fd,
            )
        except FileExistsError as error:
            raise RuntimeError("recovery attestation final name is not fresh") from error
        created_file = os.fstat(file_fd)
        named_created_file = os.stat(
            "recovery-attestation.json", dir_fd=root_fd,
            follow_symlinks=False,
        )
        if (
            _identity(created_file) != _identity(named_created_file)
            or not stat.S_ISREG(created_file.st_mode)
            or stat.S_IMODE(created_file.st_mode) != 0
            or created_file.st_nlink != 1 or created_file.st_size != 0
        ):
            raise RuntimeError("recovery file create-only identity differs")

        offset = 0
        while offset < len(raw):
            written_count = os.write(file_fd, raw[offset:])
            if written_count <= 0:
                raise RuntimeError("recovery attestation write stalled")
            offset += written_count
        os.fsync(file_fd)
        written = os.fstat(file_fd)
        os.lseek(file_fd, 0, os.SEEK_SET)
        readback = bytearray()
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            readback.extend(chunk)
        reread = os.fstat(file_fd)
        if (
            bytes(readback) != raw or _identity(written) != _identity(reread)
            or written.st_dev != created_file.st_dev
            or written.st_ino != created_file.st_ino
            or not stat.S_ISREG(written.st_mode)
            or stat.S_IMODE(written.st_mode) != 0 or written.st_nlink != 1
            or hashlib.sha256(readback).hexdigest()
                != hashlib.sha256(raw).hexdigest()
        ):
            raise RuntimeError("recovery tombstone exact1 same-FD replay differs")
        replayed = _json_no_duplicates(bytes(readback))
        replayed_unsigned = dict(replayed)
        replayed_digest = replayed_unsigned.pop("receipt_digest", None)
        if (
            replayed_unsigned != value
            or replayed_digest != sealed_value["receipt_digest"]
            or replayed_digest != _object_sha(replayed_unsigned)
        ):
            raise RuntimeError("recovery tombstone canonical self-digest differs")
        if os.listdir(root_fd) != ["recovery-attestation.json"]:
            raise RuntimeError("recovery tombstone exact1 membership differs")
        _require_attestation_file_at(
            root_fd, file_fd, raw, written, mode=0,
        )
        os.fsync(root_fd)
        os.fsync(parent_fd)

        # The official name exists, but root mode 0700 and file mode 000 are a
        # rejection state.  Rejoin every held name and revalidate the original
        # exact26 plus all detached authorities before the acceptance commit.
        _require_held_directory_chain(
            TRUSTED_PATH_ANCHOR, chain_descriptors, chain_links,
        )
        _require_directory_path_fd(
            parent, parent_fd, mode=stat.S_IMODE(parent_before.st_mode),
        )
        _require_child_directory_fd(
            parent_fd, root.name, root_fd, mode=0o700,
        )
        if os.listdir(root_fd) != ["recovery-attestation.json"]:
            raise RuntimeError("recovery tombstone exact1 membership differs")
        _require_attestation_file_at(
            root_fd, file_fd, raw, written, mode=0,
        )
        rows_after_claim, bindings_after_claim = _scan_original(
            ORIGINAL_RUN_ROOT,
        )
        original_root_after_claim = _read_directory_binding(
            ORIGINAL_RUN_ROOT, mode=0o700,
        )
        if (
            rows_after_claim != rows_before
            or bindings_after_claim != bindings_before
            or original_root_after_claim != value["original_run_root_binding"]
        ):
            raise RuntimeError("original run changed after create-only name claim")
        rows_precommit, bindings_precommit = _scan_original(ORIGINAL_RUN_ROOT)
        original_root_precommit = _read_directory_binding(
            ORIGINAL_RUN_ROOT, mode=0o700,
        )
        if (
            rows_precommit != rows_before
            or bindings_precommit != bindings_before
            or original_root_precommit != value["original_run_root_binding"]
        ):
            raise RuntimeError("original run changed before final mode commit")
        _reverify_prepublication_authorities(value)

        # Close every mutable-read interval before committing.  All fallible
        # validation and durability operations occur while the root is still
        # the 0700 rejection tombstone.
        _require_held_directory_chain(
            TRUSTED_PATH_ANCHOR, chain_descriptors, chain_links,
        )
        _require_directory_path_fd(
            parent, parent_fd, mode=stat.S_IMODE(parent_before.st_mode),
        )
        _require_child_directory_fd(
            parent_fd, root.name, root_fd, mode=0o700,
        )
        if os.listdir(root_fd) != ["recovery-attestation.json"]:
            raise RuntimeError("recovery precommit exact1 membership differs")
        _require_attestation_file_at(
            root_fd, file_fd, raw, written, mode=0,
        )
        os.fsync(file_fd)
        os.fsync(root_fd)
        os.fsync(parent_fd)
        _require_held_directory_chain(
            TRUSTED_PATH_ANCHOR, chain_descriptors, chain_links,
        )
        _require_directory_path_fd(
            parent, parent_fd, mode=stat.S_IMODE(parent_before.st_mode),
        )
        _require_child_directory_fd(
            parent_fd, root.name, root_fd, mode=0o700,
        )
        if os.listdir(root_fd) != ["recovery-attestation.json"]:
            raise RuntimeError("recovery durable precommit exact1 membership differs")
        _require_attestation_file_at(
            root_fd, file_fd, raw, written, mode=0,
        )

        # First-level commit.  The root remains the 0700 rejection tombstone,
        # so the final 0444 file can still be replayed and joined to its named
        # inode without creating an externally accepted result.
        os.fchmod(file_fd, 0o444)
        os.fsync(file_fd)
        file_final_stat = os.fstat(file_fd)
        if (
            file_final_stat.st_dev != written.st_dev
            or file_final_stat.st_ino != written.st_ino
            or not stat.S_ISREG(file_final_stat.st_mode)
            or stat.S_IMODE(file_final_stat.st_mode) != 0o444
            or file_final_stat.st_nlink != 1
        ):
            raise RuntimeError("recovery file creation-to-final identity differs")
        producer_file_binding = {
            "path": str(root / "recovery-attestation.json"),
            **_require_attestation_file_at(
                root_fd, file_fd, raw, file_final_stat, mode=0o444,
            ),
        }
        os.fsync(root_fd)
        os.fsync(parent_fd)
        _require_held_directory_chain(
            TRUSTED_PATH_ANCHOR, chain_descriptors, chain_links,
        )
        _require_directory_path_fd(
            parent, parent_fd, mode=stat.S_IMODE(parent_before.st_mode),
        )
        producer_root_binding = _require_recovery_root_binding_at(
            parent_fd, root.name, root_fd, root, mode=0o700,
        )
        producer_file_binding_replay = {
            "path": str(root / "recovery-attestation.json"),
            **_require_attestation_file_at(
                root_fd, file_fd, raw, file_final_stat, mode=0o444,
            ),
        }
        if (
            producer_file_binding_replay != producer_file_binding
            or producer_root_binding["device"] != created_root.st_dev
            or producer_root_binding["inode"] != created_root.st_ino
            or producer_file_binding["device"] != created_file.st_dev
            or producer_file_binding["inode"] != created_file.st_ino
        ):
            raise RuntimeError("recovery creation-to-precommit continuity differs")

        result = {
            "path": str(root / "recovery-attestation.json"),
            "file_sha256": producer_file_binding["sha256"],
            "size_bytes": len(raw),
            "receipt_digest": sealed_value["receipt_digest"],
            "mode_octal": "0444", "nlink": 1,
            "root_mode_octal": "0555", "exact_file_count": 1,
            "create_only_name_claim": True,
            "failure_tombstone_root_mode_octal": "0700",
            "original_run_and_source_authorities_reverified_after_name_claim": True,
            "root_and_file_same_fd_precommit_verified_and_parent_fsynced": True,
            "producer_root_precommit_binding": producer_root_binding,
            "producer_attestation_final_binding": producer_file_binding,
            "root_creation_to_precommit_device_inode_exact": True,
            "file_creation_to_final_device_inode_exact": True,
            "final_mode_commit": True,
            "final_mode_commit_order": ["file_0444", "root_0555"],
            "schema_version": RESULT_SCHEMA,
            "original_run_postverified_unchanged": True,
            "original_run_exact26_manifest_sha256": EXACT26_MANIFEST_SHA256,
            "scientific_result": "ALL_FIVE_INNER_NO_GO_ALL_OOF_UNREAD",
            "original_controller_complete": False,
        }

        # Second-level acceptance commit.  The root chmod is the final
        # fallible filesystem operation; no stat, fsync, replay, or path lookup
        # follows it.
        os.fchmod(root_fd, 0o555)
        return result
    finally:
        # Closing retained descriptors cannot alter acceptance state.  Close
        # errors are intentionally ignored, especially after the final commit.
        if file_fd is not None:
            try:
                os.close(file_fd)
            except OSError:
                pass
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass
        for descriptor in reversed(chain_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _collect_verified_attestation(
    original_root: Path, recovery_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _require_release_guard()
    if original_root != ORIGINAL_RUN_ROOT or recovery_root != RECOVERY_ROOT:
        raise RuntimeError("official recovery paths differ from frozen authority")
    if os.path.lexists(recovery_root):
        raise RuntimeError("official recovery root is not fresh")

    original_root_binding = _read_directory_binding(original_root, mode=0o700)
    release = _verify_release(RELEASE_ROOT)
    recovery_release = _verify_recovery_release_and_controller()
    controller, controller_raw = _read_regular(
        CONTROLLER_PATH, mode=0o555, nlink=1, capture=True,
    )
    python, _ = _read_regular(PYTHON_PATH, mode=0o755, nlink=1)
    _require_binding(controller, {"sha256": CONTROLLER_SHA256})
    _require_binding(python, {"sha256": PYTHON_SHA256})
    process_python = _verify_process_python()
    if any(
        process_python.get(key) != python.get(key)
        for key in ("path", "sha256", "size_bytes", "device", "inode")
    ):
        raise RuntimeError("recovery process/designated Python binding differs")
    authorities = []
    authority_raw = []
    for path, expected_sha in AUTHORITY_FILES:
        binding, raw = _read_regular(path, mode=0o444, nlink=1, capture=True)
        _require_binding(binding, {"sha256": expected_sha})
        if raw is None:
            raise RuntimeError("input authority receipt capture differs")
        authorities.append(binding)
        authority_raw.append((binding, raw))
    input_snapshot = _verify_input_snapshot(authority_raw)

    outer_stdout, stdout_raw = _read_regular(
        OUTER_STDOUT_PATH, mode=0o600, nlink=1, capture=True,
    )
    outer_stderr, stderr_raw = _read_regular(
        OUTER_STDERR_PATH, mode=0o600, nlink=1, capture=True,
    )
    _require_binding(outer_stdout, OUTER_STDOUT)
    _require_binding(outer_stderr, OUTER_STDERR)
    if stdout_raw != b"" or stderr_raw is None:
        raise RuntimeError("outer controller log payload differs")
    stderr_text = stderr_raw.decode("utf-8")
    required_errors = (
        f"signature={PARENT_STABLE_SIGNATURE_SHA256}",
        "inner/checkpoint captured semantic join differs",
        "StepId=143811.94",
        "ERROR: INNER_NO_GO_ALL_OOF0 final seal child failed",
    )
    if any(text not in stderr_text for text in required_errors):
        raise RuntimeError("outer controller failure evidence differs")

    rows_before, bindings_before = _scan_original(original_root)
    launch_plan = _verify_launch_plan(original_root, bindings_before)
    parsed_logs = []
    for fold in range(5):
        stdout_binding, raw = _read_regular(
            original_root / f"logs/train-fold{fold}.stdout",
            mode=0o444, nlink=1, capture=True,
        )
        if raw is None or stdout_binding != bindings_before[f"logs/train-fold{fold}.stdout"]:
            raise RuntimeError("train stdout scan/read binding differs")
        parsed_logs.append(_parse_train_stdout(raw, fold))
    accounting = _query_failed_step()

    v4g = _import_frozen_v4g(RELEASE_ROOT)
    recovery_parser_torch = {
        **_verify_recovery_parser_torch(v4g),
        **_verify_recovery_parser_torch_origins(v4g),
    }
    receipt_loader = v4g._load_inner_receipt_sealed
    folds = _verify_receipts(original_root, bindings_before, receipt_loader)

    # This is the exact historical bug.  Recovery accepts only the three-field
    # projection already pinned by runtime receipts, while modes/nlinks are
    # independently reverified by the exact26 scan.  It never rewrites either.
    if (
        controller_raw is None
        or b'artifact.get("physical_identity") == binding["physical_identity"]'
            not in controller_raw
    ):
        raise RuntimeError("old controller identity-schema bug source differs")

    rows_prewrite, bindings_prewrite = _scan_original(original_root)
    original_root_prewrite = _read_directory_binding(original_root, mode=0o700)
    if (
        rows_prewrite != rows_before or bindings_prewrite != bindings_before
        or original_root_prewrite != original_root_binding
    ):
        raise RuntimeError("original run changed during recovery validation")

    value = {
        "schema_version": SCHEMA,
        "authority": "burned_known_transform_development_scientific_no_go_only",
        "original_run_root": str(original_root),
        "recovery_root": str(recovery_root),
        "original_run_mutated_by_recovery": False,
        "original_run_postverified_unchanged": True,
        "original_controller_complete": False,
        "original_controller_exit_nonzero": True,
        "scientifically_verified_all_inner_no_go": True,
        "global_inner_barrier_created": False,
        "evaluate_fold_executed": False,
        "aggregate_executed": False,
        "all_fold_oof_semantic_tensor_read_count": 0,
        "all_fold_oof_semantic_tensor_materialized_count": 0,
        "recovery_ledger_reconstructed_from_burned_exact26": True,
        "burned_exact26_file_count": 26,
        "burned_exact26_manifest_sha256": EXACT26_MANIFEST_SHA256,
        "burned_exact26_manifest": rows_before,
        "burned_parent_stable_signature_sha256": PARENT_STABLE_SIGNATURE_SHA256,
        "original_run_root_binding": original_root_binding,
        "old_controller_identity_schema_bug": {
            "runtime_receipt_identity_keys": ["device", "inode", "size_bytes"],
            "phase_binding_additional_keys": [
                "mode_octal", "nlink", "mtime_ns", "ctime_ns",
            ],
            "all_ten_checkpoint_three_field_projections_exact": True,
            "all_ten_checkpoint_full_object_equal": False,
            "mode_and_nlink_verified_independently": True,
            "bug_affected_scientific_values": False,
            "bug_only_blocked_final_controller_seal": True,
        },
        "source_authority": {
            "authority_snapshot": AUTHORITY_SNAPSHOT,
            "release": release,
            "controller": controller,
            "python": python,
            "process_python": process_python,
            "input_receipts": authorities,
            "input_snapshot": input_snapshot,
            "runtime_sha256": RUNTIME_SHA256,
            "tests_sha256": TESTS_SHA256,
            "recovery_parser_torch": recovery_parser_torch,
            "recovery": recovery_release,
        },
        "launch_plan": {
            "binding": bindings_before["launch-plan.json"],
            "schema_version": launch_plan["schema_version"],
            "normal_tests_run": 36,
            "normal_tests_skipped": 0,
            "optimized_tests_run": 36,
            "optimized_tests_skipped": 0,
        },
        "original_controller_logs": {
            "stdout": outer_stdout, "stderr": outer_stderr,
        },
        "failed_seal_child_accounting": accounting,
        "folds": folds,
        "all_qualification_claims_false": True,
        "qualification_scope": {
            "known_exposed_development_gate": None,
            "known_exposed_development_gate_evaluated": False,
            "unseen_hostile_transform_gate": False,
            "unseen_hostile_transform_gate_evaluated": False,
            "latent_metric_qualified": False,
            "action_representation_qualified": False,
            "identity_disentanglement_qualified": False,
            "identity_preservation_qualified": False,
            "prior_qualified": False,
            "prior_generation_qualified": False,
            "generation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "full644_refit_authorized": False,
            "video_model_training_performed": False,
            "html_or_video_generated": False,
            "vae_necessary": None,
        },
    }

    return value, rows_before, bindings_before


def recover(original_root: Path, recovery_root: Path) -> dict[str, Any]:
    _require_release_guard()
    if original_root != ORIGINAL_RUN_ROOT or recovery_root != RECOVERY_ROOT:
        raise RuntimeError("official recovery paths differ from frozen authority")
    return _publish_attestation_create_only(recovery_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-root", required=True)
    parser.add_argument("--recovery-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if RELEASE_SEALED is not True:
        raise SystemExit("v4G recovery release is not sealed; intentional NO-GO")
    args = _parser().parse_args(argv)
    result = recover(Path(args.original_run_root), Path(args.recovery_root))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
