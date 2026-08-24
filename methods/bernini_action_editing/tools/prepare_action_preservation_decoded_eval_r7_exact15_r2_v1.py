#!/usr/bin/env python3
"""Create r7 exact15-r2 deployment inputs and print later-stage interfaces.

The commands in this helper are intentionally split at independent literal
SHA review boundaries.  ``phase-a`` only publishes the deployment request.
``phase-b`` only publishes the source/runtime spec after replaying a Phase-A
deployment receipt.  Interface commands print argv and never execute them.
No command launches Slurm, torchrun, inference, training, or a GPU process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "bernini-action-preservation-decoded-eval-deployment-request-v2"
SOURCE_RUNTIME_SCHEMA = "bernini-action-preservation-source-runtime-spec-v2"
SOURCE_PREPROCESSING_SCHEMA = (
    "bernini-action-preservation-decoded-eval-source-preprocessing-authority-v1"
)
RELEASE_GENERATION = "preservation-v2-decoded-eval-exact15-r2"
OBSOLETE_REASON = (
    "exact15-r2 is obsolete: source-preprocessing authority was not "
    "semantically projected end to end"
)
EVALUATION_ID = "apv2-r7-exact264-exact15-r2-2752c4ae"

REMOTE_BASE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
BUNDLE_ROOT = REMOTE_BASE / (
    "action_preservation_v2_decoded_eval_exact15_r2_r7_"
    "2752c4ae_207763b7_20260816"
)
ARTIFACT_ROOT = BUNDLE_ROOT / "exact15-r2-release"
CONTROLLER_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_deployment_controller_v1.py"
)
RUNTIME_SOURCE_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_verified_release_v1.py"
)
SOURCE_PREPROCESSING_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_r7_source_preprocessing_authority_v1.json"
)
INPUT_AUTHORITY_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_r7_exact15_r2_input_authority.json"
)
PREPARE_PATH = BUNDLE_ROOT / (
    "prepare_action_preservation_decoded_eval_r7_exact15_r2_v1.py"
)
WORK_ROOT = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_r7_"
    "decoded_eval_exact15_r2_2752c4ae"
)
MATERIALIZED_RELEASE_ROOT = WORK_ROOT / "materialized-release"
DEPLOYMENT_REQUEST_PATH = WORK_ROOT / "deployment-request.json"
CONTROLLER_AUTHORITY_PATH = WORK_ROOT / "controller-authority.json"
DEPLOYMENT_RECEIPT_PATH = WORK_ROOT / "deployment-receipt.json"
SOURCE_SPEC_PATH = WORK_ROOT / "source-runtime-spec.json"
SOURCE_SPEC_AUTHORITY_PATH = WORK_ROOT / "source-spec-authority.json"
EVALUATION_ROOT = WORK_ROOT / "decoded-eval"
BRIDGE_ROOT = WORK_ROOT / "bridge"
LAUNCH_ROOT = WORK_ROOT / "launch"
AGGREGATE_ROOT = WORK_ROOT / "aggregate"
BLINDING_KEY_PATH = WORK_ROOT / "blind-key.bin"

EXPERIMENT_ROOT = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_r7"
)
TRAINING_COMPLETE_PATH = EXPERIMENT_ROOT / "TRAINING_COMPLETE.json"
TRAINING_AUDIT_PATH = EXPERIMENT_ROOT / "logs/training-audit.json"
TRAINING_COMPLETE_SHA256 = (
    "2752c4aee78c833c55f7c66bf5bebf84d42f6babe00692dd6cc94b918842409b"
)
TRAINING_AUDIT_SHA256 = (
    "70b743eb566ba80406473b3dbabfcacffacd028c811aef34069cbbd3aa5c59c5"
)
SOURCE_REVISION = "54a2bafa2a09ddcd26add20c211ea9f055d339c3"
SOURCE_ARCHIVE_SHA256 = (
    "71357c8a4212fd985ffc4f73e8422ae412502756e63c014bc1c260c10c53273f"
)

SOURCE_MANIFEST_PATH = REMOTE_BASE / (
    "action_quotient_job140846_v4/source_only/manifest.json"
)
ADAPTER_RELEASE_MANIFEST_PATH = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_release_"
    "54a2bafa_nfssafe1/source.manifest.json"
)
MODEL_MANIFEST_PATH = REMOTE_BASE / (
    "bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/runtime/"
    "source_ea900d5/methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
BERNINI_ROOT = REMOTE_BASE / "motive_action_repr_auto/vendor/Bernini-2d2b4591"
VEOMNI_ROOT = REMOTE_BASE / (
    "bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
)
MODEL_ROOT = REMOTE_BASE.parent / (
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
INFERENCE_CONFIG_PATH = (
    BERNINI_ROOT / "configs/bernini_renderer_wan21_1p3b/config.json"
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

CONTROLLER_SHA256 = (
    "99003b6b942d286c44adda9a66a0c8f99d2fb7dfed72b7eb95134937c7386cf8"
)
RUNTIME_SOURCE_SHA256 = (
    "abbe1b0884e49677b4710418706ef2eca9f40ebc21b795b5781aaecad3ce6251"
)
ARCHIVE_SHA256 = (
    "12cef8d43edf3402c749149bf06317c91d4dc1ba941c0242b721058bc47b8a22"
)
MANIFEST_SHA256 = (
    "b3cb94306da704781028f9f94287175d1dd62ba03e083bebbd909e8698fecfc1"
)
MANIFEST_DIGEST = (
    "40b692681a1c532c29d08e7b90e95499da752731f88f8ad4de26207b53451898"
)
CONTENT_REVISION = "207763b76b590330631e4d0abf63865bdf44f4d6"
ENVELOPE_SHA256 = (
    "1edabae3b6b5e3130c1862b9f0cfc7b6c69d1f5613eb2654d70d91ae4b1eaca1"
)
ENVELOPE_DIGEST = (
    "f7005dd1aafb0b8e8290cfd90fd9e8ecb67ee293613ff8320dc1f286b41afb08"
)
SOURCE_PREPROCESSING_SHA256 = (
    "f0ee7196c00fb0dd0b4345707ec8a069ee2ba20a6f304b1982ef8d7945be15dd"
)
INPUT_AUTHORITY_SHA256 = (
    "cea2560b458869c22340dae8cd717cbfce3962cc0e742ba902fafe9efb5ba276"
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
SOURCE_MANIFEST_SHA256 = (
    "62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8"
)
ADAPTER_RELEASE_MANIFEST_SHA256 = (
    "ce97493465dc0d5b3733be25966f6d2ca909ac24931c4840daa4c73dc4c62198"
)
MODEL_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
INFERENCE_CONFIG_SHA256 = (
    "4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496"
)
INFER_SHA256 = (
    "dde5e3293e4fc833618c970eb51ba61fef4c66ef38dd1e67ab0e12b142f05e48"
)
DECODER_SHA256 = (
    "0b30ff6d2e4d17b20844abbeea5c26e51d376740cab092f905854279ad713fd1"
)
EXECUTOR_SHA256 = (
    "8915693b5816d7309e9f66f5a2b08975e579286c6df9e8ea410791e0ad3cce29"
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class R7DeploymentPreparationError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise R7DeploymentPreparationError("value is not finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R7DeploymentPreparationError(message)


def sha256(value: Any, *, label: str) -> str:
    require(type(value) is str and _SHA256_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def strict_json(raw: bytes, *, label: str, canonical: bool = True) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            require(key not in result, f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise R7DeploymentPreparationError(f"cannot decode {label}") from error
    require(type(value) is dict, f"{label} root differs")
    if canonical:
        require(raw == canonical_json_bytes(value) + b"\n",
                f"{label} serialization differs")
    return value


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        value.st_mode, value.st_nlink, value.st_rdev, value.st_size,
        getattr(value, "st_blocks", 0), value.st_mtime_ns, value.st_ctime_ns,
    )


def stable_file(
    path: Path, *, label: str, expected_sha256: str,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    require(path.is_absolute() and path.resolve(strict=True) == path,
            f"{label} path differs")
    require(hasattr(os, "O_NOFOLLOW"), f"{label} no-follow unavailable")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
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
    require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(named.st_mode)
        and before.st_nlink == 1
        and _identity(before) == _identity(middle) == _identity(after)
        and _identity(before) == _identity(named)
        and first == second and digest == expected_sha256
        and (expected_mode is None or stat.S_IMODE(before.st_mode) == expected_mode),
        f"{label} stable physical identity or bytes differ",
    )
    return first, {
        "path": str(path), "sha256": digest, "size": len(first),
        "mode": stat.S_IMODE(before.st_mode), "device": before.st_dev,
        "inode": before.st_ino, "uid": before.st_uid, "gid": before.st_gid,
        "nlink": before.st_nlink, "rdev": before.st_rdev,
        "blocks": getattr(before, "st_blocks", 0),
        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
    }


def pair(value: Mapping[str, Any]) -> dict[str, str]:
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(path.is_absolute() and path.parent.is_dir(), "output path differs")
    require(not os.path.lexists(path), "output path is not fresh")
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    require(hasattr(os, "O_NOFOLLOW"), "no-follow creation unavailable")
    descriptor = os.open(path, flags | os.O_NOFOLLOW, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            require(count > 0, "write made no progress")
            offset += count
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
        require(
            before.st_nlink == 1 and stat.S_IMODE(before.st_mode) == 0o444
            and _identity(before) == _identity(middle) == _identity(after)
            and _identity(before) == _identity(named)
            and first == raw == second,
            "create-only same-FD replay differs",
        )
    finally:
        os.close(descriptor)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _controller_namespace() -> dict[str, Any]:
    raw, _binding = stable_file(
        CONTROLLER_PATH, label="detached controller",
        expected_sha256=CONTROLLER_SHA256, expected_mode=0o444,
    )
    namespace: dict[str, Any] = {
        "__name__": "_apv2_r7_detached_controller",
        "__file__": str(CONTROLLER_PATH), "__package__": None,
        "__spec__": None, "__builtins__": __builtins__,
    }
    exec(compile(raw, str(CONTROLLER_PATH), "exec", dont_inherit=True), namespace)
    required = {
        "ROOT_CONTROLLER_BOOTSTRAP_SOURCE", "validate_request",
        "load_deployment_receipt", "controller_bootstrap_argv",
    }
    require(required.issubset(namespace), "detached controller API differs")
    return namespace


def controller_prefix(namespace: Mapping[str, Any]) -> list[str]:
    bootstrap = namespace["ROOT_CONTROLLER_BOOTSTRAP_SOURCE"]
    require(type(bootstrap) is str and bootstrap, "controller bootstrap differs")
    return [
        "/usr/bin/env", "-i", "HOME=/vast/users/guangyi.chen",
        "USER=guangyi.chen", "LOGNAME=guangyi.chen", "PATH=/usr/bin:/bin",
        str(ROOT_PYTHON_PATH), "-I", "-S", "-B", "-c", bootstrap,
        str(CONTROLLER_PATH), CONTROLLER_SHA256,
    ]


def _validate_bundle_and_static_inputs() -> dict[str, Any]:
    require(BUNDLE_ROOT.is_dir() and BUNDLE_ROOT.resolve(strict=True) == BUNDLE_ROOT,
            "deployment bundle differs")
    archive = stable_file(
        ARTIFACT_ROOT / "source.tar", label="r2 source archive",
        expected_sha256=ARCHIVE_SHA256, expected_mode=0o444,
    )[1]
    manifest = stable_file(
        ARTIFACT_ROOT / "source.manifest.json", label="r2 source manifest",
        expected_sha256=MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    envelope = stable_file(
        ARTIFACT_ROOT / "deployment-envelope.json", label="r2 envelope",
        expected_sha256=ENVELOPE_SHA256, expected_mode=0o444,
    )[1]
    runtime_source = stable_file(
        RUNTIME_SOURCE_PATH, label="detached r2 runtime",
        expected_sha256=RUNTIME_SOURCE_SHA256, expected_mode=0o444,
    )[1]
    controller = stable_file(
        CONTROLLER_PATH, label="detached controller",
        expected_sha256=CONTROLLER_SHA256, expected_mode=0o444,
    )[1]
    stable_file(
        SOURCE_PREPROCESSING_PATH, label="source preprocessing authority",
        expected_sha256=SOURCE_PREPROCESSING_SHA256, expected_mode=0o444,
    )
    stable_file(
        INPUT_AUTHORITY_PATH, label="r7 input authority",
        expected_sha256=INPUT_AUTHORITY_SHA256, expected_mode=0o444,
    )
    root_python = stable_file(
        ROOT_PYTHON_PATH, label="root Python",
        expected_sha256=ROOT_PYTHON_SHA256, expected_mode=0o755,
    )[1]
    frozen_python = stable_file(
        FROZEN_PYTHON_PATH, label="frozen Python",
        expected_sha256=FROZEN_PYTHON_SHA256, expected_mode=0o755,
    )[1]
    torchrun = stable_file(
        TORCHRUN_PATH, label="torchrun source",
        expected_sha256=TORCHRUN_SHA256, expected_mode=0o644,
    )[1]
    return {
        "archive": archive, "manifest": manifest, "envelope": envelope,
        "runtime_source": runtime_source, "controller": controller,
        "root_python": root_python, "frozen_python": frozen_python,
        "torchrun": torchrun,
    }


def build_phase_a_request() -> tuple[dict[str, Any], dict[str, Any]]:
    require(WORK_ROOT.is_dir() and WORK_ROOT.resolve(strict=True) == WORK_ROOT,
            "fresh work root must already exist")
    output_paths = (
        MATERIALIZED_RELEASE_ROOT, SOURCE_SPEC_PATH, SOURCE_SPEC_AUTHORITY_PATH,
        CONTROLLER_AUTHORITY_PATH, DEPLOYMENT_RECEIPT_PATH, DEPLOYMENT_REQUEST_PATH,
    )
    require(all(not os.path.lexists(path) for path in output_paths),
            "phase-A output path is not fresh")
    bindings = _validate_bundle_and_static_inputs()
    value: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "controller": pair(bindings["controller"]),
        "root_python": pair(bindings["root_python"]),
        "frozen_python": pair(bindings["frozen_python"]),
        "site_packages_path": str(SITE_PACKAGES_PATH),
        "torchrun": pair(bindings["torchrun"]),
        "release_root": str(MATERIALIZED_RELEASE_ROOT),
        "archive": pair(bindings["archive"]),
        "manifest": pair(bindings["manifest"]),
        "manifest_digest": MANIFEST_DIGEST,
        "content_revision": CONTENT_REVISION,
        "envelope": pair(bindings["envelope"]),
        "envelope_digest": ENVELOPE_DIGEST,
        "verified_runtime_source": pair(bindings["runtime_source"]),
        "source_runtime_spec_path": str(SOURCE_SPEC_PATH),
        "source_spec_authority_receipt_path": str(SOURCE_SPEC_AUTHORITY_PATH),
        "controller_authority_receipt_path": str(CONTROLLER_AUTHORITY_PATH),
        "deployment_receipt_path": str(DEPLOYMENT_RECEIPT_PATH),
        "automatic_retry": False, "network_allowed": False,
        "scientific_promotion_authorized": False,
    }
    value["request_digest"] = object_sha256(value)
    namespace = _controller_namespace()
    try:
        validated = namespace["validate_request"](value)
    except Exception as error:
        raise R7DeploymentPreparationError(str(error)) from error
    require(validated == value, "controller request projection differs")
    return value, namespace


def publish_phase_a_request() -> dict[str, Any]:
    value, namespace = build_phase_a_request()
    binding = write_create_only(DEPLOYMENT_REQUEST_PATH, value)
    argv = controller_prefix(namespace) + [
        "capture-authority", "--deployment-request", str(DEPLOYMENT_REQUEST_PATH),
        "--deployment-request-sha256", binding["sha256"],
    ]
    return {
        "status": "R7_EXACT15_R2_PHASE_A_REQUEST_PREPARED_NOT_EXECUTED",
        "deployment_request": binding, "request_digest": value["request_digest"],
        "controller_argv": argv,
        "controller_bootstrap_source_sha256": hashlib.sha256(
            namespace["ROOT_CONTROLLER_BOOTSTRAP_SOURCE"].encode("utf-8")
        ).hexdigest(),
        "remote_process_executed": False, "gpu_used": False,
    }


def _load_source_preprocessing() -> tuple[dict[str, Any], dict[str, Any]]:
    raw, binding = stable_file(
        SOURCE_PREPROCESSING_PATH, label="source preprocessing authority",
        expected_sha256=SOURCE_PREPROCESSING_SHA256, expected_mode=0o444,
    )
    value = strict_json(raw, label="source preprocessing authority")
    unsigned = dict(value)
    claimed = unsigned.pop("authority_digest", None)
    require(
        value.get("schema_version") == SOURCE_PREPROCESSING_SCHEMA
        and claimed == object_sha256(unsigned)
        and value.get("source_video_bytes_consumed_directly") is True
        and value.get("precomputed_transformed_source_artifact_used") is False
        and value.get("training_loss_read_or_used_for_selection") is False
        and value.get("source_order")
        == ["7b88a1ca1f804f41", "841b5e0080a1441d",
            "a35b590961d24694", "a66e6818e4144928"],
        "source preprocessing authority differs",
    )
    for source in value.get("sources", []):
        stable_file(
            Path(source["source_video_path"]), label=f"source video {source['iid']}",
            expected_sha256=sha256(source["source_video_sha256"], label="video SHA"),
        )
        stable_file(
            Path(source["source_receipt_path"]), label=f"source receipt {source['iid']}",
            expected_sha256=sha256(source["source_receipt_sha256"], label="receipt SHA"),
        )
    return value, binding


def build_phase_b_spec(
    *, deployment_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(not os.path.lexists(SOURCE_SPEC_PATH), "source spec path is not fresh")
    namespace = _controller_namespace()
    try:
        deployment, _runtime = namespace["load_deployment_receipt"](
            DEPLOYMENT_RECEIPT_PATH,
            expected_sha256=sha256(
                deployment_receipt_sha256, label="deployment receipt literal SHA"
            ),
        )
    except Exception as error:
        raise R7DeploymentPreparationError(str(error)) from error
    require(
        deployment["release_generation"] == RELEASE_GENERATION
        and deployment["source_runtime_spec_path"] == str(SOURCE_SPEC_PATH)
        and deployment["source_spec_authority_receipt_path"]
        == str(SOURCE_SPEC_AUTHORITY_PATH),
        "deployment receipt r7 path continuity differs",
    )
    preprocessing, preprocessing_file = _load_source_preprocessing()
    source_manifest = stable_file(
        SOURCE_MANIFEST_PATH, label="source manifest",
        expected_sha256=SOURCE_MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    adapter_manifest = stable_file(
        ADAPTER_RELEASE_MANIFEST_PATH, label="r7 adapter release manifest",
        expected_sha256=ADAPTER_RELEASE_MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    model_manifest = stable_file(
        MODEL_MANIFEST_PATH, label="model release manifest",
        expected_sha256=MODEL_MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    inference_config = stable_file(
        INFERENCE_CONFIG_PATH, label="inference config",
        expected_sha256=INFERENCE_CONFIG_SHA256, expected_mode=0o444,
    )[1]
    ffprobe = stable_file(
        FFPROBE_PATH, label="ffprobe",
        expected_sha256=FFPROBE_SHA256, expected_mode=0o755,
    )[1]
    release = deployment["release"]
    authority = deployment["controller_authority"]
    method_root = MATERIALIZED_RELEASE_ROOT / "methods/bernini_action_editing"
    infer = stable_file(
        method_root / "infer_lora.py", label="materialized infer_lora",
        expected_sha256=INFER_SHA256, expected_mode=0o444,
    )[1]
    decoder = stable_file(
        method_root / "action_preservation_decoded_eval_decoder_adapter_v1.py",
        label="materialized decoder", expected_sha256=DECODER_SHA256,
        expected_mode=0o555,
    )[1]
    pins = {
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "adapter_release_manifest_sha256": ADAPTER_RELEASE_MANIFEST_SHA256,
        "model_release_manifest_sha256": MODEL_MANIFEST_SHA256,
        "inference_source_sha256": INFER_SHA256,
        "inference_release_manifest_sha256": MANIFEST_SHA256,
        "inference_config_sha256": INFERENCE_CONFIG_SHA256,
        "source_preprocessing_sha256": SOURCE_PREPROCESSING_SHA256,
        "calibration_digest": None,
    }
    value: dict[str, Any] = {
        "schema_version": SOURCE_RUNTIME_SCHEMA,
        "pins": pins,
        "pin_files": {
            "source_manifest": pair(source_manifest),
            "adapter_release_manifest": pair(adapter_manifest),
            "model_release_manifest": pair(model_manifest),
            "inference_release_manifest": pair(release["manifest"]),
            "inference_config": pair(inference_config),
            "source_preprocessing": pair(preprocessing_file),
            "calibration": None,
        },
        "sources": preprocessing["sources"],
        "runtime": {
            "root_python": pair(deployment["root_python"]),
            "python": pair(deployment["frozen_python"]),
            "site_packages": deployment["site_packages"]["path"],
            "torchrun": pair(deployment["torchrun"]["source"]),
            "deployment_controller": pair(deployment["controller"]),
            "controller_authority": {
                "receipt": pair(authority["receipt"]),
                "authority_digest": authority["authority_digest"],
            },
            "infer_lora": pair(infer), "decoder_adapter": pair(decoder),
            "ffprobe": pair(ffprobe),
            "eval_release_root": release["release_root"]["path"],
            "eval_release_archive": pair(release["archive"]),
            "eval_release_envelope": pair(release["envelope"]),
            "eval_release_manifest_digest": release["manifest_digest"],
            "eval_release_content_revision": release["content_revision"],
            "eval_release_envelope_digest": release["envelope_digest"],
            "bernini_root": str(BERNINI_ROOT), "veomni_root": str(VEOMNI_ROOT),
            "model_checkpoint_root": str(MODEL_ROOT),
            "expected_bernini_commit": BERNINI_COMMIT,
            "expected_veomni_commit": VEOMNI_COMMIT,
            "expected_checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
            "method_source_revision": SOURCE_REVISION,
            "method_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "num_inference_steps": 40,
        },
    }
    value["spec_digest"] = object_sha256(value)
    return value, namespace


def publish_phase_b_spec(*, deployment_receipt_sha256: str) -> dict[str, Any]:
    value, namespace = build_phase_b_spec(
        deployment_receipt_sha256=deployment_receipt_sha256
    )
    binding = write_create_only(SOURCE_SPEC_PATH, value)
    argv = controller_prefix(namespace) + [
        "capture-source-spec-authority", "--deployment-receipt",
        str(DEPLOYMENT_RECEIPT_PATH), "--deployment-receipt-sha256",
        deployment_receipt_sha256, "--source-runtime-spec", str(SOURCE_SPEC_PATH),
        "--source-runtime-spec-sha256", binding["sha256"],
    ]
    return {
        "status": "R7_EXACT15_R2_PHASE_B_SPEC_PREPARED_NOT_EXECUTED",
        "source_runtime_spec": binding, "spec_digest": value["spec_digest"],
        "source_preprocessing_sha256": SOURCE_PREPROCESSING_SHA256,
        "controller_argv": argv, "remote_process_executed": False,
        "gpu_used": False,
    }


def _run_target_prefix(
    *, deployment_receipt_sha256: str, target: str, capture_receipt: Path,
    source_spec_authority_sha256: str | None = None,
) -> list[str]:
    namespace = _controller_namespace()
    value = controller_prefix(namespace) + [
        "run-target", "--deployment-receipt", str(DEPLOYMENT_RECEIPT_PATH),
        "--deployment-receipt-sha256",
        sha256(deployment_receipt_sha256, label="deployment receipt literal SHA"),
        "--target", target, "--capture-receipt", str(capture_receipt),
    ]
    if source_spec_authority_sha256 is not None:
        value += [
            "--source-spec-authority", str(SOURCE_SPEC_AUTHORITY_PATH),
            "--source-spec-authority-sha256",
            sha256(source_spec_authority_sha256,
                   label="source spec authority literal SHA"),
        ]
    return value + ["--"]


def bridge_interface(
    *, deployment_receipt_sha256: str, source_spec_authority_sha256: str,
    source_runtime_spec_sha256: str,
) -> dict[str, Any]:
    capture = WORK_ROOT / "bridge.runtime-capture.json"
    argv = _run_target_prefix(
        deployment_receipt_sha256=deployment_receipt_sha256,
        target="action_preservation_decoded_eval_bridge_v1.py",
        capture_receipt=capture,
        source_spec_authority_sha256=source_spec_authority_sha256,
    ) + [
        "--experiment-root", str(EXPERIMENT_ROOT),
        "--training-complete-sha256", TRAINING_COMPLETE_SHA256,
        "--source-runtime-spec", str(SOURCE_SPEC_PATH),
        "--source-runtime-spec-sha256",
        sha256(source_runtime_spec_sha256, label="source spec literal SHA"),
        "--evaluation-id", EVALUATION_ID,
        "--evaluation-root", str(EVALUATION_ROOT),
        "--bridge-root", str(BRIDGE_ROOT),
    ]
    return {"stage": "bridge", "argv": argv, "capture_receipt": str(capture)}


def launcher_interface(
    *, deployment_receipt_sha256: str, physical_bindings_sha256: str,
) -> dict[str, Any]:
    method_root = MATERIALIZED_RELEASE_ROOT / "methods/bernini_action_editing"
    physical = BRIDGE_ROOT / "physical_bindings.json"
    capture = WORK_ROOT / "launcher.runtime-capture.json"
    argv = _run_target_prefix(
        deployment_receipt_sha256=deployment_receipt_sha256,
        target="action_preservation_decoded_eval_launcher_v1.py",
        capture_receipt=capture,
    ) + [
        "--evaluation-root", str(EVALUATION_ROOT),
        "--launch-root", str(LAUNCH_ROOT),
        "--python", str(FROZEN_PYTHON_PATH),
        "--python-sha256", FROZEN_PYTHON_SHA256,
        "--executor", str(method_root / "action_preservation_decoded_eval_executor_v2.py"),
        "--executor-sha256", EXECUTOR_SHA256,
        "--decoder-adapter",
        str(method_root / "action_preservation_decoded_eval_decoder_adapter_v1.py"),
        "--decoder-adapter-sha256", DECODER_SHA256,
        "--ffprobe", str(FFPROBE_PATH), "--ffprobe-sha256", FFPROBE_SHA256,
        "--physical-bindings", str(physical),
        "--physical-bindings-sha256",
        sha256(physical_bindings_sha256, label="physical bindings literal SHA"),
    ]
    return {"stage": "launcher", "argv": argv, "capture_receipt": str(capture)}


def aggregate_interface(
    *, deployment_receipt_sha256: str, physical_bindings_sha256: str,
) -> dict[str, Any]:
    physical = BRIDGE_ROOT / "physical_bindings.json"
    capture = AGGREGATE_ROOT.with_name(AGGREGATE_ROOT.name + ".runtime-capture.json")
    argv = _run_target_prefix(
        deployment_receipt_sha256=deployment_receipt_sha256,
        target="action_preservation_decoded_eval_aggregate_v2.py",
        capture_receipt=capture,
    ) + [
        "--evaluation-root", str(EVALUATION_ROOT),
        "--physical-bindings", str(physical),
        "--physical-bindings-sha256",
        sha256(physical_bindings_sha256, label="physical bindings literal SHA"),
        "--blinding-key-file", str(BLINDING_KEY_PATH),
        "--aggregate-root", str(AGGREGATE_ROOT),
    ]
    return {"stage": "aggregate", "argv": argv, "capture_receipt": str(capture)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("phase-a")
    phase_b = commands.add_parser("phase-b")
    phase_b.add_argument("--deployment-receipt-sha256", required=True)
    bridge = commands.add_parser("bridge-interface")
    bridge.add_argument("--deployment-receipt-sha256", required=True)
    bridge.add_argument("--source-spec-authority-sha256", required=True)
    bridge.add_argument("--source-runtime-spec-sha256", required=True)
    launcher = commands.add_parser("launcher-interface")
    launcher.add_argument("--deployment-receipt-sha256", required=True)
    launcher.add_argument("--physical-bindings-sha256", required=True)
    aggregate = commands.add_parser("aggregate-interface")
    aggregate.add_argument("--deployment-receipt-sha256", required=True)
    aggregate.add_argument("--physical-bindings-sha256", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    raise R7DeploymentPreparationError(OBSOLETE_REASON)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_ROOT", "CONTROLLER_PATH", "DEPLOYMENT_RECEIPT_PATH",
    "DEPLOYMENT_REQUEST_PATH", "EVALUATION_ROOT", "MATERIALIZED_RELEASE_ROOT",
    "R7DeploymentPreparationError", "SOURCE_PREPROCESSING_PATH",
    "SOURCE_SPEC_PATH", "WORK_ROOT", "aggregate_interface",
    "bridge_interface", "build_phase_a_request", "build_phase_b_spec",
    "canonical_json_bytes", "launcher_interface", "object_sha256",
    "publish_phase_a_request", "publish_phase_b_spec", "write_create_only",
]
