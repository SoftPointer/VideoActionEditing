#!/usr/bin/env python3
"""Materialize the fresh r5g Shared8 full16 package without Slurm.

The production package has a 17-file local source closure.  Independently,
the r5f launcher authenticates 16 execution identities: eight package-local
files, seven external runtime files, and the plan.  The two sets therefore
overlap in eight files; their counts are not a subset relation.  The frozen
r5c launcher is one of the other package-local sources because the r5f
launcher exact-loads it.  CPU probes live under ``diagnostics`` and are not
added to either production identity set.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


BASE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments"
)
TARGETS = {
    ("143812", "auh7-1b-gpu-293"): BASE
    / "bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_"
      "full16_847b91a2_c91de7eb_d70eac5c_r1",
}
PRODUCTION_RANK_CACHE_ROOT = Path(
    "/tmp/bernini-full644-r5g-job143812-node293-full16-r1-rank-cache"
)
VACE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
VACE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
FFMPEG = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
FFPROBE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/"
    "runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
MODEL_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
BERNINI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591"
)
VEOMNI_ROOT = BERNINI_ROOT.parent / "VeOmni-f90b3dc6"
CHECKPOINT_MANIFEST = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_full644_exploratory_r64_job141620_v5/"
    "runs/full644-r64-reference-dpo-preservation-one-pass-v5/"
    "checkpoint-00000644/checkpoint_manifest.json"
)
TORCH_ROOT = VACE_PYTHON.parent.parent / "lib/python3.12/site-packages/torch"

CHECKPOINT_MANIFEST_SHA256 = (
    "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2"
)
FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
METHOD_REVISION = "ce4cffc1e8a144448c92252d9fb63087f03bbd8c"
METHOD_ARCHIVE_SHA256 = (
    "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828"
)
CAMPAIGN = "full16-production"
SELECTED_TASK_IDS = tuple(
    f"shared8-{index:02d}-{arm}"
    for index in range(8)
    for arm in ("base", "full644")
)

RELEASE_FILES = {
    "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json":
        "953933f1161b6d62826d388ba5ed42e42792fbf5f2bdeea199c1eb13cd251b4a",
    "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl":
        "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701",
    "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py":
        "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256":
        "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py":
        "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py":
        "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py":
        "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py":
        "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
    "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py":
        "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
    "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5.py":
        "cb201398940d59393fa58471dc2c3f9fdf001c7e881ec891ce892bb460cf01ba",
    "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5f.py":
        "d70eac5c0ee5fbcbfa84bc3a711fc2e836fa8cc0331555502d2b9b832e7c6b4e",
    "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py":
        "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "methods/bernini_action_editing/infer_lora.py":
        "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "methods/bernini_action_editing/self_generated_action_preservation_v2.py":
        "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py":
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "methods/bernini_action_editing/tools/materialize_vae.py":
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "methods/bernini_action_editing/train_lora.py":
        "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85",
}

ROOT_BOOTSTRAP_PROBE = (
    "methods/bernini_action_editing/"
    "full644_exploratory_matched_r5g_root_bootstrap_probe_runner_v1.py"
)
STATIC_NOMODEL_PROBE = (
    "methods/bernini_action_editing/"
    "full644_exploratory_matched_r5g_static_nomodel_probe_v1.py"
)
CPU_CONSUMPTION_PROBE = (
    "methods/bernini_action_editing/"
    "full644_exploratory_matched_r5f_cpu_consumption_probe_v1.py"
)
DIAGNOSTIC_SOURCE_PINS = {
    ROOT_BOOTSTRAP_PROBE:
        "9dd6c07e47fcb14ea50e7d1ed7aa4b5adb2e17a6320f4691aa9c8780d7791c32",
    STATIC_NOMODEL_PROBE:
        "e3a603ad1d94f4d53c7a26fcb29f9313592c571b4483431bdd04dfae5ba68d12",
    CPU_CONSUMPTION_PROBE:
        "fd64fafc9580c8f25c88d79ca603a0dbf192ea98f77403e82f14d4e17c6905f6",
}


class R5FMaterializationError(RuntimeError):
    """The fresh r5f package materialization contract differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "nlink": info.st_nlink,
        "rdev": info.st_rdev,
        "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def stable_file(
    path: Path, expected_sha256: str | None = None, expected_mode: int | None = None
) -> bytes:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
    ):
        raise R5FMaterializationError(f"noncanonical stable path: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not raw
        or len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(before) != identity(named)
        or (expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256)
        or (expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode)
    ):
        raise R5FMaterializationError(f"stable file differs: {path}")
    return raw


def create_file(path: Path, raw: bytes, mode: int) -> None:
    if not raw or path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise R5FMaterializationError(f"fresh file target differs: {path}")
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise R5FMaterializationError("fresh file write made no progress")
            offset += count
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        named = path.lstat()
        replay = os.pread(descriptor, len(raw), 0)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0
            or before.st_nlink != 1
            or identity(before) != identity(named)
            or replay != raw
        ):
            raise R5FMaterializationError("fresh file staging replay differs")
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def load_module(name: str, path: Path, expected_sha256: str) -> types.ModuleType:
    raw = stable_file(path, expected_sha256, 0o444)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = None
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    module.__builtins__ = __builtins__
    sys.modules[name] = module
    try:
        exec(
            compile(raw.decode("utf-8", "strict"), str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def mkdir_fresh(path: Path, mode: int = 0o700) -> None:
    os.mkdir(path, mode)
    os.chmod(path, mode)


def ensure_ready_pins() -> None:
    for relative, digest in DIAGNOSTIC_SOURCE_PINS.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise R5FMaterializationError(f"diagnostic source pin is not frozen: {relative}")


def launch_input(root: Path, job_id: str, node: str, plan: Path) -> dict[str, Any]:
    method_root = root / "release/methods/bernini_action_editing"
    return {
        "schema_version": "full644-exploratory-matched-root-launch-input-auh-r5f",
        "entry_mode": "trusted_stdin",
        "runner": str(method_root / "full644_exploratory_matched_runner_auh_r5.py"),
        "bridge": str(method_root / "full644_exploratory_matched_torchrun_fd_bridge_v2.py"),
        "adapter": str(method_root / "full644_exploratory_matched_infer_adapter_auh_r5f.py"),
        "base_adapter": str(method_root / "full644_exploratory_matched_infer_adapter_v2.py"),
        "eval_v1": str(method_root / "full644_exploratory_matched_eval_v1.py"),
        "eval_v2": str(method_root / "full644_exploratory_matched_eval_v2.py"),
        "model_authority": str(method_root / "action_preservation_decoded_eval_model_authority_v2.py"),
        "python": str(VACE_PYTHON),
        "ffmpeg": str(FFMPEG),
        "torchrun_source": str(TORCH_ROOT / "distributed/run.py"),
        "torchrun_handler_source": str(
            TORCH_ROOT / "distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py"
        ),
        "torch_local_agent_source": str(
            TORCH_ROOT / "distributed/elastic/agent/server/local_elastic_agent.py"
        ),
        "torch_dynamic_rendezvous_source": str(
            TORCH_ROOT / "distributed/elastic/rendezvous/dynamic_rendezvous.py"
        ),
        "torch_multiprocessing_api_source": str(
            TORCH_ROOT / "distributed/elastic/multiprocessing/api.py"
        ),
        "plan": str(plan),
        "output_report": str(root / "final/full16_report_auh_r5.json"),
        "runner_attestation": str(root / "final/full16_runner_attestation_auh_r5.json"),
        "model_root": str(MODEL_ROOT),
        "model_manifest": str(method_root / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"),
        "bernini_root": str(BERNINI_ROOT),
        "veomni_root": str(VEOMNI_ROOT),
        "authority_root": str(root / "runtime/model-authority"),
        "rank_cache_root": str(PRODUCTION_RANK_CACHE_ROOT),
        "holder_job_id": job_id,
        "expected_node": node,
        "campaign_mode": CAMPAIGN,
    }


CAPTURED_PROBE_BOOTSTRAP = r'''import hashlib,json,os,stat,sys,types
if len(sys.argv)<17: raise RuntimeError("captured probe argv differs")
pyfd_raw,srcfd_raw,python_path,python_sha,source_path,source_sha=sys.argv[1:7]
slurm_values=sys.argv[7:16]; probe_argv=sys.argv[16:]
try: pyfd=int(pyfd_raw); srcfd=int(srcfd_raw)
except ValueError as error: raise RuntimeError("captured probe FD differs") from error
if pyfd<3 or srcfd<3 or pyfd==srcfd or not os.get_inheritable(pyfd) or not os.get_inheritable(srcfd): raise RuntimeError("captured probe entry FD differs")
def ident(value): return (value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns)
def read(fd,size):
 out=[]; offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  out.append(block); offset+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("captured probe read differs")
 return raw
source_raw=None
for fd,path,pin,executable in ((pyfd,python_path,python_sha,True),(srcfd,source_path,source_sha,False)):
 before=os.fstat(fd); raw=read(fd,before.st_size); after=os.fstat(fd); named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or ident(before)!=ident(after) or ident(before)!=ident(named) or hashlib.sha256(raw).hexdigest()!=pin or (not executable and stat.S_IMODE(before.st_mode)!=0o444) or (executable and not before.st_mode&0o111): raise RuntimeError("captured probe authority differs")
 if executable and ident(before)!=ident(os.stat("/proc/self/exe")): raise RuntimeError("captured Python process differs")
 if not executable: source_raw=raw
 os.set_inheritable(fd,False)
if type(source_raw) is not bytes: raise RuntimeError("captured probe source differs")
names=("SLURM_JOB_ID","SLURM_STEP_ID","SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_NODE","SLURM_STEP_GPUS","SLURM_NNODES","SLURM_STEP_NUM_NODES","SLURM_JOB_NODELIST","SLURM_STEP_NODELIST")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"): raise RuntimeError("captured probe Python environment differs")
os.environ.clear(); os.environ.update(dict(zip(names,slurm_values,strict=True)))
module=types.ModuleType("__main__"); module.__file__=source_path; module.__package__=None; module.__loader__=None; module.__spec__=None; module.__cached__=None; module.__builtins__=__builtins__; sys.modules["__main__"]=module; sys.argv=[source_path,*probe_argv]
exec(compile(source_raw.decode("utf-8","strict"),source_path,"exec",dont_inherit=True),module.__dict__)'''


def shell_quote(value: str) -> str:
    if "\x00" in value:
        raise R5FMaterializationError("shell literal contains NUL")
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_static_payload(
    *, root: Path, job_id: str, node: str, source: Path, source_sha256: str,
    plan: Path, plan_sha256: str, launch_input_path: Path,
    launch_input_sha256: str, launch_receipt_path: Path,
    launch_receipt_sha256: str,
) -> bytes:
    receipt = root / "evidence/static_nomodel_probe_receipt_r5d.json"
    probe_args = [
        "--root", str(root), "--plan", str(plan), "--plan-sha256", plan_sha256,
        "--launch-input", str(launch_input_path),
        "--launch-input-sha256", launch_input_sha256,
        "--launch-receipt", str(launch_receipt_path),
        "--launch-receipt-sha256", launch_receipt_sha256,
        "--probe-sha256", source_sha256, "--receipt", str(receipt),
        "--holder-job-id", job_id, "--expected-node", node,
    ]
    lines = [
        "#!/bin/bash -p", "set -euo pipefail", "umask 077",
        '[[ "$-" == *p* ]] || exit 91',
        '[[ "$0" == "bash" || "$0" == "/bin/bash" || "$0" == "-bash" ]] || exit 92',
        '[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || exit 93',
        f'[[ "${{SLURM_JOB_ID-}}" == {shell_quote(job_id)} ]] || exit 94',
        '[[ "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ && "${SLURM_GPUS_ON_NODE-}" == "8" && "${SLURM_GPUS_PER_NODE-}" == "8" ]] || exit 95',
        '[[ "${SLURM_STEP_GPUS-}" == "0,1,2,3,4,5,6,7" && "${SLURM_NNODES-}" == "1" && "${SLURM_STEP_NUM_NODES-}" == "1" ]] || exit 96',
        f'[[ "${{SLURM_JOB_NODELIST-}}" == {shell_quote(node)} && "${{SLURM_STEP_NODELIST-}}" == {shell_quote(node)} ]] || exit 97',
        '[[ -z "${SLURM_JOB_GPUS+x}" && -z "${SLURM_JOB_NUM_NODES+x}" ]] || exit 98',
        "if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi",
        f"readonly R5F_PYTHON={shell_quote(str(VACE_PYTHON))}",
        f"readonly R5F_SOURCE={shell_quote(str(source))}",
        'exec {R5F_PYTHON_FD}<"$R5F_PYTHON"',
        'exec {R5F_SOURCE_FD}<"$R5F_SOURCE"',
        '[[ "$R5F_PYTHON_FD" =~ ^[0-9]+$ && "$R5F_SOURCE_FD" =~ ^[0-9]+$ ]] || exit 99',
        "exec -c \"/proc/self/fd/$R5F_PYTHON_FD\" -I -S -B -c "
        + shell_quote(CAPTURED_PROBE_BOOTSTRAP)
        + ' "$R5F_PYTHON_FD" "$R5F_SOURCE_FD" '
        + shell_quote(str(VACE_PYTHON)) + " " + shell_quote(VACE_PYTHON_SHA256)
        + " " + shell_quote(str(source)) + " " + shell_quote(source_sha256)
        + ' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_GPUS_ON_NODE" "$SLURM_GPUS_PER_NODE" "$SLURM_STEP_GPUS" "$SLURM_NNODES" "$SLURM_STEP_NUM_NODES" "$SLURM_JOB_NODELIST" "$SLURM_STEP_NODELIST" '
        + " ".join(shell_quote(value) for value in probe_args),
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_directories(root: Path) -> None:
    for relative in (
        "release/methods/action_editing_baselines/manifests",
        "release/methods/bernini_action_editing/audits",
        "release/methods/bernini_action_editing/tools",
        "plan", "launch", "diagnostics", "evidence", "outputs/media",
        "final", "logs", "runtime",
    ):
        current = root
        for part in Path(relative).parts:
            current /= part
            if not current.exists():
                mkdir_fresh(current)


def _build_plan(root: Path, release_dir: Path) -> tuple[Path, bytes, dict[str, Any]]:
    v1_path = release_dir / "full644_exploratory_matched_eval_v1.py"
    v2_path = release_dir / "full644_exploratory_matched_eval_v2.py"
    load_module("full644_exploratory_matched_eval_v1", v1_path, RELEASE_FILES[str(v1_path.relative_to(root / "release"))])
    v2 = load_module("full644_exploratory_matched_eval_v2", v2_path, RELEASE_FILES[str(v2_path.relative_to(root / "release"))])
    plan_path = root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = v2.main([
            "build-plan",
            "--input-manifest", str(root / "release/methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl"),
            "--exposure-audit", str(root / "release/methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json"),
            "--checkpoint-manifest", str(CHECKPOINT_MANIFEST),
            "--checkpoint-manifest-sha256", CHECKPOINT_MANIFEST_SHA256,
            "--infer-lora-source", str(release_dir / "infer_lora.py"),
            "--infer-lora-source-sha256", RELEASE_FILES["methods/bernini_action_editing/infer_lora.py"],
            "--method-source-revision", METHOD_REVISION,
            "--method-source-archive-sha256", METHOD_ARCHIVE_SHA256,
            "--ffprobe", str(FFPROBE), "--ffprobe-sha256", FFPROBE_SHA256,
            "--output-root", str(root / "outputs/media"),
            "--output-plan", str(plan_path),
        ])
    if result != 0:
        raise R5FMaterializationError("v2 compatibility builder failed")
    raw = stable_file(plan_path, expected_mode=0o444)
    digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    if (
        stdout.getvalue() != digest + "\n"
        or raw != canonical_json_bytes(value) + b"\n"
        or v2.load_plan(plan_path, digest) != value
        or value.get("task_count") != 16
        or value.get("pair_count") != 8
        or [row.get("task_id") for row in value.get("tasks", [])]
        != [f"shared8-{index:02d}-{arm}" for index in range(8) for arm in ("base", "full644")]
    ):
        raise R5FMaterializationError("v2 plan replay differs")
    return plan_path, raw, value


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_ready_pins()
    key = (args.job_id, args.node)
    if key not in TARGETS:
        raise R5FMaterializationError("unsupported r5f binding target")
    if os.geteuid() != 2012 or os.getegid() != 2000:
        raise R5FMaterializationError("materializer owner authority differs")
    source_root = Path(args.source_root)
    if not source_root.is_absolute() or os.path.normpath(str(source_root)) != str(source_root):
        raise R5FMaterializationError("source root path differs")
    root = TARGETS[key]
    if root.exists() or root.is_symlink():
        raise R5FMaterializationError(f"fresh r5f root exists: {root}")
    mkdir_fresh(root)
    _make_directories(root)
    release_rows: dict[str, dict[str, Any]] = {}
    for relative, digest in RELEASE_FILES.items():
        raw = stable_file(source_root / relative, digest)
        target = root / "release" / relative
        create_file(target, raw, 0o444)
        release_rows[relative] = {"sha256": digest, "size": len(raw)}
    release_dir = root / "release/methods/bernini_action_editing"
    plan_path, plan_raw, plan = _build_plan(root, release_dir)
    plan_sha256 = hashlib.sha256(plan_raw).hexdigest()

    input_value = launch_input(root, args.job_id, args.node, plan_path)
    input_raw = canonical_json_bytes(input_value) + b"\n"
    input_path = root / "launch/root_launch_input_auh_r5d.json"
    create_file(input_path, input_raw, 0o444)
    launcher = load_module(
        "full644_exploratory_matched_spooled_launcher_auh_r5f_materializer",
        release_dir / "full644_exploratory_matched_spooled_launcher_auh_r5f.py",
        RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5f.py"],
    )
    payload_path = root / "launch/root_launch_payload_auh_r5d.sh"
    receipt_path = root / "launch/root_launch_receipt_auh_r5d.json"
    production_receipt = launcher.materialize(str(input_path), str(payload_path), str(receipt_path))
    receipt_raw = stable_file(receipt_path, expected_mode=0o400)
    if (
        json.loads(receipt_raw) != production_receipt
        or production_receipt.get("release", {}).get("selected_task_ids") != list(SELECTED_TASK_IDS)
        or len(production_receipt.get("release", {}).get("identities", {})) != 16
    ):
        raise R5FMaterializationError("r5f production receipt replay differs")
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()

    diagnostic_rows: dict[str, tuple[Path, bytes]] = {}
    for relative, digest in DIAGNOSTIC_SOURCE_PINS.items():
        raw = stable_file(source_root / relative, digest)
        target = root / "diagnostics" / Path(relative).name
        create_file(target, raw, 0o444)
        diagnostic_rows[relative] = (target, raw)

    bootstrap_runner, bootstrap_runner_raw = diagnostic_rows[ROOT_BOOTSTRAP_PROBE]
    bootstrap_input = dict(input_value)
    bootstrap_input.update({
        "runner": str(bootstrap_runner),
        "output_report": str(root / "evidence/root_bootstrap_cpu_probe_receipt_r5d.json"),
        "runner_attestation": str(root / "diagnostics/unused-root-bootstrap-attestation.json"),
        "authority_root": str(root / "diagnostics/unused-root-bootstrap-authority"),
        "rank_cache_root": str(root / "diagnostics/unused-root-bootstrap-rank-cache"),
    })
    bootstrap_input_raw = canonical_json_bytes(bootstrap_input) + b"\n"
    bootstrap_input_path = root / "diagnostics/root_bootstrap_probe_input_r5d.json"
    create_file(bootstrap_input_path, bootstrap_input_raw, 0o444)
    original_runner_pin = launcher.EXPECTED_STATIC_SHA256["runner"]
    bootstrap_runner_sha = hashlib.sha256(bootstrap_runner_raw).hexdigest()
    launcher.EXPECTED_STATIC_SHA256["runner"] = bootstrap_runner_sha
    try:
        bootstrap_payload = root / "diagnostics/root_bootstrap_probe_payload_r5d.sh"
        bootstrap_receipt_path = root / "diagnostics/root_bootstrap_probe_materialization_receipt_r5d.json"
        bootstrap_receipt = launcher.materialize(
            str(bootstrap_input_path), str(bootstrap_payload), str(bootstrap_receipt_path)
        )
    finally:
        launcher.EXPECTED_STATIC_SHA256["runner"] = original_runner_pin
    bootstrap_receipt_raw = stable_file(bootstrap_receipt_path, expected_mode=0o400)
    if json.loads(bootstrap_receipt_raw) != bootstrap_receipt:
        raise R5FMaterializationError("root bootstrap materialization replay differs")

    static_source, static_source_raw = diagnostic_rows[STATIC_NOMODEL_PROBE]
    static_payload_raw = build_static_payload(
        root=root, job_id=args.job_id, node=args.node,
        source=static_source, source_sha256=hashlib.sha256(static_source_raw).hexdigest(),
        plan=plan_path, plan_sha256=plan_sha256,
        launch_input_path=input_path, launch_input_sha256=hashlib.sha256(input_raw).hexdigest(),
        launch_receipt_path=receipt_path, launch_receipt_sha256=receipt_sha256,
    )
    static_payload = root / "diagnostics/static_nomodel_probe_payload_r5d.sh"
    create_file(static_payload, static_payload_raw, 0o444)

    for directory in sorted((root / "release").rglob("*"), reverse=True):
        if directory.is_dir():
            os.chmod(directory, 0o555)
    for directory in (root / "release", root / "plan", root / "launch"):
        os.chmod(directory, 0o555)
    for relative in ("diagnostics", "evidence", "outputs", "outputs/media", "final", "logs", "runtime"):
        os.chmod(root / relative, 0o755)
    os.chmod(root, 0o755)

    expected_modes: dict[str, int] = {
        **{f"release/{relative}": 0o444 for relative in RELEASE_FILES},
        "plan/full644_exploratory_matched_plan_auh_r5d.json": 0o444,
        "launch/root_launch_input_auh_r5d.json": 0o444,
        "launch/root_launch_payload_auh_r5d.sh": 0o444,
        "launch/root_launch_receipt_auh_r5d.json": 0o400,
        f"diagnostics/{Path(ROOT_BOOTSTRAP_PROBE).name}": 0o444,
        f"diagnostics/{Path(STATIC_NOMODEL_PROBE).name}": 0o444,
        f"diagnostics/{Path(CPU_CONSUMPTION_PROBE).name}": 0o444,
        "diagnostics/root_bootstrap_probe_input_r5d.json": 0o444,
        "diagnostics/root_bootstrap_probe_payload_r5d.sh": 0o444,
        "diagnostics/root_bootstrap_probe_materialization_receipt_r5d.json": 0o400,
        "diagnostics/static_nomodel_probe_payload_r5d.sh": 0o444,
    }
    expected_sha256 = {
        **{f"release/{relative}": digest for relative, digest in RELEASE_FILES.items()},
        "plan/full644_exploratory_matched_plan_auh_r5d.json": plan_sha256,
        "launch/root_launch_input_auh_r5d.json": hashlib.sha256(input_raw).hexdigest(),
        "launch/root_launch_payload_auh_r5d.sh": production_receipt["payload_sha256"],
        "launch/root_launch_receipt_auh_r5d.json": receipt_sha256,
        f"diagnostics/{Path(ROOT_BOOTSTRAP_PROBE).name}": DIAGNOSTIC_SOURCE_PINS[ROOT_BOOTSTRAP_PROBE],
        f"diagnostics/{Path(STATIC_NOMODEL_PROBE).name}": DIAGNOSTIC_SOURCE_PINS[STATIC_NOMODEL_PROBE],
        f"diagnostics/{Path(CPU_CONSUMPTION_PROBE).name}": DIAGNOSTIC_SOURCE_PINS[CPU_CONSUMPTION_PROBE],
        "diagnostics/root_bootstrap_probe_input_r5d.json": hashlib.sha256(bootstrap_input_raw).hexdigest(),
        "diagnostics/root_bootstrap_probe_payload_r5d.sh": bootstrap_receipt["payload_sha256"],
        "diagnostics/root_bootstrap_probe_materialization_receipt_r5d.json": hashlib.sha256(bootstrap_receipt_raw).hexdigest(),
        "diagnostics/static_nomodel_probe_payload_r5d.sh": hashlib.sha256(static_payload_raw).hexdigest(),
    }
    if set(expected_modes) != set(expected_sha256):
        raise R5FMaterializationError("package expected file closure differs")
    all_entries = list(root.rglob("*"))
    if any(
        not (path.is_file() or path.is_dir() or path.is_symlink())
        for path in all_entries
    ):
        raise R5FMaterializationError("package contains a special filesystem entry")
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != set(expected_modes):
        raise R5FMaterializationError("package physical file closure differs")
    artifacts: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected_modes):
        path = root / relative
        raw = stable_file(path, expected_sha256[relative], expected_modes[relative])
        info = path.lstat()
        if info.st_uid != 2012 or info.st_gid != 2000:
            raise R5FMaterializationError(f"package artifact owner differs: {relative}")
        artifacts[relative] = {
            "sha256": expected_sha256[relative], "size": len(raw),
            "mode": expected_modes[relative], "nlink": info.st_nlink,
        }
    expected_directories = {
        ".", "release", "release/methods", "release/methods/action_editing_baselines",
        "release/methods/action_editing_baselines/manifests",
        "release/methods/bernini_action_editing",
        "release/methods/bernini_action_editing/audits",
        "release/methods/bernini_action_editing/tools", "plan", "launch",
        "diagnostics", "evidence", "outputs", "outputs/media", "final", "logs", "runtime",
    }
    actual_directories = {"."} | {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise R5FMaterializationError("package directory closure differs")
    directories: dict[str, dict[str, int]] = {}
    for relative in sorted(expected_directories):
        path = root if relative == "." else root / relative
        info = path.lstat()
        expected_mode = 0o555 if relative == "release" or relative.startswith("release/") or relative in {"plan", "launch"} else 0o755
        if (
            not stat.S_ISDIR(info.st_mode) or info.st_uid != 2012 or info.st_gid != 2000
            or stat.S_IMODE(info.st_mode) != expected_mode
        ):
            raise R5FMaterializationError(f"package directory differs: {relative}")
        directories[relative] = {"mode": expected_mode, "nlink": info.st_nlink}
    for relative in ("evidence", "outputs/media", "final", "runtime"):
        if list((root / relative).iterdir()):
            raise R5FMaterializationError(f"fresh result root is not empty: {relative}")
    if b"exact28" in plan_raw.lower() or b"auh_r5b" in plan_raw.lower() or b"auh_r5c" in plan_raw.lower():
        raise R5FMaterializationError("burned plan family entered r5f package")

    report: dict[str, Any] = {
        "schema_version": "full644-exploratory-matched-r5g-full16-materialization-v1",
        "status": "MATERIALIZED_NOT_SUBMITTED",
        "root": str(root), "holder_job_id": args.job_id, "expected_node": args.node,
        "campaign_mode": CAMPAIGN, "selected_task_ids": list(SELECTED_TASK_IDS),
        "unselected_task_count": 0,
        "physical_release_file_count": len(RELEASE_FILES),
        "production_identity_count": len(production_receipt["release"]["identities"]),
        "production_identity_decomposition": {
            "package_local": 8, "external_runtime": 7, "plan": 1,
            "package_local_intersection_with_physical_release": 8,
        },
        "plan": {"path": str(plan_path), "sha256": plan_sha256, "plan_digest": plan["plan_digest"], "size": len(plan_raw)},
        "production_launch": {
            "input": str(input_path), "input_sha256": hashlib.sha256(input_raw).hexdigest(),
            "payload": str(payload_path), "payload_sha256": production_receipt["payload_sha256"],
            "payload_size": production_receipt["payload_size"], "receipt": str(receipt_path),
            "receipt_sha256": receipt_sha256, "receipt_digest": production_receipt["receipt_digest"],
            "release_digest": production_receipt["release_digest"],
        },
        "root_bootstrap_diagnostic": {
            "runner": str(bootstrap_runner), "runner_sha256": bootstrap_runner_sha,
            "input": str(bootstrap_input_path), "input_sha256": hashlib.sha256(bootstrap_input_raw).hexdigest(),
            "payload": str(bootstrap_payload), "payload_sha256": bootstrap_receipt["payload_sha256"],
            "payload_size": bootstrap_receipt["payload_size"], "receipt": str(bootstrap_receipt_path),
            "receipt_sha256": hashlib.sha256(bootstrap_receipt_raw).hexdigest(),
            "receipt_digest": bootstrap_receipt["receipt_digest"],
        },
        "static_nomodel_diagnostic": {
            "source": str(static_source), "source_sha256": hashlib.sha256(static_source_raw).hexdigest(),
            "payload": str(static_payload), "payload_sha256": hashlib.sha256(static_payload_raw).hexdigest(),
            "payload_size": len(static_payload_raw),
        },
        "cpu_consumption_diagnostic": {
            "source": str(diagnostic_rows[CPU_CONSUMPTION_PROBE][0]),
            "source_sha256": DIAGNOSTIC_SOURCE_PINS[CPU_CONSUMPTION_PROBE],
            "methods_root": str(release_dir),
            "site_packages_root": str(TORCH_ROOT.parent),
            "external_controller_must_create_fresh_mode_0700_work_root": True,
            "receipt_must_be_inside_work_root_with_basename":
                "r5d-cpu-consumption-probe.json",
            "payload_or_controller_materialized": False,
            "executed": False,
        },
        "artifact_count": len(artifacts), "artifacts": artifacts,
        "directory_count": len(directories), "directories": directories,
        "outputs_media_empty": True, "final_empty": True, "runtime_empty": True,
        "production_rank_cache_root": str(PRODUCTION_RANK_CACHE_ROOT),
        "production_rank_cache_node_local_required": True,
        "evidence_empty": True, "slurm_step_launched": False,
    }
    report["receipt_digest"] = object_sha256(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--source-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if (
        sys.platform != "linux"
        or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or "torch" in sys.modules
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise R5FMaterializationError("isolated materializer startup differs")
    report = _materialize(build_parser().parse_args(argv))
    print((canonical_json_bytes(report) + b"\n").decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
