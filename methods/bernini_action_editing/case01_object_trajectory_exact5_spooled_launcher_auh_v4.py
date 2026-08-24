#!/usr/bin/env python3
"""Create-only HOLD launcher for the v4 implementation of the v3 exact-five ABI.

This module deliberately cannot produce a runnable canary payload.  It seals
the exact launch identities and a non-launchable plan into a captured HOLD
payload.  A future READY overlay must be a different, newly hashed artifact.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-exact5-hold-launch-release-auh-v3"
INPUT_SCHEMA = "case01-object-trajectory-exact5-hold-launch-input-auh-v3"
RECEIPT_SCHEMA = "case01-object-trajectory-exact5-hold-launch-receipt-auh-v3"
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle-v3"
ARM_ORDER = (
    "null_before",
    "route_off",
    "trajectory_bone_only",
    "trajectory_dog_bone",
    "null_after",
)
TASK_IDS = tuple(
    f"case01-object-trajectory-{arm}-full644" for arm in ARM_ORDER
)

FINAL_RUNNER_SHA256 = (
    "326ccfff1a09d6db8c93d02cfe6018e465e127263547f325cc7f18e7d16a7148"
)
FINAL_EVAL_SHA256 = (
    "381ba375147bec7580b451226b07b3d1cab9125866978602de05fbba4f16aaa3"
)
FINAL_WRAPPER_SHA256 = (
    "797c5d1e7cb8bbfda1f2e4cc3825702c248d3ce64770ddc1520155f5635c3557"
)
OBJECT_WRAPPER_INNER_SHA256 = (
    "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9"
)
EXPECTED_CAPTURED_ROOT_FAKE_SHA256 = (
    "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872"
)
EXPECTED_CAPTURED_ROOT_FAKE_SIZE = 21_596

EXPECTED_STATIC_SHA256 = {
    "runner": FINAL_RUNNER_SHA256,
    "legacy_exact5_runner": (
        "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea"
    ),
    "object_eval": FINAL_EVAL_SHA256,
    "legacy_exact5_eval": (
        "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58"
    ),
    "frozen_runner": (
        "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223"
    ),
    "bridge": (
        "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136"
    ),
    "adapter": FINAL_WRAPPER_SHA256,
    "object_wrapper_inner": OBJECT_WRAPPER_INNER_SHA256,
    "legacy_infer_alias": (
        "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
    ),
    "trajectory_projection": (
        "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e"
    ),
    "trajectory_scaffold_module": (
        "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a"
    ),
    "base_adapter": (
        "7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120"
    ),
    "eval_v1": (
        "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d"
    ),
    "eval_v2": (
        "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982"
    ),
    "model_authority": (
        "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf"
    ),
    "torchrun_source": (
        "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c"
    ),
    "torchrun_handler_source": (
        "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87"
    ),
    "torch_local_agent_source": (
        "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497"
    ),
    "torch_dynamic_rendezvous_source": (
        "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec"
    ),
    "torch_multiprocessing_api_source": (
        "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7"
    ),
    "base_model_manifest": (
        "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
    ),
    "r64_checkpoint_manifest": (
        "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2"
    ),
}

IDENTITY_ROLES = tuple(EXPECTED_STATIC_SHA256) + (
    "python", "ffmpeg", "ffprobe", "plan",
)
if len(IDENTITY_ROLES) != 26 or len(set(IDENTITY_ROLES)) != 26:
    raise RuntimeError("object trajectory exact26 role closure differs")

METHOD_ROLE_BASENAMES = {
    "runner": "case01_object_trajectory_exact5_runner_v4.py",
    "legacy_exact5_runner": "case01_source_bone_exact5_runner_v1.py",
    "object_eval": "case01_object_trajectory_exact5_eval_v4.py",
    "legacy_exact5_eval": "case01_source_bone_exact5_eval_v1.py",
    "frozen_runner": "full644_exploratory_matched_runner_auh_r5.py",
    "bridge": "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
    "adapter": "infer_case01_object_trajectory_oracle_auh_r5f_v4.py",
    "object_wrapper_inner": "infer_case01_object_trajectory_oracle_v1.py",
    "legacy_infer_alias": "infer_lora_full644_r5_frozen_acc46.py",
    "trajectory_projection": "object_trajectory_projection_v1.py",
    "trajectory_scaffold_module": "case01_oracle_object_trajectory_v1.py",
    "base_adapter": "full644_exploratory_matched_infer_adapter_v3.py",
    "eval_v1": "full644_exploratory_matched_eval_v1.py",
    "eval_v2": "full644_exploratory_matched_eval_v2.py",
    "model_authority": "action_preservation_decoded_eval_model_authority_v2.py",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}")


class HoldLaunchError(RuntimeError):
    """The sealed HOLD launch contract differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def blocked_roles() -> tuple[str, ...]:
    return tuple(
        role for role, digest in EXPECTED_STATIC_SHA256.items()
        if not SHA256_RE.fullmatch(digest)
    )


def require_final_pins() -> None:
    blocked = blocked_roles()
    if (
        blocked
        or SHA256_RE.fullmatch(EXPECTED_CAPTURED_ROOT_FAKE_SHA256) is None
        or type(EXPECTED_CAPTURED_ROOT_FAKE_SIZE) is not int
        or EXPECTED_CAPTURED_ROOT_FAKE_SIZE <= 0
    ):
        raise HoldLaunchError(
            "HOLD: final source pins are blocked: "
            + ",".join(blocked or ("captured_root_fake",))
        )


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def stable_file(
    path_value: str | Path, *, expected_sha256: str | None = None,
    expected_size: int | None = None, executable: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    path = Path(path_value)
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise HoldLaunchError(f"noncanonical authority path: {path}")
    try:
        named_before = os.lstat(path)
    except OSError as error:
        raise HoldLaunchError(f"missing authority path: {path}") from error
    # The named object is classified before resolve/open, so FIFO/socket/device
    # paths fail without a potentially blocking read.
    if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
        raise HoldLaunchError(f"authority is not one regular single-link file: {path}")
    if path.resolve(strict=True) != path:
        raise HoldLaunchError(f"noncanonical authority path: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size <= 0
            or _identity(before) != _identity(named_before)
            or (expected_size is not None and before.st_size != expected_size)
            or (executable and not before.st_mode & 0o111)
        ):
            raise HoldLaunchError(f"opened authority differs before read: {path}")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor, min(1_048_576, before.st_size - offset), offset,
            )
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        middle = os.fstat(descriptor)
        second = b"".join(
            os.pread(
                descriptor, min(1_048_576, before.st_size - at), at,
            )
            for at in range(0, before.st_size, 1_048_576)
        )
        eof = os.pread(descriptor, 1, before.st_size)
        after = os.fstat(descriptor)
        named = os.lstat(path)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    if (
        raw != second or eof != b""
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_before)
        or _identity(before) != _identity(named)
        or (expected_sha256 is not None and digest != expected_sha256)
        or (expected_size is not None and before.st_size != expected_size)
        or (executable and not before.st_mode & 0o111)
    ):
        raise HoldLaunchError(f"unstable authority: {path}")
    return raw, {
        "path": str(path), "sha256": digest, "size": len(raw),
        "mode": stat.S_IMODE(before.st_mode), "nlink": before.st_nlink,
    }


def create_file(path: Path, raw: bytes, mode: int) -> None:
    if os.path.lexists(path) or not path.parent.is_dir():
        raise HoldLaunchError(f"create-only target differs: {path}")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise HoldLaunchError("create-only write made no progress")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise HoldLaunchError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise HoldLaunchError(f"invalid JSON: {label}") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise HoldLaunchError(f"noncanonical JSON: {label}")
    return value


def _validate_hold_plan(plan: Mapping[str, Any]) -> None:
    tasks = plan.get("tasks")
    if (
        plan.get("launch_allowed") is not False
        or plan.get("production_ready") is not False
        or plan.get("status") != "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY"
        or not isinstance(plan.get("hold_reasons"), list)
        or not plan["hold_reasons"]
        or not isinstance(tasks, list) or len(tasks) != 5
        or [task.get("task_id") for task in tasks] != list(TASK_IDS)
        or [task.get("oracle_arm") for task in tasks] != list(ARM_ORDER)
        or any(task.get("source_onset_policy") != "hard1_every_step" for task in tasks)
    ):
        raise HoldLaunchError("plan is not the exact non-launchable five-arm HOLD")


# Diagnostic-only captured root.  The named HOLD payload below never invokes
# it; admission must stream this exact string to an isolated Python process.
ROOT_BOOTSTRAP = r'''import hashlib,json,os,stat,sys,types
if len(sys.argv)!=3: raise RuntimeError("root diagnostic argv differs")
spec_raw,release_digest=sys.argv[1:]
def pairs(items):
 out={}
 for key,value in items:
  if key in out: raise RuntimeError("duplicate root JSON key")
  out[key]=value
 return out
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def ident(info): return (info.st_dev,info.st_ino,info.st_uid,info.st_gid,info.st_mode,info.st_nlink,info.st_rdev,info.st_size,getattr(info,"st_blocks",0),info.st_mtime_ns,info.st_ctime_ns)
def read_fd(fd,size):
 out=[]; offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  out.append(block); offset+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("captured root read differs")
 return raw
def read_named(role,row,executable=False):
 path=row.get("path") if type(row) is dict else None
 if type(row) is not dict or set(row)!={"path","sha256","size"} or type(path) is not str or not os.path.isabs(path) or os.path.normpath(path)!=path: raise RuntimeError("identity row differs: "+role)
 named=os.lstat(path)
 if not stat.S_ISREG(named.st_mode) or named.st_nlink!=1 or (executable and not named.st_mode&0o111) or os.path.realpath(path)!=path: raise RuntimeError("named identity differs: "+role)
 fd=os.open(path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or ident(before)!=ident(named) or before.st_size!=row.get("size") or (executable and not before.st_mode&0o111): raise RuntimeError("opened identity differs before read: "+role)
  first=read_fd(fd,before.st_size); middle=os.fstat(fd); second=read_fd(fd,before.st_size); eof=os.pread(fd,1,before.st_size); after=os.fstat(fd); named_after=os.lstat(path)
 finally: os.close(fd)
 if len(first)!=before.st_size or first!=second or eof!=b"" or ident(named)!=ident(before) or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named_after) or hashlib.sha256(first).hexdigest()!=row.get("sha256") or len(first)!=row.get("size"): raise RuntimeError("identity replay differs: "+role)
 return first,ident(before)
spec=json.loads(spec_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
roles=("runner","legacy_exact5_runner","object_eval","legacy_exact5_eval","frozen_runner","bridge","adapter","object_wrapper_inner","legacy_infer_alias","trajectory_projection","trajectory_scaffold_module","base_adapter","eval_v1","eval_v2","model_authority","torchrun_source","torchrun_handler_source","torch_local_agent_source","torch_dynamic_rendezvous_source","torch_multiprocessing_api_source","base_model_manifest","r64_checkpoint_manifest","python","ffmpeg","ffprobe","plan")
tasks=["case01-object-trajectory-null_before-full644","case01-object-trajectory-route_off-full644","case01-object-trajectory-trajectory_bone_only-full644","case01-object-trajectory-trajectory_dog_bone-full644","case01-object-trajectory-null_after-full644"]
arms=["null_before","route_off","trajectory_bone_only","trajectory_dog_bone","null_after"]
if type(spec) is not dict or set(spec)!={"schema_version","campaign_mode","launch_allowed","identities","captured_runner","launch_input","result_path"} or canonical(spec)!=spec_raw or digest(spec)!=release_digest or spec.get("schema_version")!="case01-object-trajectory-exact5-root-bootstrap-diagnostic-v4" or spec.get("campaign_mode")!="case01-object-trajectory-exact5-r64-engineering-oracle-v3" or spec.get("launch_allowed") is not False: raise RuntimeError("root diagnostic release differs")
identities=spec.get("identities")
if type(identities) is not dict or set(identities)!=set(roles) or len(identities)!=26 or len({row.get("path") for row in identities.values() if type(row) is dict})!=26: raise RuntimeError("exact26 identity closure differs")
raw_by_role={}; identity_by_role={}
for role in roles:
 raw_by_role[role],identity_by_role[role]=read_named(role,identities[role],role in {"python","ffmpeg","ffprobe"})
captured_runner=spec.get("captured_runner")
captured_runner_raw,captured_runner_identity=read_named("captured_runner",captured_runner)
if captured_runner.get("sha256")!="0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872" or captured_runner.get("size")!=21596 or captured_runner.get("path") in {row["path"] for row in identities.values()} or captured_runner==identities["runner"]: raise RuntimeError("captured runner authority differs/overlaps production exact26")
launch_raw,_launch_identity=read_named("launch_input",spec.get("launch_input"))
launch=json.loads(launch_raw.decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
if launch_raw!=canonical(launch).encode()+b"\n" or type(launch) is not dict or launch.get("identities")!=identities: raise RuntimeError("validated launch input identity set differs")
plan=json.loads(raw_by_role["plan"].decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
rows=plan.get("tasks") if type(plan) is dict else None
if raw_by_role["plan"]!=canonical(plan).encode()+b"\n" or plan.get("launch_allowed") is not False or plan.get("production_ready") is not False or plan.get("status")!="HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY" or type(rows) is not list or len(rows)!=5 or [row.get("task_id") for row in rows]!=tasks or [row.get("oracle_arm") for row in rows]!=arms or any(row.get("source_onset_policy")!="hard1_every_step" for row in rows): raise RuntimeError("five-arm HOLD plan differs")
for row in rows:
 external=row.get("external_conditions")
 if row["oracle_arm"] in {"null_before","null_after"}:
  if external!={}: raise RuntimeError("null arm carries external authority")
 elif type(external) is not dict or set(external)!={"stage0_masks","g0_mouth_track","trajectory_scaffold","aux_bone_removed_source"}: raise RuntimeError("non-null external closure differs")
producer=plan.get("producer"); checkpoint=plan.get("checkpoint_manifest")
producer_roles={"legacy_infer_alias":("infer_lora_path","infer_lora_sha256","infer_lora_size"),"adapter":("inference_wrapper_path","inference_wrapper_sha256","inference_wrapper_size"),"object_wrapper_inner":("object_wrapper_inner_path","object_wrapper_inner_sha256","object_wrapper_inner_size"),"trajectory_projection":("trajectory_projection_module_path","trajectory_projection_module_sha256","trajectory_projection_module_size"),"trajectory_scaffold_module":("trajectory_scaffold_module_path","trajectory_scaffold_module_sha256","trajectory_scaffold_module_size"),"ffprobe":("ffprobe_path","ffprobe_sha256","ffprobe_size")}
if type(producer) is not dict or type(checkpoint) is not dict: raise RuntimeError("plan producer/checkpoint closure differs")
for role,keys in producer_roles.items():
 if identities[role]!={"path":producer.get(keys[0]),"sha256":producer.get(keys[1]),"size":producer.get(keys[2])}: raise RuntimeError("plan producer identity differs: "+role)
if checkpoint.get("path")!=identities["r64_checkpoint_manifest"]["path"] or checkpoint.get("sha256")!=identities["r64_checkpoint_manifest"]["sha256"] or any(row.get("adapter",{}).get("checkpoint_manifest")!=checkpoint for row in rows): raise RuntimeError("plan checkpoint identity differs")
result_path=spec.get("result_path")
if type(result_path) is not str or not os.path.isabs(result_path) or os.path.normpath(result_path)!=result_path or os.path.lexists(result_path) or not os.path.isdir(os.path.dirname(result_path)) or os.path.realpath(os.path.dirname(result_path))!=os.path.dirname(result_path): raise RuntimeError("root fake result target differs")
entry={"schema_version":"case01-object-trajectory-exact5-captured-root-entry-v4","release_digest":release_digest,"identity_roles":list(roles),"identity_set_digest":digest(identities),"launch_input_sha256":spec["launch_input"]["sha256"],"production_runner":identities["runner"],"captured_runner":captured_runner,"captured_runner_identity":list(captured_runner_identity),"plan_sha256":identities["plan"]["sha256"],"task_ids":tasks,"arm_order":arms,"all_exact26_named_identities_replayed":True,"captured_runner_outside_exact26":True,"captured_runner_bytes_compiled":True,"publication_performed":False}
entry["authority_digest"]=digest(entry)
os.environ.clear(); os.environ["CASE01_OBJECT_TRAJECTORY_CAPTURED_ROOT_ENTRY"]=canonical(entry)
sys.argv=[captured_runner["path"],"--captured-result",result_path]
module=types.ModuleType("__main__"); module.__file__=captured_runner["path"]; module.__package__=None; module.__loader__=None; module.__spec__=None; module.__cached__=None; module.__builtins__=__builtins__; sys.modules["__main__"]=module
exec(compile(captured_runner_raw.decode("utf-8","strict"),captured_runner["path"],"exec",dont_inherit=True),module.__dict__)'''


def validate_input(
    value: Mapping[str, Any], *, reopen: bool = True,
    allow_blocked_pins: bool = False,
    plan_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version", "entry_mode", "campaign_mode", "holder_job_id",
        "expected_node", "expected_allocation_gpu_count", "identities",
        "output_report", "runner_attestation", "model_root", "bernini_root",
        "veomni_root", "authority_root", "rank_cache_root",
    }
    identities = value.get("identities") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping) or set(value) != fields
        or value.get("schema_version") != INPUT_SCHEMA
        or value.get("entry_mode") != "trusted_stdin"
        or value.get("campaign_mode") != CAMPAIGN
        or value.get("expected_allocation_gpu_count") != 8
        or not isinstance(value.get("holder_job_id"), str)
        or not value["holder_job_id"]
        or not isinstance(value.get("expected_node"), str)
        or not value["expected_node"]
        or not isinstance(identities, Mapping)
        or set(identities) != set(IDENTITY_ROLES)
        or len(identities) != len(IDENTITY_ROLES)
    ):
        raise HoldLaunchError("HOLD launch input closure differs")
    if not allow_blocked_pins:
        require_final_pins()
    seen: set[str] = set()
    for role in IDENTITY_ROLES:
        row = identities[role]
        expected = EXPECTED_STATIC_SHA256.get(role)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "size"}
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
            or type(row.get("size")) is not int or row["size"] <= 0
            or row["path"] in seen
            or (expected is not None and SHA256_RE.fullmatch(expected)
                and row["sha256"] != expected)
            or (expected is not None and not SHA256_RE.fullmatch(expected)
                and not allow_blocked_pins)
        ):
            raise HoldLaunchError(f"identity row differs: {role}")
        seen.add(row["path"])
        if reopen:
            stable_file(
                row["path"], expected_sha256=row["sha256"],
                expected_size=row["size"],
                executable=role in {"python", "ffmpeg", "ffprobe"},
            )
    method_parents = {
        Path(identities[role]["path"]).parent for role in METHOD_ROLE_BASENAMES
    }
    if len(method_parents) != 1 or any(
        Path(identities[role]["path"]).name != name
        for role, name in METHOD_ROLE_BASENAMES.items()
    ):
        raise HoldLaunchError("method-source adjacency differs")
    if plan_override is None:
        plan_raw, _ = stable_file(
            identities["plan"]["path"],
            expected_sha256=identities["plan"]["sha256"],
            expected_size=identities["plan"]["size"],
        )
        plan_value = _strict_json(plan_raw, label="HOLD plan")
    else:
        plan_value = dict(plan_override)
        if (
            hashlib.sha256(canonical_json_bytes(plan_value) + b"\n").hexdigest()
            != identities["plan"]["sha256"]
            or len(canonical_json_bytes(plan_value) + b"\n")
            != identities["plan"]["size"]
        ):
            raise HoldLaunchError("supplied HOLD plan identity differs")
    _validate_hold_plan(plan_value)
    return dict(value)


def _hold_payload(release: Mapping[str, Any]) -> bytes:
    release_b64 = base64.b64encode(canonical_json_bytes(release)).decode("ascii")
    release_digest = object_sha256(release)
    script = f'''#!/bin/bash -p
set -Eeuo pipefail
umask 077
readonly RELEASE_DIGEST={release_digest!r}
readonly RELEASE_JSON_B64={release_b64!r}
/usr/bin/printf '%s\\n' 'HOLD: case01 object-trajectory exact5 is sealed but not launch-authorized' >&2
exit 88
'''
    return script.encode("utf-8")


def materialize(
    input_path: str | Path, payload_path: str | Path, receipt_path: str | Path,
    *, reopen_identities: bool = True,
    plan_override: Mapping[str, Any] | None = None,
    logical_input_path: str | Path | None = None,
    logical_payload_path: str | Path | None = None,
) -> dict[str, Any]:
    require_final_pins()
    input_raw, input_identity = stable_file(input_path)
    if logical_input_path is not None:
        input_identity = dict(input_identity)
        input_identity["path"] = str(logical_input_path)
    value = _strict_json(input_raw, label="HOLD launch input")
    validated = validate_input(
        value, reopen=reopen_identities, plan_override=plan_override,
    )
    release = {
        "schema_version": SCHEMA,
        "status": "HOLD_NOT_LAUNCHABLE",
        "launch_allowed": False,
        "campaign_mode": CAMPAIGN,
        "selected_task_ids": list(TASK_IDS),
        "identity_roles": list(IDENTITY_ROLES),
        "identities": validated["identities"],
        "input_sha256": input_identity["sha256"],
        "ready_overlay_required": True,
        "named_payload_execution_forbidden": True,
    }
    release["release_digest"] = object_sha256(release)
    payload_raw = _hold_payload(release)
    payload = Path(payload_path)
    receipt = Path(receipt_path)
    if (
        payload == receipt
        or os.path.lexists(payload)
        or os.path.lexists(receipt)
        or not payload.parent.is_dir() or not receipt.parent.is_dir()
    ):
        raise HoldLaunchError("HOLD output target closure is not fresh")
    create_file(payload, payload_raw, 0o444)
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "MATERIALIZED_HOLD_NOT_SUBMITTED",
        "launch_allowed": False,
        "slurm_step_launched": False,
        "gpu_attempt_claimed": False,
        "input": input_identity,
        "release": release,
        "payload_path": str(
            payload if logical_payload_path is None else logical_payload_path
        ),
        "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "payload_size": len(payload_raw),
    }
    result["receipt_digest"] = object_sha256(result)
    create_file(receipt, canonical_json_bytes(result) + b"\n", 0o400)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize(args.input, args.payload, args.receipt)
    except (OSError, ValueError, HoldLaunchError) as error:
        print(str(error), file=os.sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
