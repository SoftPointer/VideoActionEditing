#!/usr/bin/env python3
"""Materialize the fresh AUH r5d root-launch release.

The runner and Torchrun FD bridge already carry the adapter path and digest as
authenticated launch inputs, so their frozen bytes do not change for r5d.
This release-builder successor exact-loads the frozen r5c builder, adds the
wrapper's adjacent frozen-base dependency as an independently pinned role, and
changes the three release schemas plus their embedded root-bootstrap checks.
The r5c builder and every burned r5c release remain byte-for-byte untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


BASE_LAUNCHER_SHA256 = (
    "cb201398940d59393fa58471dc2c3f9fdf001c7e881ec891ce892bb460cf01ba"
)
R5D_ADAPTER_SHA256 = (
    "5794e1f0e5ecb84ffdb37f618fe63696ee4f87176952ac083c8c91792a9d192a"
)
RUNNER_SHA256 = (
    "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223"
)
BRIDGE_SHA256 = (
    "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136"
)
BASE_ADAPTER_SHA256 = (
    "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca"
)
MODEL_AUTHORITY_SHA256 = (
    "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf"
)

SCHEMA = "full644-exploratory-matched-root-launch-release-auh-r5d"
INPUT_SCHEMA = "full644-exploratory-matched-root-launch-input-auh-r5d"
RECEIPT_SCHEMA = "full644-exploratory-matched-root-launch-receipt-auh-r5d"

_BASE_SCHEMA = "full644-exploratory-matched-root-launch-release-auh-r5c"
_BASE_INPUT_SCHEMA = "full644-exploratory-matched-root-launch-input-auh-r5c"
_BASE_RECEIPT_SCHEMA = "full644-exploratory-matched-root-launch-receipt-auh-r5c"
_BASE_MODULE = "_full644_exploratory_matched_spooled_launcher_auh_r5c_base"
_BASE_PATH = (
    Path(__file__).resolve(strict=True).parent
    / "full644_exploratory_matched_spooled_launcher_auh_r5.py"
)
_ADAPTER_BASENAME = "full644_exploratory_matched_infer_adapter_auh_r5d.py"
_BASE_ADAPTER_BASENAME = "full644_exploratory_matched_infer_adapter_v2.py"
_MODEL_AUTHORITY_BASENAME = (
    "action_preservation_decoded_eval_model_authority_v2.py"
)


class R5DLauncherBootstrapError(RuntimeError):
    """The pinned r5c builder or the r5d pin transformation differs."""


def _read_pinned_source(path: Path, expected_sha256: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_uid,
        before.st_gid,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_uid,
            after.st_gid,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or identity
        != (
            named.st_dev,
            named.st_ino,
            named.st_uid,
            named.st_gid,
            named.st_mode,
            named.st_nlink,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        )
        or len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise R5DLauncherBootstrapError("frozen r5c launcher source differs")
    try:
        return raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise R5DLauncherBootstrapError(
            "frozen r5c launcher source is not UTF-8"
        ) from error


def _validate_adapter_dependency_paths(value: Mapping[str, Any]) -> None:
    try:
        adapter = Path(value["adapter"])
        base_adapter = Path(value["base_adapter"])
        model_authority = Path(value["model_authority"])
    except (KeyError, TypeError) as error:
        raise R5DLauncherBootstrapError(
            "r5d adapter dependency path is absent"
        ) from error
    if (
        not all(
            path.is_absolute()
            for path in (adapter, base_adapter, model_authority)
        )
        or any(
            os.path.normpath(str(path)) != str(path)
            for path in (adapter, base_adapter, model_authority)
        )
        or adapter.name != _ADAPTER_BASENAME
        or base_adapter.name != _BASE_ADAPTER_BASENAME
        or model_authority.name != _MODEL_AUTHORITY_BASENAME
        or base_adapter != adapter.parent / _BASE_ADAPTER_BASENAME
        or model_authority != adapter.parent / _MODEL_AUTHORITY_BASENAME
    ):
        raise R5DLauncherBootstrapError(
            "r5d adapter dependency adjacency differs"
        )


def _r5d_input_loader(module: types.ModuleType) -> Any:
    def load_input(
        path_value: str | Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        identity = module._stable_file(path_value, return_raw=True)
        raw = identity.pop("_raw")
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=module._pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(token)
                ),
            )
        except (UnicodeError, ValueError, TypeError) as error:
            raise module.RootLaunchReleaseError(
                "launch input is not strict JSON"
            ) from error
        fields = {
            "schema_version",
            "entry_mode",
            "runner",
            "bridge",
            "adapter",
            "base_adapter",
            "eval_v1",
            "eval_v2",
            "model_authority",
            "python",
            "ffmpeg",
            "torchrun_source",
            "torchrun_handler_source",
            "torch_local_agent_source",
            "torch_dynamic_rendezvous_source",
            "torch_multiprocessing_api_source",
            "plan",
            "output_report",
            "runner_attestation",
            "model_root",
            "model_manifest",
            "bernini_root",
            "veomni_root",
            "authority_root",
            "rank_cache_root",
            "holder_job_id",
            "expected_node",
            "campaign_mode",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or value.get("schema_version") != INPUT_SCHEMA
            or raw != module.canonical_json_bytes(value) + b"\n"
            or value.get("entry_mode") not in {"trusted_stdin", "slurm_spool"}
            or not isinstance(value.get("holder_job_id"), str)
            or not value["holder_job_id"]
            or not isinstance(value.get("expected_node"), str)
            or not value["expected_node"]
            or value.get("campaign_mode")
            not in {module.FULL16_CAMPAIGN, module.CASE00_CANARY_CAMPAIGN}
        ):
            raise module.RootLaunchReleaseError("launch input closure differs")
        try:
            _validate_adapter_dependency_paths(value)
        except R5DLauncherBootstrapError as error:
            raise module.RootLaunchReleaseError(str(error)) from error
        return value, identity

    return load_input


def _load_base() -> types.ModuleType:
    source = _read_pinned_source(_BASE_PATH, BASE_LAUNCHER_SHA256)
    module = types.ModuleType(_BASE_MODULE)
    module.__file__ = str(_BASE_PATH)
    module.__package__ = None
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    module.__builtins__ = __builtins__
    sys.modules[_BASE_MODULE] = module
    try:
        exec(
            compile(source, str(_BASE_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(_BASE_MODULE, None)
        raise
    return module


def _transform_base(module: types.ModuleType) -> types.ModuleType:
    expected_static = dict(module.EXPECTED_STATIC_SHA256)
    if (
        module.SCHEMA != _BASE_SCHEMA
        or module.INPUT_SCHEMA != _BASE_INPUT_SCHEMA
        or module.RECEIPT_SCHEMA != _BASE_RECEIPT_SCHEMA
        or expected_static.get("runner") != RUNNER_SHA256
        or expected_static.get("bridge") != BRIDGE_SHA256
        or expected_static.get("adapter") != BASE_ADAPTER_SHA256
        or expected_static.get("model_authority") != MODEL_AUTHORITY_SHA256
        or "base_adapter" in expected_static
        or len(expected_static) != 12
        or module.ROOT_BOOTSTRAP.count(_BASE_SCHEMA) != 1
        or SCHEMA in module.ROOT_BOOTSTRAP
    ):
        raise R5DLauncherBootstrapError("frozen r5c launcher contract differs")

    transformed_bootstrap = module.ROOT_BOOTSTRAP.replace(
        _BASE_SCHEMA, SCHEMA, 1
    )
    bootstrap_anchor = (
        'if not stat.S_ISREG(runner_before.st_mode) or runner_before.st_nlink!=1 '
        'or stat.S_IMODE(runner_before.st_mode)!=0o444 or '
        'ident(runner_before)!=ident(runner_after) or '
        'ident(runner_before)!=ident(runner_named) or '
        'ident(runner_before)!=runner_row.get("identity") or '
        'hashlib.sha256(runner_raw).hexdigest()!=runner_row.get("sha256"): '
        'raise RuntimeError("captured runner identity differs")'
    )
    identity_roles = tuple(
        sorted((*expected_static, "base_adapter", "python", "ffmpeg", "plan"))
    )
    bootstrap_dependency_check = f'''identity_roles=tuple(sorted(spec.get("identities",{{}})))
if identity_roles!={identity_roles!r}: raise RuntimeError("r5d exact16 release identity closure differs")
adapter_row=spec["identities"].get("adapter"); base_adapter_row=spec["identities"].get("base_adapter"); model_authority_row=spec["identities"].get("model_authority")
if type(adapter_row) is not dict or type(base_adapter_row) is not dict or type(model_authority_row) is not dict: raise RuntimeError("r5d adapter dependency rows differ")
adapter_path=adapter_row.get("path"); base_adapter_path=base_adapter_row.get("path"); model_authority_path=model_authority_row.get("path")
if not all(type(value) is str and os.path.isabs(value) and os.path.normpath(value)==value for value in (adapter_path,base_adapter_path,model_authority_path)) or os.path.basename(adapter_path)!={_ADAPTER_BASENAME!r} or os.path.basename(base_adapter_path)!={_BASE_ADAPTER_BASENAME!r} or os.path.basename(model_authority_path)!={_MODEL_AUTHORITY_BASENAME!r} or os.path.dirname(adapter_path)!=os.path.dirname(base_adapter_path) or os.path.dirname(adapter_path)!=os.path.dirname(model_authority_path): raise RuntimeError("r5d adapter dependency adjacency differs")
for source_label,source_row,source_pin in (("r5d adapter",adapter_row,{R5D_ADAPTER_SHA256!r}),("r5c base adapter",base_adapter_row,{BASE_ADAPTER_SHA256!r}),("model authority",model_authority_row,{MODEL_AUTHORITY_SHA256!r})):
 source_fd=os.open(source_row["path"],flags); source_before=os.fstat(source_fd)
 try: source_raw=read_fd(source_fd,source_before.st_size); source_after=os.fstat(source_fd); source_named=os.lstat(source_row["path"])
 finally: os.close(source_fd)
 if not stat.S_ISREG(source_before.st_mode) or source_before.st_nlink!=1 or stat.S_IMODE(source_before.st_mode)!=0o444 or ident(source_before)!=ident(source_after) or ident(source_before)!=ident(source_named) or ident(source_before)!=source_row.get("identity") or source_row.get("sha256")!=source_pin or hashlib.sha256(source_raw).hexdigest()!=source_pin: raise RuntimeError(source_label+" identity differs")'''
    if transformed_bootstrap.count(bootstrap_anchor) != 1:
        raise R5DLauncherBootstrapError(
            "frozen root-bootstrap insertion point differs"
        )
    transformed_bootstrap = transformed_bootstrap.replace(
        bootstrap_anchor,
        bootstrap_anchor + "\n" + bootstrap_dependency_check,
        1,
    )
    if (
        _BASE_SCHEMA in transformed_bootstrap
        or transformed_bootstrap.count(SCHEMA) != 1
    ):
        raise R5DLauncherBootstrapError("r5d root-bootstrap schema differs")

    expected_static["adapter"] = R5D_ADAPTER_SHA256
    expected_static["base_adapter"] = BASE_ADAPTER_SHA256
    module.SCHEMA = SCHEMA
    module.INPUT_SCHEMA = INPUT_SCHEMA
    module.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    module.EXPECTED_STATIC_SHA256 = expected_static
    module.ROOT_BOOTSTRAP = transformed_bootstrap
    module._load_input = _r5d_input_loader(module)

    original_build_release = module.build_release

    def build_release_r5d(
        value: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bytes]:
        try:
            _validate_adapter_dependency_paths(value)
        except R5DLauncherBootstrapError as error:
            raise module.RootLaunchReleaseError(str(error)) from error
        release, payload = original_build_release(value)
        identities = release.get("identities")
        expected_roles = set(expected_static) | {"python", "ffmpeg", "plan"}
        base_row = (
            identities.get("base_adapter")
            if isinstance(identities, dict)
            else None
        )
        arguments = release.get("runner_arguments")
        if (
            not isinstance(identities, dict)
            or set(identities) != expected_roles
            or len(identities) != 16
            or not isinstance(base_row, dict)
            or stat.S_IMODE(base_row.get("identity", {}).get("mode", -1))
            != 0o444
            or base_row.get("identity", {}).get("nlink") != 1
            or base_row.get("sha256") != BASE_ADAPTER_SHA256
            or not isinstance(arguments, list)
            or "--adapter-script" not in arguments
            or arguments[arguments.index("--adapter-script") + 1]
            != value["adapter"]
            or "--adapter-script-sha256" not in arguments
            or arguments[arguments.index("--adapter-script-sha256") + 1]
            != R5D_ADAPTER_SHA256
            or value["base_adapter"] in arguments
        ):
            raise module.RootLaunchReleaseError(
                "r5d exact16 adapter release closure differs"
            )
        return release, payload

    module.build_release = build_release_r5d
    return module


base = _transform_base(_load_base())

# Public compatibility surface.  The delegated functions retain ``base`` as
# their globals, where the r5d transformation above is installed exactly once.
RootLaunchReleaseError = base.RootLaunchReleaseError
FULL16_CAMPAIGN = base.FULL16_CAMPAIGN
CASE00_CANARY_CAMPAIGN = base.CASE00_CANARY_CAMPAIGN
TASK_IDS = base.TASK_IDS
CANARY_TASK_IDS = base.CANARY_TASK_IDS
ENTRY_AUTHORITY_SCHEMA = base.ENTRY_AUTHORITY_SCHEMA
EXPECTED_STATIC_SHA256 = base.EXPECTED_STATIC_SHA256
ROOT_BOOTSTRAP = base.ROOT_BOOTSTRAP
canonical_json_bytes = base.canonical_json_bytes
object_sha256 = base.object_sha256


def build_release(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    return base.build_release(value)


def materialize(
    input_path: str, payload_path: str, receipt_path: str
) -> dict[str, Any]:
    return base.materialize(input_path, payload_path, receipt_path)


def build_parser() -> Any:
    return base.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
