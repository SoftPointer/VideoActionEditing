#!/usr/bin/env python3
"""CPU-only AUH canary for an exact-23 inherited self-proc-FD model view.

This script deliberately performs no model tensor materialization, GPU access,
scheduler call, or holder allocation.  The parent retains all model file FDs,
builds a private symlink tree whose leaves name those FDs through procfs, and
starts two torchrun ranks.  Each rank exercises the real Transformers,
Diffusers, and safetensors path resolvers against that view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Sequence


SCHEMA = "apv2-eval-exact23-inherited-self-fd-torchrun-cpu-canary-v2"
MANIFEST_SHA256 = "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
TORCHRUN_HANDLER_SHA256 = (
    "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87"
)
LINE = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
FD_ENV = "APV2_EVAL_CANARY_AUTHORITY_FDS"


class CanaryError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(path: Path) -> list[tuple[str, str]]:
    if sha256_file(path) != MANIFEST_SHA256:
        raise CanaryError("checkpoint manifest SHA differs")
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE.fullmatch(line)
        if match is None:
            raise CanaryError("checkpoint manifest line differs")
        digest, raw = match.groups()
        relative = PurePosixPath(raw)
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if relative.is_absolute() or ".." in relative.parts or not normalized:
            raise CanaryError("checkpoint manifest relative path differs")
        rows.append((normalized, digest))
    if len(rows) != 23 or len({relative for relative, _ in rows}) != 23:
        raise CanaryError("checkpoint manifest is not exact-23")
    return rows


def inherited_fds(*, expected_inheritable: bool) -> tuple[int, ...]:
    raw = os.environ.get(FD_ENV)
    if raw is None:
        raise CanaryError("inherited FD environment is absent")
    try:
        fds = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CanaryError("inherited FD environment differs") from error
    if (
        not isinstance(fds, list)
        or len(fds) != 23
        or fds != sorted(fds)
        or len(set(fds)) != 23
        or any(type(fd) is not int or fd < 3 for fd in fds)
    ):
        raise CanaryError("inherited FD allowlist differs")
    for fd in fds:
        try:
            info = os.fstat(fd)
        except OSError as error:
            raise CanaryError("inherited FD is unavailable") from error
        if (
            not stat.S_ISREG(info.st_mode)
            or os.get_inheritable(fd) is not expected_inheritable
        ):
            raise CanaryError("inherited FD identity differs")
    return tuple(fds)


def rank_main(view: Path) -> int:
    if os.environ.get("WORLD_SIZE") != "2":
        raise CanaryError("torchrun world size differs")
    if not view.is_absolute() or not view.is_dir():
        raise CanaryError("FD view differs")
    leaves = sorted(path for path in view.rglob("*") if path.is_symlink())
    if len(leaves) != 23:
        raise CanaryError("rank did not observe exact-23 leaves")
    fds = inherited_fds(expected_inheritable=True)
    for fd in fds:
        os.set_inheritable(fd, False)
    inherited_fds(expected_inheritable=False)
    prefix = "/proc/self/fd/"
    for leaf in leaves:
        target = os.readlink(leaf)
        if not target.startswith(prefix):
            raise CanaryError("FD-view leaf target differs")
        descriptor = os.open(leaf, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                raise CanaryError("rank leaf identity differs")
        finally:
            os.close(descriptor)

    from transformers import AutoTokenizer, T5Config
    from diffusers.models import AutoencoderKLWan
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(
        str(view / "tokenizer"), local_files_only=True
    )
    if tokenizer.padding_side != "right":
        raise CanaryError("Transformers tokenizer resolution differs")
    text_config = T5Config.from_pretrained(
        str(view / "text_encoder"), local_files_only=True
    )
    if int(text_config.d_model) <= 0:
        raise CanaryError("Transformers config resolution differs")
    vae_config = AutoencoderKLWan.load_config(
        str(view), subfolder="vae", local_files_only=True
    )
    if int(vae_config["z_dim"]) <= 0:
        raise CanaryError("Diffusers VAE config resolution differs")

    tensor_files = sorted(view.rglob("*.safetensors"))
    if len(tensor_files) != 8:
        raise CanaryError("safetensors closure differs")
    tensor_key_counts: dict[str, int] = {}
    for tensor_file in tensor_files:
        with safe_open(str(tensor_file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if not keys:
            raise CanaryError("safetensors header has no tensors")
        tensor_key_counts[tensor_file.relative_to(view).as_posix()] = len(keys)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "rank": int(os.environ["RANK"]),
        "world_size": 2,
        "fd_binding_count": len(fds),
        "rank_fds_sealed_cloexec_before_library_load": True,
        "leaf_count": len(leaves),
        "transformers_tokenizer_loaded": True,
        "transformers_text_config_loaded": True,
        "diffusers_vae_config_loaded": True,
        "safetensors_headers_loaded": tensor_key_counts,
        "gpu_accessed": False,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def torchrun_wrapper_main(args: argparse.Namespace) -> int:
    fds = inherited_fds(expected_inheritable=True)
    for fd in fds:
        os.set_inheritable(fd, False)
    inherited_fds(expected_inheritable=False)

    import torch.distributed.elastic.multiprocessing.api as api
    import torch.distributed.elastic.multiprocessing.subprocess_handler as package
    import torch.distributed.elastic.multiprocessing.subprocess_handler.handlers as handlers
    import torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler as implementation
    import torch.distributed.run as torchrun

    handler_path = Path(implementation.__file__).resolve(strict=True)
    if sha256_file(handler_path) != TORCHRUN_HANDLER_SHA256:
        raise CanaryError("torchrun subprocess handler SHA differs")
    if not (
        package.SubprocessHandler
        is handlers.SubprocessHandler
        is api.SubprocessHandler
        is implementation.SubprocessHandler
    ):
        raise CanaryError("torchrun subprocess handler aliases differ")
    original_class = implementation.SubprocessHandler

    def exact_popen(self: Any, child_args: tuple, environment: dict[str, str]) -> Any:
        inherited_fds(expected_inheritable=False)
        process = implementation.subprocess.Popen(
            args=child_args,
            env=environment,
            stdout=self._stdout,
            stderr=self._stderr,
            start_new_session=True,
            close_fds=True,
            pass_fds=fds,
        )
        inherited_fds(expected_inheritable=False)
        return process

    original_class._popen = exact_popen
    sys.argv = [
        str(Path(torchrun.__file__).resolve(strict=True)),
        "--standalone",
        "--nproc_per_node=2",
        "--max-restarts=0",
        "--no-python",
        args.python,
        str(Path(__file__).resolve(strict=True)),
        "--rank",
        "--view",
        str(Path(args.view).resolve(strict=True)),
    ]
    torchrun.main()
    return 0


def parent_main(args: argparse.Namespace) -> int:
    if os.environ.get("SLURM_JOB_ID"):
        raise CanaryError("CPU canary must run outside a Slurm allocation")
    model = Path(args.model).resolve(strict=True)
    manifest = Path(args.manifest).resolve(strict=True)
    rows = manifest_rows(manifest)
    descriptors: list[int] = []
    directory = Path(tempfile.mkdtemp(prefix="apv2-eval-exact23-fd-canary-"))
    view = directory / "model"
    try:
        view.mkdir(mode=0o700)
        for relative, _ in rows:
            source = model / relative
            if source.is_symlink() or not source.is_file():
                raise CanaryError(f"model member is not plain: {relative}")
            descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            info = os.fstat(descriptor)
            named = source.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (info.st_dev, info.st_ino, info.st_size)
                != (named.st_dev, named.st_ino, named.st_size)
            ):
                os.close(descriptor)
                raise CanaryError(f"model member identity differs: {relative}")
            descriptors.append(descriptor)
            leaf = view / relative
            leaf.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            leaf.symlink_to(f"/proc/self/fd/{descriptor}")
            # Production capture performs two full sequential hashes and thus
            # leaves the retained descriptor at EOF.  Prove that opening the
            # Linux proc-FD magic link gives loaders an independently usable
            # description rather than relying on the parent's current offset.
            os.lseek(descriptor, 0, os.SEEK_END)

        rank_script = directory / "rank.py"
        rank_script.write_bytes(Path(__file__).read_bytes())
        rank_script.chmod(0o500)
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                FD_ENV: json.dumps(
                    descriptors, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        command = [
            args.python,
            "-B",
            str(rank_script),
            "--torchrun-wrapper",
            "--python",
            args.python,
            "--view",
            str(view),
            "--rank-script",
            str(rank_script),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            shell=False,
            close_fds=True,
            pass_fds=tuple(descriptors),
            env=environment,
        )
        if completed.returncode != 0:
            raise CanaryError(
                "torchrun canary failed: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        output = completed.stdout.decode("utf-8", errors="strict")
        # torchrun may multiplex two rank writes into one physical line.  Parse
        # concatenated JSON objects instead of treating newline boundaries as
        # an authority boundary.
        rank_rows: list[dict[str, Any]] = []
        decoder = json.JSONDecoder()
        offset = 0
        while offset < len(output):
            start = output.find("{", offset)
            if start < 0:
                break
            try:
                candidate, end = decoder.raw_decode(output, start)
            except json.JSONDecodeError:
                offset = start + 1
                continue
            offset = end
            if (
                isinstance(candidate, dict)
                and candidate.get("schema_version") == SCHEMA
            ):
                rank_rows.append(candidate)
        if len(rank_rows) != 2 or {row["rank"] for row in rank_rows} != {0, 1}:
            raise CanaryError("torchrun rank receipt closure differs")
        result = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "manifest_sha256": MANIFEST_SHA256,
            "model_member_count": 23,
            "directory_count": 7,
            "inherited_fd_count": 23,
            "parent_fds_non_inheritable_after_torchrun": all(
                not os.get_inheritable(fd) for fd in descriptors
            ),
            "torchrun_handler_sha256": TORCHRUN_HANDLER_SHA256,
            "torchrun_rank_count": 2,
            "rank_receipts": rank_rows,
            "gpu_accessed": False,
            "slurm_allocation_used": False,
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        for descriptor in descriptors:
            os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--rank", action="store_true")
    value.add_argument("--torchrun-wrapper", action="store_true")
    value.add_argument("--view")
    value.add_argument("--rank-script")
    value.add_argument("--model")
    value.add_argument("--manifest")
    value.add_argument("--python")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.rank:
        if args.view is None:
            raise CanaryError("rank arguments differ")
        return rank_main(Path(args.view))
    if args.torchrun_wrapper:
        if None in (args.view, args.rank_script, args.python):
            raise CanaryError("torchrun wrapper arguments differ")
        return torchrun_wrapper_main(args)
    if None in (args.model, args.manifest, args.python):
        raise CanaryError("parent arguments differ")
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
