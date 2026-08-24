#!/usr/bin/env python3
"""Materialize rank0-only low/high/negative prompt provenance for activation v2.

This authoring utility does not sample a video.  It validates the fixed e02/e03
caption identity before distributed initialization, tokenizes on rank zero,
loads the UMT5 text encoder only on rank zero, encodes exactly three prompts,
retires the text encoder, and broadcasts the BF16 embeddings byte-exactly over
one WORLD4/SP4 allocation.  The output is factual diagnostic material; it is
not a gate, training authority, or execution capability.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import timedelta
import gc
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
while str(METHOD_ROOT) in sys.path:
    sys.path.remove(str(METHOD_ROOT))
sys.path.insert(0, str(METHOD_ROOT))
_existing_activation = sys.modules.get("oracle_regeneration_activation_v2")
if _existing_activation is not None and Path(
    str(getattr(_existing_activation, "__file__", ""))
).resolve(strict=True) != (METHOD_ROOT / "oracle_regeneration_activation_v2.py").resolve(
    strict=True
):
    raise RuntimeError("preloaded activation-v2 origin differs")
_activation_spec = importlib.util.find_spec("oracle_regeneration_activation_v2")
if (
    _activation_spec is None
    or not isinstance(_activation_spec.origin, str)
    or Path(_activation_spec.origin).resolve(strict=True)
    != (METHOD_ROOT / "oracle_regeneration_activation_v2.py").resolve(strict=True)
):
    raise RuntimeError("activation-v2 prompt import origin differs")

import oracle_regeneration_activation_v2 as activation  # noqa: E402


SCHEMA_VERSION = "bernini-oracle-regeneration-activation-v2-prompt-authoring-run-v1"
WORLD_SIZE = 4
EXPECTED_PYTHON_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
EXPECTED_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
PROMPT_NAMES = ("low_action", "high_action", "negative")
PROMPT_MODES = {
    "low_action": "low-vr2v",
    "high_action": "high-r2v4",
    "negative": "renderer-negative",
}
PROMPT_SHAPE = (1, 512, 4096)


class PromptMaterializationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tensor_identity(value: Any) -> Mapping[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "content_sha256": activation.safe_core.tensor_content_sha256_v1(value),
    }


def _fresh_output_dir(value: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise PromptMaterializationError("output-dir must be absolute and non-root")
    parent = requested.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise PromptMaterializationError("output parent must be a plain directory")
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise PromptMaterializationError("refusing to overwrite output-dir")
    return output


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.name in ("", ".", "..") or "/" in path.name:
        raise PromptMaterializationError("receipt basename differs")
    directory = path.parent.resolve(strict=True)
    if directory != path.parent or directory.is_symlink():
        raise PromptMaterializationError("receipt directory differs")
    temporary_name = f".{path.name}.tmp-pid-{os.getpid()}"
    payload = activation.safe_core.canonical_json_bytes_v1(value)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(str(directory), directory_flags)
    descriptor: Optional[int] = None
    published = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.fsync(directory_fd)
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        check_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(check_fd)
            restored = b""
            while True:
                chunk = os.read(check_fd, 1024 * 1024)
                if not chunk:
                    break
                restored += chunk
            after = os.fstat(check_fd)
        finally:
            os.close(check_fd)
        if (
            restored != payload
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise PromptMaterializationError("published receipt identity differs")
    except FileExistsError as error:
        raise PromptMaterializationError(
            "receipt destination appeared; refusing overwrite"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _trim_host_allocator() -> bool:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = libc.malloc_trim
    except (AttributeError, OSError):
        return False
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return bool(malloc_trim(0))


def _retire_text_encoder(model: Any, *, torch_module: Any) -> bool:
    encoder = getattr(model, "t5_text_encoder", None)
    if encoder is None:
        raise PromptMaterializationError("T5 text encoder is absent before retirement")
    model.t5_text_encoder = None
    del encoder
    trimmed = _trim_host_allocator()
    torch_module.cuda.empty_cache()
    if getattr(model, "t5_text_encoder", None) is not None:
        raise PromptMaterializationError("T5 text encoder retirement failed")
    return trimmed


@contextmanager
def _nonzero_rank_t5_load_bypass(
    *,
    rank: int,
    t5_encoder_class: Any,
    checkpoint: Path,
    dtype: Any,
    placeholder_factory: Any,
) -> Any:
    """Exact scoped copy of the audited constructor-time nonzero-rank bypass."""

    if rank < 0 or rank >= WORLD_SIZE:
        raise PromptMaterializationError("prompt rank is outside WORLD4")
    audit: dict[str, Any] = {
        "rank": rank,
        "real_t5_load": rank == 0,
        "bypassed_t5_load": rank != 0,
        "call_count": 0,
        "placeholder": None,
    }
    if rank == 0:
        yield audit
        return
    own_descriptor = vars(t5_encoder_class).get("from_pretrained")
    had_own_descriptor = "from_pretrained" in vars(t5_encoder_class)

    def bypassed_from_pretrained(cls: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            cls is not t5_encoder_class
            or len(args) != 1
            or str(args[0]) != str(checkpoint)
            or kwargs.get("subfolder") != "text_encoder"
            or kwargs.get("torch_dtype") != dtype
            or set(kwargs) != {"subfolder", "torch_dtype"}
        ):
            raise PromptMaterializationError("Bernini T5 constructor ABI differs")
        audit["call_count"] += 1
        if audit["call_count"] != 1:
            raise PromptMaterializationError("Bernini loaded T5 more than once")
        placeholder = placeholder_factory()
        audit["placeholder"] = placeholder
        return placeholder

    setattr(t5_encoder_class, "from_pretrained", classmethod(bypassed_from_pretrained))
    try:
        yield audit
    finally:
        if had_own_descriptor:
            setattr(t5_encoder_class, "from_pretrained", own_descriptor)
        else:
            delattr(t5_encoder_class, "from_pretrained")
    if audit["call_count"] != 1 or audit["placeholder"] is None:
        raise PromptMaterializationError("nonzero-rank T5 bypass was not exercised")


def _all_rank_digest(value: Any, *, dist: Any) -> list[str]:
    local = activation.safe_core.tensor_content_sha256_v1(value)
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    if any(row != local for row in rows):
        raise PromptMaterializationError("prompt embedding differs across WORLD4")
    return [str(row) for row in rows]


def _encode_and_broadcast_prompts(
    model: Any,
    *,
    tokenized: Optional[Mapping[str, tuple[Any, Any]]],
    rank: int,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> tuple[Mapping[str, Any], bool]:
    if rank != 0:
        if tokenized is not None:
            raise PromptMaterializationError("nonzero rank received prompt tokens")
        _retire_text_encoder(model, torch_module=torch_module)
    dist.barrier()
    local: dict[str, Any] = {}
    status: list[Any] = [None]
    trimmed = False
    if rank == 0:
        try:
            if not isinstance(tokenized, Mapping) or tuple(tokenized) != PROMPT_NAMES:
                raise PromptMaterializationError("rank-zero prompt token bank differs")
            model.t5_text_encoder.to(device)
            with torch_module.inference_mode():
                for name in PROMPT_NAMES:
                    ids, mask = tokenized[name]
                    value = model.encode_prompt(ids.to(device), mask.to(device)).contiguous()
                    if value.dtype != torch_module.bfloat16 or tuple(value.shape) != PROMPT_SHAPE:
                        raise PromptMaterializationError(
                            f"prompt embedding contract differs for {name}"
                        )
                    local[name] = value
            status[0] = {"ok": True}
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        finally:
            trimmed = _retire_text_encoder(model, torch_module=torch_module)
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise PromptMaterializationError(f"rank-zero prompt encoding failed: {status[0]}")
    output: dict[str, Any] = {}
    for name in PROMPT_NAMES:
        value = (
            local[name]
            if rank == 0
            else torch_module.empty(
                PROMPT_SHAPE, dtype=torch_module.bfloat16, device=device
            )
        )
        dist.broadcast(value, src=0)
        value = value.detach().contiguous()
        if value.dtype != torch_module.bfloat16 or tuple(value.shape) != PROMPT_SHAPE:
            raise PromptMaterializationError(f"broadcast prompt differs for {name}")
        output[name] = value
    return output, trimmed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=activation.ALLOWED_CASES, required=True)
    parser.add_argument("--source-iid", required=True)
    parser.add_argument("--action-caption", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    expected_case = activation.EXPECTED_CASE_BINDINGS[args.case_id]
    if (
        args.source_iid != expected_case["source_iid"]
        or _text_sha256(args.action_caption) != expected_case["action_caption_sha256"]
    ):
        raise PromptMaterializationError("source iid or action caption differs")
    output = _fresh_output_dir(args.output_dir)
    checkpoint_manifest = activation._plain_absolute_file(
        args.checkpoint_content_manifest, label="checkpoint content manifest"
    )
    tool_path = Path(__file__).resolve(strict=True)
    python_path = Path(sys.executable).resolve(strict=True)
    if (
        python_path != EXPECTED_PYTHON_PATH
        or python_path.is_symlink()
        or _sha256_file(python_path) != EXPECTED_PYTHON_SHA256
    ):
        raise PromptMaterializationError("Python runtime identity differs")
    activation.verify_frozen_dependency_pins_v2()

    import infer_native_branch_homotopy_canary as prompt_builder
    import infer_native_identity_generation_canary as native
    import infer_source_kv_carrier_oracle as source_audit

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
        inference_hashes = native.legacy.validate_inference_source_files(bernini_root)
        checkpoint_identity = source_audit.validate_checkpoint_content(
            checkpoint, checkpoint_manifest
        )
    except Exception as error:
        raise PromptMaterializationError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % WORLD_SIZE:
        raise PromptMaterializationError("transformer heads do not divide WORLD4")
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel
    from transformers import __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.training.data import SYSTEM_PROMPTS

    if (
        SYSTEM_PROMPTS.get("r2v") != native.TASK_SYSTEM_PROMPTS["r2v"]
        or SYSTEM_PROMPTS.get("vr2v") != native.TASK_SYSTEM_PROMPTS["vr2v"]
        or DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT
    ):
        raise PromptMaterializationError("runtime prompt constants differ")
    low_text = prompt_builder.build_mode_native_prompt(
        "low-vr2v", args.action_caption, prompt_cleaner=prompt_clean
    )
    high_text = prompt_builder.build_mode_native_prompt(
        "high-r2v4", args.action_caption, prompt_cleaner=prompt_clean
    )
    rendered = {
        "low_action": low_text,
        "high_action": high_text,
        "negative": DEFAULT_NEG_PROMPT,
    }
    if len(set(rendered.values())) != 3:
        raise PromptMaterializationError("low/high/negative rendered prompts alias")

    paths = {
        "tokenizer_config": activation._plain_absolute_file(
            str(checkpoint / "tokenizer/tokenizer_config.json"),
            label="tokenizer config",
        ),
        "tokenizer_code": activation._plain_absolute_file(
            str(Path(native.legacy.__file__).resolve(strict=True)),
            label="tokenizer code",
        ),
        "checkpoint_content_manifest": checkpoint_manifest,
        "text_encoder_config": activation._plain_absolute_file(
            str(checkpoint / "text_encoder/config.json"),
            label="text encoder config",
        ),
        "renderer_code": activation._plain_absolute_file(
            str(bernini_root / "bernini/models/renderer.py"),
            label="renderer code",
        ),
        "prompt_builder_code": activation._plain_absolute_file(
            str(Path(prompt_builder.__file__).resolve(strict=True)),
            label="prompt builder code",
        ),
        "native_prompt_code": activation._plain_absolute_file(
            str(Path(native.__file__).resolve(strict=True)),
            label="native prompt code",
        ),
        "prompt_cleaner_code": activation._plain_absolute_file(
            str(Path(prompt_clean.__code__.co_filename).resolve(strict=True)),
            label="prompt cleaner code",
        ),
        "auto_tokenizer_module": activation._plain_absolute_file(
            str(Path(inspect.getfile(AutoTokenizer)).resolve(strict=True)),
            label="Transformers AutoTokenizer implementation",
        ),
        "text_encoder_class_module": activation._plain_absolute_file(
            str(Path(inspect.getfile(UMT5EncoderModel)).resolve(strict=True)),
            label="Transformers UMT5 implementation",
        ),
        "python_executable": python_path,
        "materializer_code": tool_path,
    }
    path_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    checkpoint_identity_sha = activation._canonical_object_sha256(checkpoint_identity)

    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != WORLD_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise PromptMaterializationError("prompt materializer requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=WORLD_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    tokenized: Optional[Mapping[str, tuple[Any, Any]]] = None
    token_metadata: list[Any] = [None]
    if distributed.rank == 0:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint),
                subfolder="tokenizer",
                **native.legacy.tokenizer_load_kwargs(),
            )
            resolved_tokenizer_path = activation._plain_absolute_file(
                str(Path(inspect.getfile(type(tokenizer))).resolve(strict=True)),
                label="resolved tokenizer class implementation",
            )
            tokenized = {
                "low_action": native.legacy._tokenize_training_prompt(
                    tokenizer, low_text
                ),
                "high_action": native.legacy._tokenize_training_prompt(
                    tokenizer, high_text
                ),
                "negative": native.legacy._tokenize_renderer_negative(
                    tokenizer, DEFAULT_NEG_PROMPT
                ),
            }
            token_metadata[0] = {
                "ok": True,
                "roles": {
                    name: {
                        "token_ids_sha256": activation.safe_core.tensor_content_sha256_v1(
                            pair[0]
                        ),
                        "attention_mask_sha256": activation.safe_core.tensor_content_sha256_v1(
                            pair[1]
                        ),
                    }
                    for name, pair in tokenized.items()
                },
                "resolved_tokenizer_class_module_path": str(
                    resolved_tokenizer_path
                ),
                "resolved_tokenizer_class_module_sha256": _sha256_file(
                    resolved_tokenizer_path
                ),
            }
            del tokenizer
        except Exception as error:
            token_metadata[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(token_metadata, src=0)
    if not isinstance(token_metadata[0], Mapping) or token_metadata[0].get("ok") is not True:
        raise PromptMaterializationError(f"rank-zero tokenization failed: {token_metadata[0]}")
    resolved_tokenizer_path = activation._plain_absolute_file(
        token_metadata[0].get("resolved_tokenizer_class_module_path"),
        label="broadcast resolved tokenizer class implementation",
    )
    resolved_tokenizer_sha256 = token_metadata[0].get(
        "resolved_tokenizer_class_module_sha256"
    )
    if (
        not isinstance(resolved_tokenizer_sha256, str)
        or _sha256_file(resolved_tokenizer_path) != resolved_tokenizer_sha256
    ):
        raise PromptMaterializationError("resolved tokenizer implementation differs")
    paths["resolved_tokenizer_class_module"] = resolved_tokenizer_path
    path_hashes["resolved_tokenizer_class_module"] = resolved_tokenizer_sha256

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise PromptMaterializationError("renderer prompt container contract differs")
    with _nonzero_rank_t5_load_bypass(
        rank=distributed.rank,
        t5_encoder_class=UMT5EncoderModel,
        checkpoint=checkpoint,
        dtype=torch.bfloat16,
        placeholder_factory=torch.nn.Identity,
    ) as load_audit:
        model = BerniniRendererModel(config)
    if distributed.rank == 0:
        if not isinstance(model.t5_text_encoder, UMT5EncoderModel):
            raise PromptMaterializationError("rank zero did not load the real T5 encoder")
    elif model.t5_text_encoder is not load_audit["placeholder"]:
        raise PromptMaterializationError("nonzero rank retained a non-placeholder T5")
    model.eval().requires_grad_(False)
    load_rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        load_rows,
        {
            "rank": distributed.rank,
            "real_t5_loaded": distributed.rank == 0,
            "bypassed_t5_load": bool(load_audit["bypassed_t5_load"]),
            "bypass_call_count": int(load_audit["call_count"]),
            "placeholder_retained": distributed.rank != 0,
        },
    )
    expected_load_rows = [
        {
            "rank": rank,
            "real_t5_loaded": rank == 0,
            "bypassed_t5_load": rank != 0,
            "bypass_call_count": 0 if rank == 0 else 1,
            "placeholder_retained": rank != 0,
        }
        for rank in range(WORLD_SIZE)
    ]
    if load_rows != expected_load_rows:
        raise PromptMaterializationError("WORLD4 T5 load closure differs")

    prompt_bank, rank0_trimmed = _encode_and_broadcast_prompts(
        model,
        tokenized=tokenized if distributed.rank == 0 else None,
        rank=distributed.rank,
        device=device,
        dist=dist,
        torch_module=torch,
    )
    del tokenized, model
    _trim_host_allocator()
    torch.cuda.empty_cache()
    values = tuple(prompt_bank[name] for name in PROMPT_NAMES)
    if (
        any(
            value.dtype != torch.bfloat16
            or tuple(value.shape) != PROMPT_SHAPE
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
            for value in values
        )
        or len({activation.safe_core.tensor_content_sha256_v1(value) for value in values})
        != 3
    ):
        raise PromptMaterializationError("materialized prompt tensor contract differs")
    try:
        activation.safe_core._require_pairwise_storage_disjoint_v1(values)
    except Exception as error:
        raise PromptMaterializationError(str(error)) from error
    identities = {name: _tensor_identity(prompt_bank[name]) for name in PROMPT_NAMES}
    all_rank = {
        name: _all_rank_digest(prompt_bank[name], dist=dist) for name in PROMPT_NAMES
    }
    if any(_sha256_file(path) != path_hashes[name] for name, path in paths.items()):
        raise PromptMaterializationError("prompt materializer dependency changed")
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_after = checkpoint_rows[0]
    if (
        not isinstance(checkpoint_after, Mapping)
        or checkpoint_after.get("ok") is not True
        or activation._canonical_object_sha256(checkpoint_after.get("identity"))
        != checkpoint_identity_sha
    ):
        raise PromptMaterializationError("checkpoint content changed during prompt materialization")

    if distributed.rank == 0:
        output.mkdir(mode=0o755)
        token_roles = token_metadata[0]["roles"]
        role_rows = {
            name: {
                "mode": PROMPT_MODES[name],
                "rendered_text": rendered[name],
                "rendered_text_sha256": _text_sha256(rendered[name]),
                "token_ids_sha256": token_roles[name]["token_ids_sha256"],
                "attention_mask_sha256": token_roles[name]["attention_mask_sha256"],
                "embedding_identity": identities[name],
            }
            for name in PROMPT_NAMES
        }
        receipt = {
            "schema_version": activation.PROMPT_RECEIPT_SCHEMA_VERSION,
            "case_id": args.case_id,
            "source_iid": args.source_iid,
            "action_caption": args.action_caption,
            "action_caption_sha256": _text_sha256(args.action_caption),
            "prompt_contract": {
                "tokenizer_config_path": str(paths["tokenizer_config"]),
                "tokenizer_config_sha256": path_hashes["tokenizer_config"],
                "tokenizer_code_path": str(paths["tokenizer_code"]),
                "tokenizer_code_sha256": path_hashes["tokenizer_code"],
                "checkpoint_content_manifest_path": str(
                    paths["checkpoint_content_manifest"]
                ),
                "checkpoint_content_manifest_sha256": path_hashes[
                    "checkpoint_content_manifest"
                ],
                "checkpoint_content_identity_sha256": checkpoint_identity_sha,
                "text_encoder_config_path": str(paths["text_encoder_config"]),
                "text_encoder_config_sha256": path_hashes["text_encoder_config"],
                "renderer_code_path": str(paths["renderer_code"]),
                "renderer_code_sha256": path_hashes["renderer_code"],
                "prompt_builder_code_path": str(paths["prompt_builder_code"]),
                "prompt_builder_code_sha256": path_hashes["prompt_builder_code"],
                "native_prompt_code_path": str(paths["native_prompt_code"]),
                "native_prompt_code_sha256": path_hashes["native_prompt_code"],
                "prompt_cleaner_code_path": str(paths["prompt_cleaner_code"]),
                "prompt_cleaner_code_sha256": path_hashes["prompt_cleaner_code"],
                "auto_tokenizer_module_path": str(paths["auto_tokenizer_module"]),
                "auto_tokenizer_module_sha256": path_hashes[
                    "auto_tokenizer_module"
                ],
                "resolved_tokenizer_class_module_path": str(
                    paths["resolved_tokenizer_class_module"]
                ),
                "resolved_tokenizer_class_module_sha256": path_hashes[
                    "resolved_tokenizer_class_module"
                ],
                "text_encoder_class_module_path": str(
                    paths["text_encoder_class_module"]
                ),
                "text_encoder_class_module_sha256": path_hashes[
                    "text_encoder_class_module"
                ],
                "transformers_version": str(transformers_version),
                "torch_version": str(torch.__version__),
                "python_executable_path": str(paths["python_executable"]),
                "python_executable_sha256": path_hashes["python_executable"],
                "python_version": str(sys.version),
                "rocm_version": str(torch.version.hip),
                "tokenizer_function": "infer_lora._tokenize_training_prompt+_tokenize_renderer_negative",
                "text_encoder_function": "bernini.models.renderer.BerniniRendererModel.encode_prompt",
                "max_length": 512,
                "embedding_dtype": "torch.bfloat16",
            },
            **role_rows,
            "materializer_code_path": str(paths["materializer_code"]),
            "materializer_code_sha256": path_hashes["materializer_code"],
            "rank_world_receipt": {
                "world_size": WORLD_SIZE,
                "sequence_parallel_size": WORLD_SIZE,
                "rank0_only_text_encode": True,
                "all_rank_text_encoder_load_roles": load_rows,
                "broadcast_exact": True,
                "all_rank_low_action_sha256": all_rank["low_action"],
                "all_rank_high_action_sha256": all_rank["high_action"],
                "all_rank_negative_sha256": all_rank["negative"],
            },
            "rank0_only_text_encoder_load": True,
            "nonzero_ranks_never_deserialized_text_encoder": True,
            "self_generated_anchor_tensor_used": False,
            "target_video_or_latent_used": False,
            "materialization_checks_passed": True,
        }
        _write_receipt(output / "prompt-receipt.json", receipt)
        run_receipt = {
            "schema_version": SCHEMA_VERSION,
            "prompt_receipt_sha256": _sha256_file(output / "prompt-receipt.json"),
            "source_tree": {
                "bernini_root": str(bernini_root),
                "veomni_root": str(veomni_root),
                "bernini_revision": str(bernini_revision),
                "veomni_revision": str(veomni_revision),
            },
            "inference_source_hashes": inference_hashes,
            "checkpoint_content_identity_sha256": checkpoint_identity_sha,
            "renderer_container_constructed": True,
            "denoising_transformer_moved_to_gpu": False,
            "sampler_or_scheduler_called": False,
            "rank0_text_encoder_retirement_trimmed_host_allocator": rank0_trimmed,
            "training": False,
            "optimizer": False,
            "diagnostic_authoring_material_only": True,
        }
        _write_receipt(output / "run-receipt.json", run_receipt)
        output.chmod(0o555)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
