#!/usr/bin/env python3
"""Run one source-only shared-8 sample through an audited baseline entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from shared8_contract import (
    Shared8ContractError,
    assert_no_privileged_cli,
    atomic_write_json,
    build_output_receipt,
    ensure_empty_directory,
    file_sha256,
    load_input_manifest,
    object_sha256,
    probe_video,
    require_81f25,
    require_sha256,
    source_aspect_bucket,
)


MODEL_IDS = {
    "lucy_official_base",
    "bernini_full644_lora_step644",
    "omnivideo2_official_base",
}
OMNI_OFFICIAL_REVISION = "adcee0a4a5b439ad3615f825298221b21177d4e3"
OMNI_TRANSFORMER_SHA256 = "f269fe8c6b35993bbb4ea340c535ee9893928ea215fb8c4be3d5e9f122d844d6"
OMNI_SPECIAL_TOKENS_SHA256 = "72129ce9ade25aa0fbf738c005d3dc090c1b6c45918580e1c683b6ecef726ad4"
OMNI_T5_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
OMNI_VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
OMNI_QWEN_MANIFEST_SHA256 = "5a0c83274293904863d5aed304385e906afa02dc04d36e9b34ac8383775b78be"
OMNI_UMT5_TOKENIZER_MANIFEST_SHA256 = "99bec3d4e1a5c50694eb95d3fbec880d32bac8c0536c3dea8080b459f90e32cd"
OMNI_UNIFIED_MODEL_SHA256 = "739addc6b46c62fb389cf037736d52837d8e0fc911d84e3ad7db05e094ee7328"
OMNI_GENERATE_SHA256 = "f7cb67288d9a860a92fa0a548b139d6a15194b1a7db837b04889dc8d75b36be6"
BERNINI_BASE_TREE_SHA256 = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
BERNINI_ADAPTER_SHA256 = "9217ff653e47f915105fe8fa64856037d63811562cec1e9fd53ae9e4613a9774"
BERNINI_INFERENCE_REVISION = "1723e509ed3f63e76105359ee22ace5648a91a26"
BERNINI_INFERENCE_ARCHIVE_SHA256 = "d1648ba75f7c1e1b76ba8823cd1697b49f1ed0100a1b78014f9ebb7b10ed403d"
_CONTEXT_RE = re.compile(
    r"\[shared8-context\] vlm=(\d+) text=(\d+) visual=(\d+) total=(\d+) max=(\d+)"
)


def _directory(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser()
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise Shared8ContractError(f"cannot resolve {label} {source}: {error}") from error
    if not resolved.is_dir():
        raise Shared8ContractError(f"{label} is not a directory: {resolved}")
    return resolved


def _file(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser()
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise Shared8ContractError(f"cannot resolve {label} {source}: {error}") from error
    if not resolved.is_file():
        raise Shared8ContractError(f"{label} is not a file: {resolved}")
    return resolved


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    assert_no_privileged_cli(command)
    printable = json.dumps(command, ensure_ascii=False)
    print(f"[shared8] command={printable}", flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise Shared8ContractError(f"baseline command failed: {error}") from error


def _verify_hash(path: Path, expected: str, *, label: str) -> str:
    require_sha256(expected, label=f"expected {label} SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise Shared8ContractError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _lucy_command(args: argparse.Namespace, *, row: Any, source_probe: Any, output: Path) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    code_root = _directory(args.code_root, label="code root")
    checkpoint = _directory(args.lucy_checkpoint, label="Lucy checkpoint")
    entry = _file(code_root / "methods" / "Lucy-Edit" / "infer.py", label="Lucy entry")
    bucket_h, bucket_w = source_aspect_bucket(
        height=source_probe.height,
        width=source_probe.width,
        max_pixels=832 * 480,
        # Lucy's expanded per-token timesteps use strided slicing while the
        # Wan patch embed uses a stride-2 convolution.  Pixel dimensions that
        # are only divisible by 16 can therefore disagree when the VAE grid is
        # odd.  Multiples of 32 keep both tokenizations identical.
        stride=32,
    )
    command = [
        str(args.python_bin),
        str(entry),
        "--input-video",
        row.source_video,
        "--prompt",
        row.instruction,
        "--model-path",
        str(checkpoint),
        "--output",
        str(output),
        "--width",
        str(bucket_w),
        "--height",
        str(bucket_h),
        "--num-frames",
        "81",
        "--fps",
        "25",
        "--guidance-scale",
        "5",
        "--num-inference-steps",
        "50",
        "--seed",
        str(row.seed),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--vae-dtype",
        "float32",
        "--vae-enable-tiling",
        "--vae-enable-slicing",
    ]
    model_identity = {
        "family": "decart-ai/Lucy-Edit-1.1-Dev",
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256_preflight": args.lucy_checkpoint_tree_sha256,
        "checkpoint_tree_identity_algorithm": "sha256-size-relpath-v1_excluding_dot_cache",
        "checkpoint_huggingface_revision": "f12c3ab18266dcc1eb97f26b9102af42dfd327c5",
        "checkpoint_tree_runtime_rehash": False,
        "adaptation": "official_base_no_project_action_adapter",
    }
    sampler = {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "seed": row.seed,
        "pipeline_dtype": "bfloat16",
        "vae_dtype": "float32",
    }
    geometry = {
        "policy": "source_aspect_sqrt_max_pixels_then_floor_to_stride32",
        "max_pixels": 832 * 480,
        "required_pixel_stride": 32,
        "height": bucket_h,
        "width": bucket_w,
    }
    return command, model_identity, sampler, geometry


def _bernini_command(
    args: argparse.Namespace, *, row: Any, sample_dir: Path, output: Path
) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_archive = _file(
        args.bernini_inference_archive,
        label="frozen Bernini inference source archive",
    )
    _verify_hash(
        source_archive,
        BERNINI_INFERENCE_ARCHIVE_SHA256,
        label="frozen Bernini inference source archive",
    )
    runtime = sample_dir / "bernini_inference_runtime"
    runtime.mkdir()
    _safe_extract_tar(source_archive, runtime)
    entries = list(runtime.rglob("methods/bernini_action_editing/infer_lora.py"))
    if len(entries) != 1:
        raise Shared8ContractError(
            f"Bernini source archive contains {len(entries)} inference entries, expected 1"
        )
    entry = _file(entries[0], label="frozen Bernini inference entry")
    bernini_root = _directory(args.bernini_root, label="Bernini official source")
    veomni_root = _directory(args.veomni_root, label="VeOmni source")
    checkpoint = _directory(args.bernini_checkpoint, label="Bernini checkpoint")
    adapter = _directory(args.bernini_adapter, label="Bernini adapter checkpoint")
    adapter_model = _file(
        adapter / "adapter" / "adapter_model.safetensors",
        label="Bernini adapter",
    )
    _verify_hash(adapter_model, BERNINI_ADAPTER_SHA256, label="Bernini adapter")
    command = [
        str(args.python_bin),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "4",
        str(entry),
        "--bernini-root",
        str(bernini_root),
        "--veomni-root",
        str(veomni_root),
        "--checkpoint",
        str(checkpoint),
        "--adapter-checkpoint",
        str(adapter),
        "--source-video",
        row.source_video,
        "--instruction",
        row.instruction,
        "--output",
        str(output),
        "--num-inference-steps",
        "40",
        "--seed",
        str(row.seed),
        "--expected-checkpoint-tree-sha256",
        BERNINI_BASE_TREE_SHA256,
        "--method-source-revision",
        BERNINI_INFERENCE_REVISION,
        "--method-source-archive-sha256",
        BERNINI_INFERENCE_ARCHIVE_SHA256,
    ]
    model_identity = {
        "family": "Bernini-R-1.3B-Diffusers",
        "base_checkpoint_path": str(checkpoint),
        "base_checkpoint_tree_sha256": BERNINI_BASE_TREE_SHA256,
        "adapter_checkpoint_path": str(adapter),
        "adapter_sha256": BERNINI_ADAPTER_SHA256,
        "adapter_step": 644,
        "adaptation": "full644_attention_lora",
        "inference_revision": BERNINI_INFERENCE_REVISION,
        "inference_archive_path": str(source_archive),
        "inference_archive_sha256": BERNINI_INFERENCE_ARCHIVE_SHA256,
    }
    sampler = {
        "name": "v2v_apg_unipc",
        "num_inference_steps": 40,
        "flow_shift": 5.0,
        "seed": row.seed,
    }
    geometry = {
        "policy": "Bernini_training_exact_source_aspect_bucket",
        "max_pixels": 245760,
        "stride": 16,
    }
    return command, model_identity, sampler, geometry


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise Shared8ContractError(f"unsafe path in Omni source archive: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise Shared8ContractError(f"unsafe member in source archive: {member.name}")
        handle.extractall(destination)


def _replace_once(path: Path, old: str, new: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise Shared8ContractError(f"{label} patch anchor count is {count}, expected 1")
    path.write_text(text.replace(old, new), encoding="utf-8")


def prepare_strict_omni_runtime(
    *, official_root: Path, sample_dir: Path
) -> tuple[Path, dict[str, Any]]:
    revision = subprocess.run(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != OMNI_OFFICIAL_REVISION:
        raise Shared8ContractError(
            f"unexpected Omni source revision: {revision} != {OMNI_OFFICIAL_REVISION}"
        )
    runtime = sample_dir / "omni_runtime"
    archive = sample_dir / "omni_official_source.tar"
    if runtime.exists() or archive.exists():
        raise Shared8ContractError("Omni runtime/archive already exists")
    with archive.open("xb") as handle:
        subprocess.run(
            ["git", "-C", str(official_root), "archive", "--format=tar", "HEAD"],
            check=True,
            stdout=handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    runtime.mkdir()
    _safe_extract_tar(archive, runtime)

    unified = runtime / "omnivideo" / "modules" / "unified_model.py"
    generate = runtime / "tools" / "inference" / "generate_omni_v2v_1_3B.py"
    _verify_hash(unified, OMNI_UNIFIED_MODEL_SHA256, label="official Omni unified_model.py")
    _verify_hash(generate, OMNI_GENERATE_SHA256, label="official Omni generate entry")

    _replace_once(
        unified,
        """                    if self.max_context_len is not None and new_context.shape[0] > self.max_context_len:\n                        new_context = new_context[:self.max_context_len]\n""",
        """                    if visual_item is None or visual_item.shape[0] != 8190:\n                        actual_visual = 0 if visual_item is None else visual_item.shape[0]\n                        raise RuntimeError(\n                            f\"shared8 requires the full 81-frame visual condition: visual={actual_visual}, expected=8190\"\n                        )\n                    if self.max_context_len is not None:\n                        vlm_len = 0 if vlm_item is None else vlm_item.shape[0]\n                        text_len = 0 if context_item is None else context_item.shape[0]\n                        logging.info(\n                            f\"[shared8-context] vlm={vlm_len} text={text_len} visual={visual_item.shape[0]} \"\n                            f\"total={new_context.shape[0]} max={self.max_context_len}\"\n                        )\n                        if new_context.shape[0] > self.max_context_len:\n                            raise RuntimeError(\n                                f\"shared8 refuses context truncation: {new_context.shape[0]} > {self.max_context_len}\"\n                            )\n""",
        label="Omni no-context-truncation",
    )
    _replace_once(
        generate,
        """        missing, unexpected = omni_video.model.load_state_dict(state_dict, strict=False)\n        if rank == 0:\n            logging.info(\n                f\"[model] load_state_dict: missing_keys={len(missing):,}, unexpected_keys={len(unexpected):,}\"\n            )\n        del state_dict\n""",
        """        missing, unexpected = omni_video.model.load_state_dict(state_dict, strict=False)\n        if rank == 0:\n            logging.info(\n                f\"[model] load_state_dict: missing_keys={len(missing):,}, unexpected_keys={len(unexpected):,}\"\n            )\n        if missing or unexpected:\n            raise RuntimeError(\n                f\"shared8 requires strict-equivalent checkpoint load; missing={missing}, unexpected={unexpected}\"\n            )\n        logging.info(\"[shared8-checkpoint] strict_equivalent_load=true\")\n        del state_dict\n""",
        label="Omni strict-equivalent checkpoint load",
    )
    source_identity = {
        "official_revision": revision,
        "official_archive_sha256": file_sha256(archive),
        "official_unified_model_sha256": OMNI_UNIFIED_MODEL_SHA256,
        "official_generate_entry_sha256": OMNI_GENERATE_SHA256,
        "patched_unified_model_sha256": file_sha256(unified),
        "patched_generate_entry_sha256": file_sha256(generate),
        "runtime_patch_contract": [
            "raise_instead_of_context_truncation_and_log_exact_total",
            "require_zero_missing_and_unexpected_checkpoint_keys",
        ],
    }
    return runtime, source_identity


def _omni_command(args: argparse.Namespace, *, row: Any, source_probe: Any, sample_dir: Path, output: Path) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, str], Path]:
    official_root = _directory(args.omni_root, label="official Omni source")
    checkpoint = _directory(args.omni_checkpoint, label="Omni checkpoint")
    qwen = _directory(args.qwen_checkpoint, label="Qwen checkpoint")
    transformer = _file(
        checkpoint / "transformer" / "pytorch_model.pt",
        label="Omni transformer checkpoint",
    )
    special_tokens = _file(checkpoint / "special_tokens.pkl", label="Omni special tokens")
    t5_checkpoint = _file(
        checkpoint / "models_t5_umt5-xxl-enc-bf16.pth",
        label="Omni UMT5 checkpoint",
    )
    vae_checkpoint = _file(checkpoint / "Wan2.1_VAE.pth", label="Omni Wan VAE")
    _verify_hash(transformer, OMNI_TRANSFORMER_SHA256, label="Omni transformer")
    _verify_hash(special_tokens, OMNI_SPECIAL_TOKENS_SHA256, label="Omni special tokens")
    _verify_hash(t5_checkpoint, OMNI_T5_SHA256, label="Omni UMT5 checkpoint")
    _verify_hash(vae_checkpoint, OMNI_VAE_SHA256, label="Omni Wan VAE")
    runtime, runtime_identity = prepare_strict_omni_runtime(
        official_root=official_root,
        sample_dir=sample_dir,
    )
    prompt_file = sample_dir / "source_only_input.jsonl"
    with prompt_file.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "id": row.iid,
                "source_clip_path": row.source_video,
                "edit_prompt": row.instruction,
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    size = "480*832" if source_probe.height > source_probe.width else "832*480"
    command = [
        str(args.python_bin),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "1",
        str(runtime / "tools" / "inference" / "generate_omni_v2v_1_3B.py"),
        "--task",
        "v2v-1.3B",
        "--size",
        size,
        "--frame_num",
        "81",
        "--sample_fps",
        "25",
        "--sample_shift",
        "5",
        "--sample_solver",
        "unipc",
        "--sample_steps",
        "40",
        "--sample_guide_scale",
        "3",
        "--base_seed",
        str(row.seed),
        "--classifier_free_ratio",
        "0",
        "--ckpt_dir",
        str(checkpoint),
        "--new_checkpoint",
        str(transformer),
        "--sampling_rate",
        "1",
        "--skip_num",
        "0",
        "--use_usp",
        "false",
        "--sp_size",
        "1",
        "--t5_fsdp",
        "false",
        "--dit_fsdp",
        "false",
        "--max_context_len",
        str(args.omni_max_context_len),
        "--qwen3vl_dtype",
        "bf16",
        "--qwen3vl_device_map",
        "balanced_low_0",
        "--qwen3vl_temperature",
        "0",
        "--video_max_duration",
        "5",
        "--qwen3vl_model_path",
        str(qwen),
        "--prompt_file",
        str(prompt_file),
    ]
    model_identity = {
        "family": "Fudan-FUXI/OmniVideo2-1.3B",
        "checkpoint_path": str(checkpoint),
        "transformer_sha256": OMNI_TRANSFORMER_SHA256,
        "special_tokens_sha256": OMNI_SPECIAL_TOKENS_SHA256,
        "umt5_checkpoint_sha256": OMNI_T5_SHA256,
        "vae_checkpoint_sha256": OMNI_VAE_SHA256,
        "umt5_plus_tokenizer_prior_identity_manifest_sha256": OMNI_UMT5_TOKENIZER_MANIFEST_SHA256,
        "qwen_checkpoint_path": str(qwen),
        "qwen_prior_identity_manifest_sha256": OMNI_QWEN_MANIFEST_SHA256,
        "qwen_prior_identity_file_count": 25,
        "qwen_prior_identity_total_bytes": 62153207155,
        "adaptation": "official_base_no_project_MARP_adapter",
        **runtime_identity,
    }
    sampler = {
        "name": "unipc",
        "num_inference_steps": 40,
        "guidance_scale": 3.0,
        "flow_shift": 5.0,
        "seed": row.seed,
    }
    geometry = {
        "policy": "official_orientation_specific_center_crop_resize",
        "size": size,
        "frame_indices": list(range(81)),
        "sampling_rate": 1,
        "max_context_len": args.omni_max_context_len,
    }
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{runtime}:{old_pythonpath}" if old_pythonpath else str(runtime)
    candidate = sample_dir / "outputs" / f"source_id{row.iid}_edited.mp4"
    return command, model_identity, sampler, geometry, candidate, env, runtime


def _validate_bernini_child_receipt(
    output: Path, *, row: Any, source_sha256: str
) -> dict[str, Any]:
    path = output.with_name(f"{output.name}.receipt.json")
    child = json.loads(_file(path, label="Bernini child receipt").read_text(encoding="utf-8"))
    digest_payload = dict(child)
    stored_digest = digest_payload.pop("receipt_digest", None)
    require_sha256(stored_digest, label="Bernini child receipt digest")
    if object_sha256(digest_payload) != stored_digest:
        raise Shared8ContractError("Bernini child receipt digest mismatch")
    input_contract = child.get("input")
    if not isinstance(input_contract, dict):
        raise Shared8ContractError("Bernini child receipt has no input object")
    conditions = input_contract.get("accepted_model_conditions")
    if conditions not in (["source_video", "edit_instruction"], ["source_video", "instruction"]):
        raise Shared8ContractError("Bernini child receipt has unexpected model conditions")
    for key in (
        "target_video_argument",
        "target_accessed_by_inference",
        "external_mask_or_swept_tube",
        "external_tracking_pose_or_trajectory",
        "reference_image_or_video",
        "external_shared_i0",
    ):
        if input_contract.get(key) is not False:
            raise Shared8ContractError(f"Bernini child receipt does not prove {key}=false")
    if input_contract.get("source_video_sha256") != source_sha256:
        raise Shared8ContractError("Bernini child source-video hash mismatch")
    instruction_sha256 = hashlib.sha256(row.instruction.encode("utf-8")).hexdigest()
    if input_contract.get("instruction_utf8_sha256") != instruction_sha256:
        raise Shared8ContractError("Bernini child instruction hash mismatch")
    if child.get("method_source_revision") != BERNINI_INFERENCE_REVISION:
        raise Shared8ContractError("Bernini child source revision mismatch")
    if child.get("method_source_archive_sha256") != BERNINI_INFERENCE_ARCHIVE_SHA256:
        raise Shared8ContractError("Bernini child source archive mismatch")
    adapter = child.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("adapter_model_sha256") != BERNINI_ADAPTER_SHA256:
        raise Shared8ContractError("Bernini child adapter hash mismatch")
    child_output = child.get("output")
    if not isinstance(child_output, dict):
        raise Shared8ContractError("Bernini child receipt has no output object")
    if Path(child_output.get("path", "")).resolve(strict=True) != output.resolve(strict=True):
        raise Shared8ContractError("Bernini child output path mismatch")
    if child_output.get("sha256") != file_sha256(output):
        raise Shared8ContractError("Bernini child output hash mismatch")
    return {
        "child_receipt_path": str(path),
        "child_receipt_sha256": file_sha256(path),
        "child_receipt_digest": stored_digest,
    }


def _validate_omni_runtime(sample_dir: Path, max_context_len: int) -> dict[str, Any]:
    log = _file(sample_dir / "log_0.log", label="Omni rank-zero log")
    text = log.read_text(encoding="utf-8", errors="replace")
    if "[shared8-checkpoint] strict_equivalent_load=true" not in text:
        raise Shared8ContractError("Omni log does not prove strict-equivalent checkpoint load")
    matches = [tuple(map(int, values)) for values in _CONTEXT_RE.findall(text)]
    if not matches:
        raise Shared8ContractError("Omni log has no exact context accounting")
    if any(
        vlm <= 0 or text <= 0 or visual != 8190 or limit != max_context_len or total > limit
        for vlm, text, visual, total, limit in matches
    ):
        raise Shared8ContractError("Omni context was over budget or used an unexpected limit")
    unique_totals = sorted({total for _vlm, _text, _visual, total, _limit in matches})
    return {
        "rank_zero_log_path": str(log),
        "rank_zero_log_sha256": file_sha256(log),
        "strict_equivalent_checkpoint_load": True,
        "context_truncation": False,
        "visual_condition_tokens": 8190,
        "context_total_tokens": unique_totals,
        "max_context_len": max_context_len,
    }


def run(args: argparse.Namespace) -> int:
    if args.model_id not in MODEL_IDS:
        raise Shared8ContractError(f"unsupported model ID: {args.model_id}")
    require_sha256(args.manifest_sha256, label="manifest SHA-256")
    require_sha256(args.source_archive_sha256, label="source archive SHA-256")
    manifest, rows = load_input_manifest(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        require_media=True,
    )
    if not 0 <= args.index < len(rows):
        raise Shared8ContractError(f"sample index out of range: {args.index}")
    row = rows[args.index]
    manifest_sha256_at_start = file_sha256(manifest)
    source_sha256_at_start = file_sha256(row.source_video)
    source_probe = probe_video(row.source_video, ffprobe=args.ffprobe)
    require_81f25(source_probe, label=f"source {row.iid}")

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        raise Shared8ContractError("output root must be absolute")
    sample_dir = ensure_empty_directory(
        output_root / args.model_id / f"sample_{row.index:03d}_{row.iid}"
    )
    output = sample_dir / "output.mp4"
    runtime_evidence: dict[str, Any] | None = None

    if args.model_id == "lucy_official_base":
        require_sha256(
            args.lucy_checkpoint_tree_sha256,
            label="Lucy checkpoint-tree SHA-256",
        )
        command, model_identity, sampler, geometry = _lucy_command(
            args,
            row=row,
            source_probe=source_probe,
            output=output,
        )
        _run(command)
    elif args.model_id == "bernini_full644_lora_step644":
        command, model_identity, sampler, geometry = _bernini_command(
            args,
            row=row,
            sample_dir=sample_dir,
            output=output,
        )
        _run(command)
        runtime_evidence = _validate_bernini_child_receipt(
            output,
            row=row,
            source_sha256=source_sha256_at_start,
        )
    else:
        (
            command,
            model_identity,
            sampler,
            geometry,
            candidate,
            environment,
            _runtime,
        ) = _omni_command(
            args,
            row=row,
            source_probe=source_probe,
            sample_dir=sample_dir,
            output=output,
        )
        _run(command, cwd=sample_dir, env=environment)
        candidate = _file(candidate, label="Omni generated video")
        if output.exists() or output.is_symlink():
            raise Shared8ContractError(f"canonical Omni output already exists: {output}")
        os.replace(candidate, output)
        runtime_evidence = _validate_omni_runtime(sample_dir, args.omni_max_context_len)

    output_probe = probe_video(output, ffprobe=args.ffprobe)
    if file_sha256(manifest) != manifest_sha256_at_start:
        raise Shared8ContractError("input manifest changed during inference")
    if file_sha256(row.source_video) != source_sha256_at_start:
        raise Shared8ContractError("source video changed during inference")
    receipt = build_output_receipt(
        model_id=args.model_id,
        row=row,
        manifest_path=manifest,
        source_probe=source_probe,
        output_path=output,
        output_probe=output_probe,
        manifest_sha256=manifest_sha256_at_start,
        source_video_sha256=source_sha256_at_start,
        source_revision=args.source_revision,
        source_archive_sha256=args.source_archive_sha256,
        model_identity=model_identity,
        sampler=sampler,
        geometry=geometry,
        runtime_evidence=runtime_evidence,
    )
    receipt_path = sample_dir / "receipt.json"
    atomic_write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "iid": row.iid,
                "output": str(output),
                "output_sha256": file_sha256(output),
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")

    parser.add_argument("--lucy-checkpoint")
    parser.add_argument("--lucy-checkpoint-tree-sha256")

    parser.add_argument("--bernini-root")
    parser.add_argument("--veomni-root")
    parser.add_argument("--bernini-checkpoint")
    parser.add_argument("--bernini-adapter")
    parser.add_argument("--bernini-inference-archive")

    parser.add_argument("--omni-root")
    parser.add_argument("--omni-checkpoint")
    parser.add_argument("--qwen-checkpoint")
    parser.add_argument("--omni-max-context-len", type=int, default=9216)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.omni_max_context_len != 9216:
        raise Shared8ContractError("Omni max context length must equal the validated 9216")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
