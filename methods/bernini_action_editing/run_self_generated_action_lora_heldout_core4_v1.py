#!/usr/bin/env python3
"""Run and verify frozen-vs-trained exact81 heldout Bernini comparisons.

This is deliberately an inference/evidence harness, not a scorer.  It uses the
SEER scoped inference wrapper for both arms (the frozen arm passes
``--base-only``) and rejects any pair whose source, instruction, seed,
preprocessing, or sampler contract differs.  Method success still requires
detached full-video action and preservation review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-self-generated-action-lora-heldout-core4-v1"
PAIR_RECEIPT_SCHEMA = "bernini-self-generated-action-lora-heldout-pair-v1"
MASTER_RECEIPT_SCHEMA = "bernini-self-generated-action-lora-heldout-core4-receipt-v1"
INFERENCE_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-inference-receipt-v1"
DEFAULT_TRAINED_INFER_RUNNER = "infer_seer_scoped_lora.py"
SAME_STATE_TRAINED_INFER_RUNNER = "infer_seer_same_state_lora.py"
SAME_STATE_FULL160_TRAINED_INFER_RUNNER = (
    "infer_seer_same_state_full160_lora.py"
)
ADMITTED_TRAINED_INFER_RUNNERS = frozenset(
    (
        DEFAULT_TRAINED_INFER_RUNNER,
        SAME_STATE_TRAINED_INFER_RUNNER,
        SAME_STATE_FULL160_TRAINED_INFER_RUNNER,
    )
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
CASE_FIELDS = {
    "iid", "actor_family", "analysis_split", "source_video",
    "source_video_sha256", "source_caption", "source_caption_utf8_sha256",
    "instruction", "instruction_utf8_sha256", "seed",
}
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[0-9a-f]{16}")


class HeldoutEvalError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HeldoutEvalError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise HeldoutEvalError(f"{label} must be an absolute plain file: {path}")
    return path.resolve(strict=True)


def _directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise HeldoutEvalError(f"{label} must be an absolute plain directory: {path}")
    return path.resolve(strict=True)


def _executable_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value)
    if not requested.is_absolute():
        raise HeldoutEvalError(f"{label} must be absolute: {requested}")
    try:
        path = requested.resolve(strict=True)
        mode = path.stat().st_mode
    except OSError as error:
        raise HeldoutEvalError(f"cannot resolve {label}: {requested}") from error
    if not stat.S_ISREG(mode) or not os.access(path, os.X_OK):
        raise HeldoutEvalError(f"{label} must resolve to an executable regular file")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HeldoutEvalError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise HeldoutEvalError(f"{label} root must be an object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise HeldoutEvalError(f"refusing to overwrite {path}")
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def load_spec(path_value: str | Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise HeldoutEvalError("expected spec SHA-256 is invalid")
    path = _plain_file(path_value, label="heldout spec")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise HeldoutEvalError("heldout spec raw SHA-256 differs")
    value = _read_json(path, label="heldout spec")
    if set(value) != {
        "schema_version", "source_registry", "training_exclusion",
        "inference_contract", "decision_contract", "cases",
    } or value["schema_version"] != SCHEMA_VERSION:
        raise HeldoutEvalError("heldout spec root closure differs")
    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) != 4:
        raise HeldoutEvalError("heldout spec must contain exactly four cases")
    seen: set[str] = set()
    families: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict) or set(raw) != CASE_FIELDS:
            raise HeldoutEvalError(f"case {index} field closure differs")
        iid = raw["iid"]
        family = raw["actor_family"]
        if not isinstance(iid, str) or _IID.fullmatch(iid) is None or iid in seen:
            raise HeldoutEvalError(f"case {index} IID differs")
        if family not in {"dog", "human"} or raw["analysis_split"] != "confirmation":
            raise HeldoutEvalError(f"case {index} split/family differs")
        source = Path(raw["source_video"])
        if not source.is_absolute() or source == Path("/"):
            raise HeldoutEvalError(f"case {index} source path differs")
        for field in ("source_video_sha256", "source_caption_utf8_sha256", "instruction_utf8_sha256"):
            if not isinstance(raw[field], str) or _SHA256.fullmatch(raw[field]) is None:
                raise HeldoutEvalError(f"case {index} {field} differs")
        for text_field, digest_field in (
            ("source_caption", "source_caption_utf8_sha256"),
            ("instruction", "instruction_utf8_sha256"),
        ):
            text = raw[text_field]
            if not isinstance(text, str) or not text.strip() or "\x00" in text:
                raise HeldoutEvalError(f"case {index} {text_field} differs")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != raw[digest_field]:
                raise HeldoutEvalError(f"case {index} {text_field} digest differs")
        if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**63:
            raise HeldoutEvalError(f"case {index} seed differs")
        seen.add(iid)
        families.append(family)
        normalized.append(dict(raw))
    if families.count("dog") != 2 or families.count("human") != 2:
        raise HeldoutEvalError("heldout core4 must be two dog and two human")
    exclusion = value["training_exclusion"]
    if (
        not isinstance(exclusion, dict)
        or exclusion.get("self_generated_owner_job") != 131524
        or exclusion.get("owner_fit_iids") != ["7b88a1ca1f804f41", "a35b590961d24694"]
        or not seen.isdisjoint(exclusion["owner_fit_iids"])
        or exclusion.get("heldout_iids_disjoint_from_owner_fit_iids") is not True
        or exclusion.get("heldout_media_or_latents_enter_optimizer") is not False
    ):
        raise HeldoutEvalError("owner131524 fit/heldout exclusion differs")
    contract = value["inference_contract"]
    if (
        not isinstance(contract, dict)
        or contract.get("runner")
        != "methods/bernini_action_editing/infer_seer_scoped_lora.py"
        or contract.get("same_source_instruction_seed_between_arms") is not True
        or contract.get("num_frames") != 81
        or contract.get("fps") != 25
        or contract.get("num_inference_steps") != 40
        or contract.get("guidance_mode") != "v2v_apg"
        or contract.get("ulysses_size") != 4
        or contract.get("target_video_available_to_inference") is not False
    ):
        raise HeldoutEvalError("inference contract differs")
    result = dict(value)
    result["cases"] = normalized
    return result, actual


def case_by_iid(spec: Mapping[str, Any], iid: str) -> dict[str, Any]:
    matches = [row for row in spec["cases"] if row["iid"] == iid]
    if len(matches) != 1:
        raise HeldoutEvalError(f"IID is not one sealed heldout case: {iid}")
    return dict(matches[0])


def verify_case_media(case: Mapping[str, Any]) -> Path:
    path = _plain_file(case["source_video"], label=f"{case['iid']} source video")
    if file_sha256(path) != case["source_video_sha256"]:
        raise HeldoutEvalError(f"{case['iid']} source video SHA-256 differs")
    return path


def inference_runner_name(*, arm: str, trained_runner: str) -> str:
    """Use one sampler wrapper for both arms; only the trained arm loads LoRA."""

    if arm == "frozen_base":
        return DEFAULT_TRAINED_INFER_RUNNER
    if arm != "trained_adapter":
        raise HeldoutEvalError(f"unknown arm: {arm}")
    if trained_runner not in ADMITTED_TRAINED_INFER_RUNNERS:
        raise HeldoutEvalError("trained inference runner is not admitted")
    return trained_runner


def _verify_inference_receipt(
    receipt_path: Path,
    *,
    case: Mapping[str, Any],
    arm: str,
    adapter_checkpoint: Path | None,
) -> dict[str, Any]:
    value = _read_json(_plain_file(receipt_path, label=f"{arm} inference receipt"), label=f"{arm} receipt")
    digest = value.pop("receipt_digest", None)
    if not isinstance(digest, str) or object_sha256(value) != digest:
        raise HeldoutEvalError(f"{arm} inference receipt digest differs")
    value["receipt_digest"] = digest
    receipt_suffix = ".receipt.json"
    if not receipt_path.name.endswith(receipt_suffix):
        raise HeldoutEvalError(f"{arm} inference receipt filename differs")
    expected_output = receipt_path.with_name(receipt_path.name[: -len(receipt_suffix)])
    input_row = value.get("input", {})
    preprocessing = value.get("preprocessing", {})
    sampling = value.get("sampling", {})
    output = value.get("output", {})
    adapter = value.get("adapter", {})
    if (
        value.get("schema_version") != INFERENCE_RECEIPT_SCHEMA
        or input_row.get("source_video_path") != str(Path(case["source_video"]).resolve())
        or input_row.get("source_video_sha256") != case["source_video_sha256"]
        or input_row.get("instruction_utf8_sha256") != case["instruction_utf8_sha256"]
        or input_row.get("accepted_model_conditions") != ["source_video", "edit_instruction"]
        or input_row.get("target_accessed_by_inference") is not False
        or preprocessing.get("frame_count") != 81
        or float(preprocessing.get("fps", -1)) != 25.0
        or float(preprocessing.get("reported_fps", -1)) != 25.0
        or preprocessing.get("temporal_policy") != "all_integer_frames_0_through_80_no_subsampling"
        or preprocessing.get("external_shared_i0") is not False
        or sampling.get("seed") != case["seed"]
        or sampling.get("num_frames") != 81
        or sampling.get("num_inference_steps") != 40
        or sampling.get("guidance_mode") != "v2v_apg"
        or sampling.get("ulysses_size") != 4
        or output.get("path") != str(expected_output)
        or output.get("frame_count") != 81
        or float(output.get("fps", -1)) != 25.0
    ):
        raise HeldoutEvalError(f"{arm} inference receipt contract differs")
    media = _plain_file(expected_output, label=f"{arm} output")
    if file_sha256(media) != output.get("sha256"):
        raise HeldoutEvalError(f"{arm} output SHA-256 differs")
    if arm == "frozen_base":
        if adapter != {
            "enabled": False,
            "mode": "frozen_base_no_adapter",
            "strictly_reloaded": False,
            "safe_merged_for_inference": False,
            "tensor_count": 0,
        }:
            raise HeldoutEvalError("frozen arm claimed adaptation")
    elif arm == "trained_adapter":
        if adapter_checkpoint is None:
            raise HeldoutEvalError("trained arm requires adapter checkpoint identity")
        if (
            adapter.get("enabled") is not True
            or adapter.get("mode") != "lora_safe_merge"
            or adapter.get("strictly_reloaded") is not True
            or adapter.get("safe_merged_for_inference") is not True
            or Path(adapter.get("checkpoint_root", "")).resolve() != adapter_checkpoint
        ):
            raise HeldoutEvalError("trained arm adapter binding differs")
    else:
        raise HeldoutEvalError(f"unknown arm: {arm}")
    return value


def _make_review(case: Mapping[str, Any], root: Path, ffmpeg: Path) -> dict[str, Any]:
    output = root / "review_source_base_trained_f0_20_40_60_80.jpg"
    if output.exists() or output.is_symlink():
        raise HeldoutEvalError(f"refusing to overwrite review mosaic: {output}")
    selector = "select=eq(n\\,0)+eq(n\\,20)+eq(n\\,40)+eq(n\\,60)+eq(n\\,80)"
    geometry = "scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2:black,tile=5x1"
    filters = ";".join(
        f"[{index}:v]{selector},{geometry}[{name}]"
        for index, name in enumerate(("source", "base", "trained"))
    ) + ";[source][base][trained]vstack=inputs=3[out]"
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-n",
        "-i", case["source_video"],
        "-i", str(root / "frozen_base.mp4"),
        "-i", str(root / "trained_adapter.mp4"),
        "-filter_complex", filters, "-map", "[out]", "-frames:v", "1", str(output),
    ]
    subprocess.run(command, check=True)
    return {
        "path": str(output),
        "sha256": file_sha256(output),
        "layout_top_to_bottom": ["source", "frozen_base", "trained_adapter"],
        "frame_indices_left_to_right": [0, 20, 40, 60, 80],
        "full_video_review_still_required": True,
    }


def torchrun_prefix(
    *,
    python_bin: Path,
    nnodes: int,
    nproc_per_node: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    no_python: bool = False,
) -> list[str]:
    if nnodes < 1 or nproc_per_node < 1 or nnodes * nproc_per_node != 4:
        raise HeldoutEvalError("torchrun topology must contain exactly four ranks")
    if node_rank < 0 or node_rank >= nnodes:
        raise HeldoutEvalError("torchrun node rank differs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", master_addr):
        raise HeldoutEvalError("torchrun master address differs")
    if not 1024 <= master_port <= 65535:
        raise HeldoutEvalError("torchrun master port differs")
    prefix = [
        str(python_bin),
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
    ]
    if no_python:
        prefix.append("--no-python")
    return prefix


def command_run_arm(args: argparse.Namespace, spec: Mapping[str, Any], spec_sha: str) -> int:
    case = case_by_iid(spec, args.iid)
    verify_case_media(case)
    method_root = _directory(args.method_root, label="method root")
    runner_name = inference_runner_name(
        arm=args.arm, trained_runner=args.trained_infer_runner
    )
    runner = _plain_file(method_root / runner_name, label=runner_name)
    python_bin = _executable_file(args.python_bin, label="Python executable")
    output_root = _directory(args.output_root, label="output root")
    case_root = output_root / case["iid"]
    case_root.mkdir(mode=0o700, exist_ok=True)
    if case_root.is_symlink():
        raise HeldoutEvalError("case output directory is a symlink")
    output = case_root / f"{args.arm}.mp4"
    receipt = output.with_name(f"{output.name}.receipt.json")
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise HeldoutEvalError(f"refusing output reuse for {case['iid']} {args.arm}")
    if _SHA1.fullmatch(args.method_source_revision) is None or _SHA256.fullmatch(args.method_source_archive_sha256) is None:
        raise HeldoutEvalError("method source identity differs")
    command = [
        *torchrun_prefix(
            python_bin=python_bin,
            nnodes=args.torchrun_nnodes,
            nproc_per_node=args.torchrun_nproc_per_node,
            node_rank=args.torchrun_node_rank,
            master_addr=args.torchrun_master_addr,
            master_port=args.master_port,
            no_python=bool(args.torchrun_worker_prefix),
        ),
        *([args.torchrun_worker_prefix] if args.torchrun_worker_prefix else []),
        str(runner),
        "--bernini-root", str(_directory(args.bernini_root, label="Bernini root")),
        "--veomni-root", str(_directory(args.veomni_root, label="VeOmni root")),
        "--checkpoint", str(_directory(args.checkpoint, label="checkpoint")),
    ]
    adapter: Path | None = None
    if args.arm == "frozen_base":
        command.append("--base-only")
    else:
        adapter = _directory(args.adapter_checkpoint, label="adapter checkpoint")
        command.extend(["--adapter-checkpoint", str(adapter)])
    command.extend([
        "--source-video", case["source_video"],
        "--instruction", case["instruction"],
        "--output", str(output),
        "--num-inference-steps", "40", "--seed", str(case["seed"]),
        "--expected-bernini-commit", BERNINI_COMMIT,
        "--expected-veomni-commit", VEOMNI_COMMIT,
        "--expected-checkpoint-tree-sha256", CHECKPOINT_TREE_SHA256,
        "--method-source-revision", args.method_source_revision,
        "--method-source-archive-sha256", args.method_source_archive_sha256,
    ])
    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "MODELING_BACKEND": "hf", "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "BERNINI_HELDOUT_SPEC_SHA256": spec_sha,
    })
    subprocess.run(command, check=True, env=environment)
    checked = _verify_inference_receipt(
        receipt, case=case, arm=args.arm, adapter_checkpoint=adapter,
    )
    print(json.dumps({
        "status": "arm_complete_not_method_success",
        "iid": case["iid"], "arm": args.arm,
        "output_sha256": checked["output"]["sha256"],
        "receipt_digest": checked["receipt_digest"],
    }, sort_keys=True))
    return 0


def command_verify_pair(args: argparse.Namespace, spec: Mapping[str, Any], spec_sha: str) -> int:
    case = case_by_iid(spec, args.iid)
    verify_case_media(case)
    root = _directory(args.output_root, label="output root") / case["iid"]
    root = _directory(root, label="case output root")
    adapter = _directory(args.adapter_checkpoint, label="adapter checkpoint")
    base = _verify_inference_receipt(
        root / "frozen_base.mp4.receipt.json", case=case,
        arm="frozen_base", adapter_checkpoint=None,
    )
    trained = _verify_inference_receipt(
        root / "trained_adapter.mp4.receipt.json", case=case,
        arm="trained_adapter", adapter_checkpoint=adapter,
    )
    equality_fields = (
        "method_source_revision", "method_source_archive_sha256", "bernini_commit",
        "veomni_commit", "checkpoint_tree_sha256", "input", "preprocessing",
        "prompt_contract", "sampling",
    )
    unequal = [field for field in equality_fields if base[field] != trained[field]]
    if unequal:
        raise HeldoutEvalError(f"base/trained paired coordinate differs: {unequal}")
    review = None
    if args.ffmpeg is not None:
        ffmpeg = _executable_file(args.ffmpeg, label="ffmpeg")
        review = _make_review(case, root, ffmpeg)
    unsigned = {
        "schema_version": PAIR_RECEIPT_SCHEMA,
        "status": "decoded_pair_ready_for_blind_review_no_method_success_claim",
        "spec_sha256": spec_sha,
        "iid": case["iid"], "actor_family": case["actor_family"],
        "analysis_split": "confirmation",
        "source_video_sha256": case["source_video_sha256"],
        "instruction_utf8_sha256": case["instruction_utf8_sha256"],
        "seed": case["seed"],
        "same_source_instruction_seed_preprocessing_sampler": True,
        "frozen_base": {
            "output": base["output"], "inference_receipt_digest": base["receipt_digest"],
        },
        "trained_adapter": {
            "output": trained["output"], "inference_receipt_digest": trained["receipt_digest"],
            "adapter_model_sha256": trained["adapter"]["adapter_model_sha256"],
            "training_global_step": trained["adapter"]["training_global_step"],
        },
        "review_mosaic": review,
        "full_video_action_and_preservation_review_complete": False,
        "method_success_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(root / "paired-receipt.json", receipt)
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


def command_verify_core4(args: argparse.Namespace, spec: Mapping[str, Any], spec_sha: str) -> int:
    root = _directory(args.output_root, label="output root")
    pairs = []
    adapter_sha: str | None = None
    for case in spec["cases"]:
        path = _plain_file(root / case["iid"] / "paired-receipt.json", label="paired receipt")
        value = _read_json(path, label="paired receipt")
        digest = value.pop("receipt_digest", None)
        if not isinstance(digest, str) or object_sha256(value) != digest:
            raise HeldoutEvalError(f"paired receipt digest differs: {case['iid']}")
        value["receipt_digest"] = digest
        if (
            value.get("schema_version") != PAIR_RECEIPT_SCHEMA
            or value.get("spec_sha256") != spec_sha
            or value.get("iid") != case["iid"]
            or value.get("same_source_instruction_seed_preprocessing_sampler") is not True
            or value.get("full_video_action_and_preservation_review_complete") is not False
            or value.get("method_success_authorized") is not False
        ):
            raise HeldoutEvalError(f"paired receipt contract differs: {case['iid']}")
        current_adapter_sha = value["trained_adapter"]["adapter_model_sha256"]
        if adapter_sha is None:
            adapter_sha = current_adapter_sha
        elif current_adapter_sha != adapter_sha:
            raise HeldoutEvalError("core4 was not evaluated with one trained adapter")
        pairs.append({
            "iid": case["iid"], "actor_family": case["actor_family"],
            "pair_receipt_path": str(path), "pair_receipt_sha256": file_sha256(path),
            "pair_receipt_digest": digest,
        })
    unsigned = {
        "schema_version": MASTER_RECEIPT_SCHEMA,
        "status": "core4_decoded_pairs_complete_pending_blind_full_video_review",
        "spec_sha256": spec_sha,
        "adapter_model_sha256": adapter_sha,
        "case_count": 4, "dog_count": 2, "human_count": 2,
        "pairs": pairs,
        "decision_contract": spec["decision_contract"],
        "training_completion_is_method_success": False,
        "decoded_generation_completion_is_method_success": False,
        "method_success_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(root / "core4-master-receipt.json", receipt)
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--verify-files", action="store_true")
    run = sub.add_parser("run-arm")
    run.add_argument("--iid", required=True)
    run.add_argument("--arm", choices=("frozen_base", "trained_adapter"), required=True)
    run.add_argument("--method-root", required=True)
    run.add_argument("--python-bin", required=True)
    run.add_argument("--bernini-root", required=True)
    run.add_argument("--veomni-root", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--adapter-checkpoint")
    run.add_argument(
        "--trained-infer-runner",
        choices=tuple(sorted(ADMITTED_TRAINED_INFER_RUNNERS)),
        default=DEFAULT_TRAINED_INFER_RUNNER,
        help=(
            "Adapter loader selected for the trained arm. The frozen control "
            "always uses the SEER wrapper in base-only mode."
        ),
    )
    run.add_argument("--output-root", required=True)
    run.add_argument("--master-port", type=int, required=True)
    run.add_argument("--torchrun-nnodes", type=int, default=1)
    run.add_argument("--torchrun-nproc-per-node", type=int, default=4)
    run.add_argument("--torchrun-node-rank", type=int, default=0)
    run.add_argument("--torchrun-master-addr", default="127.0.0.1")
    run.add_argument("--torchrun-worker-prefix")
    run.add_argument("--method-source-revision", required=True)
    run.add_argument("--method-source-archive-sha256", required=True)
    pair = sub.add_parser("verify-pair")
    pair.add_argument("--iid", required=True)
    pair.add_argument("--adapter-checkpoint", required=True)
    pair.add_argument("--output-root", required=True)
    pair.add_argument("--ffmpeg")
    core4 = sub.add_parser("verify-core4")
    core4.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec, spec_sha = load_spec(args.spec, args.expected_spec_sha256)
    if args.command == "inspect":
        if args.verify_files:
            for case in spec["cases"]:
                verify_case_media(case)
        print(json.dumps({
            "schema_version": SCHEMA_VERSION, "spec_sha256": spec_sha,
            "case_count": 4, "dog_count": 2, "human_count": 2,
            "iids": [case["iid"] for case in spec["cases"]],
            "files_verified": bool(args.verify_files),
        }, sort_keys=True))
        return 0
    if args.command == "run-arm":
        if args.arm == "trained_adapter" and not args.adapter_checkpoint:
            raise HeldoutEvalError("trained_adapter requires --adapter-checkpoint")
        if args.arm == "frozen_base" and args.adapter_checkpoint:
            raise HeldoutEvalError("frozen_base may not receive --adapter-checkpoint")
        if not 1024 <= args.master_port <= 65535:
            raise HeldoutEvalError("master port differs")
        return command_run_arm(args, spec, spec_sha)
    if args.command == "verify-pair":
        return command_verify_pair(args, spec, spec_sha)
    if args.command == "verify-core4":
        return command_verify_core4(args, spec, spec_sha)
    raise HeldoutEvalError("unreachable command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HeldoutEvalError as error:
        print(f"[heldout-core4] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
