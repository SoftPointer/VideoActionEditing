"""Prepare, validate, and summarize the Motive action-edit inference pilot.

This module is deliberately fail-closed around the expensive GPU stage.  It
freezes a small held-out manifest with absolute media paths, validates the
checkpoint payload contract, checks every generated video, builds the Qwen
blind-evaluation manifest, and finally combines semantic and motion metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PREP_SCHEMA = "motive-action-inference-contract-v1"
ARM_STATUS_SCHEMA = "motive-action-inference-arm-status-v1"
FINAL_SCHEMA = "motive-action-inference-final-summary-v1"
PREDICTION_NAME = "step_000100.mp4"
DEFAULT_SELECTED_IIDS = (
    "1852ada01d7c43a4",
    "288545b9c031491a",
    "5ae88e1170c544b8",
    "81473c034c1b4839",
    "2766a3662fbf43d1",
    "219c4c5f56e74b86",
    "2206cde2643e470a",
    "7a2f54be92024a19",
)
ARM_ORDER = (
    "e1_plain_lora",
    "e2_fixed_random",
    "e2_random_router",
    "e3_motive_frozen",
)
EXPECTED_TRANSFORMER_TENSORS = {
    "e1_plain_lora": 602,
    "e2_fixed_random": 608,
    "e2_random_router": 608,
    "e3_motive_frozen": 604,
}
RANDOM_ROUTER_KEYS = {
    "_lucy_v10_component_router.net.0.weight",
    "_lucy_v10_component_router.net.0.bias",
    "_lucy_v10_component_router.net.2.weight",
    "_lucy_v10_component_router.net.2.bias",
    "_lucy_v10_component_router.net.3.weight",
    "_lucy_v10_component_router.net.3.bias",
}
MOTIVE_HEAD_KEYS = {
    "_lucy_v10_component_router.net.3.weight",
    "_lucy_v10_component_router.net.3.bias",
}
EXPECTED_ACTION_ENCODER_DIGEST = (
    "3b8aeff4cdc583c8e5069dbe88878c7fe81d7e42630c9a82f44e9c3c3eb573eb"
)
EXPECTED_ROUTING_CONTRACT_DIGEST = (
    "56531ef8ca7805528c80db7062c620f6b62c58e863d5d8dce4bfc5ce5ad5cdb9"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _object_digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    _write_text_atomic(path, text)


def _resolve_media(row: dict[str, Any], key: str, data_root: Path) -> Path:
    raw = Path(str(row[key])).expanduser()
    path = raw if raw.is_absolute() else data_root / raw
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing/empty {key} for iid={row.get('iid')}: {path}")
    return path


def _probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    executable = shutil.which(ffprobe)
    if executable is None:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        capture.release()
        if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"OpenCV returned invalid video metadata: {path}")
        codec = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        ).strip("\x00")
        if frames < 81:
            raise RuntimeError(f"expected at least 81 frames in {path}, found {frames}")
        return {
            "codec": codec or "unknown",
            "width": width,
            "height": height,
            "frame_rate": f"{fps:.8g}",
            "frames": frames,
            "duration": frames / fps,
            "probe_backend": "opencv",
        }
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}: {completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}")
    stream = streams[0]
    frames = int(stream.get("nb_frames") or 0)
    if frames < 81:
        raise RuntimeError(f"expected at least 81 frames in {path}, found {frames}")
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frame_rate": stream.get("r_frame_rate"),
        "frames": frames,
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
        "probe_backend": "ffprobe",
    }


def _arm_checkpoint(training_run: Path, arm: str) -> Path:
    return (
        training_run
        / "lucy"
        / arm
        / "seed_2026"
        / "checkpoint_step_000100.pt"
    )


def _validate_checkpoint_payload(path: Path, arm: str) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    state = payload.get("trainable_transformer") or {}
    adapter = payload.get("concept_adapter") or {}
    extra = payload.get("extra") or {}
    expected_count = EXPECTED_TRANSFORMER_TENSORS[arm]
    if len(state) != expected_count:
        raise RuntimeError(
            f"{arm} transformer tensor count mismatch: "
            f"expected={expected_count} actual={len(state)}"
        )
    if len(adapter) != 39:
        raise RuntimeError(
            f"{arm} concept adapter tensor count mismatch: expected=39 "
            f"actual={len(adapter)}"
        )
    router_keys = {key for key in state if "_lucy_v10_component_router." in key}
    if arm == "e1_plain_lora":
        if router_keys or extra.get("v10_gate_mode") != "single":
            raise RuntimeError(f"{arm} is not a plain-LoRA checkpoint")
    elif arm in {"e2_fixed_random", "e2_random_router"}:
        if router_keys != RANDOM_ROUTER_KEYS:
            raise RuntimeError(
                f"{arm} random router state mismatch: "
                f"missing={sorted(RANDOM_ROUTER_KEYS - router_keys)} "
                f"unexpected={sorted(router_keys - RANDOM_ROUTER_KEYS)}"
            )
        if payload.get("action_encoder") is not None:
            raise RuntimeError(f"{arm} unexpectedly embeds an action encoder")
    else:
        if router_keys != MOTIVE_HEAD_KEYS:
            raise RuntimeError(
                f"{arm} Motive head state mismatch: "
                f"actual={sorted(router_keys)}"
            )
        if payload.get("action_encoder") is None:
            raise RuntimeError(f"{arm} is missing its embedded action encoder")
        if payload.get("action_routing_contract") is None:
            raise RuntimeError(f"{arm} is missing its routing contract")
        if extra.get("action_encoder_state_digest") != EXPECTED_ACTION_ENCODER_DIGEST:
            raise RuntimeError(f"{arm} action encoder digest mismatch")
        if (
            extra.get("action_routing_contract_digest")
            != EXPECTED_ROUTING_CONTRACT_DIGEST
        ):
            raise RuntimeError(f"{arm} routing contract digest mismatch")
    if arm != "e1_plain_lora":
        if extra.get("v10_gate_mode") != "learned_residual_components":
            raise RuntimeError(f"{arm} V10 gate mode mismatch")
        if float(extra.get("v10_residual_gate_scale", -1.0)) != 0.05:
            raise RuntimeError(f"{arm} V10 residual gate scale mismatch")
    validation_path = path.parent / "training_validation.json"
    validation = _read_json(validation_path)
    if validation.get("complete") is not True:
        raise RuntimeError(f"incomplete training validation: {validation_path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "transformer_tensors": len(state),
        "concept_adapter_tensors": len(adapter),
        "router_keys": sorted(router_keys),
        "action_encoder_digest": extra.get("action_encoder_state_digest"),
        "routing_contract_digest": extra.get("action_routing_contract_digest"),
        "training_validation_sha256": _sha256(validation_path),
    }


def prepare(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse inference root: {output_root}")
    output_root.mkdir(parents=True)
    try:
        training_run = args.training_run.expanduser().resolve()
        runtime_repo = (
            args.runtime_repo.expanduser().resolve()
            if args.runtime_repo
            else training_run / "runtime_repo"
        )
        data_root = args.data_root.expanduser().resolve()
        eval_manifest = args.eval_manifest.expanduser().resolve()
        selected_iids = tuple(args.selected_iid or DEFAULT_SELECTED_IIDS)
        if len(selected_iids) != 8 or len(set(selected_iids)) != 8:
            raise ValueError("exactly eight unique --selected-iid values are required")

        eval_rows = list(_iter_jsonl(eval_manifest))
        by_iid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eval_rows:
            by_iid[str(row.get("iid"))].append(row)
        selected_rows: list[dict[str, Any]] = []
        media_inventory: list[dict[str, Any]] = []
        for index, iid in enumerate(selected_iids):
            matches = by_iid.get(iid, [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one eval row for iid={iid}, found {len(matches)}"
                )
            original = matches[0]
            row = dict(original)
            source = _resolve_media(row, "src_video", data_root)
            target = _resolve_media(row, "tgt_video", data_root)
            source_probe = _probe_video(source, args.ffprobe)
            target_probe = _probe_video(target, args.ffprobe)
            if (
                source_probe["frames"] != target_probe["frames"]
                or source_probe["width"] != target_probe["width"]
                or source_probe["height"] != target_probe["height"]
            ):
                raise RuntimeError(f"source/target media contract mismatch for iid={iid}")
            row["src_video"] = str(source)
            row["tgt_video"] = str(target)
            row["base_seed"] = int(args.seed) + index
            row["evaluation_index"] = index
            row["evaluation_scope"] = (
                "heldout_test" if row.get("split") == "test" else "validation_extension"
            )
            selected_rows.append(row)
            media_inventory.append(
                {
                    "iid": iid,
                    "source": {
                        "path": str(source),
                        "sha256": _sha256(source),
                        **source_probe,
                    },
                    "target": {
                        "path": str(target),
                        "sha256": _sha256(target),
                        **target_probe,
                    },
                }
            )

        selected_prompts = {str(row.get("prompt", "")).strip() for row in selected_rows}
        overlap_report: dict[str, Any] = {}
        for manifest in args.training_manifest:
            manifest = manifest.expanduser().resolve()
            manifest_rows = list(_iter_jsonl(manifest))
            # A representation manifest may intentionally carry train,
            # validation, and test rows in one file.  Leakage checks concern
            # only examples consumed for optimization; retain rows with an
            # explicit train split, plus legacy training manifests that do not
            # encode a split at all.
            train_rows = [
                row
                for row in manifest_rows
                if row.get("split") in {None, "", "train"}
            ]
            train_prompts = {str(row.get("prompt", "")).strip() for row in train_rows}
            train_iids = {str(row.get("iid", "")) for row in train_rows}
            prompt_overlap = sorted(selected_prompts & train_prompts)
            iid_overlap = sorted(set(selected_iids) & train_iids)
            if prompt_overlap or iid_overlap:
                raise RuntimeError(
                    f"selected eval rows overlap training manifest {manifest}: "
                    f"prompt={prompt_overlap} iid={iid_overlap}"
                )
            overlap_report[str(manifest)] = {
                "sha256": _sha256(manifest),
                "rows": len(manifest_rows),
                "effective_training_rows": len(train_rows),
                "prompt_overlap": 0,
                "iid_overlap": 0,
            }

        checkpoints = {
            arm: _validate_checkpoint_payload(_arm_checkpoint(training_run, arm), arm)
            for arm in ARM_ORDER
        }
        plain_config = (
            args.plain_config.expanduser().resolve()
            if args.plain_config
            else runtime_repo
            / "lucy"
            / "configs"
            / "goku_motive_action_plain_lora_pilot.json"
        )
        router_config = (
            args.router_config.expanduser().resolve()
            if args.router_config
            else runtime_repo
            / "lucy"
            / "configs"
            / "goku_motive_action_router_pilot.json"
        )
        model_path = (
            args.model_path.expanduser().resolve()
            if args.model_path
            else runtime_repo / "checkpoints" / "LucyEdit"
        )
        for path in (plain_config, router_config, model_path):
            if not path.exists():
                raise FileNotFoundError(path)

        selected_manifest = output_root / "inputs" / "heldout8.jsonl"
        _write_jsonl_atomic(selected_manifest, selected_rows)
        contract = {
            "schema": PREP_SCHEMA,
            "created_at_utc": _utc_now(),
            "output_root": str(output_root),
            "training_run": str(training_run),
            "runtime_repo": str(runtime_repo),
            "eval_manifest": {
                "path": str(eval_manifest),
                "sha256": _sha256(eval_manifest),
                "rows": len(eval_rows),
            },
            "selected_manifest": {
                "path": str(selected_manifest),
                "sha256": _sha256(selected_manifest),
                "rows": len(selected_rows),
                "iids": list(selected_iids),
                "split_counts": dict(
                    Counter(str(row.get("split", "unspecified")) for row in selected_rows)
                ),
            },
            "training_overlap_checks": overlap_report,
            "media_inventory": media_inventory,
            "model_path": str(model_path),
            "plain_config": {
                "path": str(plain_config),
                "sha256": _sha256(plain_config),
            },
            "router_config": {
                "path": str(router_config),
                "sha256": _sha256(router_config),
            },
            "checkpoints": checkpoints,
            "inference": {
                "seed": int(args.seed),
                "width": 832,
                "height": 480,
                "num_frames": 81,
                "fps": 25,
                "sample_mode": "uniform",
                "guidance_scale": 5.0,
                "num_inference_steps": 50,
                "dtype": "bfloat16",
                "vae_dtype": "bfloat16",
                "vae_tiling": True,
                "vae_slicing": True,
                "adapter_scale": 0.0,
                "num_shards": 8,
                "max_concurrent_nodes": 2,
                "gpus_per_node": 8,
            },
            "scientific_scope": {
                "kind": "P0 held-out generation diagnostic",
                "test_samples": 5,
                "validation_extension_samples": 3,
                "content_disjoint_split": False,
                "human_reviewed_labels": False,
                "production_eligible": False,
                "warning": (
                    "Random and Motive router forwards also differ by trunk-output "
                    "L2 normalization; this is not a weights-only causal comparison."
                ),
            },
        }
        _write_json_atomic(output_root / "contract.json", contract)
        for name in ("arms", "logs", "status", "analysis", "qwen"):
            (output_root / name).mkdir()
        print(json.dumps(contract["selected_manifest"], ensure_ascii=False))
        return 0
    except BaseException:
        # A failed prepare must not leave a reusable-looking experiment root.
        import shutil

        shutil.rmtree(output_root, ignore_errors=True)
        raise


def _load_contract(output_root: Path) -> dict[str, Any]:
    contract_path = output_root / "contract.json"
    contract = _read_json(contract_path)
    if contract.get("schema") != PREP_SCHEMA:
        raise RuntimeError(f"invalid action inference contract: {contract_path}")
    selected = contract["selected_manifest"]
    selected_path = Path(selected["path"])
    if _sha256(selected_path) != selected["sha256"]:
        raise RuntimeError("selected manifest digest changed")
    return contract


def validate_arm(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    arm = args.arm
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown arm: {arm}")
    contract = _load_contract(output_root)
    arm_root = output_root / "arms" / arm
    canonical_samples = list(_iter_jsonl(Path(contract["selected_manifest"]["path"])))
    samples_path = arm_root / "samples.json"
    samples = _read_json(samples_path)
    if not isinstance(samples, list) or len(samples) != len(canonical_samples):
        raise RuntimeError(f"{arm} samples.json row count mismatch")
    canonical_identity = [
        (str(row.get("iid")), str(row.get("prompt")), int(row.get("base_seed")))
        for row in canonical_samples
    ]
    observed_identity = [
        (str(row.get("iid")), str(row.get("prompt")), int(row.get("base_seed")))
        for row in samples
    ]
    if observed_identity != canonical_identity:
        raise RuntimeError(f"{arm} sample identity/order/seed mismatch")

    probes = []
    for index, sample in enumerate(canonical_samples):
        sample_dir = arm_root / f"sample_{index:03d}"
        files = {}
        for name in ("source.mp4", "target.mp4", PREDICTION_NAME):
            path = sample_dir / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"{arm} missing/empty output: {path}")
            probe = _probe_video(path, args.ffprobe)
            if probe["frames"] != 81:
                raise RuntimeError(f"{arm} output is not exactly 81 frames: {path}")
            files[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                **probe,
            }
        probes.append({"iid": sample["iid"], "files": files})

    expected_loaded = EXPECTED_TRANSFORMER_TENSORS[arm]
    log_paths = sorted((output_root / "logs" / arm).glob("shard_*.log"))
    if len(log_paths) != 8:
        raise RuntimeError(f"{arm} expected 8 shard logs, found {len(log_paths)}")
    log_checks = []
    for log_path in log_paths:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if f"loaded {expected_loaded} trainable transformer tensors" not in text:
            raise RuntimeError(f"{arm} load-count assertion failed in {log_path}")
        if arm != "e1_plain_lora":
            if "patched_layers=301" not in text:
                raise RuntimeError(f"{arm} router patch assertion failed in {log_path}")
        if arm == "e3_motive_frozen":
            if EXPECTED_ACTION_ENCODER_DIGEST not in text:
                raise RuntimeError(f"{arm} action digest missing from {log_path}")
            if EXPECTED_ROUTING_CONTRACT_DIGEST not in text:
                raise RuntimeError(f"{arm} routing digest missing from {log_path}")
        if "[ckpt] checkpoint_step_000100.pt sample=" not in text:
            raise RuntimeError(f"{arm} generation completion line missing in {log_path}")
        log_checks.append({"path": str(log_path), "sha256": _sha256(log_path)})

    status = {
        "schema": ARM_STATUS_SCHEMA,
        "validated_at_utc": _utc_now(),
        "arm": arm,
        "complete": True,
        "expected_transformer_tensors": expected_loaded,
        "sample_count": len(canonical_samples),
        "checkpoint": contract["checkpoints"][arm],
        "logs": log_checks,
        "outputs": probes,
    }
    _write_json_atomic(output_root / "status" / f"{arm}.json", status)
    print(json.dumps({"arm": arm, "complete": True, "samples": len(probes)}))
    return 0


def validate_all(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    _load_contract(output_root)
    states = {}
    for arm in ARM_ORDER:
        path = output_root / "status" / f"{arm}.json"
        state = _read_json(path)
        if state.get("schema") != ARM_STATUS_SCHEMA or state.get("complete") is not True:
            raise RuntimeError(f"incomplete arm status: {path}")
        states[arm] = {
            "path": str(path),
            "sha256": _sha256(path),
            "sample_count": state.get("sample_count"),
        }
    _write_json_atomic(
        output_root / "status" / "all_arms.json",
        {
            "schema": "motive-action-inference-all-arms-status-v1",
            "validated_at_utc": _utc_now(),
            "complete": True,
            "arms": states,
        },
    )
    return 0


def build_qwen_manifest(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    contract = _load_contract(output_root)
    all_status = _read_json(output_root / "status" / "all_arms.json")
    if all_status.get("complete") is not True:
        raise RuntimeError("all-arm inference validation is incomplete")
    samples = list(_iter_jsonl(Path(contract["selected_manifest"]["path"])))
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        for index, sample in enumerate(samples):
            sample_dir = output_root / "arms" / arm / f"sample_{index:03d}"
            source = sample_dir / "source.mp4"
            prediction = sample_dir / PREDICTION_NAME
            row = {
                "iid": f"{arm}__{sample['iid']}",
                "arm": arm,
                "source_iid": sample["iid"],
                "split": sample.get("split", "unspecified"),
                "evaluation_scope": sample.get("evaluation_scope"),
                "prompt": sample["prompt"],
                "src_video": str(source),
                "tgt_video": str(prediction),
                "reference_target_video": sample["tgt_video"],
                "sample_index": index,
            }
            row["input_digest"] = _object_digest(
                {
                    "prompt": row["prompt"],
                    "source_sha256": _sha256(source),
                    "prediction_sha256": _sha256(prediction),
                }
            )
            rows.append(row)
    path = output_root / "qwen" / "input.jsonl"
    if path.exists():
        raise FileExistsError(path)
    _write_jsonl_atomic(path, rows)
    _write_json_atomic(
        output_root / "qwen" / "input.json",
        {
            "schema": "motive-action-inference-qwen-input-v1",
            "created_at_utc": _utc_now(),
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(rows),
            "arms": list(ARM_ORDER),
        },
    )
    print(json.dumps({"path": str(path), "rows": len(rows)}))
    return 0


def _qwen_success(row: dict[str, Any]) -> bool:
    result = row.get("result") or {}
    observation = row.get("observation") or {}
    return bool(
        row.get("status") == "ok"
        and result.get("verdict") == "valid_action"
        and observation.get("target_actor_motion") in {"clear", "weak"}
    )


def _summarize_qwen(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        iid = str(row.get("iid", ""))
        arm, separator, _ = iid.partition("__")
        if not separator or arm not in ARM_ORDER:
            raise RuntimeError(f"unexpected Qwen inference iid: {iid}")
        source = row.get("_input") or {}
        split = str(source.get("split", "unspecified"))
        by_group[(arm, split)].append(row)
        by_group[(arm, "__all__")].append(row)
    summary: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_summary = {}
        for split in ("test", "validation", "__all__"):
            group = by_group.get((arm, split), [])
            if not group:
                continue
            results = [row.get("result") or {} for row in group]
            observations = [row.get("observation") or {} for row in group]
            successes = [_qwen_success(row) for row in group]
            arm_summary[split] = {
                "n": len(group),
                "semantic_action_success": sum(successes),
                "semantic_action_success_rate": sum(successes) / len(group),
                "verdict_counts": dict(
                    Counter(str(value.get("verdict", "missing")) for value in results)
                ),
                "target_actor_motion_counts": dict(
                    Counter(
                        str(value.get("target_actor_motion", "missing"))
                        for value in observations
                    )
                ),
                "preservation_quality_counts": dict(
                    Counter(
                        str(value.get("preservation_quality", "missing"))
                        for value in observations
                    )
                ),
                "artifact_level_counts": dict(
                    Counter(
                        str(value.get("artifact_level", "missing"))
                        for value in observations
                    )
                ),
                "camera_dominance_counts": dict(
                    Counter(
                        str(value.get("camera_dominance", "missing"))
                        for value in observations
                    )
                ),
                "background_dominance_counts": dict(
                    Counter(
                        str(value.get("background_dominance", "missing"))
                        for value in observations
                    )
                ),
            }
        summary[arm] = arm_summary
    return summary


def _load_qwen_outputs(
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = output_root / "qwen" / "input.jsonl"
    input_metadata_path = output_root / "qwen" / "input.json"
    input_metadata = _read_json(input_metadata_path)
    if (
        input_metadata.get("path") != str(input_path)
        or input_metadata.get("sha256") != _sha256(input_path)
    ):
        raise RuntimeError("Qwen input manifest metadata/digest mismatch")
    input_rows = {str(row["iid"]): row for row in _iter_jsonl(input_path)}
    outputs: list[dict[str, Any]] = []
    shard_provenance = []
    output_paths = sorted(
        (output_root / "qwen" / "shards").glob("qwen-*.jsonl")
    )
    if len(output_paths) != 8:
        raise RuntimeError(f"expected eight Qwen shard outputs, found {len(output_paths)}")
    for path in output_paths:
        match = re.fullmatch(r"qwen-(\d{3})\.jsonl", path.name)
        if match is None:
            raise RuntimeError(f"unexpected Qwen shard filename: {path}")
        shard_index = int(match.group(1))
        if not 0 <= shard_index < 8:
            raise RuntimeError(f"invalid Qwen shard index: {path}")
        manifest_path = (
            output_root / "qwen" / "manifests" / f"shard-{shard_index:03d}.jsonl"
        )
        manifest_rows = list(_iter_jsonl(manifest_path))
        shard_rows = list(_iter_jsonl(path))
        if len(manifest_rows) != 4 or len(shard_rows) != 4:
            raise RuntimeError(
                f"Qwen shard {shard_index} is not balanced 4-in/4-out: "
                f"input={len(manifest_rows)} output={len(shard_rows)}"
            )
        expected_iids = {str(row["iid"]) for row in manifest_rows}
        actual_iids = {str(row.get("iid")) for row in shard_rows}
        if actual_iids != expected_iids:
            raise RuntimeError(f"Qwen shard IID mismatch: {path}")
        manifest_sha256 = _sha256(manifest_path)
        for row in shard_rows:
            if (
                row.get("execution_shard_index") != shard_index
                or row.get("execution_shard_count") != 8
                or row.get("shard_index") != 0
                or row.get("num_shards") != 1
                or row.get("execution_manifest") != str(manifest_path)
                or row.get("execution_manifest_sha256") != manifest_sha256
            ):
                raise RuntimeError(f"Qwen shard provenance mismatch: {path}")
        outputs.extend(shard_rows)
        shard_provenance.append(
            {
                "shard_index": shard_index,
                "input_path": str(manifest_path),
                "input_sha256": manifest_sha256,
                "input_rows": len(manifest_rows),
                "output_path": str(path),
                "output_sha256": _sha256(path),
                "output_rows": len(shard_rows),
            }
        )
    if len(outputs) != len(input_rows):
        raise RuntimeError(
            f"Qwen output count mismatch: expected={len(input_rows)} "
            f"actual={len(outputs)}"
        )
    seen: set[str] = set()
    for row in outputs:
        iid = str(row.get("iid"))
        if iid in seen or iid not in input_rows:
            raise RuntimeError(f"duplicate/unexpected Qwen iid: {iid}")
        seen.add(iid)
        source = input_rows[iid]
        if row.get("input_digest") != source.get("input_digest"):
            raise RuntimeError(f"Qwen input digest mismatch: {iid}")
        if row.get("status") != "ok":
            raise RuntimeError(f"Qwen evaluation failed for {iid}: {row}")
        row["_input"] = source
    invariant_fields = (
        "run_config_digest",
        "model_revision",
        "implementation_digest",
        "transformers_version",
    )
    invariants = {}
    for field in invariant_fields:
        values = sorted({str(row.get(field)) for row in outputs})
        if len(values) != 1 or values[0] in {"", "None"}:
            raise RuntimeError(f"Qwen invariant field differs across shards: {field}")
        invariants[field] = values[0]
    provenance = {
        "input_manifest": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "rows": len(input_rows),
        },
        "input_metadata": {
            "path": str(input_metadata_path),
            "sha256": _sha256(input_metadata_path),
        },
        "invariants": invariants,
        "shards": shard_provenance,
    }
    return sorted(outputs, key=lambda row: str(row["iid"])), provenance


def _verify_contract_artifacts(contract: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    eval_manifest = contract["eval_manifest"]
    eval_path = Path(eval_manifest["path"])
    if _sha256(eval_path) != eval_manifest["sha256"]:
        raise RuntimeError("source eval manifest changed after inference prepare")
    verified["eval_manifest_sha256"] = eval_manifest["sha256"]
    for config_name in ("plain_config", "router_config"):
        entry = contract[config_name]
        if _sha256(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"{config_name} changed after inference prepare")
        verified[f"{config_name}_sha256"] = entry["sha256"]
    checkpoint_hashes = {}
    for arm, entry in contract["checkpoints"].items():
        actual = _sha256(Path(entry["path"]))
        if actual != entry["sha256"]:
            raise RuntimeError(f"checkpoint changed after inference prepare: {arm}")
        checkpoint_hashes[arm] = actual
    verified["checkpoint_sha256"] = checkpoint_hashes
    media_hashes = {}
    for item in contract["media_inventory"]:
        iid = str(item["iid"])
        media_hashes[iid] = {}
        for kind in ("source", "target"):
            entry = item[kind]
            actual = _sha256(Path(entry["path"]))
            if actual != entry["sha256"]:
                raise RuntimeError(
                    f"source media changed after inference prepare: {iid}/{kind}"
                )
            media_hashes[iid][kind] = actual
    verified["media_sha256"] = media_hashes
    return verified


def _gallery_html(output_root: Path, samples: Sequence[dict[str, Any]]) -> str:
    headers = "".join(f"<th>{html.escape(arm)}</th>" for arm in ARM_ORDER)
    rows = []
    for index, sample in enumerate(samples):
        videos = []
        for arm in ARM_ORDER:
            relative = (
                Path("arms")
                / arm
                / f"sample_{index:03d}"
                / PREDICTION_NAME
            )
            videos.append(
                "<td><video controls preload='metadata' src='"
                f"{html.escape(relative.as_posix())}'></video></td>"
            )
        source = (
            Path("arms")
            / ARM_ORDER[0]
            / f"sample_{index:03d}"
            / "source.mp4"
        )
        target = (
            Path("arms")
            / ARM_ORDER[0]
            / f"sample_{index:03d}"
            / "target.mp4"
        )
        rows.append(
            "<section>"
            f"<h2>{index:03d} · {html.escape(str(sample['iid']))} · "
            f"{html.escape(str(sample.get('split')))}</h2>"
            f"<p>{html.escape(str(sample['prompt']))}</p>"
            "<table><tr><th>source</th><th>reference target</th>"
            f"{headers}</tr><tr>"
            f"<td><video controls preload='metadata' src='{html.escape(source.as_posix())}'></video></td>"
            f"<td><video controls preload='metadata' src='{html.escape(target.as_posix())}'></video></td>"
            + "".join(videos)
            + "</tr></table></section>"
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Motive action inference</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f8;color:#171717}
section{margin:0 0 36px;padding:18px;background:white;border-radius:12px}
table{border-collapse:collapse;width:100%}th,td{padding:6px;text-align:left;vertical-align:top}
video{width:240px;max-width:18vw;background:#111}p{max-width:1100px}
</style></head><body><h1>Motive held-out action-edit inference</h1>
""" + "\n".join(rows) + "\n</body></html>\n"


def finalize(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    contract = _load_contract(output_root)
    all_status_path = output_root / "status" / "all_arms.json"
    all_status = _read_json(all_status_path)
    if all_status.get("complete") is not True:
        raise RuntimeError("all-arm validation is incomplete")
    motion_summary_path = output_root / "analysis" / "motion" / "summary.json"
    motion_summary = _read_json(motion_summary_path)
    qwen_rows, qwen_provenance = _load_qwen_outputs(output_root)
    qwen_summary = _summarize_qwen(qwen_rows)
    samples = list(_iter_jsonl(Path(contract["selected_manifest"]["path"])))
    verified_contract_artifacts = _verify_contract_artifacts(contract)
    report = {
        "schema": FINAL_SCHEMA,
        "created_at_utc": _utc_now(),
        "complete": True,
        "contract": {
            "path": str(output_root / "contract.json"),
            "sha256": _sha256(output_root / "contract.json"),
        },
        "all_arms_status": {
            "path": str(all_status_path),
            "sha256": _sha256(all_status_path),
        },
        "motion_evaluation": {
            "path": str(motion_summary_path),
            "sha256": _sha256(motion_summary_path),
            "summary": motion_summary,
        },
        "qwen_evaluation": {
            "rows": len(qwen_rows),
            "provenance": qwen_provenance,
            "summary": qwen_summary,
        },
        "verified_contract_artifacts": verified_contract_artifacts,
        "limitations": [
            "Only five rows are held-out test; three are validation extensions.",
            "The legacy split is not content-derived and labels are pseudo-labels.",
            "There is one Lucy seed, 100 training steps, and only eight inference cases.",
            "Qwen is an automatic diagnostic judge, not a substitute for blinded human review.",
            (
                "The random and Motive router forward contracts differ by "
                "trunk-output L2 normalization, so E2-vs-E3 is not a pure "
                "pretrained-weights causal contrast."
            ),
        ],
    }
    _write_json_atomic(output_root / "analysis" / "final_summary.json", report)
    _write_text_atomic(output_root / "index.html", _gallery_html(output_root, samples))
    print(
        json.dumps(
            {
                "complete": True,
                "qwen_rows": len(qwen_rows),
                "final_summary": str(output_root / "analysis" / "final_summary.json"),
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--training-run", required=True, type=Path)
    prepare_parser.add_argument("--runtime-repo", type=Path)
    prepare_parser.add_argument("--eval-manifest", required=True, type=Path)
    prepare_parser.add_argument("--data-root", required=True, type=Path)
    prepare_parser.add_argument("--output-root", required=True, type=Path)
    prepare_parser.add_argument("--training-manifest", action="append", default=[], type=Path)
    prepare_parser.add_argument("--selected-iid", action="append")
    prepare_parser.add_argument("--plain-config", type=Path)
    prepare_parser.add_argument("--router-config", type=Path)
    prepare_parser.add_argument("--model-path", type=Path)
    prepare_parser.add_argument("--seed", default=2026, type=int)
    prepare_parser.add_argument("--ffprobe", default="ffprobe")
    prepare_parser.set_defaults(function=prepare)

    arm_parser = subparsers.add_parser("validate-arm")
    arm_parser.add_argument("--output-root", required=True, type=Path)
    arm_parser.add_argument("--arm", required=True, choices=ARM_ORDER)
    arm_parser.add_argument("--ffprobe", default="ffprobe")
    arm_parser.set_defaults(function=validate_arm)

    all_parser = subparsers.add_parser("validate-all")
    all_parser.add_argument("--output-root", required=True, type=Path)
    all_parser.set_defaults(function=validate_all)

    qwen_parser = subparsers.add_parser("build-qwen-manifest")
    qwen_parser.add_argument("--output-root", required=True, type=Path)
    qwen_parser.set_defaults(function=build_qwen_manifest)

    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--output-root", required=True, type=Path)
    final_parser.set_defaults(function=finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
