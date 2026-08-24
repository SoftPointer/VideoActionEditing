#!/usr/bin/env python3
"""Build a four-row mask-free fixture for the MARP DDP integration smoke.

Every tensor is synthetic and every row is ``preview_only``.  This fixture is
reporting-forbidden and tests only data/model/DDP plumbing; it is not a motion
editing dataset and cannot support a scientific quality claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from action import (  # noqa: E402
    ACTION_MANIFEST_FORMAT,
    ACTION_PAYLOAD_FORMAT,
    ACTION_PROVENANCE_FORMAT,
    action_tensor_sha256,
    load_action_config,
    validate_action_payload,
)
from pact.dataset import (  # noqa: E402
    ENCODER_CONTRACT_FORMAT,
    UMT5_EMBEDDING_DIM,
    UMT5_MAX_SEQUENCE_LENGTH,
    UMT5_PADDING_POLICY,
    UMT5_SEGMENT_ORDER,
    VLM_EMBEDDING_DIM,
    VLM_FEATURE_TENSOR,
    VLM_TOKEN_SELECTION,
    WAN21_VAE_CHANNEL_MEAN,
    WAN21_VAE_CHANNEL_STD,
    WAN21_VAE_INPUT_PIXEL_RANGE,
    WAN21_VAE_POSTERIOR_MODE,
    WAN21_VAE_STRIDE,
    encoder_contract_sha256,
)


FIXTURE_FORMAT = "marp-action-synthetic-fixture-v1"
DONE_FORMAT = "marp-action-synthetic-fixture-done-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _encoder_contract() -> dict[str, Any]:
    return {
        "format": ENCODER_CONTRACT_FORMAT,
        "vae": {
            "checkpoint_sha256": "1" * 64,
            "preprocessing_contract_sha256": "2" * 64,
            "input_pixel_range": list(WAN21_VAE_INPUT_PIXEL_RANGE),
            "posterior_mode": WAN21_VAE_POSTERIOR_MODE,
            "channel_mean": list(WAN21_VAE_CHANNEL_MEAN),
            "channel_std": list(WAN21_VAE_CHANNEL_STD),
            "stride": list(WAN21_VAE_STRIDE),
        },
        "umt5": {
            "checkpoint_manifest_sha256": "3" * 64,
            "preprocessing_contract_sha256": "4" * 64,
            "embedding_dim": UMT5_EMBEDDING_DIM,
            "max_sequence_length_per_segment": UMT5_MAX_SEQUENCE_LENGTH,
            "segment_order": list(UMT5_SEGMENT_ORDER),
            "padding_policy": UMT5_PADDING_POLICY,
        },
        "vlm": {
            "checkpoint_manifest_sha256": "5" * 64,
            "feature_extraction_contract_sha256": "6" * 64,
            "embedding_dim": VLM_EMBEDDING_DIM,
            "feature_tensor": VLM_FEATURE_TENSOR,
            "token_selection": VLM_TOKEN_SELECTION,
        },
    }


def _payload(
    sample_id: str,
    task_type: str,
    *,
    seed: int,
    motion_tokens: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randn((16, 2, 8, 8), generator=generator) * 0.05
    if task_type == "identity_reconstruction":
        target = source.clone()
    elif task_type == "native_replay":
        target = source + 0.01
    elif task_type == "native_isolation_probe":
        target = source.clone()
    else:
        # A full-tensor endpoint, deliberately not a region splice.
        temporal = torch.linspace(-0.08, 0.08, 2).reshape(1, 2, 1, 1)
        target = source + temporal
    payload = {
        "format": ACTION_PAYLOAD_FORMAT,
        "sample_id": sample_id,
        "encoder_contract": _encoder_contract(),
        "source_latent": source.float(),
        "target_latent": target.float(),
        "text_context": (
            torch.randn((4, 4096), generator=generator) * 0.01
        ).float(),
        "source_vlm_context": (
            torch.randn((6, 2048), generator=generator) * 0.01
        ).float(),
        "target_motion_tokens": (
            torch.randn((motion_tokens, 2048), generator=generator) * 0.01
        ).float(),
        "task_type": task_type,
        "preview_only": True,
    }
    return validate_action_payload(
        payload,
        expected_motion_tokens=motion_tokens,
        allowed_task_types=(
            "action_edit",
            "identity_reconstruction",
            "native_replay",
            "native_isolation_probe",
        ),
    )


def build_fixture(output_dir: str | os.PathLike[str], *, seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    payload_root = root / "payloads"
    provenance_root = root / "provenance"
    config_root = root / "configs"
    payload_root.mkdir()
    provenance_root.mkdir()
    config_root.mkdir()

    default_config_path = METHOD_ROOT / "configs" / "marp_1_3b.json"
    with default_config_path.open("r", encoding="utf-8") as handle:
        config = copy.deepcopy(json.load(handle))
    config["seed"] = seed
    config["data"].update(
        {
            "video_num_frames": 5,
            "video_fps": 1.0,
            "video_height": 64,
            "video_width": 64,
            "temporal_mode": "synthetic_fixture",
            "spatial_profile": "synthetic_fixture",
            "allow_transpose": False,
            "smoke_only": True,
            "require_materialization_metadata": False,
        }
    )
    config["model"].update(
        {"max_context_len": 6144, "context_padding_mode": "fixed_budget"}
    )
    config["lora"].update({"rank": 4, "alpha": 4.0, "dropout": 0.0})
    config["planner"].update(
        {"num_tokens": 4, "hidden_dim": 128, "depth": 1, "weight": 0.1}
    )
    config["training"].update(
        {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "num_workers": 0,
            "log_every": 1,
            "save_every": 1,
            "allow_preview": True,
        }
    )
    config_path = config_root / "marp_one_step.json"
    _write_json(config_path, config)
    checked_config = load_action_config(config_path)

    tasks = (
        "action_edit",
        "identity_reconstruction",
        "native_replay",
        "native_isolation_probe",
    )
    rows = []
    for index, task_type in enumerate(tasks):
        sample_id = f"synthetic_marp_{index:03d}"
        payload = _payload(
            sample_id,
            task_type,
            seed=seed + index,
            motion_tokens=checked_config.planner.num_tokens,
        )
        payload_path = payload_root / f"{sample_id}.pt"
        torch.save(payload, payload_path)
        payload_sha256 = _sha256(payload_path)
        encoder = payload["encoder_contract"]
        provenance = {
            "schema_version": ACTION_PROVENANCE_FORMAT,
            "sample_id": sample_id,
            "parent_id": sample_id,
            "split_group": sample_id,
            "direction": "forward",
            "task_type": task_type,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_eligible": False,
            "post_video_acceptance": "synthetic_not_applicable",
            "preview_join": {
                "manifest_path": "synthetic://fixture",
                "manifest_sha256": "7" * 64,
                "row_digest": "8" * 64,
                "row_file_sha256": "9" * 64,
                "upstream_provenance_sha256": "a" * 64,
            },
            "media": {
                "source_video_path": "synthetic://source",
                "source_video_sha256": "b" * 64,
                "target_video_path": "synthetic://target",
                "target_video_sha256": "c" * 64,
                "shared_i0_path": "synthetic://shared_i0",
                "shared_i0_sha256": "d" * 64,
                "preprocessing": {"synthetic_integration_only": True},
            },
            "conditioning": {
                "instruction": "synthetic action integration condition",
                "instruction_sha256": hashlib.sha256(
                    b"synthetic action integration condition"
                ).hexdigest(),
                "instruction_source": "synthetic_fixture",
                "generation_instruction_sha256": "e" * 64,
                "target_caption": "synthetic target caption",
                "target_caption_sha256": hashlib.sha256(
                    b"synthetic target caption"
                ).hexdigest(),
                "target_caption_origin": "synthetic_fixture",
                "motion_text": "synthetic motion-only teacher label",
                "motion_text_sha256": hashlib.sha256(
                    b"synthetic motion-only teacher label"
                ).hexdigest(),
                "motion_teacher_visual_input": "target_video_only",
                "motion_teacher_feature_input": "canonical_motion_text_only",
                "motion_pool": "synthetic_fixture",
                "target_motion_tokens_usage": "planner_loss_only",
            },
            "encoder": {
                "contract": encoder,
                "contract_sha256": encoder_contract_sha256(encoder),
                "checkpoint_identities": {"synthetic_integration_only": True},
            },
            "tensor_sha256": {
                field: action_tensor_sha256(payload[field])
                for field in (
                    "source_latent",
                    "target_latent",
                    "text_context",
                    "source_vlm_context",
                    "target_motion_tokens",
                )
            },
            "payload": {
                "path": str(Path("payloads") / payload_path.name),
                "sha256": payload_sha256,
            },
        }
        provenance_path = provenance_root / f"{sample_id}.json"
        _write_json(provenance_path, provenance)
        rows.append(
            {
                "format": ACTION_MANIFEST_FORMAT,
                "sample_id": sample_id,
                "payload_path": payload_path.name,
                "payload_sha256": payload_sha256,
                "provenance_path": str(
                    Path("provenance") / provenance_path.name
                ),
                "provenance_sha256": _sha256(provenance_path),
                "task_type": task_type,
                "preview_only": True,
            }
        )
    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    receipt = {
        "format": DONE_FORMAT,
        "fixture_format": FIXTURE_FORMAT,
        "complete": True,
        "synthetic_integration_only": True,
        "benchmark_reporting_forbidden": True,
        "production_claim_forbidden": True,
        "preview_only": True,
        "mask_or_tube_inputs": False,
        "sample_count": len(rows),
        "task_counts": {
            task_type: sum(row["task_type"] == task_type for row in rows)
            for task_type in sorted(set(tasks))
        },
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
        "config": str(config_path.relative_to(root)),
        "config_sha256": _sha256(config_path),
    }
    _write_json(root / "done.json", receipt)
    return receipt


def main() -> None:
    args = _parse_args()
    receipt = build_fixture(args.output_dir, seed=args.seed)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
