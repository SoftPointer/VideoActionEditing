#!/usr/bin/env python3
"""Run one sealed PAIR-v5 candidate through frozen native Bernini RV2V-4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import infer_native_identity_generation_canary as native
import pair_v5_native_rollout_spec as spec_contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _bind_receipt(args: argparse.Namespace, envelope: dict[str, Any]) -> None:
    output = Path(args.output_dir)
    native_path = output / "receipt.json"
    native_receipt = json.loads(native_path.read_bytes())
    candidate = envelope["candidate"]
    sampling = native_receipt.get("sampling", {}).get("rv2v", {})
    clean = native_receipt.get("outputs", {}).get("rv2v", {}).get("normalized_clean_latent")
    if (
        sampling.get("num_frames") != 81
        or sampling.get("num_inference_steps") != 40
        or sampling.get("seed") != candidate["seed"]
        or sampling.get("omega_txt") != candidate["guidance"]["omega_txt"]
        or sampling.get("omega_vid") != candidate["guidance"]["omega_vid"]
        or sampling.get("omega_img") != candidate["guidance"]["omega_img"]
        or sampling.get("target_initialization") != spec_contract.TARGET_INITIALIZATION
    ):
        raise spec_contract.PairRolloutSpecError("native receipt sampling differs from sealed candidate")
    if not isinstance(clean, dict) or clean.get("native_sampler_before_vae_decode") is not True:
        raise spec_contract.PairRolloutSpecError("predecode clean latent is not receipt-bound")
    receipt = {
        "schema_version": spec_contract.RECEIPT_SCHEMA_VERSION,
        "root_spec_raw_sha256": envelope["root_spec_raw_sha256"],
        "candidate_envelope_sha256": envelope["candidate_envelope_sha256"],
        "group_id": envelope["group_id"],
        "visible_gpus": envelope["visible_gpus"],
        "runtime_topology": {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "ordinal": envelope["ordinal"],
        "candidate": candidate,
        "sampling_contract": spec_contract.SAMPLING_CONTRACT,
        "semantic_input_closure": spec_contract.SEMANTIC_INPUT_CLOSURE,
        "native_receipt_path": str(native_path),
        "native_receipt_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "native_receipt_digest": native_receipt.get("receipt_digest"),
        "artifacts": {
            "mp4": native_receipt["outputs"]["rv2v"],
            "predecode_clean_latent": clean,
            "official_initial_gaussian": native_receipt["initial_noise_artifacts"]["rv2v"],
        },
    }
    receipt["receipt_digest"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_path = output / "pair-v5-rollout-receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise spec_contract.PairRolloutSpecError("refusing to overwrite PAIR receipt")
    receipt_path.write_bytes(_canonical(receipt) + b"\n")
    os.chmod(receipt_path, 0o400)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    envelope = spec_contract.load_candidate_envelope(
        args.candidate_spec, args.expected_root_spec_sha256
    )
    expected_visible = ",".join(str(value) for value in envelope["visible_gpus"])
    if os.environ.get("ROCR_VISIBLE_DEVICES") != expected_visible:
        raise spec_contract.PairRolloutSpecError(
            "runtime ROCR_VISIBLE_DEVICES differs from the sealed SP4 assignment"
        )
    if os.environ.get("WORLD_SIZE") not in (None, "4"):
        raise spec_contract.PairRolloutSpecError("runtime world size is not SP4")
    candidate = envelope["candidate"]
    guidance = candidate["guidance"]
    # This wrapper only changes the three native classifier-free guidance
    # coefficients.  Geometry, scheduler, condition pack, and Gaussian draw
    # remain implemented by the pinned native runner.
    native.OMEGA_TEXT = guidance["omega_txt"]
    native.OMEGA_VIDEO = guidance["omega_vid"]
    native.OMEGA_IMAGE = guidance["omega_img"]
    native_argv = [
        "--bernini-root", args.bernini_root,
        "--veomni-root", args.veomni_root,
        "--checkpoint", args.checkpoint,
        "--checkpoint-content-manifest", args.checkpoint_content_manifest,
        "--source-video", candidate["source_video"],
        "--expected-source-sha256", candidate["source_video_sha256"],
        "--action-prompt", candidate["complete_caption"],
        "--expected-action-prompt-sha256", candidate["complete_caption_sha256"],
        "--output-dir", args.output_dir,
        "--arms", "rv2v",
        "--num-inference-steps", "40",
        "--seed", str(candidate["seed"]),
        "--method-source-revision", args.method_source_revision,
        "--method-source-archive-sha256", args.method_source_archive_sha256,
    ]
    status = native.main(native_argv)
    if status == 0 and int(os.environ.get("RANK", "0")) == 0:
        _bind_receipt(args, envelope)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
