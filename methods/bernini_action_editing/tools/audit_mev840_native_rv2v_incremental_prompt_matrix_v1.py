#!/usr/bin/env python3
"""Fail-closed postflight for the fresh MEV840 native prompt matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Optional, Sequence


ARMS = ("P0", "P1", "P2")
SEEDS = (2027, 2028)
SOURCE_SHA = "a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646"
ARCHIVE_SHA = "46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115"
REVISION = "ac22e19ffd109a2d6b85c32c64463b0be8373792"
LAUNCHER_SHA = "af044659eceb3bb1cf90ead68363efcb91237a4fb320aeeb507929726b768c0a"
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
EXPECTED_LOCATION = {
    2027: {"job_id": "143808", "node": "auh7-1b-gpu-292"},
    2028: {"job_id": "147873", "node": "auh7-1b-gpu-284"},
}
EXPECTED_FILES = {
    "candidate.complete.json",
    "receipt.json",
    "rv2v.mp4",
    "rv2v.normalized-clean-latent.safetensors",
    "rv2v.official-initial-gaussian.safetensors",
    "source.normalized-clean-latent.safetensors",
}
COMPLETE_KEYS = {
    "schema",
    "complete",
    "arm",
    "seed",
    "slurm",
    "prompt_utf8_sha256",
    "video_sha256",
    "native_receipt_sha256",
    "zero_update",
    "generator_target_video_read",
    "generator_target_action_json_read",
    "generator_target_rgb_feature_embedding_latent_qkv_gaussian_read",
    "upstream_release_entrypoint_authorized",
    "immutable_release_bytes_reused_under_current_user_authorized_launcher",
    "runtime_extracted_to_node_local_scratch_exact19",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_plain(path: Path) -> bytes:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), f"not a plain file: {path}")
    return path.read_bytes()


def load_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    raw = read_plain(path)
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, raw


def canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(raw)


def verify_receipt_digest(value: Mapping[str, Any]) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    require(isinstance(declared, str) and canonical_digest(unsigned) == declared, "receipt digest differs")


def artifact_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("content_sha256"),
        row.get("raw_storage_sha256"),
        tuple(row.get("shape", ())),
        row.get("dtype"),
        row.get("finite"),
        row.get("numel"),
        row.get("byte_count"),
    )


def condition_identity(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    identities = receipt["condition_identities"]
    require(identities["full_source_video"].get("all_rank_exact") is True, "full source all-rank identity differs")
    broadcasts = identities["rank_zero_broadcasts"]
    full_broadcast = broadcasts["full_source_video"]
    require(
        full_broadcast.get("authoritative_rank") == 0
        and full_broadcast.get("broadcast_before_renderer") is True
        and full_broadcast.get("prebroadcast_metadata_all_rank_exact") is True,
        "full source broadcast authority differs",
    )
    rows = [("full_source_video", artifact_identity(identities["full_source_video"]["identity"]))]
    for index in ("0", "27", "53", "80"):
        require(identities["references"][index].get("all_rank_exact") is True, f"reference {index} all-rank identity differs")
        broadcast = broadcasts["references"][index]
        require(
            broadcast.get("authoritative_rank") == 0
            and broadcast.get("broadcast_before_renderer") is True
            and broadcast.get("prebroadcast_metadata_all_rank_exact") is True,
            f"reference {index} broadcast authority differs",
        )
        rows.append((index, artifact_identity(identities["references"][index]["identity"])))
    return tuple(rows)


def noise_identity(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    row = receipt["initial_noise_artifacts"]["rv2v"]
    return (
        row.get("content_sha256"),
        row.get("raw_value_sha256"),
        row.get("tensor_value_sha256"),
        tuple(row.get("shape", ())),
        row.get("dtype"),
        row.get("generator_initial_seed"),
    )


def configuration_projection(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    inp = receipt["input"]
    sampling = receipt["sampling"]["rv2v"]
    prompt = dict(receipt["prompt_contract"]["rv2v"])
    prompt.pop("full_prompt_sha256", None)
    return {
        "schema": receipt["schema_version"],
        "method": receipt["method"],
        "commits": [receipt["bernini_commit"], receipt["veomni_commit"]],
        "inference_files": receipt["bernini_inference_files"],
        "checkpoint": receipt["checkpoint"],
        "method_source": [receipt["method_source_archive_sha256"], receipt["method_source_revision"]],
        "arms": receipt["arms"],
        "input": {
            "accepted": inp["accepted_external_conditions"],
            "source": inp["source_video_sha256"],
            "target": inp["target_video"],
            "anchor": inp["external_first_frame_anchor"],
            "controls": inp["external_mask_flow_pose_track_trajectory"],
            "reference": inp["external_reference_image_or_video"],
        },
        "freeze": receipt["freeze_certificate"],
        "conditioning": receipt["conditioning"],
        "preprocessing": receipt["preprocessing"],
        "latent_geometry": receipt["latent_geometry"],
        "runtime_versions": receipt["runtime_versions"],
        "sampling_without_seed": {key: value for key, value in sampling.items() if key != "seed"},
        "prompt_contract_without_prompt_identity": prompt,
    }


def verify_artifact_path(candidate: Path, row: Mapping[str, Any], name: str) -> None:
    expected = candidate / name
    path = Path(row.get("path", ""))
    require(path == expected, f"artifact path differs: {name}")
    require(sha256_bytes(read_plain(path)) == row.get("sha256"), f"artifact SHA differs: {name}")


def audit(root: Path, authority_path: Path, launcher_path: Path, waves: Sequence[str]) -> Mapping[str, Any]:
    authority, _ = load_json(authority_path)
    require(sha256_bytes(read_plain(launcher_path)) == LAUNCHER_SHA, "deployed launcher SHA differs")
    require(authority.get("schema") == "mev840-native-rv2v-incremental-prompt-matrix-v1", "authority schema differs")
    require(tuple(waves) == ARMS[: len(waves)] and waves, "waves must be a nonempty P0/P1/P2 prefix")
    actual_root = {p.name for p in root.iterdir()}
    require(actual_root == set(waves), "root member closure differs")

    rows: list[Mapping[str, Any]] = []
    config_digests: list[str] = []
    noises: dict[int, list[tuple[Any, ...]]] = {seed: [] for seed in SEEDS}
    conditions: dict[int, list[tuple[Any, ...]]] = {seed: [] for seed in SEEDS}
    for arm in waves:
        arm_root = root / arm
        require(arm_root.is_dir() and not arm_root.is_symlink(), f"arm root differs: {arm}")
        require({p.name for p in arm_root.iterdir()} == {f"seed{seed}" for seed in SEEDS}, f"seed closure differs: {arm}")
        for seed in SEEDS:
            candidate = arm_root / f"seed{seed}"
            require(candidate.is_dir() and not candidate.is_symlink(), f"candidate root differs: {arm}/{seed}")
            actual = {p.name for p in candidate.iterdir()}
            require(actual == EXPECTED_FILES, f"candidate file closure differs: {arm}/{seed}: {sorted(actual)}")
            for path in candidate.iterdir():
                read_plain(path)

            complete, _ = load_json(candidate / "candidate.complete.json")
            receipt, receipt_raw = load_json(candidate / "receipt.json")
            video_raw = read_plain(candidate / "rv2v.mp4")
            require(set(complete) == COMPLETE_KEYS, f"complete key closure differs: {arm}/{seed}")
            require(complete.get("schema") == "mev840-native-rv2v-matrix-candidate-complete-v1", "complete schema differs")
            require(complete.get("complete") is True and complete.get("zero_update") is True, "complete flags differ")
            require(complete.get("arm") == arm and complete.get("seed") == seed, "complete cell identity differs")
            expected_location = EXPECTED_LOCATION[seed]
            slurm = complete.get("slurm", {})
            require(slurm.get("job_id") == expected_location["job_id"] and slurm.get("node") == expected_location["node"], "complete location differs")
            require(isinstance(slurm.get("step_id"), str) and slurm["step_id"].isdigit(), "step id differs")
            for key in (
                "generator_target_video_read",
                "generator_target_action_json_read",
                "generator_target_rgb_feature_embedding_latent_qkv_gaussian_read",
                "upstream_release_entrypoint_authorized",
            ):
                require(complete.get(key) is False, f"forbidden complete flag differs: {key}")
            require(complete.get("immutable_release_bytes_reused_under_current_user_authorized_launcher") is True, "launcher authority differs")
            require(complete.get("runtime_extracted_to_node_local_scratch_exact19") is True, "scratch closure flag differs")

            prompt_row = authority["prompts"][arm]
            prompt_sha = prompt_row["full_prompt_utf8_sha256"]
            require(complete.get("prompt_utf8_sha256") == prompt_sha, "complete prompt differs")
            require(complete.get("video_sha256") == sha256_bytes(video_raw), "complete video SHA differs")
            require(complete.get("native_receipt_sha256") == sha256_bytes(receipt_raw), "complete receipt SHA differs")
            verify_receipt_digest(receipt)
            require(receipt.get("bernini_commit") == BERNINI_COMMIT, "Bernini commit differs")
            require(receipt.get("veomni_commit") == VEOMNI_COMMIT, "VeOmni commit differs")
            require(receipt.get("checkpoint", {}).get("tree_sha256") == CHECKPOINT_TREE_SHA, "checkpoint tree differs")
            require(receipt.get("interpretation", {}).get("training_performed") is False, "training was performed")
            require(receipt.get("scientific_claim_authorized") is False, "scientific claim flag differs")

            inp = receipt["input"]
            require(receipt.get("arms") == ["rv2v"], "arm closure differs")
            require(inp.get("accepted_external_conditions") == ["source_video", "action_prompt"], "input allowlist differs")
            require(inp.get("source_video_sha256") == SOURCE_SHA, "source differs")
            require(inp.get("action_prompt_utf8_sha256") == prompt_sha, "receipt prompt differs")
            require(inp.get("action_prompt_utf8_bytes") == prompt_row["full_prompt_utf8_bytes"], "receipt prompt length differs")
            for key in (
                "target_video",
                "external_first_frame_anchor",
                "external_mask_flow_pose_track_trajectory",
                "external_reference_image_or_video",
            ):
                require(inp.get(key) is False, f"forbidden receipt input differs: {key}")
            require(receipt.get("freeze_certificate") == {"base_frozen": True, "lora_module_count": 0, "trainable_parameter_elements": 0, "trainable_parameter_tensors": 0}, "freeze differs")
            require(receipt.get("method_source_archive_sha256") == ARCHIVE_SHA and receipt.get("method_source_revision") == REVISION, "method source differs")

            sampling = receipt["sampling"]["rv2v"]
            require(sampling.get("seed") == seed and sampling.get("num_inference_steps") == 40, "sampling identity differs")
            require(sampling.get("guidance_mode") == "rv2v" and sampling.get("custom_sampler_or_scheduler") is False, "native scheduler differs")
            require(sampling.get("target_initialization") == "official_gen_wanx22_fresh_gaussian" and sampling.get("target_mixed_with_source_latent") is False, "native initialization differs")
            noise = receipt["initial_noise_artifacts"]["rv2v"]
            require(noise.get("external_initial_noise_injection") is False and noise.get("source_or_target_derived") is False and noise.get("sampler_noise_replacement") is False, "Gaussian authority differs")
            require(noise.get("all_rank_identity", {}).get("all_rank_exact") is True, "Gaussian all-rank identity differs")

            output = receipt["outputs"]["rv2v"]
            require((output.get("frame_count"), output.get("fps"), output.get("width"), output.get("height")) == (81, 25, 656, 368), "media metadata differs")
            require(Path(output.get("path", "")) == candidate / "rv2v.mp4" and output.get("sha256") == sha256_bytes(video_raw), "receipt video binding differs")
            verify_artifact_path(candidate, output["normalized_clean_latent"], "rv2v.normalized-clean-latent.safetensors")
            verify_artifact_path(candidate, noise, "rv2v.official-initial-gaussian.safetensors")
            verify_artifact_path(candidate, receipt["source_condition_artifact"], "source.normalized-clean-latent.safetensors")
            gate = receipt["resource_lifecycle"]["world4_load_completion_gate"]
            require(gate.get("hostname") == expected_location["node"], "receipt node differs")
            require(gate.get("all_four_renderer_loads_complete_before_first_native_sampling") is True, "load gate differs")

            config_digests.append(canonical_digest(configuration_projection(receipt)))
            noises[seed].append(noise_identity(receipt))
            conditions[seed].append(condition_identity(receipt))
            rows.append({
                "candidate_id": f"{arm.lower()}_s{seed}",
                "arm": arm,
                "seed": seed,
                "job_id": slurm["job_id"],
                "step_id": slurm["step_id"],
                "node": slurm["node"],
                "prompt_sha256": prompt_sha,
                "video_path": str(candidate / "rv2v.mp4"),
                "video_sha256": sha256_bytes(video_raw),
                "receipt_sha256": sha256_bytes(receipt_raw),
                "receipt_digest": receipt["receipt_digest"],
            })

    require(len(set(config_digests)) == 1, "canonical native configuration differs")
    paired = len(waves) >= 2
    if paired:
        for seed in SEEDS:
            require(len(set(noises[seed])) == 1, f"same-seed Gaussian differs: {seed}")
            require(len(set(conditions[seed])) == 1, f"same-seed five-condition identity differs: {seed}")
    return {
        "schema": "mev840-native-rv2v-incremental-prompt-matrix-postflight-v1",
        "waves": list(waves),
        "cell_count": len(rows),
        "complete": True,
        "launcher_sha256": LAUNCHER_SHA,
        "canonical_configuration_sha256": config_digests[0],
        "same_seed_gaussian_exact": True if paired else None,
        "same_seed_five_source_conditions_exact": True if paired else None,
        "generator_target_reads": False,
        "upstream_release_entrypoint_authorized": False,
        "cells": rows,
    }


def write_fresh(path: Path, value: Mapping[str, Any]) -> str:
    require(not path.exists() and not path.is_symlink(), f"output is not fresh: {path}")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".matrix-postflight-", dir=path.parent)
    try:
        os.write(fd, raw)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_bytes(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--waves", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(
        args.root.resolve(strict=True),
        args.authority.resolve(strict=True),
        args.launcher.resolve(strict=True),
        tuple(args.waves),
    )
    if args.output is None:
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    else:
        digest = write_fresh(args.output, result)
        print(json.dumps({"complete": True, "output": str(args.output), "sha256": digest}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
