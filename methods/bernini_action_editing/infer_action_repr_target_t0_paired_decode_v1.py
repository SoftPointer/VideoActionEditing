#!/usr/bin/env python3
"""Decode the complete T0 route matrix in one loaded Bernini process.

The independent v6 decodes exposed ROCm process-to-process drift even with
PyTorch deterministic algorithms enabled.  This runner therefore loads the
model once, resets the native seed/scheduler for every coordinate, and runs a
fixed paired matrix under one process lifetime.  Step-0 route-off, zero, and
correct plus step-1 route-off/zero are hard negative controls: their terminal
latents must be bit-identical before any step-1 route effect is interpreted.

The target video is never accepted by this program.  Active routes consume
only the detached, hash-bound G1 flow and middle-layer caches admitted by the
production G2a receipt.  The videos remain a one-update canary review and are
never labelled as a successful trained Ours result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

import infer_action_repr_target_t0_matched_decode_v1 as single


SCHEMA_VERSION = "bernini-action-repr-target-t0-paired-decode-v1"
COORDINATES = (
    ("s0_route_off_a", 0, "route_off"),
    ("s0_zero", 0, "zero"),
    ("s0_correct", 0, "correct"),
    ("s1_route_off", 1, "route_off"),
    ("s1_zero", 1, "zero"),
    ("s1_correct", 1, "correct"),
    ("s1_temporal_shuffle", 1, "temporal_shuffle"),
    ("s1_reverse", 1, "reverse"),
    ("s1_incomplete", 1, "incomplete"),
    ("s1_wrong_action", 1, "wrong_action"),
    ("s0_route_off_b", 0, "route_off"),
)
BASELINE_KEYS = (
    "s0_route_off_a",
    "s0_zero",
    "s0_correct",
    "s1_route_off",
    "s1_zero",
    "s0_route_off_b",
)
ACTIVE_STEP1_KEYS = (
    "s1_correct",
    "s1_temporal_shuffle",
    "s1_reverse",
    "s1_incomplete",
    "s1_wrong_action",
)


class PairedDecodeError(RuntimeError):
    """Raised before an unpaired or unauthenticated result is accepted."""


def fail(message: str) -> None:
    raise PairedDecodeError(message)


def _coordinate_map(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = row.get("key")
        if not isinstance(key, str) or key in result:
            fail("paired coordinate keys are absent or duplicated")
        result[key] = row
    expected = {key for key, _, _ in COORDINATES}
    if set(result) != expected:
        fail("paired coordinate closure differs")
    return result


def paired_gate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Evaluate the exact baseline controls and route-effect prerequisites."""

    by_key = _coordinate_map(rows)
    baseline_hash = by_key["s0_route_off_a"].get("latent_sha256")
    single.require_sha256(baseline_hash, label="paired baseline latent SHA-256")
    baseline_exact = {
        key: by_key[key].get("latent_sha256") == baseline_hash
        for key in BASELINE_KEYS
    }
    active_hashes = {
        key: by_key[key].get("latent_sha256") for key in ACTIVE_STEP1_KEYS
    }
    for key, value in active_hashes.items():
        single.require_sha256(value, label=f"{key} latent SHA-256")
    correct_changed = active_hashes["s1_correct"] != baseline_hash
    correct_control_distinct = {
        key: active_hashes[key] != active_hashes["s1_correct"]
        for key in ACTIVE_STEP1_KEYS
        if key != "s1_correct"
    }
    return {
        "baseline_latent_sha256": baseline_hash,
        "baseline_negative_controls_exact": baseline_exact,
        "baseline_gate_passed": all(baseline_exact.values()),
        "step1_correct_latent_changed_from_baseline": correct_changed,
        "step1_correct_distinct_from_controls": correct_control_distinct,
        "route_effect_detected": correct_changed,
        "route_selectivity_claimed": False,
        "quality_success_claimed": False,
    }


def _relative_metrics(value: torch.Tensor, reference: torch.Tensor) -> Mapping[str, float]:
    if value.shape != reference.shape or value.dtype != reference.dtype:
        fail("paired latent metric tensors differ in shape or dtype")
    left = value.detach().float().cpu()
    right = reference.detach().float().cpu()
    delta = left - right
    rms = float(torch.sqrt(torch.mean(delta.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(right.square())).item())
    denominator = max(reference_rms, 1.0e-12)
    cosine_denominator = float(torch.linalg.vector_norm(left).item()) * float(
        torch.linalg.vector_norm(right).item()
    )
    cosine = (
        float(torch.sum(left * right).item()) / cosine_denominator
        if cosine_denominator > 0.0
        else 1.0
    )
    result = {
        "rms": rms,
        "relative_rms": rms / denominator,
        "max_abs": float(delta.abs().max().item()),
        "cosine": cosine,
    }
    if not all(math.isfinite(number) for number in result.values()):
        fail("paired latent metrics are non-finite")
    return result


def _prepared_output_root(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        fail("paired output root must be absolute")
    root = requested.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        fail("paired output root must be one prepared real directory")
    expected_directories = {Path("cells"), *(
        Path("cells") / key for key, _, _ in COORDINATES
    )}
    observed_directories = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed_directories != expected_directories or any(
        path.is_file() or path.is_symlink() for path in root.rglob("*")
    ):
        fail("paired output root must contain only the fixed empty cell directories")
    return root


def _save_extra_video(save_output: Any, decoded: Any, path: Path) -> None:
    if path.exists() or path.is_symlink():
        fail(f"paired video coordinate is already consumed: {path}")
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    if temporary.exists() or temporary.is_symlink():
        fail(f"stale paired video temporary file exists: {temporary}")
    save_output(decoded, str(temporary), fps=25)
    if not temporary.is_file() or temporary.is_symlink() or temporary.stat().st_size <= 0:
        fail("paired video encoder did not produce one plain non-empty file")
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise PairedDecodeError(f"refusing to overwrite paired video: {path}") from error
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


class _PrimaryClampProxy:
    def __init__(self, capture: Mapping[str, Any]) -> None:
        self.capture = capture

    def as_dict(self) -> Mapping[str, Any]:
        value = self.capture.get("primary_source_onset_solver_trace")
        if not isinstance(value, Mapping):
            fail("paired primary source-onset trace is unavailable")
        return value


def validate_paired_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail("paired receipt must be one mapping")
    row = dict(value)
    declared = row.pop("receipt_digest", None)
    single.require_sha256(declared, label="paired receipt digest")
    if single.object_sha256(row) != declared:
        fail("paired receipt digest differs")
    cells = row.get("cells")
    gate = row.get("paired_gate")
    runtime = row.get("runtime")
    claims = row.get("claim_boundary")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("complete") is not True
        or row.get("case_id") != single.CASE_ID
        or not isinstance(cells, list)
        or len(cells) != len(COORDINATES)
        or not isinstance(gate, Mapping)
        or not isinstance(runtime, Mapping)
        or runtime.get("one_model_construction") is not True
        or runtime.get("one_native_main_call") is not True
        or runtime.get("sample_calls") != len(COORDINATES)
        or runtime.get("world_size") != 4
        or runtime.get("ulysses_size") != 4
        or runtime.get("strict_deterministic_algorithms") is not True
        or not isinstance(claims, Mapping)
        or claims.get("ours_claimed") is not False
        or claims.get("quality_success_claimed") is not False
        or claims.get("route_selectivity_claimed") is not False
    ):
        fail("paired receipt contract differs")
    expected = list(COORDINATES)
    for cell, (key, step, route) in zip(cells, expected):
        if (
            not isinstance(cell, Mapping)
            or cell.get("key") != key
            or cell.get("checkpoint_step") != step
            or cell.get("route_kind") != route
            or cell.get("four_rank_latent_hashes_equal") is not True
            or cell.get("shared_step_calls") != 2 * single.NUM_INFERENCE_STEPS
            or cell.get("paired_cfg_timestep_digests_equal") is not True
            or cell.get("video_generated") is not True
        ):
            fail(f"paired cell contract differs: {key}")
        for field in ("latent_sha256", "video_sha256", "decoded_rgb24_sha256"):
            single.require_sha256(cell.get(field), label=f"{key} {field}")
    observed_gate = paired_gate(cells)
    if dict(gate) != dict(observed_gate):
        fail("paired gate replay differs")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--t0-output", required=True)
    parser.add_argument("--g2a-receipt", required=True)
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--flow-cohort-receipt", required=True)
    parser.add_argument("--middle-cohort-receipt", required=True)
    parser.add_argument("--paired-output-root", required=True)
    args, native_argv = parser.parse_known_args(argv)

    output_root = _prepared_output_root(args.paired_output_root)
    primary_output = output_root / "cells" / COORDINATES[0][0] / "output.mp4"
    native_output = Path(single.option(native_argv, "--output")).expanduser()
    if native_output != primary_output:
        fail("native output must be the fixed first paired coordinate")

    from safetensors.torch import load_file

    prepared: list[dict[str, Any]] = []
    for key, step, route_kind in COORDINATES:
        validated = single.validate_decode_inputs(
            t0_output=args.t0_output,
            g2a_receipt_path=args.g2a_receipt,
            manifest_path=args.experiment_manifest,
            flow_cohort_path=args.flow_cohort_receipt,
            middle_cohort_path=args.middle_cohort_receipt,
            checkpoint_step=step,
            route_kind=route_kind,
            native_argv=native_argv,
        )
        route_cpu, route_facts = single.materialize_route(
            route_kind=route_kind,
            checkpoint_step=step,
            validated=validated,
        )
        route_cpu, route_transport = (
            single.expand_target_only_route_for_native_mv2v(route_cpu)
        )
        state = load_file(str(validated["state_path"]), device="cpu")
        prepared.append(
            {
                "key": key,
                "checkpoint_step": step,
                "route_kind": route_kind,
                "validated": validated,
                "route_cpu": route_cpu,
                "route_facts": route_facts,
                "route_transport": route_transport,
                "state": state,
                "output": output_root / "cells" / key / "output.mp4",
            }
        )

    native_argv = [value for value in native_argv if value != "--base-only"]
    native_argv.append("--base-only")

    capture: dict[str, Any] = {
        "constructed": 0,
        "patched_sample_calls": 0,
        "cells": [],
        "rank0_latents": [],
        "decoded_in_memory_sha256": {},
        "deterministic_runtime": single.enable_strict_deterministic_runtime(),
    }
    original_activate = single.native.trainer.activate_source_trees
    original_clamp = single.native.hard_phase0_source_trajectory_clamp

    @contextmanager
    def paired_outer_clamp(*_clamp_args: Any, **_clamp_kwargs: Any) -> Any:
        # Every native sample below gets its own real clamp.  This outer proxy
        # only gives infer_lora the primary sample's authenticated trace.
        yield _PrimaryClampProxy(capture)

    def patched_activate(*activate_args: Any, **activate_kwargs: Any) -> Any:
        result = original_activate(*activate_args, **activate_kwargs)
        import bernini.io_utils as io_module
        import bernini.models.renderer as renderer_module
        import bernini.pipeline as pipeline_module

        original_class = renderer_module.BerniniRendererModel
        original_vae_decode = pipeline_module._vae_decode
        original_save_output = io_module.save_output
        capture["renderer_module"] = renderer_module
        capture["original_renderer_class"] = original_class
        capture["pipeline_module"] = pipeline_module
        capture["original_vae_decode"] = original_vae_decode

        def constructed(config: Any) -> Any:
            model = original_class(config)
            handle = single.g2a.install_action_repr_g2a_adapter(
                model,
                block_indices=single.BLOCK_INDICES,
                hidden_width=1536,
                flow_width=12,
                bottleneck_width=256,
                middle_width=256,
                enable_source_copy_adapter=False,
            )
            original_sample = model.sample

            def sampled(*sample_args: Any, **sample_kwargs: Any) -> Any:
                capture["patched_sample_calls"] += 1
                if capture["patched_sample_calls"] != 1:
                    fail("native paired decoder invoked its patched sample more than once")
                source_values = sample_kwargs.get("multi_video_vae_latents")
                if not isinstance(source_values, list) or len(source_values) != 1:
                    fail("paired native sample lacks exactly one source latent")
                source_latent = source_values[0]
                device = next(model.parameters()).device
                rank = int(os.environ.get("RANK", "-1"))
                latent_values: list[torch.Tensor] = []
                primary_generated: Optional[torch.Tensor] = None

                for index, item in enumerate(prepared):
                    handle.load_state_dict_strict(item["state"])
                    state_digest = single.adapter_state_digest(handle)
                    expected_digest = item["validated"]["state_row"]["state_digest"]
                    if state_digest != expected_digest:
                        fail(f"{item['key']} loaded adapter state digest differs")
                    if item["checkpoint_step"] == 0 and not handle.output_gates_are_byte_zero():
                        fail("paired step-0 adapter is not a byte-zero gate")
                    if item["checkpoint_step"] == 1 and handle.output_gates_are_byte_zero():
                        fail("paired step-1 adapter remained a byte-zero gate")

                    route = single.route_to_device(item["route_cpu"], device)
                    diffusion = getattr(model, "diff_dec", None)
                    original_shared_step = getattr(diffusion, "shared_step", None)
                    if diffusion is None or not callable(original_shared_step):
                        fail("paired native diffusion shared_step is unavailable")
                    instance = vars(diffusion)
                    had_instance = "shared_step" in instance
                    previous_instance = instance.get("shared_step")
                    timestep_digests: list[str] = []

                    def routed_shared_step(*shared_args: Any, **shared_kwargs: Any) -> Any:
                        if single.g2a.current_action_representation_route() is not route:
                            fail("paired shared_step escaped its authenticated route")
                        timesteps = shared_kwargs.get("timesteps")
                        if timesteps is None and len(shared_args) >= 3:
                            timesteps = shared_args[2]
                        if not isinstance(timesteps, torch.Tensor):
                            fail("paired shared_step timestep tensor is absent")
                        noisy_latents = shared_kwargs.get("noisy_latents")
                        if noisy_latents is None and len(shared_args) >= 2:
                            noisy_latents = shared_args[1]
                        vae_lengths = shared_kwargs.get("batch_vae_seqlen")
                        if (
                            not isinstance(noisy_latents, torch.Tensor)
                            or noisy_latents.ndim != 3
                            or int(noisy_latents.shape[1])
                            != int(route.layout.total_tokens)
                            or vae_lengths != [int(route.layout.total_tokens)]
                        ):
                            fail("paired native mv2v source+target token layout differs")
                        timestep_digests.append(single.g2a.tensor_sha256(timesteps))
                        return original_shared_step(*shared_args, **shared_kwargs)

                    try:
                        setattr(diffusion, "shared_step", routed_shared_step)
                        with original_clamp(
                            diffusion,
                            source_latent,
                            expected_steps=single.NUM_INFERENCE_STEPS,
                        ) as clamp_trace:
                            with single.g2a.action_representation_route(route):
                                generated = original_sample(*sample_args, **sample_kwargs)
                    finally:
                        if had_instance:
                            setattr(diffusion, "shared_step", previous_instance)
                        else:
                            delattr(diffusion, "shared_step")

                    if (
                        not isinstance(generated, torch.Tensor)
                        or len(timestep_digests) != 2 * single.NUM_INFERENCE_STEPS
                        or any(
                            timestep_digests[position] != timestep_digests[position + 1]
                            for position in range(0, len(timestep_digests), 2)
                        )
                    ):
                        fail(f"{item['key']} paired sampling closure differs")
                    latent_sha = single.g2a.tensor_sha256(generated)
                    gathered: list[Any] = [None] * int(os.environ.get("WORLD_SIZE", "0"))
                    torch.distributed.all_gather_object(gathered, latent_sha)
                    if gathered != [latent_sha] * 4:
                        fail(f"{item['key']} terminal latent differs across ranks")
                    audit = single.audit_frozen_inference_parameters(
                        handle,
                        expected_adapter_state_digest=expected_digest,
                    )
                    cell = {
                        "key": item["key"],
                        "checkpoint_step": item["checkpoint_step"],
                        "route_kind": item["route_kind"],
                        "route_facts": item["route_facts"],
                        "route_transport": item["route_transport"],
                        "adapter_state_sha256": item["validated"]["state_row"]["state_sha256"],
                        "adapter_state_digest": expected_digest,
                        "latent_sha256": latent_sha,
                        "latent_shape": list(map(int, generated.shape)),
                        "latent_dtype": str(generated.dtype),
                        "four_rank_latent_hashes_equal": True,
                        "ordered_rank_latent_sha256": gathered,
                        "shared_step_calls": len(timestep_digests),
                        "paired_cfg_timestep_digests_equal": True,
                        "timestep_pair_digest": single.object_sha256(timestep_digests[::2]),
                        "source_onset_solver_trace": clamp_trace.as_dict(),
                        "base_parameter_identity_unchanged": audit[
                            "base_parameter_identity_unchanged"
                        ],
                        "base_requires_grad_false": audit["base_requires_grad_false"],
                        "adapter_requires_grad_false": audit[
                            "adapter_requires_grad_false"
                        ],
                        "adapter_state_digest_unchanged": audit[
                            "adapter_state_digest_unchanged"
                        ],
                    }
                    capture["cells"].append(cell)
                    if rank == 0:
                        progress = dict(cell)
                        progress["schema_version"] = (
                            "bernini-action-repr-target-t0-paired-latent-progress-v1"
                        )
                        progress["complete_video_cell"] = False
                        progress["claim_boundary"] = {
                            "latent_only": True,
                            "ours_claimed": False,
                            "quality_success_claimed": False,
                        }
                        progress["receipt_digest"] = single.object_sha256(progress)
                        single._write_exclusive_json(
                            item["output"].parent / "latent_progress.json", progress
                        )
                    if index == 0:
                        primary_generated = generated.detach().clone()
                        capture["primary_source_onset_solver_trace"] = cell[
                            "source_onset_solver_trace"
                        ]
                    if rank == 0:
                        latent_values.append(generated.detach().to("cpu").clone().contiguous())
                    del route

                if rank == 0:
                    if len(latent_values) != len(COORDINATES):
                        fail("rank-0 paired latent inventory differs")
                    baseline = latent_values[0]
                    correct = latent_values[5]
                    for cell, latent in zip(capture["cells"], latent_values):
                        cell["latent_delta_from_s0_route_off"] = _relative_metrics(
                            latent, baseline
                        )
                        cell["latent_delta_from_s1_correct"] = _relative_metrics(
                            latent, correct
                        )
                    capture["rank0_latents"] = latent_values
                if single.deterministic_runtime_flags() != capture["deterministic_runtime"]:
                    fail("paired native sampler changed deterministic runtime flags")
                if primary_generated is None:
                    fail("paired native sampler did not retain its primary latent")
                return primary_generated

            model.sample = sampled
            capture["constructed"] += 1
            capture["architecture"] = handle.architecture_receipt()
            return model

        def paired_vae_decode(vae: Any, primary_latent: torch.Tensor) -> Any:
            if int(os.environ.get("RANK", "-1")) != 0:
                fail("paired VAE decode ran outside rank zero")
            latents = capture.get("rank0_latents")
            if not isinstance(latents, list) or len(latents) != len(COORDINATES):
                fail("paired VAE decode lacks the complete latent inventory")
            if single.g2a.tensor_sha256(primary_latent) != capture["cells"][0]["latent_sha256"]:
                fail("native primary latent changed before VAE decode")
            primary_decoded = None
            for index, (item, latent) in enumerate(zip(prepared, latents)):
                selected = primary_latent if index == 0 else latent.to(primary_latent.device)
                decoded = original_vae_decode(vae, selected)
                if getattr(decoded, "shape", None) != (81, 368, 656, 3):
                    fail(f"{item['key']} in-memory VAE output geometry differs")
                capture["decoded_in_memory_sha256"][item["key"]] = hashlib.sha256(
                    decoded.tobytes(order="C")
                ).hexdigest()
                if index == 0:
                    primary_decoded = decoded
                else:
                    _save_extra_video(original_save_output, decoded, item["output"])
                if index != 0:
                    del selected
            capture["rank0_latents"] = []
            if primary_decoded is None:
                fail("paired primary VAE output was not produced")
            return primary_decoded

        renderer_module.BerniniRendererModel = constructed
        pipeline_module._vae_decode = paired_vae_decode
        return result

    single.native.trainer.activate_source_trees = patched_activate
    single.native.hard_phase0_source_trajectory_clamp = paired_outer_clamp
    try:
        status = single.native.main(native_argv)
    finally:
        single.native.trainer.activate_source_trees = original_activate
        single.native.hard_phase0_source_trajectory_clamp = original_clamp
        renderer_module = capture.get("renderer_module")
        if renderer_module is not None:
            renderer_module.BerniniRendererModel = capture["original_renderer_class"]
        pipeline_module = capture.get("pipeline_module")
        if pipeline_module is not None:
            pipeline_module._vae_decode = capture["original_vae_decode"]
    if (
        status != 0
        or capture["constructed"] != 1
        or capture["patched_sample_calls"] != 1
        or len(capture["cells"]) != len(COORDINATES)
    ):
        fail("paired native decode did not close one model and the fixed sample matrix")

    if int(os.environ.get("RANK", "0")) == 0:
        native_receipt = primary_output.with_name(primary_output.name + ".receipt.json")
        if not primary_output.is_file() or not native_receipt.is_file():
            fail("paired primary native output closure is incomplete")
        for item, cell in zip(prepared, capture["cells"]):
            output = item["output"]
            if not output.is_file() or output.is_symlink():
                fail(f"paired output is absent: {item['key']}")
            probe = single.validate_video_artifact(output)
            cell.update(
                {
                    "video_generated": True,
                    "output": str(output),
                    "video_sha256": single.file_sha256(output),
                    "decoded_rgb24_sha256": probe["decoded_rgb24_sha256"],
                    "decoded_video_probe": probe,
                    "decoded_in_memory_sha256": capture[
                        "decoded_in_memory_sha256"
                    ][item["key"]],
                }
            )
        gate = paired_gate(capture["cells"])
        reference = prepared[0]["validated"]
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "case_id": single.CASE_ID,
            "paired_gate": gate,
            "cells": capture["cells"],
            "native_primary_receipt": str(native_receipt),
            "native_primary_receipt_sha256": single.file_sha256(native_receipt),
            "t0_receipt_sha256": reference["t0_receipt_sha256"],
            "production_g2a_receipt_sha256": reference["g2a_receipt_sha256"],
            "manifest_sha256": reference["manifest_sha256"],
            "runtime": {
                "world_size": int(os.environ.get("WORLD_SIZE", "0")),
                "ulysses_size": 4,
                "one_model_construction": capture["constructed"] == 1,
                "one_native_main_call": True,
                "sample_calls": len(COORDINATES),
                "num_inference_steps_per_cell": single.NUM_INFERENCE_STEPS,
                "same_process_lifetime": True,
                "same_loaded_base_parameter_objects": True,
                "seed_and_scheduler_reset_per_cell": True,
                "strict_deterministic_algorithms": capture[
                    "deterministic_runtime"
                ]["deterministic_algorithms_enabled"],
                **capture["deterministic_runtime"],
            },
            "information_firewall": {
                "target_media_opened_by_renderer": False,
                "target_rgb_vae_clean_latent_received": False,
                "detached_action_cache_only": True,
                "real_target_enters_active_routes_only_as_g1_flow_middle_cache": True,
            },
            "claim_boundary": {
                "one_step_canary_decode_only": True,
                "ours_claimed": False,
                "quality_success_claimed": False,
                "route_selectivity_claimed": False,
                "manual_visual_review_required": True,
            },
            "source_lock": {
                Path(__file__).name: single.file_sha256(Path(__file__).resolve()),
                Path(single.__file__).name: single.file_sha256(
                    Path(single.__file__).resolve()
                ),
                Path(single.g2a.__file__).name: single.file_sha256(
                    Path(single.g2a.__file__).resolve()
                ),
                Path(single.native.__file__).name: single.file_sha256(
                    Path(single.native.__file__).resolve()
                ),
            },
        }
        receipt["receipt_digest"] = single.object_sha256(receipt)
        validate_paired_receipt(receipt)
        receipt_path = output_root / "paired_receipt.json"
        single._write_exclusive_json(receipt_path, receipt)
        validate_paired_receipt(json.loads(receipt_path.read_text(encoding="ascii")))
        print(
            json.dumps(
                {
                    "complete": True,
                    "paired_gate_passed": gate["baseline_gate_passed"],
                    "route_effect_detected": gate["route_effect_detected"],
                    "output_root": str(output_root),
                    "ours_or_quality_success_claimed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if gate["baseline_gate_passed"] else 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
