#!/usr/bin/env python3
"""Matched native decode for the target-action T0 step-0/step-1 canary.

This wrapper keeps ``infer_lora.py`` as the sole Bernini sampler/encoder and
injects only the hash-bound G2a action route used by the one-step T0 runner.
It accepts no target media.  The real target enters only through detached G1
flow/middle cache files already authenticated by the production G2a receipt.

The first decode deliberately reuses the single middle sigma trained by T0
(``0.55``) for every denoising call.  That is an explicit deployment
extrapolation, not a claim of per-timestep middle matching.  A later
three-sigma schedule is a separate ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

import action_repr_g2a_adapter_v1 as g2a
import audit_action_repr_g2a_world4_v1 as g2a_world4
import infer_lora as native
import train_action_repr_target_t0_canary_retry8_v1 as retry8


SCHEMA_VERSION = "bernini-action-repr-target-t0-matched-decode-v1"
CASE_ID = "0be6494dfac3"
BLOCK_INDICES = (6, 12, 18, 24)
ACTIVE_ROUTES = (
    "correct",
    "temporal_shuffle",
    "reverse",
    "incomplete",
    "wrong_action",
)
ROUTES = ("route_off", "zero", *ACTIVE_ROUTES)
MIDDLE_SIGMA_INDEX = 1
MIDDLE_SIGMA = 0.55
NUM_INFERENCE_STEPS = 40


class MatchedDecodeError(RuntimeError):
    """Raised before an unauthenticated or unmatched decode is accepted."""


def fail(message: str) -> None:
    raise MatchedDecodeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise MatchedDecodeError(f"value is not canonical finite JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} must be a lowercase SHA-256")
    return value


def read_json(path: Path | str, *, label: str) -> tuple[Path, Mapping[str, Any], str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        fail(f"{label} must be a plain file")
    try:
        value = json.loads(resolved.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatchedDecodeError(f"cannot read {label}: {error}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} must contain one JSON object")
    return resolved, value, file_sha256(resolved)


def option(argv: Sequence[str], name: str) -> str:
    try:
        index = list(argv).index(name)
        return list(argv)[index + 1]
    except (ValueError, IndexError) as error:
        raise MatchedDecodeError(f"native arguments lack {name}") from error


def _manifest_case(
    manifest_path: Path | str,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    _, manifest, observed = read_json(manifest_path, label="experiment manifest")
    if observed != require_sha256(expected_sha256, label="manifest SHA-256"):
        fail("experiment manifest differs from the T0 receipt")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        fail("experiment manifest case registry is absent")
    selected = [row for row in cases if isinstance(row, Mapping) and row.get("case_id") == CASE_ID]
    if len(selected) != 1:
        fail("experiment manifest lacks exactly one fixed T0 case")
    case = selected[0]
    source = case.get("source")
    if (
        case.get("split") != "fit"
        or not isinstance(case.get("instruction"), str)
        or isinstance(case.get("seed"), bool)
        or not isinstance(case.get("seed"), int)
        or not isinstance(source, Mapping)
    ):
        fail("fixed T0 manifest case contract differs")
    require_sha256(source.get("sha256"), label="source video SHA-256")
    return case, observed


def validate_decode_inputs(
    *,
    t0_output: Path | str,
    g2a_receipt_path: Path | str,
    manifest_path: Path | str,
    flow_cohort_path: Path | str,
    middle_cohort_path: Path | str,
    checkpoint_step: int,
    route_kind: str,
    native_argv: Sequence[str],
) -> Mapping[str, Any]:
    if checkpoint_step not in (0, 1):
        fail("matched decode checkpoint step must be 0 or 1")
    if route_kind not in ROUTES:
        fail("matched decode route differs")
    t0_root = Path(t0_output).expanduser().resolve(strict=True)
    t0_receipt = retry8.validate_published_t0_output(t0_root)
    if (
        t0_receipt.get("complete") is not True
        or t0_receipt.get("canary_execution_passed") is not True
        or t0_receipt.get("optimizer_created") is not True
        or t0_receipt.get("optimization_steps") != 1
        or t0_receipt.get("decoded_video_generated") is not False
        or t0_receipt.get("ours_model_claimed") is not False
        or t0_receipt.get("quality_success_claimed") is not False
    ):
        fail("T0 output is not a valid one-step-only canary")

    g2a_path, g2a_receipt, g2a_file_sha = read_json(
        g2a_receipt_path, label="production G2a receipt"
    )
    g2a_world4.validate_world4_receipt(g2a_receipt)
    if (
        g2a_file_sha
        != t0_receipt["upstream_authority"]["production_g2a_receipt_sha256"]
        or g2a_receipt.get("case_id") != CASE_ID
        or g2a_receipt.get("passed") is not True
    ):
        fail("production G2a receipt differs from the T0 authority")

    case, manifest_sha = _manifest_case(
        manifest_path,
        expected_sha256=t0_receipt["upstream_authority"]["manifest_sha256"],
    )
    source_path = Path(option(native_argv, "--source-video")).expanduser().resolve(strict=True)
    source = case["source"]
    if (
        str(source_path) != source.get("path")
        or file_sha256(source_path) != source.get("sha256")
        or option(native_argv, "--instruction") != case.get("instruction")
        or int(option(native_argv, "--seed")) != case.get("seed")
        or int(option(native_argv, "--num-inference-steps")) != NUM_INFERENCE_STEPS
        or option(native_argv, "--source-onset-policy") != "hard1_every_step"
    ):
        fail("native source/instruction/seed/sampler cell is not matched")
    forbidden = {
        "--target-video",
        "--target-image",
        "--reference-video",
        "--reference-image",
        "--mask",
        "--flow",
        "--pose",
        "--trajectory",
        "--adapter-checkpoint",
    }
    if any(flag in native_argv for flag in forbidden):
        fail("matched decode received a forbidden renderer condition")

    flow_path, flow_receipt, flow_sha = read_json(
        flow_cohort_path, label="target G1 flow cohort receipt"
    )
    middle_path, middle_receipt, middle_sha = read_json(
        middle_cohort_path, label="target G1 middle cohort receipt"
    )
    g1 = g2a_receipt["g1_authority"]
    if (
        flow_sha != g1.get("flow_cohort_sha256")
        or middle_sha != g1.get("middle_cohort_sha256")
        or flow_receipt.get("case_id") != CASE_ID
        or middle_receipt.get("case_id") != CASE_ID
        or flow_receipt.get("anchor_kind") != "target"
        or middle_receipt.get("anchor_kind") != "target"
    ):
        fail("selected G1 cohort receipts differ from production G2a")

    state_row = t0_receipt["adapter_states"][str(checkpoint_step)]
    state_path = (t0_root / state_row["state_path"]).resolve(strict=True)
    step_receipt_path = (t0_root / state_row["receipt_path"]).resolve(strict=True)
    if (
        state_path.parent.parent != t0_root
        or step_receipt_path.parent.parent != t0_root
        or file_sha256(state_path) != state_row["state_sha256"]
        or file_sha256(step_receipt_path) != state_row["receipt_sha256"]
    ):
        fail("selected adapter checkpoint closure differs")

    return {
        "t0_root": t0_root,
        "t0_receipt": t0_receipt,
        "t0_receipt_sha256": file_sha256(t0_root / "receipt.json"),
        "g2a_path": g2a_path,
        "g2a_receipt": g2a_receipt,
        "g2a_receipt_sha256": g2a_file_sha,
        "manifest_sha256": manifest_sha,
        "case": case,
        "flow_cohort_path": flow_path,
        "flow_cohort": flow_receipt,
        "flow_cohort_sha256": flow_sha,
        "middle_cohort_path": middle_path,
        "middle_cohort": middle_receipt,
        "middle_cohort_sha256": middle_sha,
        "state_path": state_path,
        "state_row": state_row,
        "step_receipt_path": step_receipt_path,
    }


def _route_refs(
    *,
    route_kind: str,
    flow_cohort: Mapping[str, Any],
    middle_cohort: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if route_kind in ("correct", "temporal_shuffle", "reverse"):
        flow = flow_cohort["external_bundles"][route_kind]
        middle = middle_cohort["external_caches"][route_kind]
    elif route_kind == "incomplete":
        flow = flow_cohort["generated_controls"]["incomplete"]
        middle = middle_cohort["generated_controls"]["incomplete"]
    elif route_kind == "wrong_action":
        flow = flow_cohort["generated_controls"]["wrong_action_energy_matched"]
        middle = middle_cohort["generated_controls"]["wrong_action_energy_matched"]
    else:
        fail("inactive route has no cache references")
    if not isinstance(flow, Mapping) or not isinstance(middle, Mapping):
        fail("route cache references are malformed")
    return flow, middle


def materialize_route(
    *,
    route_kind: str,
    checkpoint_step: int,
    validated: Mapping[str, Any],
) -> tuple[g2a.ActionRepresentationRoute, Mapping[str, Any]]:
    route_contract = validated["g2a_receipt"]["representation_routes"]
    layout_row = route_contract["layout"]
    layout = g2a.TokenLayout(
        total_tokens=int(layout_row["total_tokens"]),
        source_tokens=int(layout_row["source_tokens"]),
        phase_count=int(layout_row["phase_count"]),
    )
    layout.validate()
    if (
        layout.source_tokens != 0
        or layout.total_tokens != 19_803
        or layout.phase_count != 21
        or route_contract.get("selected_sigma_index") != MIDDLE_SIGMA_INDEX
        or route_contract.get("selected_sigma") != MIDDLE_SIGMA
        or route_contract.get("patch_grid") != [21, 23, 41]
    ):
        fail("production G2a route geometry/sigma differs")
    if route_kind in ("route_off", "zero"):
        route = g2a.ActionRepresentationRoute(
            kind=route_kind,
            optimizer_step=checkpoint_step,
            layout=layout,
        )
        route.validate_basic()
        return route, {
            "active": False,
            "flow_cache_path": None,
            "flow_cache_sha256": None,
            "middle_cache_path": None,
            "middle_cache_sha256": None,
            "combined_route_cache_sha256": None,
        }

    flow_ref, middle_ref = _route_refs(
        route_kind=route_kind,
        flow_cohort=validated["flow_cohort"],
        middle_cohort=validated["middle_cohort"],
    )
    flow_path = Path(flow_ref["path"]).expanduser().resolve(strict=True)
    middle_path = Path(middle_ref["path"]).expanduser().resolve(strict=True)
    flow_sha = file_sha256(flow_path)
    middle_sha = file_sha256(middle_path)
    branch = route_contract["branches"][route_kind]
    if (
        flow_sha != require_sha256(flow_ref.get("sha256"), label="flow cache SHA-256")
        or middle_sha
        != require_sha256(middle_ref.get("sha256"), label="middle cache SHA-256")
        or flow_sha != branch.get("flow_cache_sha256")
        or middle_sha != branch.get("middle_cache_sha256")
    ):
        fail("route cache bytes differ from production G2a")

    from safetensors.torch import load_file

    flow_tensors = load_file(str(flow_path), device="cpu")
    if set(flow_tensors) != {
        "backward_raw",
        "backward_camera_residual",
        "validity",
    }:
        fail("flow cache tensor closure differs")
    features_full, activity_full = g2a_world4.dense_flow.dense_flow_features_from_tensors(
        flow_tensors["backward_raw"],
        flow_tensors["backward_camera_residual"],
        flow_tensors["validity"],
    )
    if int(features_full.shape[1]) != 2 * layout.total_tokens:
        fail("flow source/target layout differs")
    if (
        bool(torch.count_nonzero(features_full[:, : layout.total_tokens]).item())
        or bool(activity_full[:, : layout.total_tokens].any().item())
    ):
        fail("flow source prefix is not a hard zero")
    flow = features_full[:, layout.total_tokens :].detach().contiguous()
    activity = activity_full[:, layout.total_tokens :].detach().contiguous()
    if list(flow.shape) != branch.get("flow_global_shape") or list(
        activity.shape
    ) != branch.get("activity_global_shape"):
        fail("flow/activity global geometry differs from production G2a")

    middle_tensors = load_file(str(middle_path), device="cpu")
    expected_middle = {f"middle_block_{index:02d}" for index in BLOCK_INDICES}
    if set(middle_tensors) != expected_middle:
        fail("middle cache tensor closure differs")
    middle_by_block: dict[int, torch.Tensor] = {}
    for index in BLOCK_INDICES:
        value = middle_tensors[f"middle_block_{index:02d}"]
        if (
            value.ndim != 4
            or tuple(value.shape[:2]) != (3, 21)
            or tuple(value.shape[2:]) != (943, 256)
            or not bool(torch.isfinite(value).all().item())
        ):
            fail(f"middle cache block {index} geometry/finiteness differs")
        selected = (
            value[MIDDLE_SIGMA_INDEX]
            .reshape(1, layout.total_tokens, 256)
            .detach()
            .contiguous()
        )
        if list(selected.shape) != branch["middle_global_shapes"][str(index)]:
            fail(f"middle block {index} global geometry differs")
        middle_by_block[index] = selected

    combined = object_sha256(
        {
            "flow_cache_sha256": flow_sha,
            "middle_cache_sha256": middle_sha,
            "middle_sigma_index": MIDDLE_SIGMA_INDEX,
            "middle_sigma": MIDDLE_SIGMA,
            "middle_capture": g2a_world4.MIDDLE_CAPTURE,
        }
    )
    if combined != branch.get("combined_route_cache_sha256"):
        fail("combined route cache digest differs from production G2a")
    route = g2a.ActionRepresentationRoute(
        kind=route_kind,
        optimizer_step=checkpoint_step,
        layout=layout,
        flow=flow,
        activity=activity,
        middle_by_block=middle_by_block,
        representation_origin=(
            "real_target_frozen_extractor"
            if route_kind == "correct"
            else "counterfactual_control"
        ),
        representation_cache_sha256=combined,
        middle_value_kind=g2a_world4.CORE_MIDDLE_ABI_KIND,
        matched_noise_timestep_rotary=True,
    )
    route.validate_basic()
    return route, {
        "active": True,
        "flow_cache_path": str(flow_path),
        "flow_cache_sha256": flow_sha,
        "middle_cache_path": str(middle_path),
        "middle_cache_sha256": middle_sha,
        "combined_route_cache_sha256": combined,
    }


def route_to_device(
    route: g2a.ActionRepresentationRoute, device: torch.device
) -> g2a.ActionRepresentationRoute:
    if route.kind in ("route_off", "zero"):
        return route
    assert route.flow is not None and route.activity is not None
    moved = g2a.ActionRepresentationRoute(
        kind=route.kind,
        optimizer_step=route.optimizer_step,
        layout=route.layout,
        flow=route.flow.to(device=device),
        activity=route.activity.to(device=device),
        middle_by_block={
            index: value.to(device=device)
            for index, value in route.middle_by_block.items()
        },
        representation_origin=route.representation_origin,
        representation_cache_sha256=route.representation_cache_sha256,
        middle_value_kind=route.middle_value_kind,
        matched_noise_timestep_rotary=route.matched_noise_timestep_rotary,
    )
    moved.validate_basic()
    return moved


def expand_target_only_route_for_native_mv2v(
    route: g2a.ActionRepresentationRoute,
) -> tuple[g2a.ActionRepresentationRoute, Mapping[str, Any]]:
    """Prefix one zero source-video extent before the noisy target payload.

    Production G2a/T0 use a target-only 19,803-token FM state. Native
    ``v2v_apg`` concatenates one 19,803-token source video before the noisy
    target, so the inference route must explicitly expand to 39,606 global
    tokens before Bernini append-padding and Ulysses slicing.
    """

    route.validate_basic()
    original = route.layout
    if original.source_tokens != 0 or original.total_tokens != 19_803:
        fail("target-only G2a route layout differs before native mv2v expansion")
    source_tokens = original.target_tokens
    expanded_layout = g2a.TokenLayout(
        total_tokens=source_tokens + original.target_tokens,
        source_tokens=source_tokens,
        phase_count=original.phase_count,
    )
    expanded_layout.validate()
    common_facts: dict[str, Any] = {
        "kind": "native_mv2v_source_zero_prefix_then_noisy_target",
        "original_layout": original.receipt(),
        "expanded_layout": expanded_layout.receipt(),
        "source_prefix_action_active": False,
        "native_concat_order": ["source_video", "noisy_target"],
    }
    if route.kind in ("route_off", "zero"):
        expanded = g2a.ActionRepresentationRoute(
            kind=route.kind,
            optimizer_step=route.optimizer_step,
            layout=expanded_layout,
        )
        expanded.validate_basic()
        return expanded, {**common_facts, "payload_expanded": False}

    assert route.flow is not None and route.activity is not None
    expanded_flow = torch.cat((torch.zeros_like(route.flow), route.flow), dim=1)
    expanded_activity = torch.cat(
        (torch.zeros_like(route.activity, dtype=torch.bool), route.activity), dim=1
    )
    expanded_middle = {
        index: torch.cat((torch.zeros_like(value), value), dim=1)
        .detach()
        .contiguous()
        for index, value in route.middle_by_block.items()
    }
    expanded = g2a.ActionRepresentationRoute(
        kind=route.kind,
        optimizer_step=route.optimizer_step,
        layout=expanded_layout,
        flow=expanded_flow.detach().contiguous(),
        activity=expanded_activity.detach().contiguous(),
        middle_by_block=expanded_middle,
        representation_origin=route.representation_origin,
        representation_cache_sha256=route.representation_cache_sha256,
        middle_value_kind=route.middle_value_kind,
        matched_noise_timestep_rotary=route.matched_noise_timestep_rotary,
    )
    expanded.validate_basic()
    if (
        bool(torch.count_nonzero(expanded.flow[:, :source_tokens]).item())
        or bool(expanded.activity[:, :source_tokens].any().item())
        or any(
            bool(torch.count_nonzero(value[:, :source_tokens]).item())
            for value in expanded.middle_by_block.values()
        )
    ):
        fail("native mv2v action expansion has a nonzero source prefix")
    transport: dict[str, Any] = {
        **common_facts,
        "payload_expanded": True,
        "expanded_flow_sha256": g2a.tensor_sha256(expanded.flow),
        "expanded_activity_sha256": g2a.tensor_sha256(expanded.activity),
        "expanded_middle_sha256": {
            str(index): g2a.tensor_sha256(value)
            for index, value in sorted(expanded.middle_by_block.items())
        },
    }
    transport["transport_digest"] = object_sha256(transport)
    return expanded, transport


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                fail("decode sidecar write made no progress")
            offset += written
        os.fsync(descriptor)
    except FileExistsError as error:
        raise MatchedDecodeError(f"refusing to overwrite decode sidecar: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if path.read_bytes() != payload:
        fail("decode sidecar replay differs")


def validate_video_artifact(path: Path | str) -> Mapping[str, Any]:
    """Decode the published MP4 and verify the exact review geometry."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        fail("matched decode video must be one plain file")
    try:
        import av

        with av.open(str(resolved), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                fail("matched decode video stream closure differs")
            stream = streams[0]
            rate = stream.average_rate
            frames = list(container.decode(stream))
    except MatchedDecodeError:
        raise
    except Exception as error:
        raise MatchedDecodeError(f"cannot decode matched video: {error}") from error
    if (
        rate is None
        or int(rate.numerator) != 25
        or int(rate.denominator) != 1
        or len(frames) != 81
        or any(int(frame.width) != 656 or int(frame.height) != 368 for frame in frames)
    ):
        fail("matched decode frame count/rate/geometry differs")
    rgb_digest = hashlib.sha256()
    for frame in frames:
        rgb = frame.to_ndarray(format="rgb24")
        if rgb.shape != (368, 656, 3) or rgb.dtype.name != "uint8":
            fail("matched decode RGB24 frame closure differs")
        rgb_digest.update(rgb.tobytes(order="C"))
    return {
        "decoded_with": "PyAV",
        "video_stream_count": 1,
        "frame_count": len(frames),
        "fps_numerator": int(rate.numerator),
        "fps_denominator": int(rate.denominator),
        "width": 656,
        "height": 368,
        "all_frames_decoded": True,
        "decoded_rgb24_sha256": rgb_digest.hexdigest(),
    }


def validate_decode_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail("decode receipt must be one mapping")
    row = dict(value)
    declared = row.pop("receipt_digest", None)
    require_sha256(declared, label="decode receipt digest")
    if object_sha256(row) != declared:
        fail("decode receipt digest differs")
    route = row.get("route")
    runtime = row.get("runtime")
    firewall = row.get("information_firewall")
    claims = row.get("claim_boundary")
    video_probe = row.get("decoded_video_probe")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("complete") is not True
        or row.get("case_id") != CASE_ID
        or row.get("checkpoint_step") not in (0, 1)
        or row.get("native_video_generated") is not True
        or not isinstance(video_probe, Mapping)
        or video_probe.get("video_stream_count") != 1
        or video_probe.get("frame_count") != 81
        or video_probe.get("fps_numerator") != 25
        or video_probe.get("fps_denominator") != 1
        or video_probe.get("width") != 656
        or video_probe.get("height") != 368
        or video_probe.get("all_frames_decoded") is not True
        or not isinstance(video_probe.get("decoded_rgb24_sha256"), str)
        or len(video_probe.get("decoded_rgb24_sha256")) != 64
        or not isinstance(route, Mapping)
        or route.get("kind") not in ROUTES
        or route.get("optimizer_step") != row.get("checkpoint_step")
        or route.get("middle_sigma_policy")
        != "fixed_single_trained_sigma_across_all_denoise_calls"
        or route.get("middle_sigma") != MIDDLE_SIGMA
        or route.get("per_timestep_middle_match_claimed") is not False
        or not isinstance(runtime, Mapping)
        or runtime.get("world_size") != 4
        or runtime.get("ulysses_size") != 4
        or runtime.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or runtime.get("shared_step_calls") != 2 * NUM_INFERENCE_STEPS
        or runtime.get("paired_cfg_timestep_digests_equal") is not True
        or runtime.get("base_parameter_identity_unchanged") is not True
        or runtime.get("base_requires_grad_false") is not True
        or runtime.get("adapter_requires_grad_false") is not True
        or runtime.get("adapter_state_digest_unchanged") is not True
        or runtime.get("deterministic_algorithms_enabled") is not True
        or runtime.get("deterministic_algorithms_warn_only") is not False
        or runtime.get("cudnn_deterministic") is not True
        or runtime.get("cudnn_benchmark") is not False
        or not isinstance(firewall, Mapping)
        or firewall.get("target_media_opened_by_renderer") is not False
        or firewall.get("target_rgb_vae_clean_latent_received") is not False
        or firewall.get("detached_action_cache_only") is not True
        or not isinstance(claims, Mapping)
        or claims.get("ours_claimed") is not False
        or claims.get("quality_success_claimed") is not False
        or claims.get("route_success_claimed") is not False
    ):
        fail("decode receipt contract differs")
    require_sha256(
        video_probe.get("decoded_rgb24_sha256"),
        label="decoded RGB24 SHA-256",
    )
    for field in (
        "output_sha256",
        "native_runtime_receipt_sha256",
        "t0_receipt_sha256",
        "adapter_state_sha256",
        "production_g2a_receipt_sha256",
    ):
        require_sha256(row.get(field), label=field)
    return value


def adapter_state_digest(handle: Any) -> str:
    return g2a.object_sha256(
        {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": g2a.tensor_sha256(value),
            }
            for name, value in handle.state_dict_cpu().items()
        }
    )


def deterministic_runtime_flags() -> Mapping[str, bool]:
    return {
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def enable_strict_deterministic_runtime() -> Mapping[str, bool]:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    observed = dict(deterministic_runtime_flags())
    expected = {
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if observed != expected:
        fail("strict deterministic decode runtime activation differs")
    return observed


def audit_frozen_inference_parameters(
    handle: Any,
    *,
    expected_adapter_state_digest: str,
) -> Mapping[str, Any]:
    """Validate inference freezing without applying the training allowlist audit."""

    allowlist = handle.parameter_allowlist()
    adapter = tuple(
        row for role in g2a.TRAINABLE_ROLES for row in allowlist[role]
    )
    adapter_ids = {id(parameter) for _, parameter in adapter}
    if (
        not adapter
        or len(adapter_ids) != len(adapter)
        or any(parameter.requires_grad for _, parameter in adapter)
    ):
        fail("inference adapter parameter freeze/identity closure differs")
    current_base = handle._current_base_named()
    if (
        tuple(name for name, _ in current_base) != handle.base_parameter_names
        or tuple(id(parameter) for _, parameter in current_base)
        != handle.base_parameter_ids
        or any(parameter.requires_grad for _, parameter in current_base)
    ):
        fail("inference frozen base parameter identity/name closure differs")
    observed_state_digest = adapter_state_digest(handle)
    if observed_state_digest != expected_adapter_state_digest:
        fail("adapter state changed during native inference")
    return {
        "base_parameter_identity_unchanged": True,
        "base_requires_grad_false": True,
        "adapter_requires_grad_false": True,
        "adapter_state_digest_unchanged": True,
        "adapter_state_digest": observed_state_digest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--t0-output", required=True)
    parser.add_argument("--g2a-receipt", required=True)
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--flow-cohort-receipt", required=True)
    parser.add_argument("--middle-cohort-receipt", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int, choices=(0, 1))
    parser.add_argument("--route-kind", required=True, choices=ROUTES)
    args, native_argv = parser.parse_known_args(argv)

    validated = validate_decode_inputs(
        t0_output=args.t0_output,
        g2a_receipt_path=args.g2a_receipt,
        manifest_path=args.experiment_manifest,
        flow_cohort_path=args.flow_cohort_receipt,
        middle_cohort_path=args.middle_cohort_receipt,
        checkpoint_step=args.checkpoint_step,
        route_kind=args.route_kind,
        native_argv=native_argv,
    )
    route_cpu, route_facts = materialize_route(
        route_kind=args.route_kind,
        checkpoint_step=args.checkpoint_step,
        validated=validated,
    )
    from safetensors.torch import load_file

    state = load_file(str(validated["state_path"]), device="cpu")
    native_output = Path(option(native_argv, "--output"))
    if not native_output.is_absolute():
        fail("native output must be absolute")
    sidecar = native_output.with_name(native_output.name + ".action-repr-t0.json")
    if native_output.exists() or native_output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        fail("matched decode output/sidecar must be fresh")
    native_argv = [value for value in native_argv if value != "--base-only"]
    native_argv.append("--base-only")

    capture: dict[str, Any] = {
        "constructed": 0,
        "sample_calls": 0,
        "shared_step_calls": 0,
        "timestep_digests": [],
        "deterministic_runtime": enable_strict_deterministic_runtime(),
    }
    original_activate = native.trainer.activate_source_trees

    def patched_activate(*activate_args: Any, **activate_kwargs: Any) -> Any:
        result = original_activate(*activate_args, **activate_kwargs)
        import bernini.models.renderer as renderer_module

        original_class = renderer_module.BerniniRendererModel

        def constructed(config: Any) -> Any:
            model = original_class(config)
            handle = g2a.install_action_repr_g2a_adapter(
                model,
                block_indices=BLOCK_INDICES,
                hidden_width=1536,
                flow_width=12,
                bottleneck_width=256,
                middle_width=256,
                enable_source_copy_adapter=False,
            )
            handle.load_state_dict_strict(state)
            selected_manifest = adapter_state_digest(handle)
            if selected_manifest != validated["state_row"]["state_digest"]:
                fail("loaded adapter state digest differs from T0 receipt")
            if args.checkpoint_step == 0 and not handle.output_gates_are_byte_zero():
                fail("step-0 decode state has a nonzero output gate")
            if args.checkpoint_step == 1 and handle.output_gates_are_byte_zero():
                fail("step-1 decode state remained a zero-effect gate")

            original_sample = model.sample

            def sampled(*sample_args: Any, **sample_kwargs: Any) -> Any:
                capture["sample_calls"] += 1
                device = next(model.parameters()).device
                route = route_to_device(route_cpu, device)
                diffusion = getattr(model, "diff_dec", None)
                original_shared_step = getattr(diffusion, "shared_step", None)
                if diffusion is None or not callable(original_shared_step):
                    fail("native diffusion shared_step is unavailable")
                instance = vars(diffusion)
                had_instance = "shared_step" in instance
                previous_instance = instance.get("shared_step")

                def routed_shared_step(*shared_args: Any, **shared_kwargs: Any) -> Any:
                    if g2a.current_action_representation_route() is not route:
                        fail("native shared_step escaped its authenticated action route")
                    timesteps = shared_kwargs.get("timesteps")
                    if timesteps is None and len(shared_args) >= 3:
                        timesteps = shared_args[2]
                    if not isinstance(timesteps, torch.Tensor):
                        fail("native shared_step timestep tensor is absent")
                    capture["shared_step_calls"] += 1
                    capture["timestep_digests"].append(g2a.tensor_sha256(timesteps))
                    return original_shared_step(*shared_args, **shared_kwargs)

                try:
                    setattr(diffusion, "shared_step", routed_shared_step)
                    with g2a.action_representation_route(route):
                        sampled_value = original_sample(*sample_args, **sample_kwargs)
                finally:
                    if had_instance:
                        setattr(diffusion, "shared_step", previous_instance)
                    else:
                        delattr(diffusion, "shared_step")
                if capture["shared_step_calls"] != 2 * NUM_INFERENCE_STEPS:
                    fail("native sampler did not execute exactly two forwards per step")
                if deterministic_runtime_flags() != capture["deterministic_runtime"]:
                    fail("native sampler changed strict deterministic runtime flags")
                pairs = capture["timestep_digests"]
                if any(pairs[index] != pairs[index + 1] for index in range(0, len(pairs), 2)):
                    fail("negative/action CFG calls did not share timesteps")
                audit = audit_frozen_inference_parameters(
                    handle,
                    expected_adapter_state_digest=validated["state_row"][
                        "state_digest"
                    ],
                )
                capture.update(audit)
                return sampled_value

            model.sample = sampled
            capture["constructed"] += 1
            capture["architecture"] = handle.architecture_receipt()
            return model

        renderer_module.BerniniRendererModel = constructed
        return result

    native.trainer.activate_source_trees = patched_activate
    status = native.main(native_argv)
    if status != 0 or capture["constructed"] != 1 or capture["sample_calls"] != 1:
        fail("native matched decode did not close exactly one model/sample call")

    if int(os.environ.get("RANK", "0")) == 0:
        native_receipt = native_output.with_name(native_output.name + ".receipt.json")
        if not native_output.is_file() or not native_receipt.is_file():
            fail("native video/runtime receipt closure is incomplete")
        pairs = capture["timestep_digests"]
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "case_id": CASE_ID,
            "checkpoint_step": int(args.checkpoint_step),
            "native_video_generated": True,
            "output": str(native_output),
            "output_sha256": file_sha256(native_output),
            "decoded_video_probe": validate_video_artifact(native_output),
            "native_runtime_receipt": str(native_receipt),
            "native_runtime_receipt_sha256": file_sha256(native_receipt),
            "t0_output": str(validated["t0_root"]),
            "t0_receipt_sha256": validated["t0_receipt_sha256"],
            "adapter_state": str(validated["state_path"]),
            "adapter_state_sha256": validated["state_row"]["state_sha256"],
            "adapter_state_digest": validated["state_row"]["state_digest"],
            "production_g2a_receipt": str(validated["g2a_path"]),
            "production_g2a_receipt_sha256": validated["g2a_receipt_sha256"],
            "manifest_sha256": validated["manifest_sha256"],
            "route": {
                "kind": args.route_kind,
                "optimizer_step": int(args.checkpoint_step),
                "facts": route_facts,
                "middle_sigma_policy": "fixed_single_trained_sigma_across_all_denoise_calls",
                "middle_sigma_index": MIDDLE_SIGMA_INDEX,
                "middle_sigma": MIDDLE_SIGMA,
                "per_timestep_middle_match_claimed": False,
                "both_cfg_calls_receive_same_route": True,
            },
            "runtime": {
                "world_size": int(os.environ.get("WORLD_SIZE", "0")),
                "ulysses_size": 4,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "shared_step_calls": capture["shared_step_calls"],
                "paired_cfg_timestep_digests_equal": all(
                    pairs[index] == pairs[index + 1]
                    for index in range(0, len(pairs), 2)
                ),
                "timestep_pair_digest": object_sha256(pairs[::2]),
                "source_onset_policy": "hard1_every_step",
                "native_base_only_label_is_transport_only": True,
                "base_parameter_identity_unchanged": capture.get(
                    "base_parameter_identity_unchanged", False
                ),
                "base_requires_grad_false": capture.get(
                    "base_requires_grad_false", False
                ),
                "adapter_requires_grad_false": capture.get(
                    "adapter_requires_grad_false", False
                ),
                "adapter_state_digest_unchanged": capture.get(
                    "adapter_state_digest_unchanged", False
                ),
                **capture["deterministic_runtime"],
            },
            "information_firewall": {
                "target_media_opened_by_renderer": False,
                "target_rgb_vae_clean_latent_received": False,
                "detached_action_cache_only": True,
                "source_video_and_instruction_only_native_renderer_inputs": True,
            },
            "claim_boundary": {
                "one_step_canary_decode_only": True,
                "ours_claimed": False,
                "quality_success_claimed": False,
                "route_success_claimed": False,
                "visual_review_required": True,
            },
            "source_lock": {
                Path(__file__).name: file_sha256(Path(__file__).resolve()),
                Path(g2a.__file__).name: file_sha256(Path(g2a.__file__).resolve()),
                Path(g2a_world4.__file__).name: file_sha256(
                    Path(g2a_world4.__file__).resolve()
                ),
                Path(retry8.__file__).name: file_sha256(Path(retry8.__file__).resolve()),
                Path(native.__file__).name: file_sha256(Path(native.__file__).resolve()),
            },
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        validate_decode_receipt(receipt)
        _write_exclusive_json(sidecar, receipt)
        replay = json.loads(sidecar.read_text(encoding="ascii"))
        validate_decode_receipt(replay)
        print(
            json.dumps(
                {
                    "complete": True,
                    "checkpoint_step": args.checkpoint_step,
                    "route_kind": args.route_kind,
                    "output": str(native_output),
                    "output_sha256": receipt["output_sha256"],
                    "ours_or_quality_success_claimed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
