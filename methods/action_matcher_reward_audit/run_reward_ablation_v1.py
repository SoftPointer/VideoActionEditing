#!/usr/bin/env python3
"""Run the fast action/preservation Best-of-N reward diagnostic.

The program deliberately separates three operations:

* ``extract-shard`` extracts frozen DINOv2 frame and dense-patch features;
* ``analyze`` applies four policies to the same immutable candidate pools;
* ``build-html`` creates a synchronized, human-auditable comparison page.

There is no VLM and no claimed ground-truth winner.  For the original
ablation, the baseline is a fixed pre-registered seed and the policies rerank
the same videos.  The same scorer may also compare matched frozen/trained
model outputs when the manifest declares that parameter updates occurred; in
that mode its ranks remain diagnostics rather than training-gain labels.  The
preservation axes are weak source-bound proxies (not face identity or
segmented safe background).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


METHOD_ROOT = Path(__file__).resolve().parent
SEMANTIC_AUDIT_ROOT = METHOD_ROOT.parent / "semantic_moments_reward_audit"
for root in (METHOD_ROOT, SEMANTIC_AUDIT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from operational_reward import (  # noqa: E402
    RewardConfig,
    average_tie_percentiles,
    score_candidate_pool,
)
from run_audit import (  # noqa: E402
    LocalDINOv2,
    compose_moments,
    load_video_frames,
    temporal_components,
)


MANIFEST_SCHEMA = "action-editing-reward-ablation-manifest-v1"
FEATURE_SCHEMA = "action-editing-reward-ablation-features-v1"
RESULT_SCHEMA = "action-editing-reward-ablation-result-v1"
HTML_SCHEMA = "action-editing-reward-ablation-html-v1"
EPS = 1.0e-8
PRESERVATION_AXES = (
    "source_appearance_set_proxy",
    "fixed_grid_background_dominant_proxy",
    "camera_translation_agreement_proxy",
    "decode_quality_proxy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(value.float(), dim=dim, eps=EPS)


def _validate_media(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} media fields differ")
    path = Path(str(value["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not an absolute plain file: {path}")
    digest = file_sha256(path)
    if digest != value["sha256"]:
        raise ValueError(f"{label} SHA-256 differs")
    return {"path": str(path.resolve()), "sha256": digest}


def validate_manifest(value: Mapping[str, Any], *, verify_media: bool) -> dict[str, Any]:
    expected = {
        "schema_version",
        "created_at",
        "experiment_id",
        "generator",
        "policy",
        "groups",
        "manifest_digest",
    }
    if set(value) != expected or value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("reward-ablation manifest field closure differs")
    unsigned = dict(value)
    claimed = unsigned.pop("manifest_digest")
    if object_sha256(unsigned) != claimed:
        raise ValueError("reward-ablation manifest digest differs")
    groups = value["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("manifest needs at least one group")
    normalized = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for group in groups:
        fields = {
            "iid",
            "node",
            "instruction",
            "source",
            "action_anchor",
            "baseline_candidate_id",
            "candidates",
        }
        if not isinstance(group, Mapping) or set(group) != fields:
            raise ValueError("manifest group fields differ")
        candidates = group["candidates"]
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("every group needs at least two candidates")
        candidate_rows = []
        for row in candidates:
            if not isinstance(row, Mapping) or set(row) != {
                "candidate_id", "seed", "origin", "media"
            }:
                raise ValueError("candidate fields differ")
            candidate_id = str(row["candidate_id"])
            if candidate_id in seen_ids:
                raise ValueError(f"duplicate candidate ID: {candidate_id}")
            seen_ids.add(candidate_id)
            media = (
                _validate_media(row["media"], label=candidate_id)
                if verify_media
                else dict(row["media"])
            )
            if str(media["path"]) in seen_paths:
                raise ValueError("candidate video path is reused")
            seen_paths.add(str(media["path"]))
            candidate_rows.append({**dict(row), "media": media})
        candidate_ids = {row["candidate_id"] for row in candidate_rows}
        if group["baseline_candidate_id"] not in candidate_ids:
            raise ValueError("baseline candidate is outside its pool")
        source = (
            _validate_media(group["source"], label=f"{group['iid']} source")
            if verify_media
            else dict(group["source"])
        )
        anchor = (
            _validate_media(group["action_anchor"], label=f"{group['iid']} anchor")
            if verify_media
            else dict(group["action_anchor"])
        )
        normalized.append(
            {
                **dict(group),
                "source": source,
                "action_anchor": anchor,
                "candidates": candidate_rows,
            }
        )
    return {**dict(value), "groups": normalized}


def manifest_items(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = []
    for group in manifest["groups"]:
        iid = group["iid"]
        for role, media in (
            ("source", group["source"]),
            ("action_anchor", group["action_anchor"]),
        ):
            items.append(
                {
                    "item_id": f"{iid}:{role}",
                    "iid": iid,
                    "role": role,
                    "path": media["path"],
                    "sha256": media["sha256"],
                }
            )
        for candidate in group["candidates"]:
            items.append(
                {
                    "item_id": candidate["candidate_id"],
                    "iid": iid,
                    "role": "candidate",
                    "path": candidate["media"]["path"],
                    "sha256": candidate["media"]["sha256"],
                }
            )
    return items


def pil_frames_to_small_tensor(frames: Sequence[Any], size: int = 96) -> torch.Tensor:
    arrays = []
    for frame in frames:
        resized = frame.resize((size, size))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        arrays.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(arrays).contiguous()


def extract_shard(args: argparse.Namespace) -> int:
    manifest = validate_manifest(load_json(args.manifest), verify_media=True)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index is outside the shard population")
    items = [
        row
        for ordinal, row in enumerate(manifest_items(manifest))
        if ordinal % args.num_shards == args.shard_index
    ]
    extractor = LocalDINOv2(args.model_root, args.device, args.frame_batch_size)
    records = []
    for ordinal, item in enumerate(items):
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "ordinal": ordinal,
                    "count": len(items),
                    "item_id": item["item_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        frames = load_video_frames(item["path"], args.num_frames)
        tokens = extractor.extract(frames).float()
        components = temporal_components(tokens)
        records.append(
            {
                **item,
                "components": components.cpu(),
                "frame_sequence": unit(tokens.mean(dim=1), dim=1).cpu(),
                "dense_sequence": unit(tokens, dim=2).to(torch.float16).cpu(),
                "raw_small": pil_frames_to_small_tensor(frames).to(torch.float16),
                "feature_geometry": {
                    "num_frames": int(tokens.shape[0]),
                    "num_patches": int(tokens.shape[1]),
                    "dimension": int(tokens.shape[2]),
                },
            }
        )
    payload = {
        "schema_version": FEATURE_SCHEMA,
        "created_at": utc_now(),
        "manifest_digest": manifest["manifest_digest"],
        "manifest_sha256": file_sha256(args.manifest),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_frames": args.num_frames,
        "record_count": len(records),
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "device": args.device,
            "model_root": str(Path(args.model_root).resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "records": records,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(f"wrote {destination} records={len(records)}", flush=True)
    return 0


def load_features(feature_roots: Iterable[str | Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = []
    for root in feature_roots:
        value = Path(root)
        if value.is_file():
            paths.append(value)
        else:
            paths.extend(sorted(value.glob("features-shard-*.pt")))
    if not paths:
        raise ValueError("no feature shards found")
    records = []
    receipts = []
    manifest_digest = None
    num_frames = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != FEATURE_SCHEMA:
            raise ValueError(f"feature schema differs: {path}")
        if manifest_digest is None:
            manifest_digest = payload["manifest_digest"]
            num_frames = payload["num_frames"]
        elif (
            payload["manifest_digest"] != manifest_digest
            or payload["num_frames"] != num_frames
        ):
            raise ValueError("feature shards have different manifest/frame bindings")
        records.extend(payload["records"])
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "record_count": payload["record_count"],
                "runtime": payload["runtime"],
            }
        )
    by_id = {row["item_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate item IDs across feature shards")
    return by_id, receipts


def mapped_cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return ((unit(left) @ unit(right).T) + 1.0).mul(0.5).clamp(0.0, 1.0)


def appearance_set_similarity(candidate: torch.Tensor, source: torch.Tensor) -> float:
    matrix = mapped_cosine_matrix(candidate.float(), source.float())
    score = 0.5 * (matrix.max(dim=1).values.mean() + matrix.max(dim=0).values.mean())
    return float(score)


def fixed_grid_similarity(candidate: torch.Tensor, source: torch.Tensor) -> float:
    if candidate.shape != source.shape:
        raise ValueError("dense source/candidate feature geometry differs")
    values = ((candidate.float() * source.float()).sum(dim=-1) + 1.0) * 0.5
    return float(values.reshape(-1).median())


def _phase_step(previous: torch.Tensor, current: torch.Tensor) -> tuple[float, float]:
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).reshape(3, 1, 1)
    left = ((previous.float() * weights).sum(dim=0) - previous.float().mean())
    right = ((current.float() * weights).sum(dim=0) - current.float().mean())
    window = torch.outer(
        torch.hann_window(left.shape[0], periodic=False),
        torch.hann_window(left.shape[1], periodic=False),
    )
    left = left * window
    right = right * window
    spectrum = torch.fft.fft2(left) * torch.fft.fft2(right).conj()
    magnitude = spectrum.abs().clamp_min(1.0e-12)
    correlation = torch.fft.ifft2(spectrum / magnitude).real
    index = int(correlation.reshape(-1).argmax())
    y, x = divmod(index, correlation.shape[1])
    if y > correlation.shape[0] // 2:
        y -= correlation.shape[0]
    if x > correlation.shape[1] // 2:
        x -= correlation.shape[1]
    return y / correlation.shape[0], x / correlation.shape[1]


def camera_agreement(candidate: torch.Tensor, source: torch.Tensor) -> float:
    if candidate.shape != source.shape or len(candidate) < 2:
        raise ValueError("raw source/candidate geometry differs")
    candidate_steps = torch.tensor(
        [_phase_step(candidate[i - 1], candidate[i]) for i in range(1, len(candidate))]
    )
    source_steps = torch.tensor(
        [_phase_step(source[i - 1], source[i]) for i in range(1, len(source))]
    )
    return float(torch.exp(-20.0 * (candidate_steps - source_steps).abs().mean()))


def quality_score(candidate: torch.Tensor, source: torch.Tensor) -> tuple[float, dict[str, float]]:
    candidate = candidate.float()
    source = source.float()

    def sharpness(value: torch.Tensor) -> torch.Tensor:
        dx = value[:, :, :, 1:] - value[:, :, :, :-1]
        dy = value[:, :, 1:, :] - value[:, :, :-1, :]
        return 0.5 * (dx.square().mean() + dy.square().mean())

    candidate_sharp = sharpness(candidate)
    source_sharp = sharpness(source)
    retention = (candidate_sharp / source_sharp.clamp_min(EPS)).clamp(0.0, 1.0)
    clipped = ((candidate <= 2.0 / 255.0) | (candidate >= 253.0 / 255.0)).float()
    exposure = (1.0 - clipped.mean()).clamp(0.0, 1.0)
    candidate_step = (candidate[1:] - candidate[:-1]).abs().mean()
    source_step = (source[1:] - source[:-1]).abs().mean()
    nonfreeze = (candidate_step / source_step.clamp_min(EPS)).clamp(0.0, 1.0)
    candidate_mean = candidate.mean(dim=(1, 2, 3))
    source_mean = source.mean(dim=(1, 2, 3))
    candidate_second = candidate_mean[2:] - 2 * candidate_mean[1:-1] + candidate_mean[:-2]
    source_second = source_mean[2:] - 2 * source_mean[1:-1] + source_mean[:-2]
    flicker = torch.exp(-10.0 * (candidate_second - source_second).abs().mean()).clamp(0.0, 1.0)
    terms = torch.stack([retention, exposure, nonfreeze, flicker])
    aggregate = torch.exp(torch.log(terms.clamp_min(1.0e-12)).mean())
    detail = {
        "sharpness_retention": float(retention),
        "exposure": float(exposure),
        "nonfreeze": float(nonfreeze),
        "flicker": float(flicker),
    }
    return float(aggregate), detail


def preservation_pool(
    source: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = []
    for candidate in candidates:
        quality, quality_detail = quality_score(
            candidate["feature"]["raw_small"], source["raw_small"]
        )
        raw = {
            "source_appearance_set_proxy": appearance_set_similarity(
                candidate["feature"]["frame_sequence"], source["frame_sequence"]
            ),
            "fixed_grid_background_dominant_proxy": fixed_grid_similarity(
                candidate["feature"]["dense_sequence"], source["dense_sequence"]
            ),
            "camera_translation_agreement_proxy": camera_agreement(
                candidate["feature"]["raw_small"], source["raw_small"]
            ),
            "decode_quality_proxy": quality,
        }
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "raw_scores": raw,
                "quality_detail": quality_detail,
            }
        )
    for axis in PRESERVATION_AXES:
        percentiles = average_tie_percentiles(
            [row["raw_scores"][axis] for row in rows]
        )
        for row, percentile in zip(rows, percentiles):
            row.setdefault("pool_percentiles", {})[axis] = percentile
    for row in rows:
        row["preservation_score"] = min(row["pool_percentiles"].values())
    ranked = sorted(
        rows, key=lambda row: (-row["preservation_score"], row["candidate_id"])
    )
    return {
        "axes": list(PRESERVATION_AXES),
        "absolute_thresholds_calibrated": False,
        "identity_isolated": False,
        "background_isolated": False,
        "diagnostic_top_candidate_id": ranked[0]["candidate_id"],
        "rows": rows,
    }


def _action_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    feature = candidate["feature"]
    return {
        "candidate_id": candidate["candidate_id"],
        "m3": feature["components"][2].float(),
        "frame_sequence": feature["frame_sequence"].float(),
    }


def _composite_selection(
    action: Mapping[str, Any], preservation: Mapping[str, Any]
) -> dict[str, Any]:
    action_by_id = {row["candidate_id"]: row for row in action["candidates"]}
    preservation_by_id = {
        row["candidate_id"]: row for row in preservation["rows"]
    }
    # With N=4, 1/3 means a candidate is not the unique worst on any
    # preservation axis.  It is a relative pilot gate, not a calibrated
    # absolute identity/background threshold.
    floor = 1.0 / 3.0
    admitted = [
        candidate_id
        for candidate_id, row in preservation_by_id.items()
        if row["preservation_score"] + EPS >= floor
        and action_by_id[candidate_id]["eligible"]
    ]
    ranked = sorted(
        admitted,
        key=lambda candidate_id: (
            -action_by_id[candidate_id]["event_score"], candidate_id
        ),
    )
    reasons = []
    if not ranked:
        reasons.append("no_candidate_passed_relative_preservation_and_action_gates")
        selected = None
        top_gap = None
    else:
        selected = ranked[0]
        top_score = action_by_id[selected]["event_score"]
        top_gap = (
            top_score - action_by_id[ranked[1]]["event_score"]
            if len(ranked) > 1
            else None
        )
        if top_score < 0.25:
            reasons.append("top_event_score_below_floor")
        if top_gap is not None and top_gap + EPS < 0.20:
            reasons.append("top_two_gap_too_small")
    return {
        "aggregation": "relative_preservation_gate_then_action_event_score",
        "weighted_sum_used": False,
        "relative_preservation_floor": floor,
        "absolute_preservation_thresholds_calibrated": False,
        "admitted_candidate_ids": ranked,
        "diagnostic_top_candidate_id": selected,
        "selected_candidate_id": selected if selected and not reasons else None,
        "abstain_required": bool(reasons),
        "abstain_reasons": reasons,
        "top_gap": top_gap,
    }


def analyze(args: argparse.Namespace) -> int:
    manifest = validate_manifest(load_json(args.manifest), verify_media=True)
    features, feature_receipts = load_features(args.feature_root)
    if set(features) != {row["item_id"] for row in manifest_items(manifest)}:
        missing = {row["item_id"] for row in manifest_items(manifest)} - set(features)
        extra = set(features) - {row["item_id"] for row in manifest_items(manifest)}
        raise ValueError(f"feature population differs: missing={missing} extra={extra}")
    groups = []
    for group in manifest["groups"]:
        iid = group["iid"]
        anchor = features[f"{iid}:action_anchor"]
        source = features[f"{iid}:source"]
        candidates = [
            {
                **candidate,
                "feature": features[candidate["candidate_id"]],
            }
            for candidate in group["candidates"]
        ]
        action = score_candidate_pool(
            reference_id=f"{iid}:action_anchor",
            reference_m3=anchor["components"][2].float(),
            reference_sequence=anchor["frame_sequence"].float(),
            candidates=[_action_candidate(row) for row in candidates],
            config=RewardConfig(contract="generic_ordered"),
            valid_candidate_prior=True,
        )
        preservation = preservation_pool(source, candidates)
        composite = _composite_selection(action, preservation)
        preservation_selected = preservation["diagnostic_top_candidate_id"]
        action_selected = action["selected_candidate_id"]
        baseline = group["baseline_candidate_id"]
        arms = {
            "baseline": {
                "candidate_id": baseline,
                "abstain": False,
                "policy": "fixed_pre_registered_seed_no_reward",
            },
            "action_only": {
                "candidate_id": action_selected or baseline,
                "diagnostic_candidate_id": action["diagnostic_top_candidate_id"],
                "abstain": action["abstain_required"],
                "fallback_to_baseline": action_selected is None,
                "policy": "operational_generic_ordered_action_reward",
            },
            "preservation_only": {
                "candidate_id": preservation_selected,
                "abstain": False,
                "policy": "relative_weak_preservation_max_min_diagnostic",
                "uncalibrated_absolute_selection": True,
            },
            "composite": {
                "candidate_id": composite["selected_candidate_id"] or baseline,
                "diagnostic_candidate_id": composite["diagnostic_top_candidate_id"],
                "abstain": composite["abstain_required"],
                "fallback_to_baseline": composite["selected_candidate_id"] is None,
                "policy": "relative_preservation_gate_then_action_no_weighted_sum",
            },
        }
        groups.append(
            {
                "iid": iid,
                "node": group["node"],
                "instruction": group["instruction"],
                "source": group["source"],
                "action_anchor": group["action_anchor"],
                "candidates": [
                    {key: value for key, value in row.items() if key != "feature"}
                    for row in candidates
                ],
                "arms": arms,
                "action_reward": action,
                "preservation_reward": preservation,
                "composite_reward": composite,
                "selection_is_not_correctness": True,
            }
        )
    selection_changes = {
        arm: sum(
            group["arms"][arm]["candidate_id"]
            != group["arms"]["baseline"]["candidate_id"]
            for group in groups
        )
        for arm in ("action_only", "preservation_only", "composite")
    }
    generator_updated = bool(
        manifest.get("generator", {}).get("parameter_update_performed", False)
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "created_at": utc_now(),
        "experiment_id": manifest["experiment_id"],
        "manifest_digest": manifest["manifest_digest"],
        "manifest_sha256": file_sha256(args.manifest),
        "authority": {
            "reranking_experiment": not generator_updated,
            "trained_model_comparison": generator_updated,
            "generator_parameter_update": generator_updated,
            "training_gain_claimed": False,
            "machine_selection_is_ground_truth": False,
            "human_review_required_for_efficacy": True,
            "qwen_or_vlm_used": False,
        },
        "candidate_pool": {
            "groups": len(groups),
            "candidates_per_group": sorted({len(group["candidates"]) for group in groups}),
            "same_pool_for_all_arms": True,
            "interpretation": (
                "matched trained/frozen model outputs under one evaluation seed; diagnostic ranks "
                "are not correctness labels"
                if generator_updated
                else "all candidates are independent native samples from the requested action prompt; "
                "this raises but does not prove the probability of at least one valid action"
            ),
        },
        "policy": manifest["policy"],
        "selection_changes_from_baseline": selection_changes,
        "feature_receipts": feature_receipts,
        "groups": groups,
    }
    result["result_digest"] = object_sha256(result)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "digest": result["result_digest"],
                "changes": selection_changes,
            },
            sort_keys=True,
        )
    )
    return 0


def _media_rel(iid: str, role: str, candidate_id: str | None = None) -> str:
    if role in {"source", "anchor"}:
        return f"media/{iid}/{role}.mp4"
    assert candidate_id is not None
    return f"media/{iid}/{candidate_id}.mp4"


def _score_table(group: Mapping[str, Any]) -> str:
    action = {row["candidate_id"]: row for row in group["action_reward"]["candidates"]}
    preservation = {
        row["candidate_id"]: row for row in group["preservation_reward"]["rows"]
    }
    rows = []
    for candidate in group["candidates"]:
        cid = candidate["candidate_id"]
        arow = action[cid]
        prow = preservation[cid]
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(cid)}</code></td>"
            f"<td>{candidate['seed']}</td>"
            f"<td>{arow['event_score']:.3f}</td>"
            f"<td>{str(arow['eligible']).lower()}</td>"
            f"<td>{prow['preservation_score']:.3f}</td>"
            f"<td>{prow['raw_scores']['source_appearance_set_proxy']:.3f}</td>"
            f"<td>{prow['raw_scores']['fixed_grid_background_dominant_proxy']:.3f}</td>"
            f"<td>{prow['raw_scores']['camera_translation_agreement_proxy']:.3f}</td>"
            f"<td>{prow['raw_scores']['decode_quality_proxy']:.3f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>candidate</th><th>seed</th><th>action event</th>"
        "<th>action gate</th><th>pres min-P</th><th>appearance*</th>"
        "<th>fixed-grid*</th><th>camera*</th><th>quality*</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _candidate_gallery(group: Mapping[str, Any]) -> str:
    iid = group["iid"]
    cards = []
    for candidate in group["candidates"]:
        candidate_id = candidate["candidate_id"]
        src = _media_rel(iid, "candidate", candidate_id)
        cards.append(
            '<article class="pool-card">'
            f'<h4>seed {candidate["seed"]}</h4>'
            f'<video controls muted loop playsinline preload="metadata" src="{html.escape(src)}"></video>'
            f'<p><code>{html.escape(candidate_id)}</code></p>'
            "</article>"
        )
    return '<h3>完整 Best-of-4 候选池</h3><div class="candidate-grid">' + "".join(cards) + "</div>"


def build_html(args: argparse.Namespace) -> int:
    result = load_json(args.result)
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("result schema differs")
    arm_labels = {
        "baseline": "Baseline（固定 seed，无 reward）",
        "action_only": "Action reward",
        "preservation_only": "Preservation proxy",
        "composite": "Composite（pres gate → action）",
    }
    sections = []
    for group in result["groups"]:
        iid = group["iid"]
        cards = [
            ("source", "Source", _media_rel(iid, "source"), "原视频"),
            ("anchor", "Self-generated action anchor", _media_rel(iid, "anchor"), "只提供动作参照"),
        ]
        for arm in ("baseline", "action_only", "preservation_only", "composite"):
            record = group["arms"][arm]
            note = f"candidate: {record['candidate_id']}"
            if record.get("abstain"):
                note += " · ABSTAIN，页面显示 baseline fallback"
            cards.append(
                (
                    arm,
                    arm_labels[arm],
                    _media_rel(iid, "candidate", record["candidate_id"]),
                    note,
                )
            )
        card_html = []
        for role, label, src, note in cards:
            audit = ""
            if role not in {"source", "anchor"}:
                audit = (
                    f'<div class="audit" data-arm="{role}">'
                    '<button data-v="action_ok">动作成功</button>'
                    '<button data-v="action_fail">动作失败</button>'
                    '<button data-v="pres_fail">主体/背景失败</button>'
                    '<button data-v="unclear">无法判断</button>'
                    "</div>"
                )
            card_html.append(
                f'<article class="video-card {role}"><h3>{html.escape(label)}</h3>'
                f'<video controls muted loop playsinline preload="metadata" src="{html.escape(src)}"></video>'
                f'<p>{html.escape(note)}</p>{audit}</article>'
            )
        candidate_options = "".join(
            f'<option value="{arm}">{html.escape(arm_labels[arm])}</option>'
            for arm in ("baseline", "action_only", "preservation_only", "composite")
        )
        sections.append(
            f'<section class="group" data-iid="{html.escape(iid)}">'
            f'<header><h2>{html.escape(iid)} · node {html.escape(group["node"])}</h2>'
            f'<p class="instruction">{html.escape(group["instruction"])}</p>'
            '<div class="controls"><button class="sync-play">同步播放/暂停</button>'
            '<button class="sync-zero">全部归零</button>'
            '<button class="sync-phase">按归一化进度对齐</button>'
            '<label>速度 <select class="rate"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label></div></header>'
            f'<div class="video-grid">{"".join(card_html)}</div>'
            '<div class="verdict"><label>整体最好：<select class="winner"><option value="">未判断</option>'
            f'{candidate_options}<option value="tie">近似相同</option><option value="none">都不可用</option></select></label>'
            '<textarea placeholder="备注：动作完成度、方向、主体、背景、camera、artifact..."></textarea></div>'
            '<details><summary>机器选择与分数（先看视频再展开；这些不是正确答案）</summary>'
            f'{_candidate_gallery(group)}{_score_table(group)}<p class="foot">* preservation 均为未校准 proxy：appearance 不是实例身份，fixed-grid 不是分割后的 safe background，camera 会受主体运动干扰。</p></details>'
            "</section>"
        )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Action-editing reward Best-of-4 ablation</title>
<style>
:root{{--bg:#f5f2ea;--card:#fffdf8;--ink:#18211e;--muted:#68716e;--accent:#136f63;--line:#d7cfbf;--warn:#9a5a18}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:1920px;margin:auto;padding:24px}} h1{{margin:0 0 8px}} .lead{{max-width:1100px;color:var(--muted)}}
.banner{{background:#fff4dc;border:1px solid #e7c98f;border-radius:12px;padding:14px 18px;margin:18px 0}}
.group{{background:var(--card);border:1px solid var(--line);border-radius:16px;margin:22px 0;padding:18px;box-shadow:0 5px 18px #453b2b12}}
.group header h2{{margin:0}} .instruction{{max-width:1200px;color:#39433f}} .controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}}
button,select,textarea{{font:inherit}} button,select{{border:1px solid #aaa08e;background:#fbf8f1;border-radius:8px;padding:7px 10px;cursor:pointer}}
button.active{{background:var(--accent);color:white;border-color:var(--accent)}}
.video-grid{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px}} .video-card{{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f8f5ee}}
.video-card h3{{font-size:15px;margin:0;padding:10px 12px}} video{{display:block;width:100%;aspect-ratio:16/10;background:#090b0a;object-fit:contain}}
.video-card p{{color:var(--muted);padding:8px 12px;margin:0;word-break:break-all;font-size:12px}} .audit{{display:flex;gap:6px;flex-wrap:wrap;padding:0 10px 10px}}
.candidate-grid{{display:grid;grid-template-columns:repeat(4,minmax(200px,1fr));gap:10px}} .pool-card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#f8f5ee}} .pool-card h4,.pool-card p{{margin:0;padding:7px 9px}} .pool-card p{{word-break:break-all}}
.audit button{{font-size:12px;padding:5px 7px}} .verdict{{display:grid;grid-template-columns:300px 1fr;gap:12px;margin-top:14px}} textarea{{min-height:72px;border:1px solid var(--line);border-radius:9px;padding:10px}}
details{{margin-top:14px;border-top:1px dashed var(--line);padding-top:10px}} table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}} th,td{{border:1px solid var(--line);padding:6px;text-align:left}} code{{font-size:11px}} .foot{{color:var(--warn)}}
.export{{position:sticky;bottom:16px;float:right;background:var(--accent);color:white;border-color:var(--accent);padding:10px 15px}}
@media(max-width:1000px){{.video-grid,.candidate-grid{{grid-template-columns:repeat(2,minmax(220px,1fr))}}}} @media(max-width:650px){{main{{padding:10px}}.video-grid,.candidate-grid{{grid-template-columns:1fr}}.verdict{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Action-editing reward Best-of-4 ablation</h1>
<p class="lead">同一 Bernini base、source、instruction 与 4 个 candidate pool；只有选择策略不同。页面没有“潜在官方正确答案”。请先盲看视频，再展开机器分数。</p>
<div class="banner"><strong>结论边界：</strong>这是 inference-time reranking 增益验证，不是参数训练增益。Action arm 可 abstain；Composite abstain 时保留 baseline。Preservation arm 只是相对 proxy 诊断，不应被当作 identity/background 真值。</div>
{"".join(sections)}
<button class="export">导出人工审计 JSON</button>
</main><script>
const state={{schema_version:'action-editing-reward-ablation-human-review-v1',created_at:new Date().toISOString(),groups:{{}}}};
document.querySelectorAll('.group').forEach(group=>{{
 const iid=group.dataset.iid, videos=[...group.querySelectorAll('.video-grid video')]; state.groups[iid]={{arms:{{}},winner:'',note:''}};
 const align=()=>{{const p=Math.max(...videos.map(v=>v.duration? v.currentTime/v.duration:0)); videos.forEach(v=>{{if(v.duration)v.currentTime=Math.min(v.duration-.01,p*v.duration)}})}};
 group.querySelector('.sync-play').onclick=()=>{{align(); const play=videos.every(v=>v.paused); videos.forEach(v=>play?v.play():v.pause())}};
 group.querySelector('.sync-zero').onclick=()=>videos.forEach(v=>{{v.pause();v.currentTime=0}});
 group.querySelector('.sync-phase').onclick=align;
 group.querySelector('.rate').onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));
 group.querySelectorAll('.audit').forEach(box=>box.querySelectorAll('button').forEach(btn=>btn.onclick=()=>{{
   box.querySelectorAll('button').forEach(x=>x.classList.remove('active'));btn.classList.add('active');state.groups[iid].arms[box.dataset.arm]=btn.dataset.v;
 }}));
 group.querySelector('.winner').onchange=e=>state.groups[iid].winner=e.target.value;
 group.querySelector('textarea').oninput=e=>state.groups[iid].note=e.target.value;
}});
document.querySelector('.export').onclick=()=>{{state.exported_at=new Date().toISOString();const blob=new Blob([JSON.stringify(state,null,2)+'\\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reward-ablation-human-review.json';a.click();URL.revokeObjectURL(a.href)}};
</script></body></html>"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    receipt = {
        "schema_version": HTML_SCHEMA,
        "created_at": utc_now(),
        "result_digest": result["result_digest"],
        "result_sha256": file_sha256(args.result),
        "html": str(output.resolve()),
        "html_sha256": file_sha256(output),
        "groups": len(result["groups"]),
        "synchronized_playback": True,
        "machine_selection_labeled_as_correct": False,
        "human_review_export": True,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    write_json(output.with_name("html-receipt.json"), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-shard")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--model-root", required=True)
    extract.add_argument("--shard-index", type=int, required=True)
    extract.add_argument("--num-shards", type=int, required=True)
    extract.add_argument("--num-frames", type=int, default=8)
    extract.add_argument("--frame-batch-size", type=int, default=8)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--output", required=True)
    extract.set_defaults(func=extract_shard)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--manifest", required=True)
    analysis.add_argument("--feature-root", action="append", required=True)
    analysis.add_argument("--output", required=True)
    analysis.set_defaults(func=analyze)
    builder = subparsers.add_parser("build-html")
    builder.add_argument("--result", required=True)
    builder.add_argument("--output", required=True)
    builder.set_defaults(func=build_html)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
