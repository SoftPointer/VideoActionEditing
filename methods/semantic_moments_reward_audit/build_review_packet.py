#!/usr/bin/env python3
"""Build a portable, blind-first SemanticMoments human review packet.

The builder consumes the frozen audit manifest, feature shards, and aggregate
receipt.  It selects preregistered success/failure examples, copies immutable
source MP4s under portable names, computes per-case diagnostic similarities,
and seals those data into a self-contained HTML page.  Video transcoding is a
separate mechanical step described by ``transcode-plan.json``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import torch


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_audit as audit  # noqa: E402


SCHEMA_VERSION = "semantic-moments-action-reward-human-review-v1"
SIMMOTION_CASES = (
    ("example_28", "strong_pass", "A"),
    ("example_20", "strong_pass", "B"),
    ("example_5", "strong_pass", "A"),
    ("example_29", "strong_pass", "B"),
    ("example_1", "strong_fail", "A"),
    ("example_13", "strong_fail", "B"),
    ("example_25", "strong_fail", "A"),
    ("example_35", "strong_fail", "B"),
)
REPRESENTATIONS = ("m1", "m2", "m3", "m23", "m123")


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.write_text(payload, encoding="utf-8")


def _safe_component(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"unsafe portable component: {value!r}")
    return value


def _load_features(feature_root: Path) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    records: dict[str, Mapping[str, Any]] = {}
    receipts = []
    for path in sorted(feature_root.glob("features-shard-*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            identifier = record["item_id"]
            if identifier in records:
                raise ValueError(f"duplicate feature record: {identifier}")
            records[identifier] = record
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": audit.file_sha256(path),
                "record_count": payload["record_count"],
            }
        )
    if not records:
        raise FileNotFoundError(f"no feature shards in {feature_root}")
    return records, receipts


def _copy_source(
    *,
    item: Mapping[str, Any],
    packet_root: Path,
    relative_source: str,
    copied: dict[str, str],
) -> None:
    identifier = item["item_id"]
    if identifier in copied:
        if copied[identifier] != relative_source:
            raise ValueError(f"item copied to two names: {identifier}")
        return
    source = Path(item["path"])
    if audit.file_sha256(source) != item["sha256"]:
        raise ValueError(f"source hash differs: {source}")
    destination = packet_root / relative_source
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if audit.file_sha256(destination) != item["sha256"]:
        raise ValueError(f"portable copy hash differs: {destination}")
    copied[identifier] = relative_source


def _add_transcode(
    plan: list[dict[str, Any]],
    *,
    source: str,
    destination: str,
    transform: str = "normal",
) -> None:
    plan.append(
        {"source": source, "destination": destination, "transform": transform}
    )


def _score_pair(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    values = {
        name: audit.cosine(
            audit.representation(reference, name),
            audit.representation(candidate, name),
        )
        for name in REPRESENTATIONS
    }
    values["dtw_order_margin"] = audit.order_margin(
        reference["frame_sequence"], candidate["frame_sequence"]
    )
    values["endpoint_score"] = audit.endpoint_score(
        reference["frame_sequence"], candidate["frame_sequence"]
    )
    return values


def _simmotion_cases(
    *,
    manifest_items: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    packet_root: Path,
    copied: dict[str, str],
    transcode_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_rows = {
        row["example_id"]: row
        for row in aggregate["simmotion_real_pairwise"]["representations"]["m3"][
            "within_dataset_recall_at_1"
        ]["rows"]
    }
    cases = []
    for example_id, stratum, positive_label in SIMMOTION_CASES:
        role_items = {
            role: manifest_items[f"simmotion:{example_id}:{role}"]
            for role in ("ref", "positive", "negative")
        }
        role_records = {role: records[item["item_id"]] for role, item in role_items.items()}
        media = {}
        for role, item in role_items.items():
            source_relative = f"originals/simmotion/{_safe_component(example_id)}__{role}.mp4"
            media_relative = f"media/simmotion/{_safe_component(example_id)}__{role}.mp4"
            _copy_source(
                item=item,
                packet_root=packet_root,
                relative_source=source_relative,
                copied=copied,
            )
            _add_transcode(
                transcode_plan, source=source_relative, destination=media_relative
            )
            media[role] = media_relative
        candidate_order = (
            ["positive", "negative"]
            if positive_label == "A"
            else ["negative", "positive"]
        )
        candidates = []
        for blind_index, role in enumerate(candidate_order):
            candidates.append(
                {
                    "blind_label": chr(ord("A") + blind_index),
                    "role": role,
                    "media": media[role],
                    "scores": _score_pair(role_records["ref"], role_records[role]),
                }
            )
        cases.append(
            {
                "case_id": f"simmotion-{example_id}",
                "example_id": example_id,
                "selection_stratum": stratum,
                "reference": {"role": "ref", "media": media["ref"]},
                "candidates": candidates,
                "official_positive_blind_label": next(
                    row["blind_label"] for row in candidates if row["role"] == "positive"
                ),
                "m3_within_dataset": aggregate_rows[example_id],
            }
        )
    return cases


def _project_cases(
    *,
    manifest_items: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    packet_root: Path,
    copied: dict[str, str],
    transcode_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in manifest_items:
        if item["group"] == "project_saic_bank":
            grouped[item["metadata"]["iid"]].append(item)
    cases = []
    branch_rotations = (
        ["forward", "reverse", "noop"],
        ["reverse", "noop", "forward"],
        ["noop", "forward", "reverse"],
    )
    for case_index, (iid, items) in enumerate(sorted(grouped.items())):
        seeds = sorted({int(item["metadata"]["seed"]) for item in items})
        if len(seeds) < 2:
            raise ValueError(f"project IID lacks two seeds: {iid}")
        query_seed, candidate_seed = seeds[:2]
        query_item = next(
            item
            for item in items
            if int(item["metadata"]["seed"]) == query_seed
            and item["metadata"]["branch"] == "forward"
        )
        candidate_items = {
            item["metadata"]["branch"]: item
            for item in items
            if int(item["metadata"]["seed"]) == candidate_seed
        }
        if set(candidate_items) != set(audit.PROJECT_BRANCHES):
            raise ValueError(f"project candidate branches differ: {iid}")

        query_source = f"originals/project/{iid}__query-forward-s{query_seed}.mp4"
        query_media = f"media/project/{iid}__query-forward-s{query_seed}.mp4"
        _copy_source(
            item=query_item,
            packet_root=packet_root,
            relative_source=query_source,
            copied=copied,
        )
        _add_transcode(
            transcode_plan, source=query_source, destination=query_media
        )
        query_record = records[query_item["item_id"]]
        candidate_order = branch_rotations[case_index % len(branch_rotations)]
        candidates = []
        for blind_index, branch in enumerate(candidate_order):
            item = candidate_items[branch]
            source_relative = f"originals/project/{iid}__{branch}-s{candidate_seed}.mp4"
            media_relative = f"media/project/{iid}__{branch}-s{candidate_seed}.mp4"
            _copy_source(
                item=item,
                packet_root=packet_root,
                relative_source=source_relative,
                copied=copied,
            )
            _add_transcode(
                transcode_plan, source=source_relative, destination=media_relative
            )
            candidates.append(
                {
                    "blind_label": chr(ord("A") + blind_index),
                    "branch": branch,
                    "candidate_id": item["metadata"]["candidate_id"],
                    "media": media_relative,
                    "scores": _score_pair(query_record, records[item["item_id"]]),
                }
            )
        cases.append(
            {
                "case_id": f"project-{iid}",
                "iid": iid,
                "actor_family": query_item["metadata"]["actor_family"],
                "action_family_id": query_item["metadata"]["action_family_id"],
                "query": {
                    "candidate_id": query_item["metadata"]["candidate_id"],
                    "seed": query_seed,
                    "branch": "forward",
                    "media": query_media,
                },
                "candidate_seed": candidate_seed,
                "candidates": candidates,
                "forward_blind_label": next(
                    row["blind_label"] for row in candidates if row["branch"] == "forward"
                ),
            }
        )
    return cases


def _canary_cases(
    *,
    manifest_items: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    packet_root: Path,
    copied: dict[str, str],
    transcode_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = []
    for item in manifest_items:
        if item["group"] != "project_probe":
            continue
        name = _safe_component(item["metadata"]["name"])
        source_relative = f"originals/canary/{name}.mp4"
        natural_media = f"media/canary/{name}__natural.mp4"
        reverse_media = f"media/canary/{name}__reverse.mp4"
        _copy_source(
            item=item,
            packet_root=packet_root,
            relative_source=source_relative,
            copied=copied,
        )
        _add_transcode(
            transcode_plan, source=source_relative, destination=natural_media
        )
        _add_transcode(
            transcode_plan,
            source=source_relative,
            destination=reverse_media,
            transform="reverse",
        )
        record = records[item["item_id"]]
        cases.append(
            {
                "case_id": f"canary-{name}",
                "name": name,
                "natural_media": natural_media,
                "reverse_media": reverse_media,
                "reverse_diagnostics": record["variants"]["reverse"],
            }
        )
    return sorted(cases, key=lambda row: row["case_id"])


def build_packet(args: argparse.Namespace) -> int:
    manifest = audit.load_json(args.manifest)
    aggregate = audit.load_json(args.aggregate_results)
    if manifest.get("manifest_digest") != aggregate["manifest"]["digest"]:
        raise ValueError("manifest and aggregate receipt differ")
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"review output must be fresh: {output}")
    output.mkdir(parents=True)
    records, shard_receipts = _load_features(Path(args.features_dir))
    manifest_items = {item["item_id"]: item for item in manifest["items"]}
    if set(records) != set(manifest_items):
        raise ValueError("feature coverage differs from manifest")

    copied: dict[str, str] = {}
    transcode_plan: list[dict[str, Any]] = []
    review = {
        "schema_version": SCHEMA_VERSION,
        "authority": {
            "human_review_only": True,
            "reward_authorized": False,
            "reranking_authorized": False,
            "training_authorized": False,
        },
        "source_receipts": {
            "manifest_digest": manifest["manifest_digest"],
            "aggregate_receipt_digest": aggregate["receipt_digest"],
            "feature_shards": shard_receipts,
        },
        "simmotion_cases": _simmotion_cases(
            manifest_items=manifest_items,
            records=records,
            aggregate=aggregate,
            packet_root=output,
            copied=copied,
            transcode_plan=transcode_plan,
        ),
        "project_cases": _project_cases(
            manifest_items=manifest["items"],
            records=records,
            packet_root=output,
            copied=copied,
            transcode_plan=transcode_plan,
        ),
        "canary_cases": _canary_cases(
            manifest_items=manifest["items"],
            records=records,
            packet_root=output,
            copied=copied,
            transcode_plan=transcode_plan,
        ),
        "population_summary": {
            "simmotion_m3_pairwise_accuracy": aggregate["simmotion_real_pairwise"][
                "representations"
            ]["m3"]["pairwise_positive_over_negative"]["accuracy"],
            "simmotion_m3_within_dataset_r1": aggregate["simmotion_real_pairwise"][
                "representations"
            ]["m3"]["within_dataset_recall_at_1"]["accuracy"],
            "project_m3_forward_over_reverse": aggregate[
                "project_saic_prompt_branch_bank"
            ]["representations"]["m3"]["forward_anchor_preference"][
                "against_reverse"
            ]["accuracy"],
            "project_m3_other_seed_same_iid_top1": aggregate[
                "project_saic_prompt_branch_bank"
            ]["representations"]["m3"]["global_other_seed_top1"]["rates"][
                "same_iid"
            ],
            "exact_reverse_semantic_moments_cosine_mean": aggregate[
                "controlled_temporal_variants_all_videos"
            ]["reverse"]["semantic_moments_cosine"]["mean"],
        },
    }
    unsigned = dict(review)
    review["review_digest"] = audit.object_sha256(unsigned)
    _write_json(output / "review-manifest.json", review)
    _write_json(output / "transcode-plan.json", transcode_plan)

    template = Path(args.html_template).read_text(encoding="utf-8")
    marker = "__REVIEW_DATA__"
    if template.count(marker) != 1:
        raise ValueError("HTML template must contain one review-data marker")
    embedded = json.dumps(review, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    rendered = template.replace(marker, embedded)
    (output / "index.html").write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "review_digest": review["review_digest"],
                "copied_source_count": len(copied),
                "transcode_count": len(transcode_plan),
                "output_root": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--aggregate-results", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--html-template", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return build_packet(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
