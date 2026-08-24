"""Auditable high-recall selection from a completed Goku source census.

The distributed census intentionally used the historical, conservative
``spatial_energy_entropy <= 0.94`` gate.  That statistic measures how spread
motion energy is over the image; it is not, by itself, evidence that a clip is
static or unsuitable.  This module consumes (and SHA-binds) the census final
artifacts without modifying their receipts.  It may waive *only* the recorded
``spatial_motion_entropy_too_high`` reason up to an explicit bound, then ranks
freshly analysed clips by score in deterministic family round-robin order.

Every selected row retains the upstream verdict and waiver in
``source_gate_policy``.  Old-IID/group exclusions, media geometry, scene cuts,
weak dynamics, and every other census rejection remain hard failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import goku_action_anchor_prefilter as prefilter
from .goku_action_anchor_qwen import validate_input_row


SCHEMA_VERSION = "motive-goku-atomic-source-high-recall-v1"
SUMMARY_SCHEMA = "motive-goku-atomic-source-high-recall-summary-v1"
DONE_SCHEMA = "motive-goku-atomic-source-high-recall-done-v1"
POLICY_SCHEMA = "motive-goku-source-gate-high-recall-v1"
ADVISORY_REASON = "spatial_motion_entropy_too_high"
UPSTREAM_SELECTION_ANNOTATION = "duplicate_fresh_group_not_selected"
ELIGIBILITY_NAME = "eligibility.jsonl"
SELECTED_NAME = "selected.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
ANCHOR_DIR = "anchors"
FINAL_ENTRIES = frozenset(
    {ELIGIBILITY_NAME, SELECTED_NAME, SUMMARY_NAME, DONE_NAME, ANCHOR_DIR}
)


class HighRecallError(RuntimeError):
    """Fail-closed high-recall finalization error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha_field(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HighRecallError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise HighRecallError(f"invalid JSON in {context}: {error}") from error
    if not isinstance(value, dict):
        raise HighRecallError(f"{context} is not a JSON object")
    return value


def _read_plain(path: Path, *, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise HighRecallError(f"{context} must be a plain file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise HighRecallError(f"{context} is empty: {path}")
    return raw


def _read_jsonl(raw: bytes, *, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise HighRecallError(f"blank line in {context}:{line_number}")
        rows.append(_parse_json(line, context=f"{context}:{line_number}"))
    return rows


def _implementation_bundle() -> dict[str, Any]:
    modules = {
        "goku_atomic_source_high_recall": Path(__file__).resolve(strict=True),
        "goku_action_anchor_prefilter": Path(
            str(prefilter.__file__)
        ).resolve(strict=True),
    }
    records = {
        name: {"path": str(path), "sha256": _file_sha256(path)}
        for name, path in sorted(modules.items())
    }
    return {"modules": records, "bundle_digest": _digest(records)}


def _load_bound_upstream(
    source_final_dir: str | Path,
    *,
    expected_source_done_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(source_final_dir).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise HighRecallError(f"source_final_dir must be a plain directory: {root}")
    done_raw = _read_plain(root / DONE_NAME, context="upstream done")
    actual_done_sha = _sha256(done_raw)
    if actual_done_sha != _sha_field(
        expected_source_done_sha256, name="expected source done SHA"
    ):
        raise HighRecallError(
            "upstream done SHA differs: "
            f"expected={expected_source_done_sha256} actual={actual_done_sha}"
        )
    done = _parse_json(done_raw, context="upstream done")
    if done.get("status") != "complete":
        raise HighRecallError("upstream census is not complete")
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HighRecallError("upstream done lacks artifacts")
    if done.get("artifact_digest") != _digest(artifacts):
        raise HighRecallError("upstream artifact digest differs")

    evaluated_raw = _read_plain(root / "evaluated.jsonl", context="evaluated")
    summary_raw = _read_plain(root / SUMMARY_NAME, context="upstream summary")
    for name, raw in (
        ("evaluated.jsonl", evaluated_raw),
        (SUMMARY_NAME, summary_raw),
    ):
        expected = artifacts.get(name)
        if expected != _sha256(raw):
            raise HighRecallError(f"upstream artifact SHA differs: {name}")
    summary = _parse_json(summary_raw, context="upstream summary")
    if summary.get("status") != "complete":
        raise HighRecallError("upstream summary is not complete")
    semantics = summary.get("semantics")
    if not isinstance(semantics, dict) or not all(
        semantics.get(key) is True
        for key in (
            "fresh_media_geometry_motion_analysis",
            "old_iid_and_group_exclusion",
        )
    ):
        raise HighRecallError("upstream exclusion/fresh-census semantics are missing")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise HighRecallError("upstream summary lacks inputs")
    _sha_field(inputs.get("old_selected_sha256"), name="old selected SHA")
    if done.get("input_binding_digest") != inputs.get("binding_digest"):
        raise HighRecallError("upstream input binding digest differs")
    if done.get("config_digest") != summary.get("config_digest"):
        raise HighRecallError("upstream config digest differs")

    rows = _read_jsonl(evaluated_raw, context="evaluated")
    counts = summary.get("counts")
    if not isinstance(counts, dict) or int(counts.get("mother", -1)) != len(rows):
        raise HighRecallError("upstream evaluated closure count differs")
    ranks = [row.get("mother_rank") for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        raise HighRecallError("upstream evaluated mother ranks are not complete")
    iids = [str(row.get("iid", "")) for row in rows]
    if any(not iid for iid in iids) or len(set(iids)) != len(iids):
        raise HighRecallError("upstream evaluated IIDs are missing or duplicated")

    binding = {
        "source_final_dir": str(root),
        "source_done_sha256": actual_done_sha,
        "source_evaluated_sha256": _sha256(evaluated_raw),
        "source_summary_sha256": _sha256(summary_raw),
        "source_artifact_digest": done["artifact_digest"],
        "source_input_binding_digest": inputs["binding_digest"],
        "old_selected_sha256": inputs["old_selected_sha256"],
        "source_implementation_bundle_digest": done.get(
            "implementation_bundle_digest"
        ),
        "source_config_digest": summary["config_digest"],
    }
    binding["binding_digest"] = _digest(binding)
    return rows, binding


def _eligibility_decision(
    row: Mapping[str, Any], *, max_spatial_energy_entropy: float
) -> dict[str, Any]:
    reasons = row.get("rejection_reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise HighRecallError(f"iid={row.get('iid')} has invalid rejection_reasons")
    entropy = None
    actor_motion = row.get("actor_motion")
    if isinstance(actor_motion, dict):
        entropy = actor_motion.get("spatial_energy_entropy")
    entropy_ok = (
        isinstance(entropy, (int, float))
        and not isinstance(entropy, bool)
        and math.isfinite(float(entropy))
        and 0.0 <= float(entropy) <= max_spatial_energy_entropy
    )
    advisory: list[str] = []
    upstream_annotations: list[str] = []
    hard: list[str] = []
    for reason in reasons:
        if reason == ADVISORY_REASON and entropy_ok:
            advisory.append(reason)
        elif reason == UPSTREAM_SELECTION_ANNOTATION and bool(row.get("eligible")):
            # This is injected by the strict finalizer after receipt validation;
            # group uniqueness is re-applied below from score-ranked candidates.
            upstream_annotations.append(reason)
        else:
            hard.append(reason)
    upstream_eligible = row.get("eligible")
    if not isinstance(upstream_eligible, bool):
        hard.append("high_recall_invalid_upstream_eligible")
    elif upstream_eligible:
        unexpected = set(reasons) - {UPSTREAM_SELECTION_ANNOTATION}
        if unexpected:
            hard.append("high_recall_upstream_eligibility_inconsistent")
    elif not reasons:
        hard.append("high_recall_upstream_eligibility_inconsistent")
    required_analysis = all(
        isinstance(row.get(key), dict) for key in ("media", "motion", "actor_motion")
    )
    score = row.get("prefilter_score")
    score_ok = (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
    )
    if not required_analysis:
        hard.append("high_recall_missing_fresh_analysis")
    if not score_ok:
        hard.append("high_recall_invalid_prefilter_score")
    eligible = not hard
    return {
        "upstream_eligible": upstream_eligible,
        "upstream_rejection_reasons": list(reasons),
        "advisory_waivers": advisory,
        "upstream_selection_annotations": upstream_annotations,
        "hard_rejection_reasons": list(dict.fromkeys(hard)),
        "spatial_energy_entropy": entropy,
        "high_recall_eligible": eligible,
    }


def _anchor_payload(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the minimal picklable contract consumed by one anchor worker."""

    return {
        "index": index,
        "iid": row.get("iid"),
        "resolved_src_video": row.get("resolved_src_video"),
        "source_video_sha256": row.get("source_video_sha256"),
        "media": row.get("media"),
    }


def _extract_bound_anchor(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one immutable source and decode its lossless frame-zero anchor."""

    index = payload.get("index")
    iid = str(payload.get("iid", ""))
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise HighRecallError(f"invalid anchor task index for iid={iid}")
    media = payload.get("media")
    if not isinstance(media, dict):
        raise HighRecallError(f"selected row lacks bound media: {iid}")
    try:
        source = Path(str(payload.get("resolved_src_video", ""))).resolve(
            strict=True
        )
    except (FileNotFoundError, OSError) as error:
        raise HighRecallError(f"selected source cannot be resolved: {iid}") from error
    if source.is_symlink() or not source.is_file():
        raise HighRecallError(f"selected source is not a plain file: {source}")
    before = source.stat()
    if _file_sha256(source) != payload.get("source_video_sha256"):
        raise HighRecallError(f"source SHA changed: {iid}")
    anchor_bytes, width, height = prefilter._extract_anchor_png_bytes(source)
    after = source.stat()
    expected_size = int(media.get("file_size_bytes", -1))
    expected_mtime = int(media.get("mtime_ns_at_analysis", -1))
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or after.st_size != expected_size
        or after.st_mtime_ns != expected_mtime
    ):
        raise HighRecallError(f"source stat changed: {iid}")
    return {
        "index": index,
        "iid": iid,
        "source": str(source),
        "source_size": after.st_size,
        "source_mtime_ns": after.st_mtime_ns,
        "anchor_bytes": anchor_bytes,
        "width": int(width),
        "height": int(height),
    }


def _ordered_anchor_results(
    chosen: Sequence[Mapping[str, Any]], *, local_workers: int
) -> Iterable[dict[str, Any]]:
    """Yield dynamically computed anchors in the original chosen order.

    At most ``2 * local_workers`` results are in flight or buffered.  This
    preserves bounded memory even if an early task is slow while later tasks
    finish, and permits a new task to be submitted whenever ordered output
    drains buffer capacity.
    """

    payloads = [_anchor_payload(index, row) for index, row in enumerate(chosen)]
    if local_workers == 1:
        for payload in payloads:
            yield _extract_bound_anchor(payload)
        return
    limit = 2 * local_workers
    next_submit = 0
    next_output = 0
    ready: dict[int, dict[str, Any]] = {}
    in_flight: dict[Any, int] = {}
    with ProcessPoolExecutor(max_workers=local_workers) as executor:
        while next_output < len(payloads):
            while (
                next_submit < len(payloads)
                and len(in_flight) + len(ready) < limit
            ):
                payload = payloads[next_submit]
                future = executor.submit(_extract_bound_anchor, payload)
                in_flight[future] = next_submit
                next_submit += 1
            if next_output in ready:
                yield ready.pop(next_output)
                next_output += 1
                continue
            if not in_flight:
                raise HighRecallError("anchor executor ended before ordered closure")
            completed, unused_pending = wait(
                in_flight, return_when=FIRST_COMPLETED
            )
            for future in completed:
                expected_index = in_flight.pop(future)
                result = future.result()
                actual_index = result.get("index")
                if actual_index != expected_index or actual_index in ready:
                    raise HighRecallError("anchor worker result identity differs")
                ready[actual_index] = result


def _publish_directory(output_dir: Path, writer: Any) -> None:
    target = output_dir.expanduser().resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"create-only output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        writer(stage, target)
        if {entry.name for entry in stage.iterdir()} != FINAL_ENTRIES:
            raise HighRecallError("high-recall staging closure differs")
        if os.path.lexists(target):
            raise FileExistsError(f"output appeared during publication: {target}")
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def finalize(
    *,
    source_final_dir: str | Path,
    expected_source_done_sha256: str,
    output_dir: str | Path,
    sample_size: int,
    max_spatial_energy_entropy: float = 1.0,
    max_per_family: int | None = None,
    local_workers: int = 1,
) -> dict[str, Any]:
    """Publish score-ranked, family-diverse candidates from a bound census."""

    if sample_size <= 0:
        raise HighRecallError("sample_size must be positive")
    if not 0.94 <= max_spatial_energy_entropy <= 1.0:
        raise HighRecallError("max_spatial_energy_entropy must be in [0.94, 1.0]")
    if max_per_family is not None and max_per_family <= 0:
        raise HighRecallError("max_per_family must be positive when provided")
    if (
        isinstance(local_workers, bool)
        or not isinstance(local_workers, int)
        or local_workers <= 0
    ):
        raise HighRecallError("local_workers must be positive")
    rows, upstream = _load_bound_upstream(
        source_final_dir,
        expected_source_done_sha256=expected_source_done_sha256,
    )
    effective_max_per_family = sample_size if max_per_family is None else max_per_family
    policy = {
        "schema_version": POLICY_SCHEMA,
        "advisory_reason": ADVISORY_REASON,
        "max_spatial_energy_entropy": max_spatial_energy_entropy,
        "all_other_census_rejections_remain_hard": True,
        "old_iid_and_group_exclusion_remains_hard": True,
        "selection": "fresh_score_descending_within_family_round_robin_unique_group",
        "max_per_family": effective_max_per_family,
    }
    policy["policy_digest"] = _digest(policy)

    eligibility_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for upstream_row in rows:
        decision = _eligibility_decision(
            upstream_row,
            max_spatial_energy_entropy=max_spatial_energy_entropy,
        )
        audit = {
            "schema_version": POLICY_SCHEMA,
            "policy_digest": policy["policy_digest"],
            **decision,
        }
        eligibility_rows.append(
            {
                "iid": upstream_row["iid"],
                "group_id": upstream_row.get("group_id"),
                "family": upstream_row.get("family"),
                "mother_rank": upstream_row["mother_rank"],
                "prefilter_score": upstream_row.get("prefilter_score"),
                "source_gate_policy": audit,
                "selected": False,
                "selection_rank": None,
            }
        )
        if decision["high_recall_eligible"]:
            candidate = dict(upstream_row)
            candidate["eligible"] = True
            candidate["rejection_reasons"] = []
            candidate["source_gate_policy"] = audit
            candidates.append(candidate)

    chosen = prefilter.select_diverse(
        candidates,
        sample_size=sample_size,
        max_per_family=effective_max_per_family,
    )
    chosen_iids = {str(row["iid"]): row for row in chosen}
    for row in eligibility_rows:
        selected = chosen_iids.get(str(row["iid"]))
        if selected is not None:
            row["selected"] = True
            row["selection_rank"] = selected["selection_rank"]

    implementation = _implementation_bundle()
    target = Path(output_dir).expanduser().resolve(strict=False)

    def writer(stage: Path, final_output: Path) -> None:
        (stage / ANCHOR_DIR).mkdir()
        selected: list[dict[str, Any]] = []
        family_counts: Counter[str] = Counter()
        for anchor in _ordered_anchor_results(chosen, local_workers=local_workers):
            index = int(anchor["index"])
            if index != len(selected):
                raise HighRecallError("anchor output order differs from chosen order")
            chosen_row = chosen[index]
            row = dict(chosen_row)
            if str(row["iid"]) != anchor["iid"]:
                raise HighRecallError("anchor IID differs from chosen row")
            source = Path(str(anchor["source"]))
            current = source.stat()
            if (
                current.st_size != int(anchor["source_size"])
                or current.st_mtime_ns != int(anchor["source_mtime_ns"])
            ):
                raise HighRecallError(
                    f"source stat changed before ordered write: {row['iid']}"
                )
            anchor_bytes = anchor["anchor_bytes"]
            width = int(anchor["width"])
            height = int(anchor["height"])
            media = row.get("media")
            if not isinstance(media, dict):
                raise HighRecallError(f"selected row lacks bound media: {row['iid']}")
            relative = Path(ANCHOR_DIR) / f"{row['iid']}.png"
            (stage / relative).write_bytes(anchor_bytes)
            family_counts[str(row["family"])] += 1
            row.update(
                {
                    "selected": True,
                    "within_family_rank": family_counts[str(row["family"])],
                    "anchor_image": relative.as_posix(),
                    "resolved_anchor_image": str(
                        (final_output / relative).resolve(strict=False)
                    ),
                    "anchor_sha256": _sha256(anchor_bytes),
                    "media": {
                        **media,
                        "anchor_width": width,
                        "anchor_height": height,
                        "anchor_frame_index": 0,
                        "anchor_encoding": "lossless_png",
                    },
                }
            )
            validate_input_row(row)
            selected.append(row)

        eligibility_raw = _jsonl_bytes(eligibility_rows)
        selected_raw = _jsonl_bytes(selected)
        (stage / ELIGIBILITY_NAME).write_bytes(eligibility_raw)
        (stage / SELECTED_NAME).write_bytes(selected_raw)
        source_counts: Counter[str] = Counter()
        for item in eligibility_rows:
            decision = item["source_gate_policy"]
            if decision["high_recall_eligible"]:
                source_counts["high_recall_eligible"] += 1
            if decision["upstream_eligible"]:
                source_counts["upstream_eligible"] += 1
            if ADVISORY_REASON in decision["advisory_waivers"]:
                source_counts["entropy_advisory_present"] += 1
                if decision["high_recall_eligible"]:
                    source_counts["entropy_advisory_waived_eligible"] += 1
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "upstream": upstream,
            "policy": policy,
            "implementation": implementation,
            "anchor_extraction": {
                "local_workers": local_workers,
                "executor": (
                    "in_process_serial"
                    if local_workers == 1
                    else "bounded_dynamic_process_pool"
                ),
                "max_in_flight_or_buffered": 1 if local_workers == 1 else 2 * local_workers,
                "output_order": "chosen_index_ascending",
            },
            "counts": {
                "evaluated": len(rows),
                **dict(sorted(source_counts.items())),
                "requested": sample_size,
                "selected": len(selected),
                "selection_shortfall": max(sample_size - len(selected), 0),
            },
            "selected_iids": [row["iid"] for row in selected],
            "selected_families": dict(
                sorted(Counter(str(row["family"]) for row in selected).items())
            ),
            "production_eligible": False,
        }
        summary_raw = _json_bytes(summary)
        (stage / SUMMARY_NAME).write_bytes(summary_raw)
        anchors = {row["anchor_image"]: row["anchor_sha256"] for row in selected}
        artifacts = {
            ELIGIBILITY_NAME: _sha256(eligibility_raw),
            SELECTED_NAME: _sha256(selected_raw),
            SUMMARY_NAME: _sha256(summary_raw),
            ANCHOR_DIR: _digest(anchors),
        }
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "upstream_binding_digest": upstream["binding_digest"],
            "policy_digest": policy["policy_digest"],
            "implementation_bundle_digest": implementation["bundle_digest"],
            "counts": summary["counts"],
            "artifacts": artifacts,
            "anchor_sha256": anchors,
            "artifact_digest": _digest(artifacts),
        }
        (stage / DONE_NAME).write_bytes(_json_bytes(done))

    _publish_directory(target, writer)
    return json.loads((target / SUMMARY_NAME).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-recall score/family selection from a bound Goku census."
    )
    parser.add_argument("--source-final-dir", required=True, type=Path)
    parser.add_argument("--expected-source-done-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", required=True, type=int)
    parser.add_argument("--max-spatial-energy-entropy", type=float, default=1.0)
    parser.add_argument("--max-per-family", type=int, default=None)
    parser.add_argument("--local-workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = finalize(
        source_final_dir=args.source_final_dir,
        expected_source_done_sha256=args.expected_source_done_sha256,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_spatial_energy_entropy=args.max_spatial_energy_entropy,
        max_per_family=args.max_per_family,
        local_workers=args.local_workers,
    )
    print(_canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
