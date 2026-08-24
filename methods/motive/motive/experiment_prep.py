"""Prepare provenance-bound Motive action experiment manifests.

The first supported profile is intentionally a *legacy pseudo-label pilot*.
It extracts only conservative Qwen positives from the immutable v6 calibration
run.  It must not be confused with a human-approved production manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .qwen_filter import (
    _object_digest,
    _validate_observation,
    _validate_visual,
)


EXPERIMENT_PSEUDO_LABEL_SCHEMA = "motive-experiment-pseudo-label-v1"
LEGACY_QWEN_POLICY = "legacy-qwen-original-valid-action-v1"
INPUT_DIGEST_FIELDS = (
    "iid",
    "prompt",
    "src_video",
    "tgt_video",
    "source_caption",
    "edited_caption",
)
SPLITS = {"train", "validation", "test"}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with _atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    with _atomic_text_writer(path) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _validate_input_row(row: dict[str, Any], *, context: str) -> None:
    missing = [key for key in INPUT_DIGEST_FIELDS if key not in row]
    if missing:
        raise ValueError(f"{context} is missing input fields {missing}")
    expected = row.get("input_digest")
    actual = _canonical_digest({key: row[key] for key in INPUT_DIGEST_FIELDS})
    if expected != actual:
        raise ValueError(
            f"{context} input_digest does not bind its prompt/video/caption fields"
        )
    if str(row.get("split")) not in SPLITS:
        raise ValueError(f"{context} has invalid split={row.get('split')!r}")


def _strict_legacy_qwen_positive(
    row: dict[str, Any],
    *,
    source_manifest_sha256: str,
    context: str,
) -> tuple[dict[str, Any] | None, str]:
    """Return a derived pilot row or a conservative rejection reason."""

    _validate_input_row(row, context=context)
    evidence = row.get("qwen_evidence")
    if not isinstance(evidence, dict):
        return None, "missing_qwen_evidence"
    visual = evidence.get("visual")
    if not isinstance(visual, dict):
        return None, "missing_visual_evidence"
    if visual.get("iid") != row.get("iid"):
        raise ValueError(f"{context} Qwen iid disagrees with fused row")
    if visual.get("input_digest") != row.get("input_digest"):
        raise ValueError(f"{context} Qwen input_digest disagrees with fused row")
    if visual.get("status") != "ok":
        return None, "qwen_status_not_ok"
    if visual.get("observation_validated_from") != "original":
        return None, "observation_not_original"
    if visual.get("result_validated_from") != "original":
        return None, "result_not_original"
    if visual.get("observation_repairs"):
        return None, "observation_has_repairs"
    if visual.get("alignment_repairs"):
        return None, "alignment_has_repairs"

    observation = visual.get("observation")
    result = visual.get("result")
    if not isinstance(observation, dict) or not isinstance(result, dict):
        raise ValueError(f"{context} lacks structured Qwen observation/result")
    _validate_observation(observation)
    _validate_visual(result, observation=observation)

    observation_digest = _object_digest(observation)
    if visual.get("observation_digest") != observation_digest:
        raise ValueError(f"{context} Qwen observation digest mismatch")
    result_object_digest = _object_digest(result)
    declared_result_digest = visual.get("result_digest")
    if (
        declared_result_digest is not None
        and declared_result_digest != result_object_digest
    ):
        raise ValueError(f"{context} Qwen result digest mismatch")
    if result.get("verdict") != "valid_action":
        return None, f"verdict:{result.get('verdict', '<missing>')}"
    if observation.get("target_actor_motion") != "clear":
        return None, (
            "quality:target_actor_motion="
            f"{observation.get('target_actor_motion', '<missing>')}"
        )
    if result.get("confidence") not in {"medium", "high"}:
        return None, (
            f"quality:confidence={result.get('confidence', '<missing>')}"
        )

    quality_requirements = {
        "camera_dominance": "low",
        "background_dominance": "low",
        "artifact_level": "low",
        "preservation_quality": "acceptable",
    }
    for key, expected in quality_requirements.items():
        if observation.get(key) != expected:
            return None, f"quality:{key}={observation.get(key, '<missing>')}"

    action_signature = str(result.get("action_signature") or "").strip()
    if not action_signature:
        raise ValueError(f"{context} valid_action has an empty action_signature")
    derived = dict(row)
    derived["experiment_pseudo_label"] = {
        "schema_version": EXPERIMENT_PSEUDO_LABEL_SCHEMA,
        "policy": LEGACY_QWEN_POLICY,
        "action_signature": action_signature,
        "source_manifest_sha256": source_manifest_sha256,
        "observation_digest": observation_digest,
        "result_object_digest": result_object_digest,
        "legacy_result_digest_missing": declared_result_digest is None,
        "human_approved": False,
        "production_eligible": False,
    }
    return derived, "selected"


def _priority(row: dict[str, Any], seed: int) -> tuple[str, str]:
    iid = str(row["iid"])
    return (
        hashlib.sha256(f"{seed}\0{iid}".encode("utf-8")).hexdigest(),
        iid,
    )


def prepare_legacy_qwen_pilot(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser()
    output_dir = args.output_dir.expanduser()
    source_digest_before = _file_digest(input_path)
    output_paths = {
        "representation": output_dir / "representation_manifest.jsonl",
        "lucy_train": output_dir / "lucy_train_manifest.jsonl",
        "lucy_eval": output_dir / "lucy_eval_manifest.jsonl",
        "summary": output_dir / "summary.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "experiment outputs already exist; use a new output directory or "
            f"--overwrite: {[str(path) for path in existing]}"
        )
    max_lucy_train = int(args.max_lucy_train)
    if max_lucy_train < 0:
        raise ValueError("--max-lucy-train must be non-negative")

    selected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    seen: set[str] = set()
    total = 0
    for line_number, row in enumerate(_iter_jsonl(input_path), start=1):
        total += 1
        iid = str(row.get("iid") or "")
        if not iid:
            raise ValueError(f"{input_path}:{line_number} has an empty iid")
        if iid in seen:
            raise ValueError(f"{input_path}:{line_number} duplicates iid={iid}")
        seen.add(iid)
        derived, reason = _strict_legacy_qwen_positive(
            row,
            source_manifest_sha256=source_digest_before,
            context=f"{input_path}:{line_number}",
        )
        reason_counts[reason] += 1
        if derived is not None:
            selected.append(derived)

    source_digest_after = _file_digest(input_path)
    if source_digest_after != source_digest_before:
        raise RuntimeError(f"{input_path} changed while experiment inputs were read")
    selected.sort(key=lambda row: str(row["iid"]))
    split_counts = Counter(str(row["split"]) for row in selected)
    if split_counts["train"] < 2:
        raise RuntimeError(
            f"pilot has fewer than two training rows: {dict(split_counts)}"
        )
    if not split_counts["validation"] or not split_counts["test"]:
        raise RuntimeError(
            "pilot must retain non-empty validation and test splits: "
            f"{dict(split_counts)}"
        )

    train_rows = sorted(
        (row for row in selected if row["split"] == "train"),
        key=lambda row: _priority(row, int(args.seed)),
    )
    if max_lucy_train:
        train_rows = train_rows[:max_lucy_train]
    eval_rows = [
        row for row in selected if row["split"] in {"validation", "test"}
    ]

    _write_jsonl(output_paths["representation"], selected)
    _write_jsonl(output_paths["lucy_train"], train_rows)
    _write_jsonl(output_paths["lucy_eval"], eval_rows)
    output_digests = {
        name: {
            "path": str(path),
            "sha256": _file_digest(path),
            "rows": (
                len(selected)
                if name == "representation"
                else len(train_rows)
                if name == "lucy_train"
                else len(eval_rows)
            ),
        }
        for name, path in output_paths.items()
        if name != "summary"
    }
    split_provenance_counts = Counter(
        str((row.get("split_provenance") or {}).get("version") or "<missing>")
        for row in selected
    )
    summary = {
        "schema_version": "motive-action-experiment-prep-v1",
        "profile": LEGACY_QWEN_POLICY,
        "status": "pseudo_label_interface_pilot_only",
        "production_eligible": False,
        "source_manifest": str(input_path),
        "source_manifest_sha256": source_digest_before,
        "source_rows": total,
        "selected_rows": len(selected),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_provenance_counts": dict(
            sorted(split_provenance_counts.items())
        ),
        "legacy_non_content_split": any(
            version not in {
                "source-sampled-phash-v1",
                "source-visual-cluster-v1",
            }
            for version in split_provenance_counts
        ),
        "legacy_missing_result_digest_rows": sum(
            bool(row["experiment_pseudo_label"]["legacy_result_digest_missing"])
            for row in selected
        ),
        "lucy_train_seed": int(args.seed),
        "lucy_train_limit": max_lucy_train,
        "outputs": output_digests,
        "required_training_flags": [
            "--allow-unreviewed-pseudo-labels",
            "--allow-non-content-splits",
        ],
        "limitations": [
            "No human verdict is asserted or synthesized.",
            "The source Qwen run predates post-generation result_digest.",
            "The inherited split is not content-derived.",
            "This pilot may validate interfaces but not representation generalization.",
        ],
    }
    _write_json(output_paths["summary"], summary)
    print(
        "[motive-experiment-prep] "
        f"source={total} selected={len(selected)} "
        f"lucy_train={len(train_rows)} eval={len(eval_rows)} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare provenance-bound Motive action experiment inputs."
    )
    parser.add_argument(
        "profile",
        choices=[LEGACY_QWEN_POLICY],
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-lucy-train", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260108828)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.profile == LEGACY_QWEN_POLICY:
        return prepare_legacy_qwen_pilot(args)
    raise AssertionError(args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
