"""Deterministically select unseen high-yield candidates for R7 visual review.

This stage is intentionally text/rule-only.  It excludes every IID from prior
development manifests before selecting rows, so the resulting queue can be
used to construct a new visual-cluster split after Qwen/feature screening.
It does not assign train/validation/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


R7_EXPANSION_SCHEMA = "motive-r7-expansion-selection-v1"
DEFAULT_LABELS = ("temporal_action",)
DEFAULT_MIN_SCORE = 0.60
DEFAULT_FAMILY_CAP = 2_000
DEFAULT_SEED = 260108829
LEGACY_SPLIT_VALUES = frozenset({"train", "validation", "test"})
LEGACY_SPLIT_FIELDS = frozenset({"split", "split_provenance"})
LEGACY_SPLIT_PROVENANCE_FIELDS = frozenset({"seed", "version"})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _binding_digest(values: Iterable[tuple[str, str]]) -> str:
    """Bind IIDs to canonical pair digests independently of input order."""

    records = [
        _canonical_json({"iid": iid, "canonical_sha256": digest})
        for iid, digest in sorted(values)
    ]
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _iid(row: Mapping[str, Any], *, path: Path | None = None) -> str:
    value = str(row.get("iid") or row.get("id") or "").strip()
    if not value:
        location = "" if path is None else f" in {path}"
        raise ValueError(f"row has no IID{location}")
    return value


def _rule(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("auto_rule")
    if not isinstance(value, Mapping):
        raise ValueError(f"IID {_iid(row)} has no auto_rule object")
    return value


def _primary_family(rule: Mapping[str, Any]) -> str:
    families = rule.get("action_families")
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        return "unknown"
    values = [str(value).strip().lower() for value in families if str(value).strip()]
    return values[0] if values else "unknown"


def _actor_present(rule: Mapping[str, Any]) -> bool:
    actors = rule.get("actors")
    return (
        isinstance(actors, Sequence)
        and not isinstance(actors, (str, bytes))
        and any(str(value).strip() for value in actors)
    )


def _validate_legacy_split_pair(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> dict[str, Any] | None:
    """Validate legacy split metadata before quarantining it from output.

    Selection deliberately creates an unsplit queue.  A complete, canonical
    legacy pair may be audited and removed, but a partial or type-confused
    pair is ambiguous and therefore fails closed.
    """

    has_split = "split" in row
    has_provenance = "split_provenance" in row
    if has_split != has_provenance:
        missing = "split_provenance" if has_split else "split"
        raise ValueError(
            f"IID {iid} has partial legacy split metadata; missing {missing}"
        )
    if not has_split:
        return None

    split = row["split"]
    if type(split) is not str or split not in LEGACY_SPLIT_VALUES:
        raise ValueError(f"IID {iid} has invalid legacy split")

    provenance = row["split_provenance"]
    if type(provenance) is not dict:
        raise ValueError(
            f"IID {iid} legacy split_provenance must be an object"
        )
    if set(provenance) != LEGACY_SPLIT_PROVENANCE_FIELDS:
        raise ValueError(
            f"IID {iid} legacy split_provenance must have exactly "
            "seed/version"
        )
    seed = provenance["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError(
            f"IID {iid} legacy split_provenance seed must be a "
            "non-negative integer"
        )
    version = provenance["version"]
    if (
        type(version) is not str
        or not version
        or version.strip() != version
        or "\x00" in version
    ):
        raise ValueError(
            f"IID {iid} legacy split_provenance version must be a "
            "canonical non-empty string"
        )

    canonical_provenance = {"seed": seed, "version": version}
    pair = {
        "split": split,
        "split_provenance": canonical_provenance,
    }
    return {
        "split": split,
        "provenance_seed": seed,
        "provenance_version": version,
        "provenance_sha256": _object_digest(canonical_provenance),
        "canonical_sha256": _object_digest(pair),
    }


def _parse_family_caps(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in values:
        family, separator, raw_cap = item.partition("=")
        family = family.strip().lower()
        if not separator or not family:
            raise ValueError(f"invalid --family-cap {item!r}; expected FAMILY=COUNT")
        try:
            cap = int(raw_cap)
        except ValueError as exc:
            raise ValueError(f"invalid --family-cap {item!r}") from exc
        if cap < 0:
            raise ValueError("family caps must be non-negative")
        result[family] = cap
    return result


def load_excluded_iids(paths: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    excluded: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = 0
        before = len(excluded)
        for row in _iter_jsonl(path):
            excluded.add(_iid(row, path=path))
            rows += 1
        provenance.append(
            {
                "path": str(path),
                "sha256": _file_digest(path),
                "rows": rows,
                "new_unique_iids": len(excluded) - before,
            }
        )
    return excluded, provenance


def select_expansion_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    excluded_iids: set[str],
    sample_size: int,
    labels: set[str] | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    require_actor: bool = True,
    default_family_cap: int = DEFAULT_FAMILY_CAP,
    family_caps: Mapping[str, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select rows by deterministic family-balanced round robin."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not math.isfinite(min_score):
        raise ValueError("min_score must be finite")
    if default_family_cap < 0:
        raise ValueError("default_family_cap must be non-negative")
    labels = set(DEFAULT_LABELS if labels is None else labels)
    if not labels or any(not str(value).strip() for value in labels):
        raise ValueError("labels must be non-empty strings")
    overrides = {
        str(key).strip().lower(): int(value)
        for key, value in (family_caps or {}).items()
    }
    if any(value < 0 for value in overrides.values()):
        raise ValueError("family caps must be non-negative")

    seen_iids: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    buckets: dict[
        str,
        list[
            tuple[
                tuple[Any, ...],
                dict[str, Any],
                dict[str, Any] | None,
            ]
        ],
    ] = {}
    source_legacy_split_counts: Counter[str] = Counter()
    source_legacy_version_counts: Counter[str] = Counter()
    source_legacy_seed_counts: Counter[int] = Counter()
    source_legacy_provenance_digest_counts: Counter[str] = Counter()
    source_legacy_pair_digest_counts: Counter[str] = Counter()
    source_legacy_bindings: list[tuple[str, str]] = []
    scanned = 0
    for raw_row in rows:
        scanned += 1
        row = dict(raw_row)
        iid = _iid(row)
        if iid in seen_iids:
            raise ValueError(f"duplicate input IID: {iid}")
        seen_iids.add(iid)
        legacy_split = _validate_legacy_split_pair(row, iid=iid)
        if legacy_split is not None:
            source_legacy_split_counts[str(legacy_split["split"])] += 1
            source_legacy_version_counts[
                str(legacy_split["provenance_version"])
            ] += 1
            source_legacy_seed_counts[
                int(legacy_split["provenance_seed"])
            ] += 1
            source_legacy_provenance_digest_counts[
                str(legacy_split["provenance_sha256"])
            ] += 1
            source_legacy_pair_digest_counts[
                str(legacy_split["canonical_sha256"])
            ] += 1
            source_legacy_bindings.append(
                (iid, str(legacy_split["canonical_sha256"]))
            )
        if iid in excluded_iids:
            rejection_counts["prior_iid"] += 1
            continue
        rule = _rule(row)
        label = str(rule.get("label") or "").strip()
        if label not in labels:
            rejection_counts["rule_label"] += 1
            continue
        try:
            score = float(rule.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"IID {iid} has invalid rule score") from exc
        if not math.isfinite(score):
            raise ValueError(f"IID {iid} has non-finite rule score")
        if score < min_score:
            rejection_counts["rule_score"] += 1
            continue
        if require_actor and not _actor_present(rule):
            rejection_counts["actor_missing"] += 1
            continue
        family = _primary_family(rule)
        cap = overrides.get(family, default_family_cap)
        if cap == 0:
            rejection_counts["family_disabled"] += 1
            continue
        tier = str(rule.get("tier") or "").strip().lower()
        tier_rank = 0 if tier == "high" else 1
        priority = hashlib.sha256(f"{seed}\0{iid}".encode("utf-8")).hexdigest()
        key = (tier_rank, -score, priority, iid, _canonical_json(row))
        sanitized_row = dict(row)
        sanitized_row.pop("split", None)
        sanitized_row.pop("split_provenance", None)
        buckets.setdefault(family, []).append(
            (key, sanitized_row, legacy_split)
        )

    for values in buckets.values():
        values.sort(key=lambda value: value[0])

    family_order = sorted(
        buckets,
        key=lambda family: (
            hashlib.sha256(f"{seed}\0family\0{family}".encode("utf-8")).hexdigest(),
            family,
        ),
    )
    cursors = {family: 0 for family in family_order}
    selected_counts: Counter[str] = Counter()
    selected_legacy_split_counts: Counter[str] = Counter()
    selected_legacy_version_counts: Counter[str] = Counter()
    selected_legacy_seed_counts: Counter[int] = Counter()
    selected_legacy_provenance_digest_counts: Counter[str] = Counter()
    selected_legacy_pair_digest_counts: Counter[str] = Counter()
    selected_legacy_bindings: list[tuple[str, str]] = []
    selected: list[dict[str, Any]] = []
    while len(selected) < sample_size:
        progress = False
        for family in family_order:
            cap = overrides.get(family, default_family_cap)
            cursor = cursors[family]
            values = buckets[family]
            if selected_counts[family] >= cap or cursor >= len(values):
                continue
            row = dict(values[cursor][1])
            legacy_split = values[cursor][2]
            cursors[family] = cursor + 1
            selected_counts[family] += 1
            row["r7_expansion_selection"] = {
                "schema_version": R7_EXPANSION_SCHEMA,
                "seed": seed,
                "primary_family": family,
                "within_family_rank": selected_counts[family],
                "prior_iid_excluded": True,
                "split_assigned": False,
                "legacy_split_quarantine": {
                    "present": legacy_split is not None,
                    "canonical_sha256": (
                        None
                        if legacy_split is None
                        else str(legacy_split["canonical_sha256"])
                    ),
                },
            }
            selected.append(row)
            if legacy_split is not None:
                selected_iid = _iid(row)
                selected_legacy_split_counts[
                    str(legacy_split["split"])
                ] += 1
                selected_legacy_version_counts[
                    str(legacy_split["provenance_version"])
                ] += 1
                selected_legacy_seed_counts[
                    int(legacy_split["provenance_seed"])
                ] += 1
                selected_legacy_provenance_digest_counts[
                    str(legacy_split["provenance_sha256"])
                ] += 1
                selected_legacy_pair_digest_counts[
                    str(legacy_split["canonical_sha256"])
                ] += 1
                selected_legacy_bindings.append(
                    (selected_iid, str(legacy_split["canonical_sha256"]))
                )
            progress = True
            if len(selected) >= sample_size:
                break
        if not progress:
            break
    if len(selected) < sample_size:
        capacity = sum(
            min(len(values), overrides.get(family, default_family_cap))
            for family, values in buckets.items()
        )
        raise ValueError(
            f"only {len(selected)} eligible rows for requested {sample_size}; "
            f"family-capped capacity={capacity}"
        )
    selected_iids = [_iid(row) for row in selected]
    if len(set(selected_iids)) != len(selected_iids):
        raise AssertionError("selection contains duplicate IIDs")
    if set(selected_iids) & excluded_iids:
        raise AssertionError("selection contains a prior IID")
    if any(
        LEGACY_SPLIT_FIELDS & set(row)
        for row in selected
    ):
        raise AssertionError("selection contains quarantined legacy split fields")
    source_legacy_rows = sum(source_legacy_split_counts.values())
    selected_legacy_rows = sum(selected_legacy_split_counts.values())
    audit = {
        "scanned_rows": scanned,
        "eligible_families": len(buckets),
        "eligible_before_caps": sum(len(values) for values in buckets.values()),
        "selected_rows": len(selected),
        "selected_family_counts": dict(sorted(selected_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_iid_digest": hashlib.sha256(
            "\n".join(selected_iids).encode("utf-8")
        ).hexdigest(),
        "legacy_split_quarantine": {
            "source_rows_with_pair": source_legacy_rows,
            "source_rows_without_pair": scanned - source_legacy_rows,
            "removed_row_count": selected_legacy_rows,
            "removed_top_level_field_count": 2 * selected_legacy_rows,
            "split_value_counts": dict(
                sorted(source_legacy_split_counts.items())
            ),
            "provenance_version_counts": dict(
                sorted(source_legacy_version_counts.items())
            ),
            "provenance_seed_counts": {
                str(key): value
                for key, value in sorted(source_legacy_seed_counts.items())
            },
            "provenance_sha256_counts": dict(
                sorted(source_legacy_provenance_digest_counts.items())
            ),
            "canonical_pair_sha256_counts": dict(
                sorted(source_legacy_pair_digest_counts.items())
            ),
            "source_iid_pair_binding_sha256": _binding_digest(
                source_legacy_bindings
            ),
            "selected_split_value_counts": dict(
                sorted(selected_legacy_split_counts.items())
            ),
            "selected_provenance_version_counts": dict(
                sorted(selected_legacy_version_counts.items())
            ),
            "selected_provenance_seed_counts": {
                str(key): value
                for key, value in sorted(selected_legacy_seed_counts.items())
            },
            "selected_provenance_sha256_counts": dict(
                sorted(selected_legacy_provenance_digest_counts.items())
            ),
            "selected_canonical_pair_sha256_counts": dict(
                sorted(selected_legacy_pair_digest_counts.items())
            ),
            "selected_iid_pair_binding_sha256": _binding_digest(
                selected_legacy_bindings
            ),
            "output_rows_have_top_level_split_fields": False,
        },
    }
    return selected, audit


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select an unseen family-balanced R7 visual-review queue."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=16_000)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--allow-missing-actor", action="store_true")
    parser.add_argument("--default-family-cap", type=int, default=DEFAULT_FAMILY_CAP)
    parser.add_argument(
        "--family-cap",
        action="append",
        default=[],
        metavar="FAMILY=COUNT",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve(strict=True)
    output_path = args.output.expanduser()
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("output or summary already exists; use a fresh run directory")
    excluded, exclusion_provenance = load_excluded_iids(args.exclude)
    caps = _parse_family_caps(args.family_cap)
    rows, audit = select_expansion_rows(
        _iter_jsonl(input_path),
        excluded_iids=excluded,
        sample_size=int(args.sample_size),
        labels={str(value) for value in args.labels},
        min_score=float(args.min_score),
        require_actor=not bool(args.allow_missing_actor),
        default_family_cap=int(args.default_family_cap),
        family_caps=caps,
        seed=int(args.seed),
    )
    _atomic_write_jsonl(output_path, rows)
    summary = {
        "schema_version": R7_EXPANSION_SCHEMA,
        "stage": "selection",
        "status": "complete",
        "input": str(input_path),
        "input_sha256": _file_digest(input_path),
        "exclusions": exclusion_provenance,
        "excluded_unique_iids": len(excluded),
        "output": str(output_path.resolve()),
        "output_sha256": _file_digest(output_path),
        "config": {
            "sample_size": int(args.sample_size),
            "labels": sorted(str(value) for value in args.labels),
            "min_score": float(args.min_score),
            "require_actor": not bool(args.allow_missing_actor),
            "default_family_cap": int(args.default_family_cap),
            "family_caps": dict(sorted(caps.items())),
            "seed": int(args.seed),
        },
        "audit": audit,
        "split_assigned": False,
        "human_labels_asserted": False,
        "production_eligible": False,
    }
    _atomic_write_json(summary_path, summary)
    print(
        f"[r7-select] rows={len(rows)} families="
        f"{audit['eligible_families']} output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
