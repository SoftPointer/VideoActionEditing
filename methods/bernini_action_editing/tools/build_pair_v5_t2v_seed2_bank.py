#!/usr/bin/env python3
"""Derive the preregistered second-seed PAIR-v5 T2V bank.

The source core4-v2 bank already closes four actor/scene cells, two action
families, fit/confirmation isolation, and ten semantic branches per cell.  A
second independent official-Gaussian replicate must change only:

* the native sampler seed;
* the seed-bearing calibration-group identifier; and
* the globally unique candidate identifier.

Captions, geometry probes, split axes, action families, branch ordering, and
all artifact-use prohibitions remain byte-for-byte semantic copies.  The
result is still calibration/critic evidence only.  It never authorizes using
generated media or latents as an editor target, donor, condition, or noise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_calibration_bank_spec as contract  # noqa: E402


@dataclass(frozen=True)
class ReplicationProfile:
    source_spec_sha256: str
    seed_map: Mapping[int, int]
    source_id_prefix: str
    topup_id_prefix: str


REPLICATION_PROFILES = {
    "core4-v2": ReplicationProfile(
        source_spec_sha256="a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95",
        seed_map={
            2026080825: 2026080925,
            2026080826: 2026080926,
            2026080827: 2026080927,
            2026080828: 2026080928,
        },
        source_id_prefix="pair5-t2v-core4-v2-",
        topup_id_prefix="pair5-t2v-core4-seed2-",
    ),
    "reserve4-v1": ReplicationProfile(
        source_spec_sha256="2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab",
        seed_map={
            2026080821: 2026080921,
            2026080822: 2026080922,
            2026080823: 2026080923,
            2026080824: 2026080924,
        },
        source_id_prefix="pair5-t2v-reserve4-v1-",
        topup_id_prefix="pair5-t2v-reserve4-seed2-",
    ),
}
DEFAULT_PROFILE = "core4-v2"
# Backward-compatible aliases for the already launched core4 replicate.
SOURCE_SPEC_SHA256 = REPLICATION_PROFILES[DEFAULT_PROFILE].source_spec_sha256
SEED_MAP = REPLICATION_PROFILES[DEFAULT_PROFILE].seed_map
SOURCE_ID_PREFIX = REPLICATION_PROFILES[DEFAULT_PROFILE].source_id_prefix
TOPUP_ID_PREFIX = REPLICATION_PROFILES[DEFAULT_PROFILE].topup_id_prefix
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PairV5T2VSeed2Error(RuntimeError):
    """Raised before an ambiguous or post-hoc top-up can be materialized."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(name: str) -> ReplicationProfile:
    try:
        return REPLICATION_PROFILES[name]
    except KeyError as error:
        raise PairV5T2VSeed2Error("replication profile is not preregistered") from error


def _load_source(
    path: Path, expected_sha256: str, profile_name: str = DEFAULT_PROFILE
) -> Mapping[str, Any]:
    profile = _profile(profile_name)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise PairV5T2VSeed2Error("source spec path or SHA-256 differs")
    if (
        expected_sha256 != profile.source_spec_sha256
        or file_sha256(path) != expected_sha256
    ):
        raise PairV5T2VSeed2Error("source is not the preregistered bank authority")
    try:
        raw = json.loads(path.read_bytes())
        normalized = contract.validate_root_spec(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, contract.PairT2VCalibrationSpecError) as error:
        raise PairV5T2VSeed2Error("source core4-v2 spec failed validation") from error
    return normalized


def _replace_seed_suffix(value: str, old_seed: int, new_seed: int) -> str:
    suffix = f"-s{old_seed}"
    if not value.endswith(suffix) or value.count(suffix) != 1:
        raise PairV5T2VSeed2Error("calibration group does not bind the source seed once")
    return value[: -len(suffix)] + f"-s{new_seed}"


def derive_seed2_spec(
    source: Mapping[str, Any], profile_name: str = DEFAULT_PROFILE
) -> dict[str, Any]:
    """Return the deterministic second-seed bank and prove the change surface."""

    profile = _profile(profile_name)
    try:
        normalized = contract.validate_root_spec(source)
    except contract.PairT2VCalibrationSpecError as error:
        raise PairV5T2VSeed2Error("source core4-v2 topology differs") from error
    result = json.loads(json.dumps(normalized, ensure_ascii=False))
    observed_old_seeds: set[int] = set()
    observed_new_seeds: set[int] = set()
    ids: set[str] = set()
    for group in result["groups"]:
        for candidate in group["candidates"]:
            old_seed = candidate["seed"]
            if old_seed not in profile.seed_map:
                raise PairV5T2VSeed2Error("source contains an unregistered seed")
            new_seed = profile.seed_map[old_seed]
            observed_old_seeds.add(old_seed)
            observed_new_seeds.add(new_seed)
            candidate_id = candidate["candidate_id"]
            if not candidate_id.startswith(profile.source_id_prefix):
                raise PairV5T2VSeed2Error("source candidate prefix differs")
            new_id = profile.topup_id_prefix + candidate_id[len(profile.source_id_prefix) :]
            if new_id in ids:
                raise PairV5T2VSeed2Error("derived candidate identifiers alias")
            ids.add(new_id)
            candidate["candidate_id"] = new_id
            candidate["calibration_group_id"] = _replace_seed_suffix(
                candidate["calibration_group_id"], old_seed, new_seed
            )
            candidate["seed"] = new_seed
    if (
        observed_old_seeds != set(profile.seed_map)
        or observed_new_seeds != set(profile.seed_map.values())
    ):
        raise PairV5T2VSeed2Error("seed population is incomplete")
    validated = contract.validate_root_spec(result)
    _prove_only_registered_fields_changed(normalized, validated, profile=profile)
    return validated


def _prove_only_registered_fields_changed(
    source: Mapping[str, Any],
    derived: Mapping[str, Any],
    *,
    profile: ReplicationProfile,
) -> None:
    for key in (
        "schema_version",
        "sampling_contract",
        "semantic_input_closure",
        "artifact_use_contract",
        "split_contract",
    ):
        if source[key] != derived[key]:
            raise PairV5T2VSeed2Error(f"root field changed outside seed top-up: {key}")
    source_groups = source["groups"]
    derived_groups = derived["groups"]
    if len(source_groups) != len(derived_groups):
        raise PairV5T2VSeed2Error("group count changed")
    mutable = {"candidate_id", "calibration_group_id", "seed"}
    for old_group, new_group in zip(source_groups, derived_groups):
        if (
            old_group["group_id"] != new_group["group_id"]
            or old_group["visible_gpus"] != new_group["visible_gpus"]
            or len(old_group["candidates"]) != len(new_group["candidates"])
        ):
            raise PairV5T2VSeed2Error("SP4 topology changed")
        for old, new in zip(old_group["candidates"], new_group["candidates"]):
            for key in set(old) | set(new):
                if key not in mutable and old.get(key) != new.get(key):
                    raise PairV5T2VSeed2Error(
                        f"candidate field changed outside seed top-up: {key}"
                    )
            if (
                new["seed"] != profile.seed_map[old["seed"]]
                or new["candidate_id"]
                != profile.topup_id_prefix
                + old["candidate_id"][len(profile.source_id_prefix) :]
                or new["calibration_group_id"]
                != _replace_seed_suffix(
                    old["calibration_group_id"], old["seed"], new["seed"]
                )
            ):
                raise PairV5T2VSeed2Error("registered seed rewrite differs")


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        raise PairV5T2VSeed2Error("output must be a fresh absolute plain-file path")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise PairV5T2VSeed2Error("output parent must be an existing plain directory")
    payload = contract.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    observed = hashlib.sha256(payload).hexdigest()
    if file_sha256(path) != observed:
        raise PairV5T2VSeed2Error("published seed2 spec failed byte replay")
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(REPLICATION_PROFILES),
        default=DEFAULT_PROFILE,
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    profile = _profile(args.profile)
    source = _load_source(
        Path(args.source_spec), args.expected_source_sha256, args.profile
    )
    derived = derive_seed2_spec(source, args.profile)
    output = Path(args.output)
    digest = _write_create_only(output, derived)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "source_spec_sha256": profile.source_spec_sha256,
                "seed_map": {
                    str(key): value for key, value in profile.seed_map.items()
                },
                "candidate_count": sum(
                    len(group["candidates"]) for group in derived["groups"]
                ),
                "output": str(output),
                "output_sha256": digest,
                "artifact_use": "critic_calibration_only_never_editor_input",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PairV5T2VSeed2Error",
    "ReplicationProfile",
    "REPLICATION_PROFILES",
    "DEFAULT_PROFILE",
    "SEED_MAP",
    "SOURCE_SPEC_SHA256",
    "derive_seed2_spec",
    "file_sha256",
    "main",
]
