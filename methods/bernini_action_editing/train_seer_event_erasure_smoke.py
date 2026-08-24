#!/usr/bin/env python3
"""Four-step Bernini SEER-FM B0 smoke with a verifiable parameter update.

This is deliberately a thin specialization of :mod:`train_lora`.  It changes
neither Bernini's renderer nor its flow-matching target.  It narrows the LoRA
to cross-attention Q/O projections and binds the two-row training dataset to
the independently audited self-generated owner videos.  The smoke answers
only whether the new supervision can update parameters; method success still
requires the separate held-out decoded evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_lora as core  # noqa: E402


SCHEMA = "bernini-seer-owner-core2-v1"
METHOD = "self-generated-event-erasure-flow-matching-b0"
EXPECTED_ROWS = 2
EXPECTED_TARGET_MODULES = 60
EXPECTED_INCLUSION_POLICY = "strict_single_actor"
_CROSS_Q_OUT = re.compile(r".+\.blocks\.\d+\.attn2\.(?:to_q|to_out\.0)$")


class SeerSmokeError(RuntimeError):
    """Raised before a misleading optimizer update."""


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SeerSmokeError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SeerSmokeError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise SeerSmokeError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_owner_spec(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise SeerSmokeError("owner spec SHA-256 is invalid")
    if _sha256(path) != expected_sha256:
        raise SeerSmokeError("owner spec SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SeerSmokeError(f"cannot read owner spec: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise SeerSmokeError("owner spec schema differs")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise SeerSmokeError("owner spec must contain exactly two rows")
    if {row.get("actor_family") for row in rows if isinstance(row, Mapping)} != {
        "dog",
        "human",
    }:
        raise SeerSmokeError("owner spec must contain one dog and one human")
    event = value.get("event_erasure_contract")
    authority = value.get("fresh_experiment_authority")
    if (
        not isinstance(event, Mapping)
        or event.get("frame_count") != 81
        or event.get("fps") != 25
        or event.get("erasure_cutoff_exclusive") != 32
        or event.get("source_must_not_contain_transition_or_terminal_frames") is not True
        or not isinstance(authority, Mapping)
        or authority.get("user_directed_parameter_update") is not True
        or authority.get("production_claim_authorized") is not False
        or authority.get("method_success_claim_authorized_by_training_completion") is not False
        or authority.get("heldout_decoded_review_required") is not True
    ):
        raise SeerSmokeError("owner event-erasure or authority contract differs")
    for row in rows:
        if not isinstance(row, Mapping):
            raise SeerSmokeError("owner row must be an object")
        instruction = row.get("instruction")
        if (
            not isinstance(instruction, str)
            or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            != row.get("instruction_utf8_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("target_video_sha256")))
        ):
            raise SeerSmokeError("owner instruction or target hash differs")
    return value


def _dataset_rows(directory: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SeerSmokeError("pyarrow is required to validate the SEER dataset") from error
    files = sorted(directory.glob("*.parquet"))
    if len(files) != EXPECTED_ROWS or any(path.is_symlink() for path in files):
        raise SeerSmokeError("SEER dataset must contain exactly two plain shards")
    rows: list[dict[str, Any]] = []
    for path in files:
        values = pq.read_table(path).to_pylist()
        if len(values) != 1 or not isinstance(values[0], dict):
            raise SeerSmokeError(f"SEER shard must contain one row: {path}")
        rows.append(values[0])
    return sorted(rows, key=lambda row: str(row.get("iid")))


def _cross_bind_dataset(owner: Mapping[str, Any], directory: Path) -> None:
    expected = {str(row["iid"]): row for row in owner["rows"]}
    rows = _dataset_rows(directory)
    if {str(row.get("iid")) for row in rows} != set(expected):
        raise SeerSmokeError("materialized SEER IID set differs from owner spec")
    for row in rows:
        wanted = expected[str(row["iid"])]
        if row.get("target_video_sha256") != wanted.get("target_video_sha256"):
            raise SeerSmokeError("materialized target is not the audited owner target")
        if row.get("source_video_sha256") == row.get("target_video_sha256"):
            raise SeerSmokeError("event-erasure source unexpectedly equals its target")
        try:
            messages = json.loads(str(row.get("inputs")))
        except json.JSONDecodeError as error:
            raise SeerSmokeError("renderer messages are invalid") from error
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or messages[1].get("text") != wanted.get("instruction")
        ):
            raise SeerSmokeError("renderer instruction differs from owner spec")


def _install_specialization(owner: Mapping[str, Any], owner_path: Path) -> None:
    core.EXPECTED_DATASET_ROWS = EXPECTED_ROWS
    core.EXPECTED_STRICT_ROWS = EXPECTED_ROWS
    core.EXPECTED_NON_STRICT_ROWS = 0
    core.EXPECTED_INCLUSION_POLICY = EXPECTED_INCLUSION_POLICY
    core.EXPECTED_LORA_TARGET_MODULES = EXPECTED_TARGET_MODULES
    original_select = core.select_attention_projection_names
    original_receipt = core.build_receipt
    original_save = core.save_training_checkpoint

    def select_cross_q_out(model: Any) -> list[str]:
        selected = [name for name in original_select(model) if _CROSS_Q_OUT.fullmatch(name)]
        if len(selected) != EXPECTED_TARGET_MODULES:
            raise SeerSmokeError(
                f"cross-attention Q/O scope has {len(selected)} modules, expected 60"
            )
        return selected

    def build_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        receipt = original_receipt(*args, **kwargs)
        receipt["method"] = METHOD
        receipt["training_contract"]["lora_scope"] = (
            "all_30_blocks_attn2_cross_attention_q_and_out_rank8"
        )
        receipt["seer"] = {
            "owner_spec_path": str(owner_path),
            "owner_spec_sha256": _sha256(owner_path),
            "owner_job": owner["source_owner_job"],
            "event_erasure_cutoff_exclusive": 32,
            "self_generated_target_supervision": True,
            "training_completion_is_method_success": False,
            "heldout_decoded_review_required": True,
        }
        receipt["receipt_digest"] = core.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        return receipt

    def save_checkpoint(*args: Any, **kwargs: Any) -> Path:
        model = kwargs["model"]
        receipt = dict(kwargs["receipt"])
        named = core.trainable_lora_parameters(model)
        final_digest = core.trainable_parameters_digest(named)
        initial_digest = receipt["distributed"]["lora_initialization_digest"]
        if final_digest == initial_digest:
            raise SeerSmokeError("optimizer completed but LoRA parameters did not change")
        receipt["parameter_update_evidence"] = {
            "initial_trainable_parameter_digest": initial_digest,
            "final_trainable_parameter_digest": final_digest,
            "exact_parameter_bytes_changed": True,
            "method_success_claimed": False,
        }
        receipt["receipt_digest"] = core.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        kwargs["receipt"] = receipt
        return original_save(*args, **kwargs)

    core.select_attention_projection_names = select_cross_q_out
    core.build_receipt = build_receipt
    core.save_training_checkpoint = save_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--seer-owner-spec", required=True)
    wrapper.add_argument("--expected-seer-owner-spec-sha256", required=True)
    known, remaining = wrapper.parse_known_args(argv)
    owner_path = _plain_file(known.seer_owner_spec, label="seer_owner_spec")
    owner = _load_owner_spec(owner_path, known.expected_seer_owner_spec_sha256)
    core_args = core.build_parser().parse_args(remaining)
    if core_args.max_steps > 4:
        raise SeerSmokeError("B0 smoke is limited to at most four optimizer steps")
    if float(core_args.learning_rate) != 1.0e-6:
        raise SeerSmokeError("B0 smoke fixes learning_rate to 1e-6")
    dataset_dir = Path(core_args.preprocessed_parquet_dir).expanduser().resolve(strict=True)
    _cross_bind_dataset(owner, dataset_dir)
    _install_specialization(owner, owner_path)
    return core.main(remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SeerSmokeError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
