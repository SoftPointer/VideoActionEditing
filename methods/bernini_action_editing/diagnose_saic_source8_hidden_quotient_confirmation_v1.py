#!/usr/bin/env python3
"""Evaluate the single preregistered source8 hidden-quotient candidate.

The representation and model coordinate were fixed before source8 hidden
materialization: block-15 action-minus-noop residual at native schedule index
33, temporal centering, per-spatial-sketch temporal cosine self-similarity, and
averaging over the fixed 16 spatial sketches.  No representation, rank, block,
sigma, threshold, or seed sweep is implemented here.

For each action family, the two registered fit sources define one common
forward direction from their forward-vs-noop and forward-vs-reverse contrasts.
The two registered confirmation sources contribute four held-out margins per
family and never enter a direction.  No-op and reverse are separately treated
as pseudo-positive controls.  The candidate replication gate requires all
eight forward margins to be positive and requires neither pseudo-positive
control to pass all eight comparisons.  The gate does not authorize training,
an optimizer, an editor update, or a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_starc_core4_hidden_temporal_quotient_v1 as prior_diagnostic  # noqa: E402
import materialize_saic_source8_hidden_quotient_v1 as materializer  # noqa: E402


SCHEMA_VERSION = "bernini-saic-source8-hidden-quotient-confirmation-v1"
REPRESENTATION_NAME = "centered_sketch_self_similarity"
EXPECTED_SHAPE = (1, 21, 16, 1536)
BRANCH_ORDER = materializer.BRANCH_ORDER
NEGATIVES_BY_POSITIVE = {
    branch: tuple(other for other in BRANCH_ORDER if other != branch)
    for branch in BRANCH_ORDER
}


class Source8HiddenConfirmationError(RuntimeError):
    """A sealed input or fit/confirmation invariant failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Source8HiddenConfirmationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise Source8HiddenConfirmationError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise Source8HiddenConfirmationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _json_file(path: Path, *, expected_sha256: str | None, label: str) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Source8HiddenConfirmationError(f"{label} must be absolute plain file")
    observed = file_sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise Source8HiddenConfirmationError(f"{label} SHA-256 differs")

    def reject_constant(token: str) -> Any:
        raise Source8HiddenConfirmationError(f"{label} contains {token}")

    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Source8HiddenConfirmationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Source8HiddenConfirmationError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise Source8HiddenConfirmationError(f"{label} root differs")
    return value


def _verify_seal(value: Mapping[str, Any], *, schema: str, label: str) -> None:
    row = dict(value)
    declared = row.pop("receipt_digest", None)
    if row.get("schema_version") != schema or declared != object_sha256(row):
        raise Source8HiddenConfirmationError(f"{label} schema/digest differs")


def _unit(value: Any) -> Any:
    import torch

    flat = value.float().reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1.0e-12:
        raise Source8HiddenConfirmationError("zero/non-finite vector")
    return flat / norm


def preregistered_representation(value: Any) -> Any:
    """Use exactly the sole winner from the opened core4 representation audit."""

    representations = prior_diagnostic.temporal_representations(value)
    if REPRESENTATION_NAME not in representations:
        raise Source8HiddenConfirmationError("preregistered representation is absent")
    result = representations[REPRESENTATION_NAME]
    if tuple(int(item) for item in result.shape) != (210,):
        raise Source8HiddenConfirmationError("representation geometry differs")
    return result


def _load_population(master_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch
    from safetensors import safe_open

    master = _json_file(master_path, expected_sha256=None, label="source8 master")
    _verify_seal(
        master,
        schema=materializer.MASTER_SCHEMA_VERSION,
        label="source8 master",
    )
    if (
        master.get("actor_family_order") != list(materializer.ACTOR_FAMILIES)
        or master.get("branch_order") != list(BRANCH_ORDER)
        or master.get("selection_rule") != "minimum_registered_seed_per_source"
        or master.get("source_count") != 8
        or master.get("arm_count") != 24
        or master.get("model_forward_count") != 48
        or master.get("fit_source_count") != 4
        or master.get("confirmation_source_count") != 4
        or master.get("all_60_generation_receipts_authenticated_before_selection")
        is not True
        or master.get("candidate_rgb_opened") is not False
        or master.get("confirmation_used_for_structure_or_parameter_selection")
        is not False
        or master.get("training_performed") is not False
        or master.get("optimizer_authorized") is not False
        or master.get("editor_optimizer_authorized") is not False
        or master.get("representation_selection_authorized") is not False
        or master.get("scientific_claim_authorized") is not False
        or master.get("single_preregistered_coordinate")
        != {
            "hook": "block.15.output",
            "schedule_index": 33,
            "sigma": 0.5161304473876953,
            "native_timestep": 516,
        }
    ):
        raise Source8HiddenConfirmationError("master topology/authority differs")

    sources: dict[str, dict[str, Any]] = {}
    group_bindings = master.get("group_bindings")
    if not isinstance(group_bindings, list) or len(group_bindings) != 2:
        raise Source8HiddenConfirmationError("master group count differs")
    for expected_family, binding in zip(materializer.ACTOR_FAMILIES, group_bindings):
        if binding.get("actor_family") != expected_family:
            raise Source8HiddenConfirmationError("master group family order differs")
        group_path = Path(binding["path"])
        group = _json_file(
            group_path,
            expected_sha256=binding["file_sha256"],
            label=f"{expected_family} group",
        )
        _verify_seal(
            group,
            schema=materializer.GROUP_SCHEMA_VERSION,
            label=f"{expected_family} group",
        )
        if (
            group.get("receipt_digest") != binding.get("receipt_digest")
            or group.get("actor_family") != expected_family
            or group.get("source_order") != binding.get("source_order")
            or group.get("split_by_iid") != binding.get("split_by_iid")
            or group.get("arm_count") != binding.get("arm_count")
            or group.get("model_forward_count") != binding.get("model_forward_count")
            or group.get("candidate_rgb_opened") is not False
            or group.get("confirmation_used_for_structure_or_parameter_selection")
            is not False
            or group.get("training_performed") is not False
            or group.get("optimizer_authorized") is not False
        ):
            raise Source8HiddenConfirmationError("master-to-group binding differs")
        arm_bindings = group.get("arm_bindings")
        if not isinstance(arm_bindings, list) or len(arm_bindings) != 12:
            raise Source8HiddenConfirmationError("group arm population differs")
        expected_pairs = [
            (iid, branch) for iid in group["source_order"] for branch in BRANCH_ORDER
        ]
        if [(row.get("iid"), row.get("branch")) for row in arm_bindings] != expected_pairs:
            raise Source8HiddenConfirmationError("group arm order differs")
        for arm_binding in arm_bindings:
            receipt = _json_file(
                Path(arm_binding["receipt_path"]),
                expected_sha256=arm_binding["receipt_file_sha256"],
                label="source8 arm",
            )
            _verify_seal(
                receipt,
                schema=materializer.SCHEMA_VERSION,
                label="source8 arm",
            )
            if (
                receipt.get("receipt_digest") != arm_binding.get("receipt_digest")
                or receipt.get("iid") != arm_binding.get("iid")
                or receipt.get("analysis_split") != arm_binding.get("analysis_split")
                or receipt.get("branch") != arm_binding.get("branch")
                or receipt.get("candidate_binding", {}).get("candidate_id")
                != arm_binding.get("candidate_id")
                or receipt.get("minimum_registered_seed") is not True
                or receipt.get("candidate_rgb_opened") is not False
                or receipt.get("training_performed") is not False
                or receipt.get("optimizer_authorized") is not False
                or receipt.get("editor_optimizer_authorized") is not False
                or receipt.get("representation_selection_authorized") is not False
                or receipt.get("scientific_claim_authorized") is not False
            ):
                raise Source8HiddenConfirmationError("group-to-arm binding differs")
            artifact = receipt.get("artifact")
            if (
                not isinstance(artifact, Mapping)
                or artifact.get("path") != arm_binding.get("artifact_path")
                or artifact.get("file_sha256")
                != arm_binding.get("artifact_file_sha256")
                or artifact.get("tensor_sha256")
                != arm_binding.get("artifact_tensor_sha256")
                or artifact.get("tensor_key") != materializer.starc.TENSOR_KEY
                or artifact.get("tensor_shape") != list(EXPECTED_SHAPE)
                or artifact.get("tensor_dtype") != "torch.float32"
                or artifact.get("detached_finite_fp32") is not True
            ):
                raise Source8HiddenConfirmationError("arm artifact binding differs")
            artifact_path = Path(artifact["path"])
            if file_sha256(artifact_path) != artifact["file_sha256"]:
                raise Source8HiddenConfirmationError("artifact file SHA-256 differs")
            with safe_open(artifact_path, framework="pt", device="cpu") as opened:
                if list(opened.keys()) != [materializer.starc.TENSOR_KEY]:
                    raise Source8HiddenConfirmationError("artifact tensor key differs")
                tensor = opened.get_tensor(materializer.starc.TENSOR_KEY).contiguous()
            if (
                tuple(int(item) for item in tensor.shape) != EXPECTED_SHAPE
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not bool(torch.isfinite(tensor).all().item())
                or prior_diagnostic.tensor_sha256(tensor) != artifact["tensor_sha256"]
            ):
                raise Source8HiddenConfirmationError("artifact tensor values differ")
            iid = receipt["iid"]
            source = sources.setdefault(
                iid,
                {
                    "iid": iid,
                    "actor_family": receipt["actor_family"],
                    "analysis_split": receipt["analysis_split"],
                    "action_family_id": receipt["action_family_id"],
                    "actor_group_id": receipt["actor_group_id"],
                    "scene_group_id": receipt["scene_group_id"],
                    "seed": receipt["seed"],
                    "branches": {},
                },
            )
            for name in (
                "actor_family",
                "analysis_split",
                "action_family_id",
                "actor_group_id",
                "scene_group_id",
                "seed",
            ):
                if source[name] != receipt[name]:
                    raise Source8HiddenConfirmationError(
                        "source metadata differs across branches"
                    )
            branch = receipt["branch"]
            if branch in source["branches"]:
                raise Source8HiddenConfirmationError("source branch repeats")
            source["branches"][branch] = {
                "tensor": tensor,
                "artifact_file_sha256": artifact["file_sha256"],
                "artifact_tensor_sha256": artifact["tensor_sha256"],
                "arm_receipt_digest": receipt["receipt_digest"],
                "candidate_id": arm_binding["candidate_id"],
            }
    if (
        len(sources) != 8
        or list(sources) != master["source_order"]
        or any(tuple(source["branches"]) != BRANCH_ORDER for source in sources.values())
        or sum(source["analysis_split"] == "fit" for source in sources.values()) != 4
        or sum(source["analysis_split"] == "confirmation" for source in sources.values())
        != 4
    ):
        raise Source8HiddenConfirmationError("loaded source population differs")
    return master, sources


def fit_direction(
    fit_sources: Sequence[Mapping[str, Any]], *, positive_branch: str
) -> tuple[Any, list[dict[str, Any]]]:
    """Fit one common direction from exactly two sources and two contrasts each."""

    import torch

    negatives = NEGATIVES_BY_POSITIVE.get(positive_branch)
    if negatives is None or len(fit_sources) != 2:
        raise Source8HiddenConfirmationError("fit direction topology differs")
    contrasts = []
    rows = []
    for source in fit_sources:
        positive = _unit(source["features"][positive_branch])
        for negative_branch in negatives:
            contrast = _unit(positive - _unit(source["features"][negative_branch]))
            contrasts.append(contrast)
            rows.append(
                {
                    "iid": source["iid"],
                    "positive_branch": positive_branch,
                    "negative_branch": negative_branch,
                }
            )
    direction = _unit(torch.stack(contrasts).mean(dim=0))
    fit_margins = [float(torch.dot(direction, contrast).item()) for contrast in contrasts]
    if sum(fit_margins) < 0.0:
        direction = -direction
        fit_margins = [-value for value in fit_margins]
    for row, margin in zip(rows, fit_margins):
        row["margin"] = margin
        row["positive"] = margin > 0.0
    return direction, rows


def evaluate_positive_branch(
    *,
    fit_sources: Sequence[Mapping[str, Any]],
    confirmation_sources: Sequence[Mapping[str, Any]],
    positive_branch: str,
) -> dict[str, Any]:
    import torch

    if len(confirmation_sources) != 2:
        raise Source8HiddenConfirmationError("confirmation source count differs")
    direction, fit_rows = fit_direction(
        fit_sources, positive_branch=positive_branch
    )
    rows = []
    for source in confirmation_sources:
        positive = _unit(source["features"][positive_branch])
        for negative_branch in NEGATIVES_BY_POSITIVE[positive_branch]:
            negative = _unit(source["features"][negative_branch])
            margin = float(torch.dot(direction, positive - negative).item())
            if not math.isfinite(margin):
                raise Source8HiddenConfirmationError("confirmation margin is non-finite")
            rows.append(
                {
                    "iid": source["iid"],
                    "positive_branch": positive_branch,
                    "negative_branch": negative_branch,
                    "margin": margin,
                    "positive": margin > 0.0,
                }
            )
    margins = [row["margin"] for row in rows]
    return {
        "positive_branch": positive_branch,
        "fit_rows": fit_rows,
        "fit_positive_count": sum(row["positive"] for row in fit_rows),
        "fit_count": len(fit_rows),
        "fit_all_positive": all(row["positive"] for row in fit_rows),
        "fit_minimum_margin": min(row["margin"] for row in fit_rows),
        "confirmation_rows": rows,
        "confirmation_positive_count": sum(row["positive"] for row in rows),
        "confirmation_count": len(rows),
        "confirmation_all_positive": all(row["positive"] for row in rows),
        "confirmation_mean_margin": sum(margins) / len(margins),
        "confirmation_minimum_margin": min(margins),
    }


def run_diagnostic(master_path: Path) -> dict[str, Any]:
    import safetensors
    import torch

    master, sources = _load_population(master_path)
    prepared = []
    for source in sources.values():
        prepared.append(
            {
                **{key: value for key, value in source.items() if key != "branches"},
                "features": {
                    branch: preregistered_representation(
                        source["branches"][branch]["tensor"]
                    )
                    for branch in BRANCH_ORDER
                },
                "artifact_bindings": {
                    branch: {
                        key: value
                        for key, value in source["branches"][branch].items()
                        if key != "tensor"
                    }
                    for branch in BRANCH_ORDER
                },
            }
        )

    family_results = {}
    aggregate_rows = {branch: [] for branch in BRANCH_ORDER}
    for actor_family in materializer.ACTOR_FAMILIES:
        family_sources = [
            source for source in prepared if source["actor_family"] == actor_family
        ]
        fit_sources = [
            source for source in family_sources if source["analysis_split"] == "fit"
        ]
        confirmation_sources = [
            source
            for source in family_sources
            if source["analysis_split"] == "confirmation"
        ]
        if len(fit_sources) != 2 or len(confirmation_sources) != 2:
            raise Source8HiddenConfirmationError("family split topology differs")
        by_positive = {
            branch: evaluate_positive_branch(
                fit_sources=fit_sources,
                confirmation_sources=confirmation_sources,
                positive_branch=branch,
            )
            for branch in BRANCH_ORDER
        }
        for branch in BRANCH_ORDER:
            aggregate_rows[branch].extend(by_positive[branch]["confirmation_rows"])
        family_results[actor_family] = {
            "action_family_id": fit_sources[0]["action_family_id"],
            "fit_iids": [source["iid"] for source in fit_sources],
            "confirmation_iids": [source["iid"] for source in confirmation_sources],
            "by_positive_branch": by_positive,
        }

    aggregate = {}
    for branch in BRANCH_ORDER:
        rows = aggregate_rows[branch]
        margins = [row["margin"] for row in rows]
        aggregate[branch] = {
            "positive_count": sum(row["positive"] for row in rows),
            "count": len(rows),
            "all_positive": all(row["positive"] for row in rows),
            "mean_margin": sum(margins) / len(margins),
            "minimum_margin": min(margins),
            "rows": rows,
        }
    forward_gate = aggregate["forward"]["all_positive"]
    noop_control_passed = aggregate["noop"]["all_positive"]
    reverse_control_passed = aggregate["reverse"]["all_positive"]
    specificity_gate = (
        forward_gate and not noop_control_passed and not reverse_control_passed
    )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "source8_independent_population_hidden_quotient_confirmation_no_authority",
        "master_binding": {
            "path": str(master_path),
            "file_sha256": file_sha256(master_path),
            "receipt_digest": master["receipt_digest"],
        },
        "preregistration": {
            "prior_exploratory_result_sha256": (
                "12dc4b194082f73711e1282a1fc116ce581d2c7193b76afa5faa3b4f943585b8"
            ),
            "representation_name": REPRESENTATION_NAME,
            "representation_count_evaluated": 1,
            "rank_sweep_performed": False,
            "block_sweep_performed": False,
            "sigma_sweep_performed": False,
            "seed_selection_performed": False,
            "fixed_coordinate": master["single_preregistered_coordinate"],
            "direction_rule": (
                "per-family unit mean of four fit-only unit forward-vs-"
                "noop/reverse contrasts"
            ),
            "forward_gate_rule": "all eight held-out forward margins strictly positive",
            "specificity_gate_rule": (
                "forward gate passes and neither noop nor reverse pseudo-positive "
                "control passes all eight held-out comparisons"
            ),
            "margin_threshold_beyond_strict_positivity": None,
        },
        "population": {
            "source_count": 8,
            "fit_source_count": 4,
            "confirmation_source_count": 4,
            "actor_family_count": 2,
            "branches_per_source": 3,
            "hidden_artifact_count": 24,
            "forward_confirmation_margin_count": 8,
            "all_registered_rows_consumed": True,
            "minimum_registered_seed_per_source": True,
        },
        "split_policy": {
            "two_fit_and_two_confirmation_sources_per_actor_family": True,
            "fit_only_defines_every_direction": True,
            "confirmation_never_enters_direction_or_basis": True,
            "confirmation_consumed_by_optimizer": False,
        },
        "family_results": family_results,
        "aggregate_by_positive_branch": aggregate,
        "gates": {
            "forward_replication_gate_passed": forward_gate,
            "noop_pseudo_positive_control_passed": noop_control_passed,
            "reverse_pseudo_positive_control_passed": reverse_control_passed,
            "action_specific_replication_gate_passed": specificity_gate,
            "next_preregistered_editor_probe_eligible": specificity_gate,
        },
        "runtime_binding": {
            "diagnostic_source_sha256": file_sha256(Path(__file__).resolve()),
            "prior_representation_source_sha256": file_sha256(
                Path(prior_diagnostic.__file__).resolve(strict=True)
            ),
            "materializer_source_sha256": file_sha256(
                Path(materializer.__file__).resolve(strict=True)
            ),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "safetensors_version": safetensors.__version__,
            "device": "cpu",
            "optimizer_constructed": False,
            "editor_forward_performed": False,
        },
        "limitations": {
            "source8_was_used_by_prior_decoded_diagnostics": True,
            "source8_is_new_for_this_hidden_representation_but_not_virgin_data": True,
            "single_block_and_single_sigma_only": True,
            "hidden_residuals_are_self_generated_t2v_not_current_rv2v": True,
            "only_noop_and_reverse_semantic_controls_available": True,
            "protected_entity_nuisance_projection_not_tested": True,
            "decoded_editor_confirmation_still_required": True,
        },
        "authority": {
            "data_selection": False,
            "representation_selection": False,
            "editor_feature_target": False,
            "training": False,
            "optimizer": False,
            "editor_update": False,
            "scientific_claim": False,
        },
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise Source8HiddenConfirmationError("output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Source8HiddenConfirmationError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_diagnostic(args.master.resolve(strict=True))
    _write_create_only(args.output.resolve(strict=False), result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "receipt_digest": result["receipt_digest"],
                "gates": result["gates"],
                "authority": result["authority"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPRESENTATION_NAME",
    "SCHEMA_VERSION",
    "Source8HiddenConfirmationError",
    "evaluate_positive_branch",
    "fit_direction",
    "main",
    "preregistered_representation",
    "run_diagnostic",
]
