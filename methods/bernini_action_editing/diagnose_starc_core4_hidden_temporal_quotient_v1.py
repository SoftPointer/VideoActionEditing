#!/usr/bin/env python3
"""Audit frozen STARC hidden residuals with fit-only temporal quotients.

The historical STARC scalar critic overfit its two fit episodes and passed only
11/24 held-out positive-vs-negative comparisons.  This diagnostic asks a
different, optimizer-free question: does a fixed temporal representation of
``block.15.output(action) - block.15.output(noop)`` preserve the *direction* of
each action-vs-hard-negative contrast across source identities?

All 52 sealed core4 tensors are authenticated before use.  One fit episode and
one confirmation episode are paired within each registered action family.
Confirmation tensors never define a template or subspace.  No editor forward,
parameter mutation, data selection, or optimizer is present in this program.
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


SCHEMA_VERSION = "bernini-starc-core4-hidden-temporal-quotient-diagnostic-v1"
MASTER_SCHEMA = "bernini-starc-core4-same-state-hidden-master-v1"
GROUP_SCHEMA = "bernini-starc-core4-same-state-hidden-group-v1"
ARM_SCHEMA = "bernini-starc-core4-same-state-hidden-arm-v1"
TENSOR_KEY = "sketched_action_minus_noop_hidden_residual"
EXPECTED_SHAPE = (1, 21, 16, 1536)
EXPECTED_SPLITS = ("fit", "confirmation")
EXPECTED_ROLES = (
    "positive",
    "same_video_reverse",
    "same_video_freeze_first",
    "same_video_phase_shuffle",
    "semantic_noop",
    "semantic_incomplete",
    "semantic_reverse",
    "semantic_shuffle",
    "semantic_wrong_actor",
    "semantic_wrong_object",
    "semantic_camera_only",
    "semantic_appearance_only",
    "semantic_generic_wrong_motion",
)
NEGATIVE_ROLES = EXPECTED_ROLES[1:]
REGISTERED_RANKS = (1, 2, 3, 4)
FOLLOWUP_NUISANCE_AXIS_SETS = {
    "phase_actor_shortcut2": (
        "same_video_phase_shuffle",
        "semantic_wrong_actor",
    ),
    "protected_entity4": (
        "semantic_wrong_actor",
        "semantic_wrong_object",
        "semantic_camera_only",
        "semantic_appearance_only",
    ),
    "temporal_transform4": (
        "same_video_reverse",
        "same_video_freeze_first",
        "same_video_phase_shuffle",
        "semantic_shuffle",
    ),
    "temporal_plus_protected8": (
        "same_video_reverse",
        "same_video_freeze_first",
        "same_video_phase_shuffle",
        "semantic_shuffle",
        "semantic_wrong_actor",
        "semantic_wrong_object",
        "semantic_camera_only",
        "semantic_appearance_only",
    ),
}


class HiddenTemporalQuotientError(RuntimeError):
    """A sealed input or fit/confirmation diagnostic violated its closure."""


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
        raise HiddenTemporalQuotientError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise HiddenTemporalQuotientError(f"not a regular file: {path}")
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
        raise HiddenTemporalQuotientError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise HiddenTemporalQuotientError("tensor hash requires a material tensor")
    owned = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in owned.shape],
        "dtype": str(owned.dtype),
        "layout": str(owned.layout),
    }
    raw = owned.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    return hashlib.sha256(canonical_json_bytes(metadata) + b"\x00" + raw).hexdigest()


def _json_file(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise HiddenTemporalQuotientError(f"JSON path must be absolute plain file: {path}")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise HiddenTemporalQuotientError(f"JSON file SHA-256 differs: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HiddenTemporalQuotientError(f"invalid ASCII JSON: {path}") from error
    if not isinstance(value, dict):
        raise HiddenTemporalQuotientError(f"JSON root must be object: {path}")
    return value


def _verify_seal(value: Mapping[str, Any], *, schema: str, label: str) -> None:
    row = dict(value)
    declared = row.pop("receipt_digest", None)
    if row.get("schema_version") != schema or declared != object_sha256(row):
        raise HiddenTemporalQuotientError(f"{label} schema or receipt digest differs")


def _unit(value: Any) -> Any:
    import torch

    flat = value.float().reshape(-1)
    norm = torch.linalg.vector_norm(flat)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1.0e-12:
        raise HiddenTemporalQuotientError("representation has zero/non-finite norm")
    return flat / norm


def temporal_representations(value: Any) -> dict[str, Any]:
    """Return preregistered frozen temporal views of one hidden residual."""

    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(int(item) for item in value.shape) != EXPECTED_SHAPE
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise HiddenTemporalQuotientError("hidden residual tensor contract differs")
    hidden = value[0]
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    velocity = hidden[1:] - hidden[:-1]
    phase_flat = hidden.reshape(21, -1)
    centered_flat = centered.reshape(21, -1)

    def self_similarity(rows: Any) -> Any:
        normalized = rows / torch.linalg.vector_norm(rows, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        gram = normalized @ normalized.transpose(-1, -2)
        upper = torch.triu_indices(21, 21, offset=1)
        return gram[..., upper[0], upper[1]].mean(dim=0).reshape(-1)

    sketch_rows = hidden.permute(1, 0, 2)
    centered_sketch_rows = centered.permute(1, 0, 2)
    phase_energy = torch.linalg.vector_norm(phase_flat, dim=1)
    velocity_energy = torch.linalg.vector_norm(velocity.reshape(20, -1), dim=1)
    singular_values = torch.linalg.svdvals(centered_flat)
    phase_mean = hidden.mean(dim=1)
    centered_phase_mean = phase_mean - phase_mean.mean(dim=0, keepdim=True)
    return {
        "raw_hidden": _unit(hidden),
        "centered_hidden": _unit(centered),
        "temporal_velocity": _unit(velocity),
        "endpoint_arrow": _unit(hidden[-1] - hidden[0]),
        "phase_energy": _unit(phase_energy - phase_energy.mean()),
        "velocity_energy": _unit(velocity_energy - velocity_energy.mean()),
        "temporal_singular_values": _unit(singular_values),
        "global_temporal_self_similarity": _unit(self_similarity(phase_flat[None])),
        "sketch_temporal_self_similarity": _unit(self_similarity(sketch_rows)),
        "centered_sketch_self_similarity": _unit(
            self_similarity(centered_sketch_rows)
        ),
        "centered_phase_mean": _unit(centered_phase_mean),
    }


def _cosine(left: Any, right: Any) -> float:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        raise HiddenTemporalQuotientError("cosine representation shape differs")
    value = float(torch.dot(_unit(left), _unit(right)).item())
    if not math.isfinite(value):
        raise HiddenTemporalQuotientError("cosine is non-finite")
    return value


def common_direction_ranking(
    *,
    fit_by_role: Mapping[str, Any],
    confirmation_by_role: Mapping[str, Any],
    positive_role: str,
) -> dict[str, Any]:
    """Orient one fit-only direction and rank a held-out role against all others."""

    import torch

    if (
        tuple(fit_by_role) != EXPECTED_ROLES
        or tuple(confirmation_by_role) != EXPECTED_ROLES
        or positive_role not in EXPECTED_ROLES
    ):
        raise HiddenTemporalQuotientError("common-direction role closure differs")
    negative_roles = tuple(role for role in EXPECTED_ROLES if role != positive_role)
    fit_positive = _unit(fit_by_role[positive_role])
    confirmation_positive = _unit(confirmation_by_role[positive_role])
    fit_contrasts = [
        _unit(fit_positive - _unit(fit_by_role[role])) for role in negative_roles
    ]
    direction = _unit(torch.stack(fit_contrasts).mean(dim=0))
    fit_margins = [
        float(torch.dot(direction, contrast).item()) for contrast in fit_contrasts
    ]
    if sum(fit_margins) < 0.0:
        direction = -direction
        fit_margins = [-value for value in fit_margins]
    rows = []
    for role in negative_roles:
        contrast = confirmation_positive - _unit(confirmation_by_role[role])
        margin = float(torch.dot(direction, contrast).item())
        rows.append(
            {
                "negative_role": role,
                "confirmation_margin": margin,
                "positive": margin > 0.0,
            }
        )
    margins = [row["confirmation_margin"] for row in rows]
    return {
        "positive_role": positive_role,
        "fit_minimum_margin": min(fit_margins),
        "positive_count": sum(row["positive"] for row in rows),
        "count": len(rows),
        "all_positive": all(row["positive"] for row in rows),
        "mean_confirmation_margin": sum(margins) / len(margins),
        "min_confirmation_margin": min(margins),
        "rows": rows,
        "direction": direction,
    }


def nuisance_projected_action_ranking(
    *,
    fit_by_role: Mapping[str, Any],
    confirmation_by_role: Mapping[str, Any],
    nuisance_roles: Sequence[str],
) -> dict[str, Any]:
    """Project the fit action axis off fit-only pseudo-positive nuisance axes."""

    import torch

    roles = tuple(nuisance_roles)
    if (
        not roles
        or len(set(roles)) != len(roles)
        or any(role not in NEGATIVE_ROLES for role in roles)
    ):
        raise HiddenTemporalQuotientError("nuisance-axis role set differs")
    action = common_direction_ranking(
        fit_by_role=fit_by_role,
        confirmation_by_role=confirmation_by_role,
        positive_role="positive",
    )["direction"]
    basis = []
    for role in roles:
        axis = common_direction_ranking(
            fit_by_role=fit_by_role,
            confirmation_by_role=confirmation_by_role,
            positive_role=role,
        )["direction"].clone()
        for unit in basis:
            axis = axis - unit * torch.dot(unit, axis)
        norm = float(torch.linalg.vector_norm(axis).item())
        if norm > 1.0e-8:
            basis.append(axis / norm)
    if not basis:
        raise HiddenTemporalQuotientError("nuisance-axis basis is empty")
    residual = action.clone()
    for unit in basis:
        residual = residual - unit * torch.dot(unit, residual)
    retained_norm = float(torch.linalg.vector_norm(residual).item())
    if not math.isfinite(retained_norm) or retained_norm <= 1.0e-8:
        return {
            "nuisance_roles": list(roles),
            "nuisance_effective_rank": len(basis),
            "retained_action_norm_fraction": retained_norm,
            "fit_positive_count": 0,
            "confirmation_positive_count": 0,
            "count": len(NEGATIVE_ROLES),
            "all_fit_positive": False,
            "all_confirmation_positive": False,
            "mean_confirmation_margin": None,
            "min_confirmation_margin": None,
            "rows": [],
        }
    direction = residual / retained_norm
    fit_positive = _unit(fit_by_role["positive"])
    confirmation_positive = _unit(confirmation_by_role["positive"])
    rows = []
    for role in NEGATIVE_ROLES:
        fit_margin = float(
            torch.dot(direction, fit_positive - _unit(fit_by_role[role])).item()
        )
        confirmation_margin = float(
            torch.dot(
                direction,
                confirmation_positive - _unit(confirmation_by_role[role]),
            ).item()
        )
        rows.append(
            {
                "negative_role": role,
                "fit_margin": fit_margin,
                "fit_positive": fit_margin > 0.0,
                "confirmation_margin": confirmation_margin,
                "confirmation_positive": confirmation_margin > 0.0,
            }
        )
    margins = [row["confirmation_margin"] for row in rows]
    return {
        "nuisance_roles": list(roles),
        "nuisance_effective_rank": len(basis),
        "retained_action_norm_fraction": retained_norm,
        "fit_positive_count": sum(row["fit_positive"] for row in rows),
        "confirmation_positive_count": sum(
            row["confirmation_positive"] for row in rows
        ),
        "count": len(rows),
        "all_fit_positive": all(row["fit_positive"] for row in rows),
        "all_confirmation_positive": all(
            row["confirmation_positive"] for row in rows
        ),
        "mean_confirmation_margin": sum(margins) / len(margins),
        "min_confirmation_margin": min(margins),
        "rows": rows,
    }


def diagnose_representation(
    *,
    fit_by_role: Mapping[str, Any],
    confirmation_by_role: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare matched action-vs-negative arrows without fitting confirmation."""

    import torch

    if tuple(fit_by_role) != EXPECTED_ROLES or tuple(confirmation_by_role) != EXPECTED_ROLES:
        raise HiddenTemporalQuotientError("role order differs")
    fit_positive = _unit(fit_by_role["positive"])
    confirmation_positive = _unit(confirmation_by_role["positive"])
    fit_contrasts = []
    confirmation_contrasts = []
    template_margins = []
    role_rows = []
    for role in NEGATIVE_ROLES:
        fit_negative = _unit(fit_by_role[role])
        confirmation_negative = _unit(confirmation_by_role[role])
        fit_contrast = _unit(fit_positive - fit_negative)
        confirmation_contrast = _unit(confirmation_positive - confirmation_negative)
        alignment = _cosine(fit_contrast, confirmation_contrast)
        template_margin = float(
            torch.dot(fit_positive, confirmation_positive).item()
            - torch.dot(fit_positive, confirmation_negative).item()
        )
        fit_contrasts.append(fit_contrast)
        confirmation_contrasts.append(confirmation_contrast)
        template_margins.append(template_margin)
        role_rows.append(
            {
                "negative_role": role,
                "matched_contrast_cosine": alignment,
                "matched_contrast_positive": alignment > 0.0,
                "positive_template_margin": template_margin,
                "positive_template_margin_positive": template_margin > 0.0,
            }
        )

    fit_matrix = torch.stack(fit_contrasts)
    gram = fit_matrix @ fit_matrix.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    total_fit_contrast_energy = float(eigenvalues.sum().item())
    if not math.isfinite(total_fit_contrast_energy) or total_fit_contrast_energy <= 0.0:
        raise HiddenTemporalQuotientError("fit contrast energy is invalid")
    low_rank = {}
    for rank in REGISTERED_RANKS:
        kept = eigenvalues[:rank]
        valid = kept > 1.0e-10
        if int(valid.sum().item()) != rank:
            raise HiddenTemporalQuotientError("fit contrast rank is below registration")
        left = eigenvectors[:, :rank]
        singular = torch.sqrt(kept)
        rank_rows = []
        for index, role in enumerate(NEGATIVE_ROLES):
            fit_coordinates = singular * left[index]
            cross = fit_matrix @ confirmation_contrasts[index]
            confirmation_coordinates = (left.T @ cross) / singular
            alignment = _cosine(fit_coordinates, confirmation_coordinates)
            fit_coordinate_norm = float(torch.linalg.vector_norm(fit_coordinates).item())
            confirmation_support = float(
                torch.linalg.vector_norm(confirmation_coordinates).item()
            )
            signed_support = float(
                torch.dot(_unit(fit_coordinates), confirmation_coordinates).item()
            )
            rank_rows.append(
                {
                    "negative_role": role,
                    "coordinate_alignment": alignment,
                    "positive": alignment > 0.0,
                    "fit_coordinate_norm": fit_coordinate_norm,
                    "confirmation_support_fraction": confirmation_support,
                    "signed_confirmation_support": signed_support,
                }
            )
        alignments = [row["coordinate_alignment"] for row in rank_rows]
        supports = [row["confirmation_support_fraction"] for row in rank_rows]
        signed_supports = [row["signed_confirmation_support"] for row in rank_rows]
        low_rank[str(rank)] = {
            "rank": rank,
            "explained_fit_contrast_energy": float(kept.sum().item())
            / total_fit_contrast_energy,
            "positive_count": sum(row["positive"] for row in rank_rows),
            "all_positive": all(row["positive"] for row in rank_rows),
            "mean_alignment": sum(alignments) / len(alignments),
            "min_alignment": min(alignments),
            "mean_confirmation_support_fraction": sum(supports) / len(supports),
            "min_confirmation_support_fraction": min(supports),
            "mean_signed_confirmation_support": sum(signed_supports)
            / len(signed_supports),
            "min_signed_confirmation_support": min(signed_supports),
            "rows": rank_rows,
        }

    matched = [row["matched_contrast_cosine"] for row in role_rows]
    common = common_direction_ranking(
        fit_by_role=fit_by_role,
        confirmation_by_role=confirmation_by_role,
        positive_role="positive",
    )
    return {
        "negative_count": len(NEGATIVE_ROLES),
        "matched_contrast": {
            "positive_count": sum(row["matched_contrast_positive"] for row in role_rows),
            "all_positive": all(row["matched_contrast_positive"] for row in role_rows),
            "mean_cosine": sum(matched) / len(matched),
            "min_cosine": min(matched),
        },
        "positive_template_ranking": {
            "positive_count": sum(
                row["positive_template_margin_positive"] for row in role_rows
            ),
            "all_positive": all(
                row["positive_template_margin_positive"] for row in role_rows
            ),
            "mean_margin": sum(template_margins) / len(template_margins),
            "min_margin": min(template_margins),
        },
        "fit_only_role_contrast_subspace": low_rank,
        "fit_only_common_direction_ranking": {
            key: value for key, value in common.items() if key != "direction"
        },
        "rows": role_rows,
    }


def _load_rows(master_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch
    from safetensors import safe_open

    master = _json_file(master_path)
    _verify_seal(master, schema=MASTER_SCHEMA, label="master")
    if (
        master.get("arm_count") != 52
        or master.get("episode_count") != 4
        or master.get("fit_episode_count") != 2
        or master.get("confirmation_episode_count") != 2
        or master.get("arm_order") != list(EXPECTED_ROLES)
        or master.get("training_performed") is not False
        or master.get("optimizer_authorized") is not False
        or master.get("editor_optimizer_authorized") is not False
    ):
        raise HiddenTemporalQuotientError("master scientific/row closure differs")

    episodes: dict[str, dict[str, Any]] = {}
    for binding in master.get("group_bindings", []):
        group_path = Path(binding["manifest_path"])
        group = _json_file(group_path, expected_sha256=binding["manifest_file_sha256"])
        _verify_seal(group, schema=GROUP_SCHEMA, label=f"group {binding['group_id']}")
        if (
            group.get("receipt_digest") != binding.get("receipt_digest")
            or group.get("arm_order") != list(EXPECTED_ROLES)
            or group.get("training_performed") is not False
            or group.get("optimizer_authorized") is not False
        ):
            raise HiddenTemporalQuotientError("group binding/authority differs")
        for arm in group.get("arm_bindings", []):
            receipt_path = Path(arm["receipt_path"])
            receipt = _json_file(
                receipt_path, expected_sha256=arm["receipt_file_sha256"]
            )
            _verify_seal(receipt, schema=ARM_SCHEMA, label="arm")
            if any(
                receipt.get(name) != arm.get(name)
                for name in ("episode_id", "role", "split", "label", "receipt_digest")
            ):
                raise HiddenTemporalQuotientError("arm receipt binding differs")
            artifact_path = Path(arm["artifact_path"])
            if (
                file_sha256(artifact_path) != arm["artifact_file_sha256"]
                or receipt.get("artifact", {}).get("file_sha256")
                != arm["artifact_file_sha256"]
                or receipt.get("artifact", {}).get("tensor_sha256")
                != arm["artifact_tensor_sha256"]
            ):
                raise HiddenTemporalQuotientError("arm artifact binding differs")
            with safe_open(artifact_path, framework="pt", device="cpu") as handle:
                if list(handle.keys()) != [TENSOR_KEY]:
                    raise HiddenTemporalQuotientError("artifact tensor key differs")
                tensor = handle.get_tensor(TENSOR_KEY)
            tensor = tensor.detach().contiguous()
            if tensor_sha256(tensor) != arm["artifact_tensor_sha256"]:
                raise HiddenTemporalQuotientError("artifact tensor value differs")
            episode_id = receipt["episode_id"]
            episode = episodes.setdefault(
                episode_id,
                {
                    "episode_id": episode_id,
                    "split": receipt["split"],
                    "action_family_id": receipt["action_family_id"],
                    "actor_group_id": receipt["actor_group_id"],
                    "scene_group_id": receipt["scene_group_id"],
                    "roles": {},
                },
            )
            if any(
                episode[name] != receipt[name]
                for name in ("split", "action_family_id", "actor_group_id", "scene_group_id")
            ):
                raise HiddenTemporalQuotientError("episode metadata differs across roles")
            if receipt["role"] in episode["roles"]:
                raise HiddenTemporalQuotientError("duplicate episode role")
            episode["roles"][receipt["role"]] = {
                "tensor": tensor,
                "artifact_file_sha256": arm["artifact_file_sha256"],
                "artifact_tensor_sha256": arm["artifact_tensor_sha256"],
                "receipt_digest": arm["receipt_digest"],
            }

    if len(episodes) != 4:
        raise HiddenTemporalQuotientError("episode count differs")
    for episode in episodes.values():
        if tuple(episode["roles"]) != EXPECTED_ROLES:
            raise HiddenTemporalQuotientError("episode role closure/order differs")
    if {episode["split"] for episode in episodes.values()} != set(EXPECTED_SPLITS):
        raise HiddenTemporalQuotientError("split closure differs")
    return master, episodes


def run_diagnostic(master_path: Path) -> dict[str, Any]:
    import safetensors
    import torch

    master, episodes = _load_rows(master_path)
    families: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in episodes.values():
        family = families.setdefault(episode["action_family_id"], {})
        if episode["split"] in family:
            raise HiddenTemporalQuotientError("family has duplicate split episode")
        family[episode["split"]] = episode
    if len(families) != 2 or any(set(rows) != set(EXPECTED_SPLITS) for rows in families.values()):
        raise HiddenTemporalQuotientError("family fit/confirmation closure differs")

    episode_features: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in episodes.values():
        episode_features[episode["episode_id"]] = {
            role: temporal_representations(episode["roles"][role]["tensor"])
            for role in EXPECTED_ROLES
        }
    representation_names = tuple(
        episode_features[next(iter(episode_features))]["positive"]
    )
    diagnostics: dict[str, Any] = {}
    for representation in representation_names:
        per_family = {}
        all_matched = []
        all_template = []
        ranks = {str(rank): [] for rank in REGISTERED_RANKS}
        rank_supports = {str(rank): [] for rank in REGISTERED_RANKS}
        rank_signed_supports = {str(rank): [] for rank in REGISTERED_RANKS}
        rank_fit_energy = {str(rank): [] for rank in REGISTERED_RANKS}
        common_direction_margins = []
        pseudo_positive_margins = {role: [] for role in EXPECTED_ROLES}
        nuisance_projection_results = {
            name: [] for name in FOLLOWUP_NUISANCE_AXIS_SETS
        }
        global_fit_contrasts = []
        global_confirmation_contrasts = []
        for family_name, split_rows in sorted(families.items()):
            fit = split_rows["fit"]
            confirmation = split_rows["confirmation"]
            fit_features = {
                role: episode_features[fit["episode_id"]][role][representation]
                for role in EXPECTED_ROLES
            }
            confirmation_features = {
                role: episode_features[confirmation["episode_id"]][role][representation]
                for role in EXPECTED_ROLES
            }
            result = diagnose_representation(
                fit_by_role=fit_features,
                confirmation_by_role=confirmation_features,
            )
            per_family[family_name] = {
                "fit_episode_id": fit["episode_id"],
                "confirmation_episode_id": confirmation["episode_id"],
                **result,
            }
            all_matched.extend(row["matched_contrast_cosine"] for row in result["rows"])
            all_template.extend(row["positive_template_margin"] for row in result["rows"])
            common_direction_margins.extend(
                row["confirmation_margin"]
                for row in result["fit_only_common_direction_ranking"]["rows"]
            )
            for pseudo_role in EXPECTED_ROLES:
                pseudo = common_direction_ranking(
                    fit_by_role=fit_features,
                    confirmation_by_role=confirmation_features,
                    positive_role=pseudo_role,
                )
                pseudo_positive_margins[pseudo_role].extend(
                    row["confirmation_margin"] for row in pseudo["rows"]
                )
            for name, nuisance_roles in FOLLOWUP_NUISANCE_AXIS_SETS.items():
                projected = nuisance_projected_action_ranking(
                    fit_by_role=fit_features,
                    confirmation_by_role=confirmation_features,
                    nuisance_roles=nuisance_roles,
                )
                nuisance_projection_results[name].append(
                    {
                        "action_family_id": family_name,
                        **projected,
                    }
                )
            fit_positive = _unit(fit_features["positive"])
            confirmation_positive = _unit(confirmation_features["positive"])
            for negative_role in NEGATIVE_ROLES:
                global_fit_contrasts.append(
                    _unit(fit_positive - _unit(fit_features[negative_role]))
                )
                global_confirmation_contrasts.append(
                    confirmation_positive - _unit(confirmation_features[negative_role])
                )
            for rank in REGISTERED_RANKS:
                rank_result = result["fit_only_role_contrast_subspace"][str(rank)]
                ranks[str(rank)].extend(
                    row["coordinate_alignment"] for row in rank_result["rows"]
                )
                rank_supports[str(rank)].extend(
                    row["confirmation_support_fraction"] for row in rank_result["rows"]
                )
                rank_signed_supports[str(rank)].extend(
                    row["signed_confirmation_support"] for row in rank_result["rows"]
                )
                rank_fit_energy[str(rank)].append(
                    rank_result["explained_fit_contrast_energy"]
                )
        global_direction = _unit(torch.stack(global_fit_contrasts).mean(dim=0))
        global_fit_margins = [
            float(torch.dot(global_direction, value).item())
            for value in global_fit_contrasts
        ]
        if sum(global_fit_margins) < 0.0:
            global_direction = -global_direction
        global_margins = [
            float(torch.dot(global_direction, value).item())
            for value in global_confirmation_contrasts
        ]
        diagnostics[representation] = {
            "per_family": per_family,
            "aggregate": {
                "matched_contrast_positive_count": sum(value > 0.0 for value in all_matched),
                "matched_contrast_count": len(all_matched),
                "matched_contrast_all_positive": all(value > 0.0 for value in all_matched),
                "matched_contrast_mean_cosine": sum(all_matched) / len(all_matched),
                "matched_contrast_min_cosine": min(all_matched),
                "positive_template_margin_positive_count": sum(
                    value > 0.0 for value in all_template
                ),
                "positive_template_margin_all_positive": all(
                    value > 0.0 for value in all_template
                ),
                "positive_template_mean_margin": sum(all_template) / len(all_template),
                "positive_template_min_margin": min(all_template),
                "family_specific_common_direction": {
                    "positive_count": sum(value > 0.0 for value in common_direction_margins),
                    "count": len(common_direction_margins),
                    "all_positive": all(value > 0.0 for value in common_direction_margins),
                    "mean_confirmation_margin": sum(common_direction_margins)
                    / len(common_direction_margins),
                    "min_confirmation_margin": min(common_direction_margins),
                },
                "global_common_direction": {
                    "positive_count": sum(value > 0.0 for value in global_margins),
                    "count": len(global_margins),
                    "all_positive": all(value > 0.0 for value in global_margins),
                    "mean_confirmation_margin": sum(global_margins) / len(global_margins),
                    "min_confirmation_margin": min(global_margins),
                },
                "pseudo_positive_control": {
                    role: {
                        "positive_count": sum(value > 0.0 for value in values),
                        "count": len(values),
                        "all_positive": all(value > 0.0 for value in values),
                        "mean_confirmation_margin": sum(values) / len(values),
                        "min_confirmation_margin": min(values),
                    }
                    for role, values in pseudo_positive_margins.items()
                },
                "followup_nuisance_projection": {
                    name: {
                        "exploratory_post_screen_control": True,
                        "nuisance_roles": list(FOLLOWUP_NUISANCE_AXIS_SETS[name]),
                        "family_results": rows,
                        "fit_positive_count": sum(
                            row["fit_positive_count"] for row in rows
                        ),
                        "confirmation_positive_count": sum(
                            row["confirmation_positive_count"] for row in rows
                        ),
                        "count": sum(row["count"] for row in rows),
                        "all_fit_positive": all(
                            row["all_fit_positive"] for row in rows
                        ),
                        "all_confirmation_positive": all(
                            row["all_confirmation_positive"] for row in rows
                        ),
                        "minimum_retained_action_norm_fraction": min(
                            row["retained_action_norm_fraction"] for row in rows
                        ),
                        "minimum_confirmation_margin": min(
                            row["min_confirmation_margin"]
                            for row in rows
                            if row["min_confirmation_margin"] is not None
                        ),
                    }
                    for name, rows in nuisance_projection_results.items()
                },
                "rank_sweep": {
                    rank: {
                        "positive_count": sum(value > 0.0 for value in values),
                        "count": len(values),
                        "all_positive": all(value > 0.0 for value in values),
                        "mean_alignment": sum(values) / len(values),
                        "min_alignment": min(values),
                        "mean_explained_fit_contrast_energy": sum(
                            rank_fit_energy[rank]
                        )
                        / len(rank_fit_energy[rank]),
                        "minimum_explained_fit_contrast_energy": min(
                            rank_fit_energy[rank]
                        ),
                        "mean_confirmation_support_fraction": sum(
                            rank_supports[rank]
                        )
                        / len(rank_supports[rank]),
                        "min_confirmation_support_fraction": min(
                            rank_supports[rank]
                        ),
                        "mean_signed_confirmation_support": sum(
                            rank_signed_supports[rank]
                        )
                        / len(rank_signed_supports[rank]),
                        "min_signed_confirmation_support": min(
                            rank_signed_supports[rank]
                        ),
                    }
                    for rank, values in ranks.items()
                },
            },
        }

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_hidden_temporal_quotient_diagnostic_no_authority",
        "master_binding": {
            "path": str(master_path),
            "file_sha256": file_sha256(master_path),
            "receipt_digest": master["receipt_digest"],
        },
        "population": {
            "episode_count": 4,
            "arm_count": 52,
            "action_family_count": 2,
            "negative_role_count_per_family": 12,
            "confirmation_contrast_count": 24,
            "all_registered_rows_consumed": True,
            "seed_selection_performed": False,
        },
        "split_policy": {
            "one_fit_and_one_confirmation_episode_per_action_family": True,
            "fit_only_defines_templates_and_subspaces": True,
            "confirmation_vectors_never_extend_basis": True,
            "confirmation_consumed_by_optimizer": False,
        },
        "representation_names": list(representation_names),
        "registered_rank_sweep": list(REGISTERED_RANKS),
        "followup_nuisance_axis_sets": {
            name: list(roles) for name, roles in FOLLOWUP_NUISANCE_AXIS_SETS.items()
        },
        "diagnostics": diagnostics,
        "runtime_binding": {
            "diagnostic_source_sha256": file_sha256(Path(__file__).resolve()),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "safetensors_version": safetensors.__version__,
            "device": "cpu",
            "optimizer_constructed": False,
            "editor_forward_performed": False,
        },
        "limitations": {
            "core4_has_one_fit_and_one_confirmation_source_per_family": True,
            "single_block_and_single_sigma_only": True,
            "hidden_residuals_are_self_generated_t2v_not_current_rv2v": True,
            "spatial_coordinates_are_fixed_random_sketches": True,
            "decoded_confirmation_still_required": True,
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
        raise HiddenTemporalQuotientError("output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise HiddenTemporalQuotientError("output write made no progress")
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
    print(json.dumps({
        "output": str(args.output),
        "receipt_digest": result["receipt_digest"],
        "authority": result["authority"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
