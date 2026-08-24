#!/usr/bin/env python3
"""Strict loader for the instance-level E00 source-role observer asset.

The labels in this asset name visible source instances.  They are not a
ground-truth claim that the requested edit succeeded and they authorize no
training, routing, or decoding.  Only the three peer vessel instances share a
mutual-exclusion group; agent and support evidence remain independent.
"""

from __future__ import annotations

import json
from pathlib import Path
import stat
from typing import Any, Mapping

try:
    from . import source_owned_role_locator_v15 as locator
except ImportError:  # pragma: no cover - module:factory deployment path
    import source_owned_role_locator_v15 as locator


ASSET_SCHEMA_VERSION = "bernini-e00-instance-role-token-spans-v15b"
ASSET_SHA256 = "883d2bf7d6ce4caa0222904d5de892963b7d1641f91197ecef6832a8d0ba2709"
EVENT_SHA256 = "61e33c1be7b59ce16b344424cbebafd7ca073303e2d56e3059bddbd4695f65fd"
SOURCE_VIDEO_SHA256 = "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
ROLE_NAMES = ("agent", "old_actor", "new_actor", "recipient", "support")
VESSEL_COMPETITION_GROUP = ("old_actor", "new_actor", "recipient")
INDEPENDENT_ROLES = ("agent", "support")
DEFAULT_ASSET = (
    Path(__file__).resolve().parent
    / "assets"
    / "interaction_e00_source_instance_role_token_spans_v15b.json"
)


class E00V15BAssetError(RuntimeError):
    """Fail-closed instance-role asset violation."""


def _plain_file(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise E00V15BAssetError("v15b asset must be an absolute non-symlink file")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise E00V15BAssetError("v15b asset is missing") from error
    if not stat.S_ISREG(mode):
        raise E00V15BAssetError("v15b asset is not a plain file")
    return path.resolve(strict=True)


def load_e00_v15b_asset(path: str | Path = DEFAULT_ASSET) -> tuple[
    locator.SourceRoleEventSpec, Mapping[str, Any]
]:
    """Authenticate the whole sidecar and return its core observer event."""

    asset_path = _plain_file(Path(path))
    try:
        raw = json.loads(asset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E00V15BAssetError("cannot parse v15b asset") from error
    required = {
        "asset_sha256",
        "competition_groups",
        "decode_authorized",
        "event",
        "independent_roles",
        "mutual_exclusion_scope",
        "required_localization_roles",
        "route_authorized",
        "schema_version",
        "semantic_contract",
        "source_video_sha256",
        "status",
        "training_authorized",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise E00V15BAssetError("v15b asset fields differ")
    payload = dict(raw)
    asset_sha = payload.pop("asset_sha256", None)
    if (
        raw["schema_version"] != ASSET_SCHEMA_VERSION
        or asset_sha != ASSET_SHA256
        or locator.object_sha256(payload) != ASSET_SHA256
        or raw["source_video_sha256"] != SOURCE_VIDEO_SHA256
        or raw["status"] != "observer_only_diagnostic_not_route"
        or raw["route_authorized"] is not False
        or raw["training_authorized"] is not False
        or raw["decode_authorized"] is not False
    ):
        raise E00V15BAssetError("v15b asset identity/authority differs")
    if (
        raw["competition_groups"] != {
            "vessel_instances": list(VESSEL_COMPETITION_GROUP)
        }
        or tuple(raw["independent_roles"]) != INDEPENDENT_ROLES
        or tuple(raw["required_localization_roles"]) != VESSEL_COMPETITION_GROUP
        or raw["mutual_exclusion_scope"]
        != "only_members_within_each_competition_group"
    ):
        raise E00V15BAssetError("v15b competition scope differs")
    semantic = raw["semantic_contract"]
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("labels_are_source_instance_descriptors_not_action_ground_truth")
        is not True
        or semantic.get("prompt_action_success_authorized") is not False
        or set(semantic)
        != {
            "agent",
            "old_actor",
            "new_actor",
            "recipient",
            "support",
            "requested_graph",
            "labels_are_source_instance_descriptors_not_action_ground_truth",
            "prompt_action_success_authorized",
        }
    ):
        raise E00V15BAssetError("v15b semantic scope differs")

    row = raw["event"]
    event_keys = {
        "event_id",
        "source_iid",
        "source_caption",
        "source_caption_sha256",
        "model_text",
        "model_text_sha256",
        "source_caption_char_start",
        "tokenizer_tree_sha256",
        "roles",
        "event_sha256",
    }
    if not isinstance(row, Mapping) or set(row) != event_keys:
        raise E00V15BAssetError("v15b event fields differ")
    try:
        roles = tuple(
            locator.LockedRoleSpan(
                role=item["role"],
                kind=item["kind"],
                protected=item["protected"],
                substring=item["substring"],
                char_start=item["char_start"],
                char_end=item["char_end"],
                substring_sha256=item["substring_sha256"],
                token_start=item["token_start"],
                token_end=item["token_end"],
                token_ids=tuple(item["token_ids"]),
                token_ids_sha256=item["token_ids_sha256"],
                span_sha256=item["span_sha256"],
            )
            for item in row["roles"]
        )
        spec = locator.SourceRoleEventSpec(
            event_id=row["event_id"],
            source_iid=row["source_iid"],
            source_caption=row["source_caption"],
            source_caption_sha256=row["source_caption_sha256"],
            model_text=row["model_text"],
            model_text_sha256=row["model_text_sha256"],
            source_caption_char_start=row["source_caption_char_start"],
            tokenizer_tree_sha256=row["tokenizer_tree_sha256"],
            roles=roles,
            event_sha256=row["event_sha256"],
        )
    except (KeyError, TypeError, locator.SourceOwnedRoleLocatorError) as error:
        raise E00V15BAssetError("v15b role/token lock differs") from error
    if (
        spec.event_sha256 != EVENT_SHA256
        or spec.role_names != ROLE_NAMES
        or tuple(item.kind for item in spec.roles)
        != (
            "agent_instance",
            "vessel_instance",
            "vessel_instance",
            "vessel_instance",
            "support",
        )
        or spec.tokenizer_tree_sha256 != locator.PINNED_TOKENIZER_TREE_SHA256
    ):
        raise E00V15BAssetError("v15b event/role registry differs")
    return spec, raw


__all__ = [
    "ASSET_SCHEMA_VERSION",
    "ASSET_SHA256",
    "DEFAULT_ASSET",
    "E00V15BAssetError",
    "EVENT_SHA256",
    "INDEPENDENT_ROLES",
    "ROLE_NAMES",
    "VESSEL_COMPETITION_GROUP",
    "load_e00_v15b_asset",
]
