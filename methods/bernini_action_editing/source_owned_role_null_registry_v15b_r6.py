#!/usr/bin/env python3
"""Strict loader/runtime verifier for the preregistered E00 r6 null bank."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

try:
    from . import source_owned_role_locator_v15 as locator
    from . import source_owned_role_locator_v15b_e00_asset as role_asset
except ImportError:  # pragma: no cover - flat AUH deployment
    import source_owned_role_locator_v15 as locator
    import source_owned_role_locator_v15b_e00_asset as role_asset


SCHEMA_VERSION = "bernini-e00-null-token-span-registry-v15b-r6"
REGISTRY_SHA256 = "67fd82119c77b2444b072395fde427b5b95fe0d9b49ae38bc66a2ff044d371a4"
SPAN_COUNT = 64
ACTIVE_TOKEN_COUNT = 56
TOKEN_INPUT_IDS_SHA256 = "29c64e1005bc625c64a194d7056c6c1d9b15b78bb994c14793d46fa71d00983e"
TOKEN_ATTENTION_MASK_SHA256 = "86c7129afa1c6cc35f5104ce2bf534dff382c6a6e3c2c063b284ed328bcd14a3"
DEFAULT_ASSET = (
    Path(__file__).resolve().parent
    / "assets"
    / "interaction_e00_null_token_span_registry_v15b_r6.json"
)


class NullRegistryV15BR6Error(RuntimeError):
    """Fail-closed null registry or runtime-token mismatch."""


@dataclass(frozen=True)
class NullTokenSpanV15BR6:
    index: int
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    char_start: int
    char_end: int
    text: str
    text_sha256: str
    span_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
            or isinstance(self.token_start, bool)
            or not isinstance(self.token_start, int)
            or isinstance(self.token_end, bool)
            or not isinstance(self.token_end, int)
            or not 0 <= self.token_start < self.token_end <= ACTIVE_TOKEN_COUNT
            or len(self.token_ids) != self.token_end - self.token_start
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.token_ids)
            or isinstance(self.char_start, bool)
            or not isinstance(self.char_start, int)
            or isinstance(self.char_end, bool)
            or not isinstance(self.char_end, int)
            or not 0 <= self.char_start < self.char_end
            or not isinstance(self.text, str)
            or not self.text
        ):
            raise NullRegistryV15BR6Error("null span scalar geometry differs")
        if locator.object_sha256(list(self.token_ids)) != self.token_ids_sha256:
            raise NullRegistryV15BR6Error("null span token digest differs")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise NullRegistryV15BR6Error("null span text digest differs")
        payload = {
            "index": self.index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_ids": list(self.token_ids),
            "token_ids_sha256": self.token_ids_sha256,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
            "text_sha256": self.text_sha256,
        }
        if locator.object_sha256(payload) != self.span_sha256:
            raise NullRegistryV15BR6Error("null span receipt digest differs")


@dataclass(frozen=True)
class NullTokenRegistryV15BR6:
    event_id: str
    role_asset_sha256: str
    role_event_sha256: str
    source_video_sha256: str
    model_text_sha256: str
    tokenizer_tree_sha256: str
    token_input_ids_sha256: str
    token_attention_mask_sha256: str
    active_token_ids: tuple[int, ...]
    active_token_offsets: tuple[tuple[int, int], ...]
    occupied_token_indices: tuple[int, ...]
    spans: tuple[NullTokenSpanV15BR6, ...]
    registry_sha256: str

    def __post_init__(self) -> None:
        if (
            self.event_id != "pour-liquid-into-cup"
            or self.role_asset_sha256 != role_asset.ASSET_SHA256
            or self.role_event_sha256 != role_asset.EVENT_SHA256
            or self.source_video_sha256 != role_asset.SOURCE_VIDEO_SHA256
            or self.tokenizer_tree_sha256 != locator.PINNED_TOKENIZER_TREE_SHA256
            or self.token_input_ids_sha256 != TOKEN_INPUT_IDS_SHA256
            or self.token_attention_mask_sha256 != TOKEN_ATTENTION_MASK_SHA256
            or len(self.active_token_ids) != ACTIVE_TOKEN_COUNT
            or len(self.active_token_offsets) != ACTIVE_TOKEN_COUNT
            or len(self.spans) != SPAN_COUNT
            or tuple(item.index for item in self.spans) != tuple(range(SPAN_COUNT))
            or self.registry_sha256 != REGISTRY_SHA256
        ):
            raise NullRegistryV15BR6Error("null registry identity differs")
        if locator.object_sha256(list(self.active_token_ids)) != TOKEN_INPUT_IDS_SHA256:
            raise NullRegistryV15BR6Error("active token ID digest differs")
        role_occupied = set(self.occupied_token_indices)
        if any(
            index in role_occupied
            for span in self.spans
            for index in range(span.token_start, span.token_end)
        ):
            raise NullRegistryV15BR6Error("null span overlaps a locked role token")


def _plain_file(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise NullRegistryV15BR6Error("null registry must be absolute/non-symlink")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise NullRegistryV15BR6Error("null registry asset is missing") from error
    if not stat.S_ISREG(mode):
        raise NullRegistryV15BR6Error("null registry asset is not a plain file")
    return path.resolve(strict=True)


def _tokenizer_fields(value: Mapping[str, Any]) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    try:
        ids = value["input_ids"]
        mask = value["attention_mask"]
        offsets = value["offset_mapping"]
    except (KeyError, TypeError) as error:
        raise NullRegistryV15BR6Error("runtime tokenizer fields differ") from error
    if ids and isinstance(ids[0], Sequence) and not isinstance(ids[0], (str, bytes)):
        ids, mask, offsets = ids[0], mask[0], offsets[0]
    try:
        return (
            [int(item) for item in ids],
            [int(item) for item in mask],
            [(int(item[0]), int(item[1])) for item in offsets],
        )
    except (TypeError, ValueError, IndexError) as error:
        raise NullRegistryV15BR6Error("runtime tokenizer values differ") from error


def load_null_registry_v15b_r6(
    path: str | Path = DEFAULT_ASSET,
) -> NullTokenRegistryV15BR6:
    asset = _plain_file(Path(path))
    try:
        raw = json.loads(asset.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NullRegistryV15BR6Error("cannot parse null registry") from error
    required = {
        "active_token_count",
        "active_token_ids",
        "active_token_offsets",
        "anchor_consumed",
        "decode_authorized",
        "event_id",
        "model_text_sha256",
        "registration_policy",
        "registry_sha256",
        "role_asset_sha256",
        "role_event_sha256",
        "role_occupied_token_indices",
        "route_authorized",
        "schema_version",
        "source_video_sha256",
        "span_count",
        "spans",
        "status",
        "token_attention_mask_sha256",
        "token_input_ids_sha256",
        "tokenizer_tree_sha256",
        "training_authorized",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise NullRegistryV15BR6Error("null registry fields differ")
    payload = dict(raw)
    digest = payload.pop("registry_sha256", None)
    if (
        raw["schema_version"] != SCHEMA_VERSION
        or digest != REGISTRY_SHA256
        or locator.object_sha256(payload) != REGISTRY_SHA256
        or raw["status"] != "observer_only_null_distribution_not_route"
        or raw["anchor_consumed"] is not False
        or raw["route_authorized"] is not False
        or raw["training_authorized"] is not False
        or raw["decode_authorized"] is not False
        or raw["active_token_count"] != ACTIVE_TOKEN_COUNT
        or raw["span_count"] != SPAN_COUNT
        or raw["registration_policy"]
        != {
            "candidate_order": "span_length_then_token_start",
            "excludes_all_locked_role_tokens": True,
            "first_exact_count": SPAN_COUNT,
            "requires_active_attention_mask": True,
            "requires_nonzero_character_offset": True,
            "span_lengths": [1, 2, 3, 4],
        }
    ):
        raise NullRegistryV15BR6Error("null registry hash/authority differs")
    try:
        spans = tuple(
            NullTokenSpanV15BR6(
                index=item["index"],
                token_start=item["token_start"],
                token_end=item["token_end"],
                token_ids=tuple(item["token_ids"]),
                token_ids_sha256=item["token_ids_sha256"],
                char_start=item["char_start"],
                char_end=item["char_end"],
                text=item["text"],
                text_sha256=item["text_sha256"],
                span_sha256=item["span_sha256"],
            )
            for item in raw["spans"]
        )
        registry = NullTokenRegistryV15BR6(
            event_id=raw["event_id"],
            role_asset_sha256=raw["role_asset_sha256"],
            role_event_sha256=raw["role_event_sha256"],
            source_video_sha256=raw["source_video_sha256"],
            model_text_sha256=raw["model_text_sha256"],
            tokenizer_tree_sha256=raw["tokenizer_tree_sha256"],
            token_input_ids_sha256=raw["token_input_ids_sha256"],
            token_attention_mask_sha256=raw["token_attention_mask_sha256"],
            active_token_ids=tuple(raw["active_token_ids"]),
            active_token_offsets=tuple(tuple(item) for item in raw["active_token_offsets"]),
            occupied_token_indices=tuple(raw["role_occupied_token_indices"]),
            spans=spans,
            registry_sha256=raw["registry_sha256"],
        )
    except (KeyError, TypeError, NullRegistryV15BR6Error) as error:
        raise NullRegistryV15BR6Error("null registry row construction differs") from error
    spec, _raw = role_asset.load_e00_v15b_asset()
    if registry.model_text_sha256 != spec.model_text_sha256:
        raise NullRegistryV15BR6Error("registry model text differs")
    expected_occupied = sorted(
        index
        for item in spec.roles
        for index in range(item.token_start, item.token_end)
    )
    if list(registry.occupied_token_indices) != expected_occupied:
        raise NullRegistryV15BR6Error("registry role occupancy differs")
    for span in registry.spans:
        if (
            registry.active_token_ids[span.token_start : span.token_end] != span.token_ids
            or registry.active_token_offsets[span.token_start][0] != span.char_start
            or registry.active_token_offsets[span.token_end - 1][1] != span.char_end
            or spec.model_text[span.char_start : span.char_end] != span.text
        ):
            raise NullRegistryV15BR6Error("registered null span token/text lock differs")
    return registry


def validate_runtime_null_registry_v15b_r6(
    tokenizer: Any,
    registry: NullTokenRegistryV15BR6,
) -> Mapping[str, Any]:
    if not isinstance(registry, NullTokenRegistryV15BR6):
        raise NullRegistryV15BR6Error("runtime validation lacks registry")
    spec, _raw = role_asset.load_e00_v15b_asset()
    try:
        encoded = tokenizer(
            spec.model_text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except Exception as error:
        raise NullRegistryV15BR6Error("runtime null tokenization failed") from error
    ids, mask, offsets = _tokenizer_fields(encoded)
    if (
        ids != list(registry.active_token_ids)
        or mask != [1] * ACTIVE_TOKEN_COUNT
        or offsets != list(registry.active_token_offsets)
        or locator.object_sha256(ids) != registry.token_input_ids_sha256
        or locator.object_sha256(mask) != registry.token_attention_mask_sha256
    ):
        raise NullRegistryV15BR6Error("runtime null tokenization differs from preregistration")
    eligible = {
        index
        for index, ((start, stop), active) in enumerate(zip(offsets, mask))
        if active == 1 and stop > start and index not in registry.occupied_token_indices
    }
    candidates: list[tuple[int, int]] = []
    for length in (1, 2, 3, 4):
        for start in range(ACTIVE_TOKEN_COUNT - length + 1):
            stop = start + length
            if all(index in eligible for index in range(start, stop)):
                candidates.append((start, stop))
                if len(candidates) == SPAN_COUNT:
                    break
        if len(candidates) == SPAN_COUNT:
            break
    if tuple(candidates) != tuple((item.token_start, item.token_end) for item in registry.spans):
        raise NullRegistryV15BR6Error("runtime deterministic null enumeration differs")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": registry.registry_sha256,
        "span_count": SPAN_COUNT,
        "token_input_ids_sha256": registry.token_input_ids_sha256,
        "token_attention_mask_sha256": registry.token_attention_mask_sha256,
        "runtime_exact": True,
        "anchor_consumed": False,
        "route_authorized": False,
    }
    return {**payload, "receipt_sha256": locator.object_sha256(payload)}


__all__ = [
    "DEFAULT_ASSET",
    "NullRegistryV15BR6Error",
    "NullTokenRegistryV15BR6",
    "NullTokenSpanV15BR6",
    "REGISTRY_SHA256",
    "SCHEMA_VERSION",
    "SPAN_COUNT",
    "load_null_registry_v15b_r6",
    "validate_runtime_null_registry_v15b_r6",
]
