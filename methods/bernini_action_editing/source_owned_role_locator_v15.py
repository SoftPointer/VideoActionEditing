#!/usr/bin/env python3
"""Source-owned semantic role localization for Bernini-R cross-attention.

This module is deliberately a *locator*, not an action route.  It observes one
real-source-only ``attn2`` call and derives visual-token affinities from

    official visual Q  x  official source-caption text K.

The public observation API has no action-anchor argument.  The source caption,
its frozen text embedding and the source visual stream are the complete data
authority.  This v15 asset and its masks are observer-only diagnostics; they
are explicitly unauthorized for action routing, training, or decoding.

The wrapper delegates to the exact official processor first and returns that
exact output object.  In observer mode it then calls the official
``_project_qkv`` path on detached inputs solely to record Q/K affinity.  It
does not add a residual, replace attention, alter model parameters, or execute
an implicit distributed collective.

Ulysses ABI
-----------
For cross-attention, Bernini exposes rank-local visual Q and replicated full
text K.  Each rank therefore emits one :class:`RoleAffinityShard` containing
its global half-open interval and only its valid (non append-padding) tokens.
The caller may exchange ``padded_collective_tensor()`` plus
``collective_metadata()`` using its already-authenticated Ulysses process
group, reconstruct shards with :meth:`RoleAffinityShard.from_collective`, and
call :func:`assemble_global_role_affinity`.  Keeping the collective outside
the processor prevents an observer from silently changing the model's
collective schedule.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch


SCHEMA_VERSION = "bernini-source-owned-role-locator-v15"
ASSET_SCHEMA_VERSION = "bernini-complex4-source-role-token-spans-v15"
MASK_SCHEMA_VERSION = "bernini-source-owned-role-masks-v15"
EXPECTED_BLOCK_COUNT = 30
LATENT_PHASES = 21
MAX_TEXT_TOKENS = 512
PINNED_TRANSFORMERS_VERSION = "5.5.4"
PINNED_TOKENIZER_TREE_SHA256 = (
    "0e7e4b06b2c321420e2fb97c07d2329837539b09a39bdf4bcbaa6ec1977da616"
)
PINNED_TOKENIZER_FILES = (
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
OFFICIAL_ATTN2_PROCESSOR_MODULE = "bernini.models.transformer_wan"
OFFICIAL_ATTN2_PROCESSOR_CLASS = "WanAttnProcessor2_0"
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE_SEAL = object()


class SourceOwnedRoleLocatorError(RuntimeError):
    """Raised instead of accepting ambiguous source/text/geometry authority."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceOwnedRoleLocatorError(
            f"value is not canonical finite JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SourceOwnedRoleLocatorError("text hash input must be non-empty UTF-8 text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor values together with their exact logical dtype/shape.

    The bytes are taken from a detached contiguous logical copy, not from the
    entire backing storage.  Consequently a full tensor and a rank-local view
    cannot collide merely because they share one allocation.
    """

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise SourceOwnedRoleLocatorError("tensor digest requires a material tensor")
    try:
        source = value.detach()
        logical = torch.empty(
            tuple(int(item) for item in source.shape),
            dtype=source.dtype,
            device="cpu",
        )
        logical.copy_(source, non_blocking=False)
        header = canonical_json_bytes(
            {"dtype": str(logical.dtype), "shape": list(_shape(logical))}
        )
        raw = logical.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    except Exception as error:
        raise SourceOwnedRoleLocatorError("cannot materialize tensor digest") from error
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


def tokenizer_tree_sha256(tokenizer_dir: str | Path) -> str:
    """Return the pinned tokenizer-tree digest used by the v15 assets.

    Definition: recursively enumerate every plain file below ``tokenizer_dir``
    in relative POSIX path order; emit ``{path,size,sha256}`` rows; canonical
    JSON encode the row list; SHA-256 those UTF-8 bytes.  Symlinks and an
    incomplete/extended tree fail closed.
    """

    root = Path(tokenizer_dir)
    if not root.is_dir() or root.is_symlink():
        raise SourceOwnedRoleLocatorError("tokenizer directory is missing or symlinked")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    relative = tuple(path.relative_to(root).as_posix() for path in paths)
    if relative != PINNED_TOKENIZER_FILES or any(path.is_symlink() for path in paths):
        raise SourceOwnedRoleLocatorError("tokenizer tree file registry differs")
    rows = []
    for path, name in zip(paths, relative):
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise SourceOwnedRoleLocatorError("cannot read tokenizer tree") from error
        rows.append(
            {
                "path": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _exact_int(value: Any, *, label: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceOwnedRoleLocatorError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise SourceOwnedRoleLocatorError(f"{label} must be >= {minimum}")
    return value


def _exact_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SourceOwnedRoleLocatorError(f"{label} must be a lowercase SHA-256")
    return value


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise SourceOwnedRoleLocatorError("expected a torch.Tensor")
    return tuple(int(item) for item in value.shape)


def _one_length(value: Any, *, label: str) -> int:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value]
    if len(values) != 1:
        raise SourceOwnedRoleLocatorError(f"{label} must describe batch size one")
    item = values[0]
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) != 1:
            raise SourceOwnedRoleLocatorError(f"{label} nested length differs")
        item = item[0]
    try:
        integer = int(item)
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as error:
        raise SourceOwnedRoleLocatorError(f"{label} must contain one integer") from error
    if not math.isfinite(numeric) or numeric != float(integer) or integer <= 0:
        raise SourceOwnedRoleLocatorError(f"{label} must contain one positive integer")
    return integer


@dataclass(frozen=True)
class SourceVisualGeometry:
    """The only supported source-video token layout: phase-major 21 x H x W."""

    height: int
    width: int
    phases: int = LATENT_PHASES

    def __post_init__(self) -> None:
        height = _exact_int(self.height, label="source latent height", minimum=1)
        width = _exact_int(self.width, label="source latent width", minimum=1)
        phases = _exact_int(self.phases, label="source latent phases", minimum=1)
        if phases != LATENT_PHASES:
            raise SourceOwnedRoleLocatorError("source video must have exactly 21 phases")
        if height * width < 2:
            raise SourceOwnedRoleLocatorError("source spatial grid must contain >=2 sites")

    @property
    def spatial_tokens(self) -> int:
        return self.height * self.width

    @property
    def global_tokens(self) -> int:
        return self.phases * self.spatial_tokens

    def reshape_global(self, value: torch.Tensor, *, leading: int) -> torch.Tensor:
        if _shape(value) != (leading, self.global_tokens):
            raise SourceOwnedRoleLocatorError(
                "global affinity does not satisfy role x 21 x H x W geometry"
            )
        return value.reshape(leading, self.phases, self.height, self.width)


@dataclass(frozen=True)
class UlyssesVisualShard:
    """Append-pad/contiguous rank-local layout used by Bernini Ulysses."""

    geometry: SourceVisualGeometry
    rank: int
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, SourceVisualGeometry):
            raise SourceOwnedRoleLocatorError("Ulysses shard lacks source geometry")
        size = _exact_int(self.size, label="Ulysses size", minimum=1)
        rank = _exact_int(self.rank, label="Ulysses rank", minimum=0)
        if rank >= size:
            raise SourceOwnedRoleLocatorError("Ulysses rank lies outside its group")
        if size not in (1, 2, 4, 8):
            raise SourceOwnedRoleLocatorError("only SP1/SP2/SP4/SP8 are registered")

    @property
    def padded_local_tokens(self) -> int:
        return math.ceil(self.geometry.global_tokens / self.size)

    @property
    def global_start(self) -> int:
        return self.rank * self.padded_local_tokens

    @property
    def valid_local_tokens(self) -> int:
        return max(
            0,
            min(
                self.geometry.global_tokens,
                self.global_start + self.padded_local_tokens,
            )
            - self.global_start,
        )

    @property
    def global_stop(self) -> int:
        return self.global_start + self.valid_local_tokens


@dataclass(frozen=True)
class LockedRoleSpan:
    """A character and tokenizer identity lock for one source-caption role."""

    role: str
    kind: str
    protected: bool
    substring: str
    char_start: int
    char_end: int
    substring_sha256: str
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    span_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or _ROLE_RE.fullmatch(self.role) is None:
            raise SourceOwnedRoleLocatorError("role name is invalid")
        if not isinstance(self.kind, str) or _ROLE_RE.fullmatch(self.kind) is None:
            raise SourceOwnedRoleLocatorError("role kind is invalid")
        if not isinstance(self.protected, bool):
            raise SourceOwnedRoleLocatorError("protected must be boolean")
        if not isinstance(self.substring, str) or not self.substring or "\x00" in self.substring:
            raise SourceOwnedRoleLocatorError("role substring is invalid")
        start = _exact_int(self.char_start, label="role char_start", minimum=0)
        end = _exact_int(self.char_end, label="role char_end", minimum=1)
        if end <= start or end - start != len(self.substring):
            raise SourceOwnedRoleLocatorError("role character interval differs")
        _exact_sha(self.substring_sha256, label="substring_sha256")
        if text_sha256(self.substring) != self.substring_sha256:
            raise SourceOwnedRoleLocatorError("role substring hash differs")
        token_start = _exact_int(self.token_start, label="role token_start", minimum=0)
        token_end = _exact_int(self.token_end, label="role token_end", minimum=1)
        if token_end <= token_start:
            raise SourceOwnedRoleLocatorError("role token interval is empty")
        if (
            not isinstance(self.token_ids, tuple)
            or len(self.token_ids) != token_end - token_start
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.token_ids)
        ):
            raise SourceOwnedRoleLocatorError("role token IDs differ from interval")
        _exact_sha(self.token_ids_sha256, label="token_ids_sha256")
        if object_sha256(list(self.token_ids)) != self.token_ids_sha256:
            raise SourceOwnedRoleLocatorError("role token-ID hash differs")
        _exact_sha(self.span_sha256, label="span_sha256")
        if object_sha256(self._span_payload()) != self.span_sha256:
            raise SourceOwnedRoleLocatorError("role span hash differs")

    def _span_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "kind": self.kind,
            "protected": self.protected,
            "substring": self.substring,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "substring_sha256": self.substring_sha256,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_ids": list(self.token_ids),
            "token_ids_sha256": self.token_ids_sha256,
        }

    @classmethod
    def create(
        cls,
        *,
        role: str,
        kind: str,
        protected: bool,
        substring: str,
        char_start: int,
        token_start: int,
        token_ids: Sequence[int],
    ) -> "LockedRoleSpan":
        token_tuple = tuple(token_ids)
        payload = {
            "role": role,
            "kind": kind,
            "protected": protected,
            "substring": substring,
            "char_start": char_start,
            "char_end": char_start + len(substring),
            "substring_sha256": text_sha256(substring),
            "token_start": token_start,
            "token_end": token_start + len(token_tuple),
            "token_ids": list(token_tuple),
            "token_ids_sha256": object_sha256(list(token_tuple)),
        }
        constructor = dict(payload)
        constructor["token_ids"] = token_tuple
        return cls(**constructor, span_sha256=object_sha256(payload))

    def as_dict(self) -> dict[str, Any]:
        return {**self._span_payload(), "span_sha256": self.span_sha256}


@dataclass(frozen=True)
class SourceRoleEventSpec:
    event_id: str
    source_iid: str
    source_caption: str
    source_caption_sha256: str
    model_text: str
    model_text_sha256: str
    source_caption_char_start: int
    tokenizer_tree_sha256: str
    roles: tuple[LockedRoleSpan, ...]
    event_sha256: str

    def __post_init__(self) -> None:
        for label, value in (("event_id", self.event_id), ("source_iid", self.source_iid)):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise SourceOwnedRoleLocatorError(f"{label} is invalid")
        if not isinstance(self.source_caption, str) or not self.source_caption:
            raise SourceOwnedRoleLocatorError("source caption is empty")
        if text_sha256(self.source_caption) != _exact_sha(
            self.source_caption_sha256, label="source_caption_sha256"
        ):
            raise SourceOwnedRoleLocatorError("source caption hash differs")
        if not isinstance(self.model_text, str) or not self.model_text:
            raise SourceOwnedRoleLocatorError("model text is empty")
        if text_sha256(self.model_text) != _exact_sha(
            self.model_text_sha256, label="model_text_sha256"
        ):
            raise SourceOwnedRoleLocatorError("model text hash differs")
        offset = _exact_int(
            self.source_caption_char_start,
            label="source_caption_char_start",
            minimum=0,
        )
        if self.model_text[offset : offset + len(self.source_caption)] != self.source_caption:
            raise SourceOwnedRoleLocatorError("model text does not contain caption at locked offset")
        _exact_sha(self.tokenizer_tree_sha256, label="tokenizer_tree_sha256")
        if not isinstance(self.roles, tuple) or len(self.roles) < 2:
            raise SourceOwnedRoleLocatorError("event must register at least two roles")
        if len({item.role for item in self.roles}) != len(self.roles):
            raise SourceOwnedRoleLocatorError("event role names are not unique")
        if not any(item.protected for item in self.roles):
            raise SourceOwnedRoleLocatorError("event must register protected roles")
        intervals = sorted((item.char_start, item.char_end, item.role) for item in self.roles)
        for index, (start, end, role) in enumerate(intervals):
            if self.model_text[start:end] != next(item.substring for item in self.roles if item.role == role):
                raise SourceOwnedRoleLocatorError(f"locked substring differs for role {role}")
            if index and start < intervals[index - 1][1]:
                raise SourceOwnedRoleLocatorError("role character spans overlap")
        token_intervals = sorted((item.token_start, item.token_end) for item in self.roles)
        for index, (start, _end) in enumerate(token_intervals):
            if index and start < token_intervals[index - 1][1]:
                raise SourceOwnedRoleLocatorError("role token spans overlap")
        _exact_sha(self.event_sha256, label="event_sha256")
        if object_sha256(self._event_payload()) != self.event_sha256:
            raise SourceOwnedRoleLocatorError("event role specification hash differs")

    def _event_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_iid": self.source_iid,
            "source_caption": self.source_caption,
            "source_caption_sha256": self.source_caption_sha256,
            "model_text": self.model_text,
            "model_text_sha256": self.model_text_sha256,
            "source_caption_char_start": self.source_caption_char_start,
            "tokenizer_tree_sha256": self.tokenizer_tree_sha256,
            "roles": [item.as_dict() for item in self.roles],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._event_payload(), "event_sha256": self.event_sha256}

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.roles)

    @property
    def protected_role_names(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.roles if item.protected)


def _tokenizer_fields(encoded: Any) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    def field(name: str) -> Any:
        value = getattr(encoded, name, None)
        if isinstance(encoded, Mapping):
            value = encoded.get(name, value)
        return value

    ids = field("input_ids")
    mask = field("attention_mask")
    offsets = field("offset_mapping")
    for name, value in (("input_ids", ids), ("attention_mask", mask), ("offset_mapping", offsets)):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if name == "input_ids":
            ids = value
        elif name == "attention_mask":
            mask = value
        else:
            offsets = value
    if isinstance(ids, Sequence) and len(ids) == 1 and isinstance(ids[0], Sequence):
        ids = list(ids[0])
    if isinstance(mask, Sequence) and len(mask) == 1 and isinstance(mask[0], Sequence):
        mask = list(mask[0])
    if (
        isinstance(offsets, Sequence)
        and len(offsets) == 1
        and isinstance(offsets[0], Sequence)
        and offsets[0]
        and isinstance(offsets[0][0], Sequence)
    ):
        offsets = list(offsets[0])
    if (
        not isinstance(ids, Sequence)
        or not isinstance(mask, Sequence)
        or not isinstance(offsets, Sequence)
        or len(ids) != len(mask)
        or len(ids) != len(offsets)
        or not ids
    ):
        raise SourceOwnedRoleLocatorError("tokenizer must return aligned IDs/mask/offsets")
    ids_list = [int(item) for item in ids]
    mask_list = [int(item) for item in mask]
    offset_list: list[tuple[int, int]] = []
    for value in offsets:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise SourceOwnedRoleLocatorError("token offset mapping is malformed")
        start, end = (int(value[0]), int(value[1]))
        if start < 0 or end < start:
            raise SourceOwnedRoleLocatorError("token offset lies outside its domain")
        offset_list.append((start, end))
    if any(item not in (0, 1) for item in mask_list):
        raise SourceOwnedRoleLocatorError("tokenizer attention mask is not binary")
    return ids_list, mask_list, offset_list


def resolve_exact_substring_token_span(
    tokenizer: Any,
    *,
    model_text: str,
    role: str,
    kind: str,
    protected: bool,
    substring: str,
    expected_char_start: Optional[int] = None,
    expected_lock: Optional[LockedRoleSpan] = None,
) -> LockedRoleSpan:
    """Resolve one unique substring and lock its exact tokenizer interval.

    A token that crosses either substring boundary is rejected.  Every
    non-whitespace character in the substring must be covered by at least one
    selected token offset.  Long inputs are rejected before a role can be
    silently truncated at Bernini's 512-token boundary.
    """

    if not callable(tokenizer):
        raise SourceOwnedRoleLocatorError("tokenizer must be callable")
    if not isinstance(model_text, str) or not model_text or "\x00" in model_text:
        raise SourceOwnedRoleLocatorError("model text is invalid")
    if not isinstance(substring, str) or not substring or "\x00" in substring:
        raise SourceOwnedRoleLocatorError("role substring is invalid")
    start = model_text.find(substring)
    if start < 0:
        raise SourceOwnedRoleLocatorError("role substring is missing from model text")
    if model_text.find(substring, start + 1) >= 0:
        raise SourceOwnedRoleLocatorError("role substring is not unique")
    if expected_char_start is not None and start != _exact_int(
        expected_char_start, label="expected_char_start", minimum=0
    ):
        raise SourceOwnedRoleLocatorError("role character offset changed")
    end = start + len(substring)
    try:
        encoded = tokenizer(
            model_text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except Exception as error:
        raise SourceOwnedRoleLocatorError("tokenizer cannot provide exact offsets") from error
    ids, mask, offsets = _tokenizer_fields(encoded)
    active = sum(mask)
    if (
        active <= 0
        or active > MAX_TEXT_TOKENS
        or any(mask[index] == 0 for index in range(active))
        or any(mask[index] == 1 for index in range(active, len(mask)))
    ):
        raise SourceOwnedRoleLocatorError("tokenizer active-token/truncation contract differs")
    if len(ids) > MAX_TEXT_TOKENS and active == len(ids):
        raise SourceOwnedRoleLocatorError("model text exceeds Bernini's token budget")
    selected: list[int] = []
    coverage: set[int] = set()
    for index, (token_start, token_end) in enumerate(offsets[:active]):
        if token_start == token_end:
            continue
        intersects = token_start < end and token_end > start
        if not intersects:
            continue
        if (
            (token_start < start and model_text[token_start:start].strip())
            or (token_end > end and model_text[end:token_end].strip())
        ):
            raise SourceOwnedRoleLocatorError("a tokenizer token crosses the role boundary")
        selected.append(index)
        coverage.update(range(token_start, token_end))
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise SourceOwnedRoleLocatorError("role token span is empty or non-contiguous")
    required = {index for index in range(start, end) if not model_text[index].isspace()}
    if not required.issubset(coverage):
        raise SourceOwnedRoleLocatorError("token offsets do not cover the exact role text")
    lock = LockedRoleSpan.create(
        role=role,
        kind=kind,
        protected=protected,
        substring=substring,
        char_start=start,
        token_start=selected[0],
        token_ids=ids[selected[0] : selected[-1] + 1],
    )
    if expected_lock is not None and lock != expected_lock:
        raise SourceOwnedRoleLocatorError("runtime role tokenization differs from lock")
    return lock


def validate_runtime_tokenization(
    tokenizer: Any,
    spec: SourceRoleEventSpec,
) -> tuple[LockedRoleSpan, ...]:
    """Re-tokenize a registered event and prove every stored lock exactly."""

    if not isinstance(spec, SourceRoleEventSpec):
        raise SourceOwnedRoleLocatorError("runtime tokenization requires an event spec")
    return tuple(
        resolve_exact_substring_token_span(
            tokenizer,
            model_text=spec.model_text,
            role=item.role,
            kind=item.kind,
            protected=item.protected,
            substring=item.substring,
            expected_char_start=item.char_start,
            expected_lock=item,
        )
        for item in spec.roles
    )


@dataclass(frozen=True)
class RuntimeTokenizationReceipt:
    """Proof that runtime text uses the exact tokenizer contract of the asset."""

    tokenizer_tree_sha256: str
    transformers_version: str
    fix_mistral_regex: bool
    input_ids_sha256: str
    attention_mask_sha256: str
    active_token_count: int

    def __post_init__(self) -> None:
        for label, value in (
            ("tokenizer_tree_sha256", self.tokenizer_tree_sha256),
            ("input_ids_sha256", self.input_ids_sha256),
            ("attention_mask_sha256", self.attention_mask_sha256),
        ):
            _exact_sha(value, label=label)
        if self.transformers_version != PINNED_TRANSFORMERS_VERSION:
            raise SourceOwnedRoleLocatorError("transformers runtime version differs")
        if self.fix_mistral_regex is not True:
            raise SourceOwnedRoleLocatorError("fix_mistral_regex must be exactly True")
        count = _exact_int(self.active_token_count, label="active token count", minimum=1)
        if count > MAX_TEXT_TOKENS:
            raise SourceOwnedRoleLocatorError("active token count exceeds Bernini limit")


def validate_pinned_tokenizer_runtime(
    tokenizer: Any,
    spec: SourceRoleEventSpec,
    *,
    tokenizer_dir: str | Path,
    transformers_version: str,
) -> RuntimeTokenizationReceipt:
    """Validate version, tokenizer bytes, regex fix, and every exact role span.

    This function is mandatory in :func:`bind_source_text_provenance`; callers
    cannot construct an observer invocation from a tensor alone.
    """

    if transformers_version != PINNED_TRANSFORMERS_VERSION:
        raise SourceOwnedRoleLocatorError(
            f"transformers must be exactly {PINNED_TRANSFORMERS_VERSION}"
        )
    if spec.tokenizer_tree_sha256 != PINNED_TOKENIZER_TREE_SHA256:
        raise SourceOwnedRoleLocatorError("event spec does not use the pinned tokenizer tree")
    try:
        installed_transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise SourceOwnedRoleLocatorError("transformers runtime is not installed") from error
    if installed_transformers_version != PINNED_TRANSFORMERS_VERSION:
        raise SourceOwnedRoleLocatorError(
            "installed transformers runtime differs from the pinned version"
        )
    if getattr(tokenizer, "padding_side", None) != "right":
        raise SourceOwnedRoleLocatorError("runtime tokenizer must use right padding")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, Mapping) or init_kwargs.get("fix_mistral_regex") is not True:
        raise SourceOwnedRoleLocatorError("runtime tokenizer lacks fix_mistral_regex=True")
    tree_digest = tokenizer_tree_sha256(tokenizer_dir)
    if tree_digest != spec.tokenizer_tree_sha256:
        raise SourceOwnedRoleLocatorError("runtime tokenizer tree differs from event lock")
    validate_runtime_tokenization(tokenizer, spec)
    try:
        encoded = tokenizer(
            spec.model_text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except Exception as error:
        raise SourceOwnedRoleLocatorError("cannot fingerprint runtime tokenization") from error
    ids, mask, _offsets = _tokenizer_fields(encoded)
    active = sum(mask)
    if (
        active <= 0
        or active > MAX_TEXT_TOKENS
        or any(mask[index] == 0 for index in range(active))
        or any(mask[index] == 1 for index in range(active, len(mask)))
    ):
        raise SourceOwnedRoleLocatorError("runtime token mask is truncated/non-prefix")
    if any(item.token_end > active for item in spec.roles):
        raise SourceOwnedRoleLocatorError("runtime tokenization truncates a locked role")
    return RuntimeTokenizationReceipt(
        tokenizer_tree_sha256=tree_digest,
        transformers_version=transformers_version,
        fix_mistral_regex=True,
        input_ids_sha256=object_sha256(ids),
        attention_mask_sha256=object_sha256(mask),
        active_token_count=active,
    )


def load_role_span_asset(path: str | Path) -> tuple[SourceRoleEventSpec, ...]:
    asset_path = Path(path)
    try:
        raw = json.loads(asset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceOwnedRoleLocatorError("cannot load role-span asset") from error
    required_top = {
        "schema_version",
        "purpose",
        "status",
        "route_authorized",
        "training_authorized",
        "role_semantics",
        "source_authority_only",
        "forbidden_inputs",
        "transformers_version",
        "fix_mistral_regex",
        "tokenizer_tree_sha256",
        "event_count",
        "events",
        "asset_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required_top:
        raise SourceOwnedRoleLocatorError("role-span asset fields differ")
    if raw["schema_version"] != ASSET_SCHEMA_VERSION:
        raise SourceOwnedRoleLocatorError("role-span asset schema differs")
    if raw["source_authority_only"] is not True:
        raise SourceOwnedRoleLocatorError("role-span asset is not source-authoritative")
    if (
        raw["status"] != "observer_only_diagnostic_not_route"
        or raw["route_authorized"] is not False
        or raw["training_authorized"] is not False
        or not isinstance(raw["role_semantics"], str)
        or not raw["role_semantics"]
    ):
        raise SourceOwnedRoleLocatorError("role-span asset is not observer-only")
    if raw["forbidden_inputs"] != ["action_anchor", "appearance_donor", "anchor_text"]:
        raise SourceOwnedRoleLocatorError("role-span forbidden-input registry differs")
    if (
        raw["transformers_version"] != PINNED_TRANSFORMERS_VERSION
        or raw["fix_mistral_regex"] is not True
    ):
        raise SourceOwnedRoleLocatorError("role-span tokenizer runtime lock differs")
    tokenizer_sha = _exact_sha(raw["tokenizer_tree_sha256"], label="tokenizer_tree_sha256")
    if tokenizer_sha != PINNED_TOKENIZER_TREE_SHA256:
        raise SourceOwnedRoleLocatorError("role-span asset tokenizer is not the pinned tree")
    payload = dict(raw)
    asset_sha = payload.pop("asset_sha256")
    if object_sha256(payload) != _exact_sha(asset_sha, label="asset_sha256"):
        raise SourceOwnedRoleLocatorError("role-span asset hash differs")
    rows = raw["events"]
    if not isinstance(rows, list) or raw["event_count"] != len(rows) or len(rows) != 4:
        raise SourceOwnedRoleLocatorError("role-span asset must contain exact complex4")
    specs: list[SourceRoleEventSpec] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SourceOwnedRoleLocatorError("event row must be an object")
        expected = {
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
        if set(row) != expected or row["tokenizer_tree_sha256"] != tokenizer_sha:
            raise SourceOwnedRoleLocatorError("event role-span fields/tokenizer differ")
        roles = tuple(
            LockedRoleSpan(
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
        specs.append(
            SourceRoleEventSpec(
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
        )
    if tuple(item.event_id for item in specs) != (
        "pour-liquid-into-cup",
        "twist-pull-mushroom",
        "close-door-then-drawer",
        "players-contact-then-separate",
    ):
        raise SourceOwnedRoleLocatorError("complex4 event order differs")
    return tuple(specs)


@dataclass(frozen=True)
class SourceTextProvenance:
    """Pre-SP source-text authority retained across replicated SP text K.

    The full conditioned tensor is retained deliberately.  After
    ``prepare_inputs_for_sp`` official Bernini Ulysses shards visual Q but
    leaves full text K replicated.  The attn2 observer therefore accepts only
    that exact root tensor or a logical prefix sharing its allocation, offset,
    dtype and stride.  Equal-valued clones and unrelated views fail closed
    without relying on Python object identity.
    """

    event_id: str
    tokenization: RuntimeTokenizationReceipt
    raw_source_text_hidden_states: torch.Tensor
    conditioned_source_text_hidden_states: torch.Tensor
    renderer_text_length: int
    raw_embedding_sha256: str
    conditioned_embedding_sha256: str
    conditioned_view_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _PROVENANCE_SEAL:
            raise SourceOwnedRoleLocatorError(
                "source text provenance must come from bind_source_text_provenance"
            )
        if not isinstance(self.event_id, str) or not self.event_id:
            raise SourceOwnedRoleLocatorError("source text provenance event is invalid")
        if not isinstance(self.tokenization, RuntimeTokenizationReceipt):
            raise SourceOwnedRoleLocatorError("source text provenance lacks tokenizer proof")
        raw = self.raw_source_text_hidden_states
        conditioned = self.conditioned_source_text_hidden_states
        for label, tensor in (("raw", raw), ("conditioned", conditioned)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 3
                or int(tensor.shape[0]) != 1
                or int(tensor.shape[1]) != MAX_TEXT_TOKENS
                or int(tensor.shape[2]) <= 0
                or tensor.device.type == "meta"
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise SourceOwnedRoleLocatorError(
                    f"{label} source text must be detached finite [1,512,D]"
                )
        length = _exact_int(
            self.renderer_text_length, label="renderer text length", minimum=1
        )
        if length > MAX_TEXT_TOKENS:
            raise SourceOwnedRoleLocatorError("renderer text length exceeds Bernini limit")
        if self.tokenization.active_token_count > length:
            raise SourceOwnedRoleLocatorError("renderer text view truncates active source text")
        for label, digest in (
            ("raw_embedding_sha256", self.raw_embedding_sha256),
            ("conditioned_embedding_sha256", self.conditioned_embedding_sha256),
            ("conditioned_view_sha256", self.conditioned_view_sha256),
        ):
            _exact_sha(digest, label=label)
        if tensor_sha256(raw) != self.raw_embedding_sha256:
            raise SourceOwnedRoleLocatorError("raw source text digest differs")
        if tensor_sha256(conditioned) != self.conditioned_embedding_sha256:
            raise SourceOwnedRoleLocatorError("conditioned source text digest differs")
        if tensor_sha256(conditioned[:, :length]) != self.conditioned_view_sha256:
            raise SourceOwnedRoleLocatorError("conditioned source text view digest differs")

    def validate_rank_local_view(
        self,
        value: torch.Tensor,
        *,
        require_sp_view: bool,
    ) -> None:
        full = self.conditioned_source_text_hidden_states
        length = self.renderer_text_length
        if (
            not isinstance(value, torch.Tensor)
            or _shape(value) != (1, length, int(full.shape[2]))
            or value.dtype != full.dtype
            or value.device != full.device
            or value.requires_grad
            or value.grad_fn is not None
            or value.storage_offset() != full.storage_offset()
            or tuple(value.stride()) != tuple(full.stride())
        ):
            raise SourceOwnedRoleLocatorError("rank-local source text view descriptor differs")
        try:
            value_storage = value.untyped_storage()
            full_storage = full.untyped_storage()
            same_storage = (
                value_storage.data_ptr() == full_storage.data_ptr()
                and value_storage.nbytes() == full_storage.nbytes()
                and getattr(value_storage, "_cdata", None)
                == getattr(full_storage, "_cdata", None)
            )
        except Exception as error:
            raise SourceOwnedRoleLocatorError("cannot authenticate source text storage") from error
        if not same_storage:
            raise SourceOwnedRoleLocatorError("attn2 text does not share pre-SP source storage")
        # `require_sp_view` preserves the call ABI, but must not require
        # `Tensor._base`: official Bernini prepare_inputs_for_sp shards visual
        # Q only and returns replicated full text K as the root tensor.  The
        # storage/offset/stride/value checks above are the actual provenance
        # boundary and continue to reject equal-valued clones.
        # Re-hash both the retained full tensor and the exact consumed view to
        # catch mutation between condition embedding and attn2 observation.
        if tensor_sha256(self.raw_source_text_hidden_states) != self.raw_embedding_sha256:
            raise SourceOwnedRoleLocatorError("raw source T5 embedding was mutated")
        if tensor_sha256(full) != self.conditioned_embedding_sha256:
            raise SourceOwnedRoleLocatorError("pre-SP source text was mutated")
        if tensor_sha256(value) != self.conditioned_view_sha256:
            raise SourceOwnedRoleLocatorError("rank-local source text values differ")

    def receipt_payload(self) -> dict[str, Any]:
        raw = self.raw_source_text_hidden_states
        conditioned = self.conditioned_source_text_hidden_states
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "tokenizer_tree_sha256": self.tokenization.tokenizer_tree_sha256,
            "transformers_version": self.tokenization.transformers_version,
            "fix_mistral_regex": self.tokenization.fix_mistral_regex,
            "input_ids_sha256": self.tokenization.input_ids_sha256,
            "attention_mask_sha256": self.tokenization.attention_mask_sha256,
            "active_token_count": self.tokenization.active_token_count,
            "renderer_text_length": self.renderer_text_length,
            "raw_shape": list(_shape(raw)),
            "raw_dtype": str(raw.dtype),
            "conditioned_shape": list(_shape(conditioned)),
            "conditioned_dtype": str(conditioned.dtype),
            "conditioned_stride": list(conditioned.stride()),
            "conditioned_storage_offset": conditioned.storage_offset(),
            "conditioned_storage_nbytes": conditioned.untyped_storage().nbytes(),
            "raw_embedding_sha256": self.raw_embedding_sha256,
            "conditioned_embedding_sha256": self.conditioned_embedding_sha256,
            "conditioned_view_sha256": self.conditioned_view_sha256,
            "binding": "pre_SP_storage_to_replicated_full_text_K",
        }

    @property
    def receipt_sha256(self) -> str:
        return object_sha256(self.receipt_payload())


def bind_source_text_provenance(
    *,
    tokenizer: Any,
    tokenizer_dir: str | Path,
    transformers_version: str,
    event_spec: SourceRoleEventSpec,
    raw_source_text_hidden_states: torch.Tensor,
    derive_conditioned_source_text: Callable[[torch.Tensor], torch.Tensor],
    renderer_text_length: int = MAX_TEXT_TOKENS,
) -> tuple[SourceTextProvenance, torch.Tensor]:
    """Authenticate source tokenization and derive the pre-SP text tensor.

    Call this after obtaining the real source T5 embedding and *before*
    ``prepare_inputs_for_sp``.  The supplied derivation callback is invoked
    internally under ``no_grad`` so a caller cannot substitute a separately
    prepared conditioned tensor.  The returned tensor is the only tensor whose
    storage may later appear at observed attn2 calls.
    """

    if not isinstance(event_spec, SourceRoleEventSpec):
        raise SourceOwnedRoleLocatorError("provenance requires an event spec")
    receipt = validate_pinned_tokenizer_runtime(
        tokenizer,
        event_spec,
        tokenizer_dir=tokenizer_dir,
        transformers_version=transformers_version,
    )
    raw = raw_source_text_hidden_states
    if (
        not isinstance(raw, torch.Tensor)
        or _shape(raw)[:2] != (1, MAX_TEXT_TOKENS)
        or raw.device.type == "meta"
        or raw.requires_grad
        or raw.grad_fn is not None
        or not bool(torch.isfinite(raw).all().item())
    ):
        raise SourceOwnedRoleLocatorError("raw source T5 embedding contract differs")
    if not callable(derive_conditioned_source_text):
        raise SourceOwnedRoleLocatorError("conditioned source derivation must be callable")
    with torch.no_grad():
        conditioned = derive_conditioned_source_text(raw)
    if (
        not isinstance(conditioned, torch.Tensor)
        or _shape(conditioned)[:2] != (1, MAX_TEXT_TOKENS)
        or conditioned.device != raw.device
        or conditioned.requires_grad
        or conditioned.grad_fn is not None
        or not bool(torch.isfinite(conditioned).all().item())
    ):
        raise SourceOwnedRoleLocatorError("conditioned source text derivation differs")
    length = _exact_int(renderer_text_length, label="renderer text length", minimum=1)
    if length > MAX_TEXT_TOKENS or any(item.token_end > length for item in event_spec.roles):
        raise SourceOwnedRoleLocatorError("renderer text length truncates a role")
    provenance = SourceTextProvenance(
        event_id=event_spec.event_id,
        tokenization=receipt,
        raw_source_text_hidden_states=raw,
        conditioned_source_text_hidden_states=conditioned,
        renderer_text_length=length,
        raw_embedding_sha256=tensor_sha256(raw),
        conditioned_embedding_sha256=tensor_sha256(conditioned),
        conditioned_view_sha256=tensor_sha256(conditioned[:, :length]),
        _seal=_PROVENANCE_SEAL,
    )
    return provenance, conditioned


@dataclass(frozen=True)
class SourceRoleObserverInvocation:
    """All authority needed by one source-only observer trajectory."""

    capture_bank: "SourceRoleCaptureBank"
    event_spec: SourceRoleEventSpec
    geometry: SourceVisualGeometry
    source_text_provenance: SourceTextProvenance
    step_index: int
    ulysses: UlyssesVisualShard

    def __post_init__(self) -> None:
        if not isinstance(self.capture_bank, SourceRoleCaptureBank):
            raise SourceOwnedRoleLocatorError("observer capture bank has wrong type")
        if not isinstance(self.event_spec, SourceRoleEventSpec):
            raise SourceOwnedRoleLocatorError("observer event spec has wrong type")
        if not isinstance(self.geometry, SourceVisualGeometry):
            raise SourceOwnedRoleLocatorError("observer geometry has wrong type")
        if not isinstance(self.ulysses, UlyssesVisualShard) or self.ulysses.geometry != self.geometry:
            raise SourceOwnedRoleLocatorError("observer Ulysses geometry differs")
        if (
            not isinstance(self.source_text_provenance, SourceTextProvenance)
            or self.source_text_provenance.event_id != self.event_spec.event_id
        ):
            raise SourceOwnedRoleLocatorError("observer source-text provenance differs")
        length = self.source_text_provenance.renderer_text_length
        if any(item.token_end > length for item in self.event_spec.roles):
            raise SourceOwnedRoleLocatorError("a locked role lies beyond source text length")
        _exact_int(self.step_index, label="observer step index", minimum=0)


_ACTIVE_OBSERVER: ContextVar[Optional[SourceRoleObserverInvocation]] = ContextVar(
    "bernini_source_owned_role_observer_v15", default=None
)


@contextmanager
def observe_source_roles(invocation: SourceRoleObserverInvocation) -> Iterator[None]:
    if not isinstance(invocation, SourceRoleObserverInvocation):
        raise SourceOwnedRoleLocatorError("observer context requires an invocation")
    if _ACTIVE_OBSERVER.get() is not None:
        raise SourceOwnedRoleLocatorError("nested source-role observers are forbidden")
    token: Token[Optional[SourceRoleObserverInvocation]] = _ACTIVE_OBSERVER.set(invocation)
    try:
        yield
    finally:
        _ACTIVE_OBSERVER.reset(token)


def current_source_role_observer() -> Optional[SourceRoleObserverInvocation]:
    return _ACTIVE_OBSERVER.get()


@dataclass(frozen=True)
class RoleAffinityShard:
    event_id: str
    source_text_provenance_sha256: str
    step_index: int
    block_index: int
    role_names: tuple[str, ...]
    layout: UlyssesVisualShard
    affinity: torch.Tensor
    null_affinity: torch.Tensor
    shuffled_affinity: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise SourceOwnedRoleLocatorError("affinity shard event is invalid")
        _exact_sha(
            self.source_text_provenance_sha256,
            label="source_text_provenance_sha256",
        )
        _exact_int(self.step_index, label="affinity step", minimum=0)
        _exact_int(self.block_index, label="affinity block", minimum=0)
        if (
            not isinstance(self.role_names, tuple)
            or not self.role_names
            or len(set(self.role_names)) != len(self.role_names)
            or any(_ROLE_RE.fullmatch(item) is None for item in self.role_names)
        ):
            raise SourceOwnedRoleLocatorError("affinity role order is invalid")
        if not isinstance(self.layout, UlyssesVisualShard):
            raise SourceOwnedRoleLocatorError("affinity shard layout is invalid")
        contracts = (
            (self.affinity, (len(self.role_names), self.layout.valid_local_tokens)),
            (self.null_affinity, (self.layout.valid_local_tokens,)),
            (self.shuffled_affinity, (len(self.role_names), self.layout.valid_local_tokens)),
        )
        for tensor, expected_shape in contracts:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not tensor.is_contiguous()
                or _shape(tensor) != expected_shape
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise SourceOwnedRoleLocatorError("affinity/control shard contract differs")
        if any(tensor.device != self.affinity.device for tensor, _shape_value in contracts):
            raise SourceOwnedRoleLocatorError("affinity/control shard devices differ")

    def padded_collective_tensor(self) -> torch.Tensor:
        roles = len(self.role_names)
        # Explicit ABI rows: [real role rows, one null row, cyclic-shuffle rows].
        result = self.affinity.new_zeros((2 * roles + 1, self.layout.padded_local_tokens))
        if self.layout.valid_local_tokens:
            stop = self.layout.valid_local_tokens
            result[:roles, :stop].copy_(self.affinity)
            result[roles, :stop].copy_(self.null_affinity)
            result[roles + 1 :, :stop].copy_(self.shuffled_affinity)
        return result

    def collective_metadata(self) -> dict[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "source_text_provenance_sha256": self.source_text_provenance_sha256,
            "step_index": self.step_index,
            "block_index": self.block_index,
            "role_names": list(self.role_names),
            "height": self.layout.geometry.height,
            "width": self.layout.geometry.width,
            "phases": self.layout.geometry.phases,
            "rank": self.layout.rank,
            "size": self.layout.size,
            "global_start": self.layout.global_start,
            "global_stop": self.layout.global_stop,
            "padded_local_tokens": self.layout.padded_local_tokens,
            "valid_local_tokens": self.layout.valid_local_tokens,
            "collective_channels": {
                "real": [0, len(self.role_names)],
                "null": len(self.role_names),
                "cyclic_shuffled": [len(self.role_names) + 1, 2 * len(self.role_names) + 1],
            },
        }
        return {**value, "metadata_sha256": object_sha256(value)}

    @classmethod
    def from_collective(
        cls,
        tensor: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> "RoleAffinityShard":
        required = {
            "schema_version",
            "event_id",
            "source_text_provenance_sha256",
            "step_index",
            "block_index",
            "role_names",
            "height",
            "width",
            "phases",
            "rank",
            "size",
            "global_start",
            "global_stop",
            "padded_local_tokens",
            "valid_local_tokens",
            "collective_channels",
            "metadata_sha256",
        }
        if not isinstance(metadata, Mapping) or set(metadata) != required:
            raise SourceOwnedRoleLocatorError("collective shard metadata fields differ")
        payload = dict(metadata)
        digest = payload.pop("metadata_sha256", None)
        if object_sha256(payload) != _exact_sha(digest, label="metadata_sha256"):
            raise SourceOwnedRoleLocatorError("collective shard metadata hash differs")
        if payload.pop("schema_version", None) != SCHEMA_VERSION:
            raise SourceOwnedRoleLocatorError("collective shard metadata schema differs")
        geometry = SourceVisualGeometry(
            height=metadata["height"], width=metadata["width"], phases=metadata["phases"]
        )
        layout = UlyssesVisualShard(
            geometry=geometry, rank=metadata["rank"], size=metadata["size"]
        )
        if (
            metadata["global_start"] != layout.global_start
            or metadata["global_stop"] != layout.global_stop
            or metadata["padded_local_tokens"] != layout.padded_local_tokens
            or metadata["valid_local_tokens"] != layout.valid_local_tokens
            or metadata["collective_channels"]
            != {
                "real": [0, len(metadata["role_names"])],
                "null": len(metadata["role_names"]),
                "cyclic_shuffled": [
                    len(metadata["role_names"]) + 1,
                    2 * len(metadata["role_names"]) + 1,
                ],
            }
            or _shape(tensor)
            != (2 * len(metadata["role_names"]) + 1, layout.padded_local_tokens)
        ):
            raise SourceOwnedRoleLocatorError("collective shard geometry differs")
        roles = len(metadata["role_names"])
        stop = layout.valid_local_tokens
        return cls(
            event_id=metadata["event_id"],
            source_text_provenance_sha256=metadata["source_text_provenance_sha256"],
            step_index=metadata["step_index"],
            block_index=metadata["block_index"],
            role_names=tuple(metadata["role_names"]),
            layout=layout,
            affinity=tensor[:roles, :stop].detach().float().contiguous(),
            null_affinity=tensor[roles, :stop].detach().float().contiguous(),
            shuffled_affinity=tensor[roles + 1 :, :stop].detach().float().contiguous(),
        )


class SourceRoleCaptureBank:
    def __init__(self, selected_block_indices: Sequence[int]) -> None:
        indices = tuple(selected_block_indices)
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                or item >= EXPECTED_BLOCK_COUNT
                for item in indices
            )
        ):
            raise SourceOwnedRoleLocatorError("selected blocks must be an increasing subset of 0..29")
        self.selected_block_indices = indices
        self._shards: dict[tuple[str, int, int, int], RoleAffinityShard] = {}
        self.capture_count = 0

    def capture(self, shard: RoleAffinityShard) -> None:
        if not isinstance(shard, RoleAffinityShard):
            raise SourceOwnedRoleLocatorError("capture requires a role-affinity shard")
        if shard.block_index not in self.selected_block_indices:
            raise SourceOwnedRoleLocatorError("affinity block is outside capture scope")
        key = (shard.event_id, shard.step_index, shard.block_index, shard.layout.rank)
        if key in self._shards:
            raise SourceOwnedRoleLocatorError("duplicate source-role affinity capture")
        self._shards[key] = shard
        self.capture_count += 1

    def shards_for(
        self, *, event_id: str, step_index: int, block_index: int
    ) -> tuple[RoleAffinityShard, ...]:
        rows = [
            value
            for (event, step, block, _rank), value in self._shards.items()
            if (event, step, block) == (event_id, step_index, block_index)
        ]
        return tuple(sorted(rows, key=lambda item: item.layout.rank))

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "selected_block_indices": list(self.selected_block_indices),
            "capture_count": self.capture_count,
            "stored_shards": len(self._shards),
            "implicit_collective_calls": 0,
        }


def _source_role_affinity(
    query: torch.Tensor,
    key: torch.Tensor,
    roles: Sequence[LockedRoleSpan],
    *,
    valid_local_tokens: int,
    active_source_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        query.ndim != 4
        or key.ndim != 4
        or int(query.shape[0]) != 1
        or int(key.shape[0]) != 1
        or int(query.shape[2]) != int(key.shape[2])
        or int(query.shape[3]) != int(key.shape[3])
        or int(query.shape[1]) < valid_local_tokens
        or valid_local_tokens <= 0
    ):
        raise SourceOwnedRoleLocatorError("official cross-attention Q/K geometry differs")
    if query.device != key.device or not all(bool(torch.isfinite(item).all().item()) for item in (query, key)):
        raise SourceOwnedRoleLocatorError("official cross-attention Q/K device/finiteness differs")
    maximum = max(item.token_end for item in roles)
    text_length = _exact_int(active_source_tokens, label="active source tokens", minimum=1)
    if maximum > text_length or text_length > int(key.shape[1]):
        raise SourceOwnedRoleLocatorError("locked role span exceeds projected text K")
    q = torch.nn.functional.normalize(
        query[:, :valid_local_tokens].detach().float(), dim=-1, eps=1e-12
    )
    rows: list[torch.Tensor] = []
    role_keys: list[torch.Tensor] = []
    for item in roles:
        role_tokens = key[:, item.token_start : item.token_end].detach().float()
        role_key = torch.nn.functional.normalize(role_tokens, dim=-1, eps=1e-12).mean(dim=1)
        role_key = torch.nn.functional.normalize(role_key, dim=-1, eps=1e-12)
        score = torch.einsum("blhd,bhd->blh", q, role_key).mean(dim=-1).squeeze(0)
        rows.append(score)
        role_keys.append(role_key)
    result = torch.stack(rows, dim=0).detach().float().contiguous()
    occupied = torch.zeros(text_length, dtype=torch.bool, device=key.device)
    for item in roles:
        occupied[item.token_start : item.token_end] = True
    null_indices = torch.nonzero(~occupied, as_tuple=False).flatten()
    if int(null_indices.numel()) == 0:
        raise SourceOwnedRoleLocatorError("no unclaimed source tokens exist for null control")
    null_tokens = key[:, null_indices].detach().float()
    null_key = torch.nn.functional.normalize(null_tokens, dim=-1, eps=1e-12).mean(dim=1)
    null_key = torch.nn.functional.normalize(null_key, dim=-1, eps=1e-12)
    null_affinity = (
        torch.einsum("blhd,bhd->blh", q, null_key)
        .mean(dim=-1)
        .squeeze(0)
        .detach()
        .float()
        .contiguous()
    )
    # Deterministic wrong-span control: role r is compared to the next locked
    # role's key.  It is semantics-preserving under distributed assembly and
    # cannot consume any anchor/donor state.
    shuffled_rows = []
    for role_index in range(len(role_keys)):
        wrong_key = role_keys[(role_index + 1) % len(role_keys)]
        shuffled_rows.append(
            torch.einsum("blhd,bhd->blh", q, wrong_key).mean(dim=-1).squeeze(0)
        )
    shuffled = torch.stack(shuffled_rows, dim=0).detach().float().contiguous()
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (result, null_affinity, shuffled)
    ):
        raise SourceOwnedRoleLocatorError("source role affinity/control is non-finite")
    return result, null_affinity, shuffled


def _validate_frozen_stateless_attention(attn: Any) -> None:
    """Reject training/stochastic/stateful official-attention projections."""

    if not isinstance(attn, torch.nn.Module):
        # Unit-level protocol doubles are allowed; the SP4 GPU probe requires
        # the runtime adapter to expose an actual frozen nn.Module.
        return
    if attn.training:
        raise SourceOwnedRoleLocatorError("observer attn2 must be in eval mode")
    if any(parameter.requires_grad for parameter in attn.parameters()):
        raise SourceOwnedRoleLocatorError("observer attn2 parameters must be frozen")
    for module in attn.modules():
        if module.training:
            raise SourceOwnedRoleLocatorError("observer projection submodule is training")
        if any(
            bool(getattr(module, attribute, {}))
            for attribute in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        ):
            raise SourceOwnedRoleLocatorError(
                "observer projection has stateful forward/backward hooks"
            )
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and float(module.p) != 0.0:
            # Eval dropout is deterministic, but the explicit ban keeps the
            # second _project_qkv call stateless under accidental mode changes.
            raise SourceOwnedRoleLocatorError("observer projection contains dropout")


class SourceOwnedRoleAttn2Observer:
    """Bit-exact-output wrapper around one official Bernini ``attn2`` processor."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        capture_bank: SourceRoleCaptureBank,
    ) -> None:
        processor_type = type(base_processor)
        if (
            not callable(base_processor)
            or not callable(getattr(base_processor, "_project_qkv", None))
            or processor_type.__module__ != OFFICIAL_ATTN2_PROCESSOR_MODULE
            or processor_type.__name__ != OFFICIAL_ATTN2_PROCESSOR_CLASS
        ):
            raise SourceOwnedRoleLocatorError("base attn2 processor lacks official _project_qkv")
        index = _exact_int(block_index, label="attn2 block index", minimum=0)
        if index not in capture_bank.selected_block_indices:
            raise SourceOwnedRoleLocatorError("attn2 block is outside capture scope")
        self.base_processor = base_processor
        self.block_index = index
        self.capture_bank = capture_bank
        self.base_calls = 0
        self.observer_calls = 0

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        batch_image_vae_seqlen=None,
        text_features_length=None,
        origin_hidden_states_seq_len: Optional[int] = None,
        split_hidden_states_seq_len: Optional[int] = None,
        cu_seqlens_q_cache=None,
        max_seqlen_q_cache=None,
        cu_seqlens_k_cross_cache=None,
        cu_seqlens_q_cross_cache=None,
        max_seqlen_k_cross_cache=None,
        max_seqlen_q_cross_cache=None,
    ) -> torch.Tensor:
        output = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            text_features_length=text_features_length,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            split_hidden_states_seq_len=split_hidden_states_seq_len,
            cu_seqlens_q_cache=cu_seqlens_q_cache,
            max_seqlen_q_cache=max_seqlen_q_cache,
            cu_seqlens_k_cross_cache=cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache=cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache=max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache=max_seqlen_q_cross_cache,
        )
        self.base_calls += 1
        invocation = current_source_role_observer()
        if invocation is None:
            return output
        if invocation.capture_bank is not self.capture_bank:
            raise SourceOwnedRoleLocatorError("processor/capture-bank ownership differs")
        _validate_frozen_stateless_attention(attn)
        if encoder_hidden_states is None or attention_mask is not None or rotary_emb is not None:
            raise SourceOwnedRoleLocatorError("source-role observer requires unmasked non-RoPE attn2")
        provenance = invocation.source_text_provenance
        provenance.validate_rank_local_view(
            encoder_hidden_states,
            require_sp_view=invocation.ulysses.size > 1,
        )
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or int(hidden_states.shape[0]) != 1
            or int(hidden_states.shape[1]) != invocation.ulysses.padded_local_tokens
            or hidden_states.device != encoder_hidden_states.device
            or hidden_states.dtype != encoder_hidden_states.dtype
            or not bool(torch.isfinite(hidden_states).all().item())
        ):
            raise SourceOwnedRoleLocatorError("source visual hidden shard contract differs")
        if _one_length(batch_image_vae_seqlen, label="batch_image_vae_seqlen") != invocation.geometry.global_tokens:
            raise SourceOwnedRoleLocatorError("source visual global length differs")
        if origin_hidden_states_seq_len is None or _exact_int(
            origin_hidden_states_seq_len, label="origin_hidden_states_seq_len", minimum=1
        ) != invocation.geometry.global_tokens:
            raise SourceOwnedRoleLocatorError("source visual origin length differs")
        if text_features_length is None or _one_length(
            text_features_length, label="text_features_length"
        ) != provenance.renderer_text_length:
            raise SourceOwnedRoleLocatorError("source text feature length differs")
        # Detached observation cannot attach a second autograd graph to the
        # model call.  It deliberately executes after base output is complete.
        # fork_rng additionally prevents an unexpected projection-side dropout
        # from advancing the caller's CPU/GPU RNG stream.
        fork_devices: list[int] = []
        if hidden_states.device.type == "cuda":
            device_index = hidden_states.device.index
            if device_index is None:
                device_index = int(torch.cuda.current_device())
            fork_devices.append(device_index)
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            with torch.no_grad():
                query, key, _value = self.base_processor._project_qkv(
                    attn,
                    hidden_states.detach(),
                    encoder_hidden_states.detach(),
                    None,
                    invocation.geometry.global_tokens,
                    True,
                )
                affinity, null_affinity, shuffled_affinity = _source_role_affinity(
                    query,
                    key,
                    invocation.event_spec.roles,
                    valid_local_tokens=invocation.ulysses.valid_local_tokens,
                    active_source_tokens=provenance.tokenization.active_token_count,
                )
        shard = RoleAffinityShard(
            event_id=invocation.event_spec.event_id,
            source_text_provenance_sha256=provenance.receipt_sha256,
            step_index=invocation.step_index,
            block_index=self.block_index,
            role_names=invocation.event_spec.role_names,
            layout=invocation.ulysses,
            affinity=affinity,
            null_affinity=null_affinity,
            shuffled_affinity=shuffled_affinity,
        )
        self.capture_bank.capture(shard)
        self.observer_calls += 1
        # Object identity, dtype, storage, values and autograd path are those
        # returned by the official processor.  No arithmetic touches output.
        return output

    def statistics(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "block_index": self.block_index,
            "base_calls": self.base_calls,
            "observer_calls": self.observer_calls,
            "output_modified": False,
            "implicit_collective_calls": 0,
        }


@dataclass(frozen=True)
class GlobalRoleAffinity:
    event_id: str
    source_text_provenance_sha256: str
    step_index: int
    block_index: int
    role_names: tuple[str, ...]
    geometry: SourceVisualGeometry
    affinity: torch.Tensor
    null_affinity: torch.Tensor
    shuffled_affinity: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise SourceOwnedRoleLocatorError("global affinity event is invalid")
        _exact_sha(
            self.source_text_provenance_sha256,
            label="source_text_provenance_sha256",
        )
        _exact_int(self.step_index, label="global affinity step", minimum=0)
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, int) or self.block_index < -1:
            raise SourceOwnedRoleLocatorError("global affinity block is invalid")
        if (
            not isinstance(self.role_names, tuple)
            or not self.role_names
            or len(set(self.role_names)) != len(self.role_names)
            or any(_ROLE_RE.fullmatch(item) is None for item in self.role_names)
        ):
            raise SourceOwnedRoleLocatorError("global affinity role order is invalid")
        if not isinstance(self.geometry, SourceVisualGeometry):
            raise SourceOwnedRoleLocatorError("global affinity source geometry is invalid")
        contracts = (
            (
                self.affinity,
                (len(self.role_names), self.geometry.phases, self.geometry.height, self.geometry.width),
            ),
            (
                self.null_affinity,
                (self.geometry.phases, self.geometry.height, self.geometry.width),
            ),
            (
                self.shuffled_affinity,
                (len(self.role_names), self.geometry.phases, self.geometry.height, self.geometry.width),
            ),
        )
        for tensor, expected_shape in contracts:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not tensor.is_contiguous()
                or _shape(tensor) != expected_shape
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise SourceOwnedRoleLocatorError("global role affinity/control contract differs")
        if any(tensor.device != self.affinity.device for tensor, _shape_value in contracts):
            raise SourceOwnedRoleLocatorError("global affinity/control devices differ")


def assemble_global_role_affinity(
    shards: Sequence[RoleAffinityShard],
) -> GlobalRoleAffinity:
    rows = tuple(shards)
    if not rows:
        raise SourceOwnedRoleLocatorError("no Ulysses affinity shards were supplied")
    first = rows[0]
    if not all(isinstance(item, RoleAffinityShard) for item in rows):
        raise SourceOwnedRoleLocatorError("global assembly received a non-shard")
    size = first.layout.size
    if len(rows) != size or tuple(item.layout.rank for item in rows) != tuple(range(size)):
        raise SourceOwnedRoleLocatorError("global assembly requires ordered exact Ulysses ranks")
    shared = (
        first.event_id,
        first.source_text_provenance_sha256,
        first.step_index,
        first.block_index,
        first.role_names,
        first.layout.geometry,
        first.layout.size,
        first.affinity.device,
    )
    if any(
        (
            item.event_id,
            item.source_text_provenance_sha256,
            item.step_index,
            item.block_index,
            item.role_names,
            item.layout.geometry,
            item.layout.size,
            item.affinity.device,
        )
        != shared
        for item in rows
    ):
        raise SourceOwnedRoleLocatorError("Ulysses affinity shards do not share one authority")
    expected_start = 0
    for item in rows:
        if item.layout.global_start != expected_start:
            raise SourceOwnedRoleLocatorError("Ulysses global intervals are not contiguous")
        expected_start = item.layout.global_stop
    if expected_start != first.layout.geometry.global_tokens:
        raise SourceOwnedRoleLocatorError("Ulysses affinity assembly is incomplete")
    flat = torch.cat([item.affinity for item in rows], dim=1).contiguous()
    null_flat = torch.cat([item.null_affinity for item in rows], dim=0).contiguous()
    shuffled_flat = torch.cat([item.shuffled_affinity for item in rows], dim=1).contiguous()
    shaped = first.layout.geometry.reshape_global(flat, leading=len(first.role_names)).contiguous()
    null_shaped = null_flat.reshape(
        first.layout.geometry.phases,
        first.layout.geometry.height,
        first.layout.geometry.width,
    ).contiguous()
    shuffled_shaped = first.layout.geometry.reshape_global(
        shuffled_flat, leading=len(first.role_names)
    ).contiguous()
    return GlobalRoleAffinity(
        event_id=first.event_id,
        source_text_provenance_sha256=first.source_text_provenance_sha256,
        step_index=first.step_index,
        block_index=first.block_index,
        role_names=first.role_names,
        geometry=first.layout.geometry,
        affinity=shaped,
        null_affinity=null_shaped,
        shuffled_affinity=shuffled_shaped,
    )


def _resolve_wan_transformer(model: Any) -> Any:
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            if len(blocks) != EXPECTED_BLOCK_COUNT:
                raise SourceOwnedRoleLocatorError(
                    f"Bernini-R transformer must have {EXPECTED_BLOCK_COUNT} blocks"
                )
            return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("diff_dec", "transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise SourceOwnedRoleLocatorError("cannot resolve the 30-block Wan transformer")


@dataclass
class SourceRoleObserverPatchHandle:
    transformer: Any
    block_indices: tuple[int, ...]
    processors: tuple[SourceOwnedRoleAttn2Observer, ...]
    original_processors: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        for index, processor in zip(self.block_indices, self.processors):
            if getattr(self.transformer.blocks[index].attn2, "processor", None) is not processor:
                raise SourceOwnedRoleLocatorError(
                    f"block {index} attn2 processor changed behind observer handle"
                )
        for index, original in zip(self.block_indices, self.original_processors):
            attn2 = self.transformer.blocks[index].attn2
            setter = getattr(attn2, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn2.processor = original
        self.restored = True

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "block_indices": list(self.block_indices),
            "attn2_only": True,
            "parameters_added": 0,
            "output_modified": False,
            "restored": self.restored,
            "processors": [item.statistics() for item in self.processors],
        }

    def __enter__(self) -> "SourceRoleObserverPatchHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def install_source_owned_role_observer(
    model: Any,
    *,
    capture_bank: SourceRoleCaptureBank,
) -> SourceRoleObserverPatchHandle:
    """Reversibly wrap exactly the bank-selected official ``attn2`` processors."""

    if not isinstance(capture_bank, SourceRoleCaptureBank):
        raise SourceOwnedRoleLocatorError("installation requires a source-role capture bank")
    transformer = _resolve_wan_transformer(model)
    originals: list[Any] = []
    processors: list[SourceOwnedRoleAttn2Observer] = []
    installed: list[int] = []
    try:
        for index in capture_bank.selected_block_indices:
            attn2 = transformer.blocks[index].attn2
            original = getattr(attn2, "processor", None)
            processor = SourceOwnedRoleAttn2Observer(
                original, block_index=index, capture_bank=capture_bank
            )
            setter = getattr(attn2, "set_processor", None)
            if callable(setter):
                setter(processor)
            else:
                attn2.processor = processor
            if getattr(attn2, "processor", None) is not processor:
                raise SourceOwnedRoleLocatorError("attn2 observer installation did not stick")
            originals.append(original)
            processors.append(processor)
            installed.append(index)
    except Exception:
        for index, original in zip(installed, originals):
            attn2 = transformer.blocks[index].attn2
            setter = getattr(attn2, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn2.processor = original
        raise
    return SourceRoleObserverPatchHandle(
        transformer=transformer,
        block_indices=tuple(installed),
        processors=tuple(processors),
        original_processors=tuple(originals),
    )


def aggregate_global_role_affinities(
    values: Sequence[GlobalRoleAffinity],
) -> GlobalRoleAffinity:
    rows = tuple(values)
    if not rows:
        raise SourceOwnedRoleLocatorError("no block affinities were supplied")
    first = rows[0]
    shared = (
        first.event_id,
        first.source_text_provenance_sha256,
        first.step_index,
        first.role_names,
        first.geometry,
        first.affinity.device,
    )
    if any(
        (
            item.event_id,
            item.source_text_provenance_sha256,
            item.step_index,
            item.role_names,
            item.geometry,
            item.affinity.device,
        )
        != shared
        for item in rows
    ):
        raise SourceOwnedRoleLocatorError("block affinities do not share one source authority")
    if len({item.block_index for item in rows}) != len(rows):
        raise SourceOwnedRoleLocatorError("block affinity list contains duplicates")
    stacked = torch.stack([item.affinity for item in rows], dim=0)
    aggregate = stacked.mean(dim=0).detach().float().contiguous()
    null_aggregate = (
        torch.stack([item.null_affinity for item in rows], dim=0)
        .mean(dim=0)
        .detach()
        .float()
        .contiguous()
    )
    shuffled_aggregate = (
        torch.stack([item.shuffled_affinity for item in rows], dim=0)
        .mean(dim=0)
        .detach()
        .float()
        .contiguous()
    )
    return GlobalRoleAffinity(
        event_id=first.event_id,
        source_text_provenance_sha256=first.source_text_provenance_sha256,
        step_index=first.step_index,
        block_index=-1,
        role_names=first.role_names,
        geometry=first.geometry,
        affinity=aggregate,
        null_affinity=null_aggregate,
        shuffled_affinity=shuffled_aggregate,
    )


@dataclass(frozen=True)
class RoleMaskPolicy:
    keep_fraction: float = 0.08
    minimum_spatial_std: float = 0.01
    minimum_absolute_affinity: float = 0.05
    minimum_null_margin: float = 0.03
    minimum_shuffled_margin: float = 0.03
    minimum_other_role_margin: float = 0.01
    minimum_pixels_per_role_phase: int = 1
    minimum_confident_phases: int = 3

    def __post_init__(self) -> None:
        for label, value in (
            ("keep_fraction", self.keep_fraction),
            ("minimum_spatial_std", self.minimum_spatial_std),
            ("minimum_absolute_affinity", self.minimum_absolute_affinity),
            ("minimum_null_margin", self.minimum_null_margin),
            ("minimum_shuffled_margin", self.minimum_shuffled_margin),
            ("minimum_other_role_margin", self.minimum_other_role_margin),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SourceOwnedRoleLocatorError(f"{label} must be finite")
        if not 0.0 < float(self.keep_fraction) <= 1.0:
            raise SourceOwnedRoleLocatorError("keep_fraction must be in (0,1]")
        if float(self.minimum_spatial_std) <= 0.0:
            raise SourceOwnedRoleLocatorError("minimum_spatial_std must be positive")
        for label, value in (
            ("minimum_pixels_per_role_phase", self.minimum_pixels_per_role_phase),
            ("minimum_confident_phases", self.minimum_confident_phases),
        ):
            _exact_int(value, label=label, minimum=1)
        if self.minimum_confident_phases > LATENT_PHASES:
            raise SourceOwnedRoleLocatorError("minimum_confident_phases exceeds 21")

    def as_dict(self) -> dict[str, Any]:
        return {
            "keep_fraction": float(self.keep_fraction),
            "minimum_spatial_std": float(self.minimum_spatial_std),
            "minimum_absolute_affinity": float(self.minimum_absolute_affinity),
            "minimum_null_margin": float(self.minimum_null_margin),
            "minimum_shuffled_margin": float(self.minimum_shuffled_margin),
            "minimum_other_role_margin": float(self.minimum_other_role_margin),
            "minimum_pixels_per_role_phase": self.minimum_pixels_per_role_phase,
            "minimum_confident_phases": self.minimum_confident_phases,
        }


@dataclass(frozen=True)
class SourceRoleMaskSet:
    event_id: str
    source_text_provenance_sha256: str
    step_index: int
    role_names: tuple[str, ...]
    protected_role_names: tuple[str, ...]
    geometry: SourceVisualGeometry
    masks: torch.Tensor
    protected_union: torch.Tensor
    phase0_masks: torch.Tensor
    confident_role_phases: torch.Tensor
    qualified: bool
    policy: RoleMaskPolicy
    affinity_sha256: str
    mask_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise SourceOwnedRoleLocatorError("role-mask event is invalid")
        _exact_sha(
            self.source_text_provenance_sha256,
            label="source_text_provenance_sha256",
        )
        _exact_int(self.step_index, label="role-mask step", minimum=0)
        if (
            not isinstance(self.role_names, tuple)
            or not self.role_names
            or len(set(self.role_names)) != len(self.role_names)
            or any(_ROLE_RE.fullmatch(item) is None for item in self.role_names)
        ):
            raise SourceOwnedRoleLocatorError("role-mask role order is invalid")
        if (
            not isinstance(self.protected_role_names, tuple)
            or not self.protected_role_names
            or len(set(self.protected_role_names)) != len(self.protected_role_names)
            or any(item not in self.role_names for item in self.protected_role_names)
        ):
            raise SourceOwnedRoleLocatorError("protected-role registry is invalid")
        if not isinstance(self.geometry, SourceVisualGeometry):
            raise SourceOwnedRoleLocatorError("role-mask source geometry is invalid")
        expected = (len(self.role_names), self.geometry.phases, self.geometry.height, self.geometry.width)
        if (
            not isinstance(self.masks, torch.Tensor)
            or self.masks.dtype != torch.bool
            or _shape(self.masks) != expected
        ):
            raise SourceOwnedRoleLocatorError("role mask geometry differs")
        if self.masks.requires_grad or self.masks.grad_fn is not None or not self.masks.is_contiguous():
            raise SourceOwnedRoleLocatorError("role masks must be detached contiguous bool")
        if bool((self.masks.sum(dim=0) > 1).any().item()):
            raise SourceOwnedRoleLocatorError("role masks overlap")
        counts = self.masks.reshape(len(self.role_names), self.geometry.phases, -1).sum(dim=2)
        if not isinstance(self.policy, RoleMaskPolicy):
            raise SourceOwnedRoleLocatorError("role-mask policy differs")
        expected_confident = counts >= self.policy.minimum_pixels_per_role_phase
        if (
            not isinstance(self.confident_role_phases, torch.Tensor)
            or self.confident_role_phases.dtype != torch.bool
            or not self.confident_role_phases.is_contiguous()
            or _shape(self.confident_role_phases)
            != (len(self.role_names), self.geometry.phases)
            or not torch.equal(self.confident_role_phases, expected_confident)
        ):
            raise SourceOwnedRoleLocatorError("confident role-phase registry differs")
        expected_qualified = bool(
            (expected_confident.sum(dim=1) >= self.policy.minimum_confident_phases)
            .all()
            .item()
        )
        if not isinstance(self.qualified, bool) or self.qualified != expected_qualified:
            raise SourceOwnedRoleLocatorError("role-mask qualification differs")
        if _shape(self.phase0_masks) != (
            len(self.role_names), self.geometry.height, self.geometry.width
        ) or not torch.equal(self.phase0_masks, self.masks[:, 0]):
            raise SourceOwnedRoleLocatorError("phase-zero role masks differ")
        protected_indices = [self.role_names.index(name) for name in self.protected_role_names]
        expected_protected = self.masks[protected_indices].any(dim=0)
        if self.protected_union.dtype != torch.bool or not torch.equal(self.protected_union, expected_protected):
            raise SourceOwnedRoleLocatorError("protected-role union differs")
        _exact_sha(self.affinity_sha256, label="affinity_sha256")
        _exact_sha(self.mask_sha256, label="mask_sha256")
        if tensor_sha256(self.masks) != self.mask_sha256:
            raise SourceOwnedRoleLocatorError("role-mask content hash differs")
        _exact_sha(self.receipt_sha256, label="mask receipt_sha256")
        if self.receipt_sha256 != object_sha256(self.receipt_payload()):
            raise SourceOwnedRoleLocatorError("role-mask receipt hash differs")

    def receipt_payload(self) -> dict[str, Any]:
        counts = self.masks.reshape(len(self.role_names), self.geometry.phases, -1).sum(dim=2)
        return {
            "schema_version": MASK_SCHEMA_VERSION,
            "event_id": self.event_id,
            "source_text_provenance_sha256": self.source_text_provenance_sha256,
            "step_index": self.step_index,
            "role_names": list(self.role_names),
            "protected_role_names": list(self.protected_role_names),
            "geometry": [self.geometry.phases, self.geometry.height, self.geometry.width],
            "phase_role_pixel_counts": counts.detach().cpu().tolist(),
            "confident_role_phases": self.confident_role_phases.detach().cpu().tolist(),
            "qualified": self.qualified,
            "policy": self.policy.as_dict(),
            "affinity_sha256": self.affinity_sha256,
            "mask_sha256": self.mask_sha256,
            "mutually_exclusive": True,
            "phase_zero_explicit": True,
            "source_authority_only": True,
            "observer_only_not_route": True,
        }


def build_source_owned_role_masks(
    affinity: GlobalRoleAffinity,
    spec: SourceRoleEventSpec,
    *,
    policy: RoleMaskPolicy = RoleMaskPolicy(),
) -> SourceRoleMaskSet:
    """Turn source Q/text-K affinity into sparse, exclusive role masks.

    A site remains unassigned unless the real role score beats an unclaimed
    source-token null, the cyclic wrong-span control, every competing role,
    and an absolute floor.  There is deliberately no per-phase seed and no
    forced nonempty role: weak/noisy evidence yields an unqualified empty mask.
    """

    if not isinstance(affinity, GlobalRoleAffinity) or not isinstance(spec, SourceRoleEventSpec):
        raise SourceOwnedRoleLocatorError("mask construction requires affinity and event spec")
    if affinity.event_id != spec.event_id or affinity.role_names != spec.role_names:
        raise SourceOwnedRoleLocatorError("mask affinity/event role authority differs")
    if not isinstance(policy, RoleMaskPolicy):
        raise SourceOwnedRoleLocatorError("mask policy has wrong type")
    scores = affinity.affinity.detach().float()
    roles, phases, height, width = _shape(scores)
    spatial = height * width
    flattened = scores.reshape(roles, phases, spatial)
    null_scores = affinity.null_affinity.detach().float().reshape(phases, spatial)
    shuffled_scores = affinity.shuffled_affinity.detach().float().reshape(
        roles, phases, spatial
    )
    std = flattened.std(dim=2, unbiased=False)
    quota = max(
        policy.minimum_pixels_per_role_phase,
        min(spatial, math.ceil(spatial * float(policy.keep_fraction))),
    )
    masks = torch.zeros_like(flattened, dtype=torch.bool)
    for phase in range(phases):
        phase_scores = flattened[:, phase]
        winners = phase_scores.argmax(dim=0)
        for role_index in range(roles):
            other = torch.cat(
                (phase_scores[:role_index], phase_scores[role_index + 1 :]), dim=0
            ).max(dim=0).values
            real = phase_scores[role_index]
            allowed = (
                (std[role_index, phase] >= float(policy.minimum_spatial_std))
                & (real >= float(policy.minimum_absolute_affinity))
                & (
                    real - null_scores[phase]
                    >= float(policy.minimum_null_margin)
                )
                & (
                    real - shuffled_scores[role_index, phase]
                    >= float(policy.minimum_shuffled_margin)
                )
                & (real - other >= float(policy.minimum_other_role_margin))
                & (winners == role_index)
            )
            candidates = torch.nonzero(allowed, as_tuple=False).flatten()
            if int(candidates.numel()) < policy.minimum_pixels_per_role_phase:
                continue
            take = min(quota, int(candidates.numel()))
            choice = torch.topk(
                phase_scores[role_index, candidates], k=take, largest=True, sorted=False
            ).indices
            masks[role_index, phase, candidates[choice]] = True
    masks = masks.reshape(roles, phases, height, width).detach().contiguous()
    protected_names = spec.protected_role_names
    protected_indices = [spec.role_names.index(name) for name in protected_names]
    protected_union = masks[protected_indices].any(dim=0).detach().contiguous()
    phase0 = masks[:, 0].detach().contiguous()
    counts = masks.reshape(roles, phases, -1).sum(dim=2)
    confident = (counts >= policy.minimum_pixels_per_role_phase).detach().contiguous()
    qualified = bool(
        (confident.sum(dim=1) >= policy.minimum_confident_phases).all().item()
    )
    affinity_digest = object_sha256(
        {
            "real": tensor_sha256(affinity.affinity),
            "null": tensor_sha256(affinity.null_affinity),
            "shuffled": tensor_sha256(affinity.shuffled_affinity),
        }
    )
    mask_digest = tensor_sha256(masks)
    payload = {
        "schema_version": MASK_SCHEMA_VERSION,
        "event_id": affinity.event_id,
        "source_text_provenance_sha256": affinity.source_text_provenance_sha256,
        "step_index": affinity.step_index,
        "role_names": list(affinity.role_names),
        "protected_role_names": list(protected_names),
        "geometry": [phases, height, width],
        "phase_role_pixel_counts": counts.detach().cpu().tolist(),
        "confident_role_phases": confident.detach().cpu().tolist(),
        "qualified": qualified,
        "policy": policy.as_dict(),
        "affinity_sha256": affinity_digest,
        "mask_sha256": mask_digest,
        "mutually_exclusive": True,
        "phase_zero_explicit": True,
        "source_authority_only": True,
        "observer_only_not_route": True,
    }
    return SourceRoleMaskSet(
        event_id=affinity.event_id,
        source_text_provenance_sha256=affinity.source_text_provenance_sha256,
        step_index=affinity.step_index,
        role_names=affinity.role_names,
        protected_role_names=protected_names,
        geometry=affinity.geometry,
        masks=masks,
        protected_union=protected_union,
        phase0_masks=phase0,
        confident_role_phases=confident,
        qualified=qualified,
        policy=policy,
        affinity_sha256=affinity_digest,
        mask_sha256=mask_digest,
        receipt_sha256=object_sha256(payload),
    )


def source_owned_locator_contract() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": "independent_observer_core_not_connected_to_training_or_decode",
        "authority": ["real_source_visual_hidden", "real_source_caption_text_embedding"],
        "forbidden_inputs": ["action_anchor", "appearance_donor", "anchor_text"],
        "tokenizer": {
            "transformers_version": PINNED_TRANSFORMERS_VERSION,
            "fix_mistral_regex": True,
            "tree_sha256": PINNED_TOKENIZER_TREE_SHA256,
        },
        "source_text_provenance": (
            "pre_SP_conditioned_storage_plus_replicated_full_text_K_descriptor_and_value_digest"
        ),
        "attention": "official_attn2_visual_Q_by_source_text_K",
        "official_processor_type": (
            f"{OFFICIAL_ATTN2_PROCESSOR_MODULE}.{OFFICIAL_ATTN2_PROCESSOR_CLASS}"
        ),
        "official_base_output": "same_object_returned_without_arithmetic",
        "observer_projection": "official__project_qkv_on_detached_source_inputs",
        "projection_preconditions": [
            "eval",
            "all_attn2_parameters_frozen",
            "no_hooks",
            "no_dropout_modules",
        ],
        "implicit_collectives": 0,
        "ulysses": "rank_local_Q_replicated_text_K_explicit_external_assembly",
        "geometry": "role_x_21_x_H_x_W_with_explicit_phase0",
        "mask_constraints": [
            "mutually_exclusive",
            "unassigned_allowed",
            "null_margin",
            "cyclic_shuffled_span_margin",
            "confidence_gate",
            "protected_union",
        ],
        "route_authorized": False,
        "training_authorized": False,
        "trained_parameters": 0,
    }
    return {**value, "contract_sha256": object_sha256(value)}


__all__ = [
    "ASSET_SCHEMA_VERSION",
    "EXPECTED_BLOCK_COUNT",
    "GlobalRoleAffinity",
    "LATENT_PHASES",
    "LockedRoleSpan",
    "MASK_SCHEMA_VERSION",
    "MAX_TEXT_TOKENS",
    "OFFICIAL_ATTN2_PROCESSOR_CLASS",
    "OFFICIAL_ATTN2_PROCESSOR_MODULE",
    "PINNED_TOKENIZER_FILES",
    "PINNED_TOKENIZER_TREE_SHA256",
    "PINNED_TRANSFORMERS_VERSION",
    "RoleAffinityShard",
    "RoleMaskPolicy",
    "RuntimeTokenizationReceipt",
    "SCHEMA_VERSION",
    "SourceOwnedRoleAttn2Observer",
    "SourceOwnedRoleLocatorError",
    "SourceRoleCaptureBank",
    "SourceRoleEventSpec",
    "SourceRoleMaskSet",
    "SourceRoleObserverInvocation",
    "SourceTextProvenance",
    "SourceRoleObserverPatchHandle",
    "SourceVisualGeometry",
    "UlyssesVisualShard",
    "aggregate_global_role_affinities",
    "assemble_global_role_affinity",
    "bind_source_text_provenance",
    "build_source_owned_role_masks",
    "canonical_json_bytes",
    "current_source_role_observer",
    "install_source_owned_role_observer",
    "load_role_span_asset",
    "object_sha256",
    "observe_source_roles",
    "resolve_exact_substring_token_span",
    "source_owned_locator_contract",
    "text_sha256",
    "tensor_sha256",
    "tokenizer_tree_sha256",
    "validate_pinned_tokenizer_runtime",
    "validate_runtime_tokenization",
]
