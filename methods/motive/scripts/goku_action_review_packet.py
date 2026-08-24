#!/usr/bin/env python3
"""Build a fail-closed, static human-review packet for Goku action proposals.

This command deliberately does *not* create the production approval schema.
The browser UI can export only a draft review document whose decisions still
need to be converted, attested, and proposal-bound by a separate human step.

The packet contains an exact byte-for-byte copy of ``proposed_128.jsonl``, an
exact verified copy of every source video and I0 image, a static review UI, a
closed packet manifest, checksums, and a terminal readiness marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


PACKET_SCHEMA = "motive-goku-action-review-packet-v1"
READY_SCHEMA = "motive-goku-action-review-packet-ready-v1"
DRAFT_SCHEMA = "motive-goku-action-review-draft-v1"
PROPOSAL_ROW_SCHEMA = "motive-goku-action-anchor-final-row-v1"
OFFICIAL_APPROVAL_SCHEMA = "motive-goku-action-anchor-approval-v1"

PROPOSAL_COPY_NAME = "proposed_128.jsonl"
MANIFEST_NAME = "packet_manifest.json"
SUMS_NAME = "SHA256SUMS"
READY_NAME = "PACKET_READY.json"
INDEX_NAME = "index.html"
APP_NAME = "app.js"
STYLES_NAME = "styles.css"


class ReviewPacketError(ValueError):
    """The proposal, media, path mapping, or output contract is invalid."""


def _reject_constant(value: str) -> None:
    raise ReviewPacketError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewPacketError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReviewPacketError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ReviewPacketError) as error:
        if isinstance(error, ReviewPacketError):
            raise
        raise ReviewPacketError(f"{context} is not strict JSON: {error}") from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReviewPacketError(f"value is not canonical JSON: {error}") from error


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _builder_implementation() -> tuple[Path, str]:
    implementation = _regular_file(
        _absolute_lexical_path(
            os.path.abspath(__file__),
            context="builder implementation",
        ),
        context="builder implementation",
    )
    return implementation, _sha256_file(implementation)


def _canonical_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReviewPacketError(f"{context} must be a non-empty string")
    if value != value.strip() or "\x00" in value:
        raise ReviewPacketError(f"{context} is not a canonical string")
    return value


def _sha256_field(value: Any, *, context: str) -> str:
    digest = _canonical_string(value, context=context)
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ReviewPacketError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return digest


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewPacketError(f"{context} must be an object")
    return value


def _string_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ReviewPacketError(f"{context} must be a non-empty string array")
    return [
        _canonical_string(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    ]


def _absolute_lexical_path(value: str | Path, *, context: str) -> Path:
    text = os.fspath(value)
    if not text or "\x00" in text or not os.path.isabs(text):
        raise ReviewPacketError(f"{context} must be an absolute path")
    if os.path.normpath(text) != text:
        raise ReviewPacketError(f"{context} is not lexically canonical: {text}")
    return Path(text)


def _absolute_remote_path(value: Any, *, context: str) -> PurePosixPath:
    text = _canonical_string(value, context=context)
    if not text.startswith("/") or posixpath.normpath(text) != text:
        raise ReviewPacketError(
            f"{context} must be a canonical absolute POSIX path"
        )
    return PurePosixPath(text)


def _assert_no_symlink_components(
    path: Path,
    *,
    context: str,
    include_leaf: bool = True,
) -> None:
    path = _absolute_lexical_path(path, context=context)
    parts = path.parts
    current = Path(parts[0])
    stop = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:stop]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ReviewPacketError(
                f"{context} component does not exist: {current}"
            ) from error
        if stat.S_ISLNK(mode):
            raise ReviewPacketError(
                f"{context} contains a symlink component: {current}"
            )


def _regular_file(path: Path, *, context: str) -> Path:
    lexical = _absolute_lexical_path(path, context=context)
    _assert_no_symlink_components(lexical, context=context)
    try:
        metadata = os.lstat(lexical)
    except FileNotFoundError as error:
        raise ReviewPacketError(f"{context} does not exist: {lexical}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReviewPacketError(
            f"{context} must be a regular non-symlink file: {lexical}"
        )
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ReviewPacketError(
            f"{context} does not resolve to the same canonical path: {lexical}"
        )
    return lexical


def _regular_directory(path: Path, *, context: str) -> Path:
    lexical = _absolute_lexical_path(path, context=context)
    _assert_no_symlink_components(lexical, context=context)
    try:
        metadata = os.lstat(lexical)
    except FileNotFoundError as error:
        raise ReviewPacketError(f"{context} does not exist: {lexical}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReviewPacketError(
            f"{context} must be a regular non-symlink directory: {lexical}"
        )
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ReviewPacketError(
            f"{context} does not resolve to the same canonical path: {lexical}"
        )
    return lexical


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class _PathMap:
    def __init__(self, remote_prefix: PurePosixPath, local_prefix: Path) -> None:
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix

    @property
    def remote_text(self) -> str:
        return self.remote_prefix.as_posix()

    def matches(self, remote_path: PurePosixPath) -> bool:
        prefix = self.remote_prefix.parts
        return remote_path.parts[: len(prefix)] == prefix

    def apply(self, remote_path: PurePosixPath) -> Path:
        tail = remote_path.parts[len(self.remote_prefix.parts) :]
        return self.local_prefix.joinpath(*tail)

    def manifest_record(self) -> dict[str, str]:
        return {
            "remote_prefix": self.remote_text,
            "local_prefix": str(self.local_prefix),
        }


def parse_path_maps(values: Sequence[str]) -> list[_PathMap]:
    mappings: list[_PathMap] = []
    seen_remote: set[str] = set()
    for index, value in enumerate(values, start=1):
        if not isinstance(value, str) or "=" not in value:
            raise ReviewPacketError(
                f"path map {index} must use REMOTE_PREFIX=LOCAL_PREFIX"
            )
        remote_text, local_text = value.split("=", 1)
        remote = _absolute_remote_path(
            remote_text,
            context=f"path map {index} remote prefix",
        )
        local = _regular_directory(
            _absolute_lexical_path(
                local_text,
                context=f"path map {index} local prefix",
            ),
            context=f"path map {index} local prefix",
        )
        normalized_remote = remote.as_posix()
        if normalized_remote in seen_remote:
            raise ReviewPacketError(
                f"duplicate path-map remote prefix: {normalized_remote}"
            )
        seen_remote.add(normalized_remote)
        mappings.append(_PathMap(remote, local))
    return sorted(
        mappings,
        key=lambda mapping: (
            -len(mapping.remote_prefix.parts),
            -len(mapping.remote_text),
            mapping.remote_text,
        ),
    )


def _map_remote_path(
    remote_path: PurePosixPath,
    *,
    path_maps: Sequence[_PathMap],
    context: str,
) -> Path:
    for mapping in path_maps:
        if mapping.matches(remote_path):
            candidate = mapping.apply(remote_path)
            return _regular_file(candidate, context=f"{context} mapped file")
    return _regular_file(
        Path(remote_path.as_posix()),
        context=f"{context} unmapped file",
    )


def _load_proposal(
    proposal_path: Path,
) -> tuple[list[dict[str, Any]], list[bytes], bytes]:
    proposal = _regular_file(proposal_path, context="proposal input")
    if proposal.name != PROPOSAL_COPY_NAME:
        raise ReviewPacketError(
            f"proposal input basename must be {PROPOSAL_COPY_NAME!r}"
        )
    raw = proposal.read_bytes()
    if not raw:
        raise ReviewPacketError("proposal input is empty")
    if not raw.endswith(b"\n"):
        raise ReviewPacketError("proposal input must end with a newline")
    line_payloads: list[bytes] = []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ReviewPacketError(
                f"proposal input contains a blank line at {line_number}"
            )
        line_payload = line + b"\n"
        value = _parse_json(
            line,
            context=f"proposal input line {line_number}",
        )
        if not isinstance(value, dict):
            raise ReviewPacketError(
                f"proposal input line {line_number} is not an object"
            )
        rows.append(value)
        line_payloads.append(line_payload)
    return rows, line_payloads, raw


def _required_text(
    mapping: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> str:
    if field not in mapping:
        raise ReviewPacketError(f"{context} is missing {field!r}")
    return _canonical_string(mapping[field], context=f"{context} {field}")


def _validate_review_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    path_maps: Sequence[_PathMap],
) -> dict[str, Any]:
    context = f"proposal row {line_number}"
    iid = _required_text(row, "iid", context=context)
    for field in ("source_caption", "edited_caption", "prompt"):
        _required_text(row, field, context=f"{context} iid={iid}")
    source_sha = _sha256_field(
        row.get("source_video_sha256"),
        context=f"{context} iid={iid} source_video_sha256",
    )
    anchor_sha = _sha256_field(
        row.get("anchor_sha256"),
        context=f"{context} iid={iid} anchor_sha256",
    )
    source_remote = _absolute_remote_path(
        row.get("resolved_src_video"),
        context=f"{context} iid={iid} resolved_src_video",
    )
    anchor_remote = _absolute_remote_path(
        row.get("resolved_anchor_image"),
        context=f"{context} iid={iid} resolved_anchor_image",
    )
    if source_remote.suffix.casefold() != ".mp4":
        raise ReviewPacketError(f"{context} iid={iid} source is not an MP4")
    if anchor_remote.suffix.casefold() != ".png":
        raise ReviewPacketError(f"{context} iid={iid} anchor is not a PNG")

    source_local = _map_remote_path(
        source_remote,
        path_maps=path_maps,
        context=f"{context} iid={iid} source video",
    )
    anchor_local = _map_remote_path(
        anchor_remote,
        path_maps=path_maps,
        context=f"{context} iid={iid} anchor image",
    )
    actual_source_sha = _sha256_file(source_local)
    actual_anchor_sha = _sha256_file(anchor_local)
    if actual_source_sha != source_sha:
        raise ReviewPacketError(
            f"{context} iid={iid} source video SHA-256 differs"
        )
    if actual_anchor_sha != anchor_sha:
        raise ReviewPacketError(
            f"{context} iid={iid} anchor image SHA-256 differs"
        )

    finalization = _mapping(
        row.get("action_anchor_finalization"),
        context=f"{context} iid={iid} action_anchor_finalization",
    )
    if finalization.get("schema_version") != PROPOSAL_ROW_SCHEMA:
        raise ReviewPacketError(
            f"{context} iid={iid} is not a final proposal row"
        )
    expected_state = {
        "hard_gate_passed": True,
        "selection_bucket": "proposed",
        "human_review_status": "pending",
        "human_label": False,
        "generation_authorized": False,
    }
    for field, expected in expected_state.items():
        if finalization.get(field) != expected:
            raise ReviewPacketError(
                f"{context} iid={iid} {field} must be {expected!r}"
            )
    if finalization.get("hard_gate_failures") != []:
        raise ReviewPacketError(
            f"{context} iid={iid} hard_gate_failures must be empty"
        )
    target_support = _mapping(
        finalization.get("target_support_evidence"),
        context=f"{context} iid={iid} target_support_evidence",
    )
    if type(
        target_support.get("requires_proposal_bound_human_review")
    ) is not bool:
        raise ReviewPacketError(
            f"{context} iid={iid} target-support review flag must be boolean"
        )

    qwen = _mapping(
        row.get("qwen_action_anchor"),
        context=f"{context} iid={iid} qwen_action_anchor",
    )
    observation = _mapping(
        qwen.get("anchor_observation"),
        context=f"{context} iid={iid} anchor_observation",
    )
    compatibility = _mapping(
        qwen.get("compatibility"),
        context=f"{context} iid={iid} compatibility",
    )
    initial_state = _required_text(
        observation,
        "initial_state",
        context=f"{context} iid={iid} anchor_observation",
    )
    source_action = _required_text(
        observation,
        "source_action",
        context=f"{context} iid={iid} anchor_observation",
    )
    source_action_normalized = _required_text(
        compatibility,
        "source_action_normalized",
        context=f"{context} iid={iid} compatibility",
    )
    target_action = _required_text(
        compatibility,
        "target_action_normalized",
        context=f"{context} iid={iid} compatibility",
    )
    target_verb = _required_text(
        compatibility,
        "target_action_verb",
        context=f"{context} iid={iid} compatibility",
    )
    instruction = _required_text(
        compatibility,
        "rewritten_edit_instruction",
        context=f"{context} iid={iid} compatibility",
    )
    causal_bridge = _required_text(
        compatibility,
        "causal_bridge",
        context=f"{context} iid={iid} compatibility",
    )
    bridge_description = _required_text(
        compatibility,
        "causal_bridge_description",
        context=f"{context} iid={iid} compatibility",
    )
    causal_stages = _string_list(
        compatibility.get("causal_stages"),
        context=f"{context} iid={iid} compatibility causal_stages",
    )
    absolute_prompt = _required_text(
        compatibility,
        "absolute_target_prompt",
        context=f"{context} iid={iid} compatibility",
    )

    return {
        "iid": iid,
        "source_local": source_local,
        "anchor_local": anchor_local,
        "source_sha256": source_sha,
        "anchor_sha256": anchor_sha,
        "source_original_path": source_remote.as_posix(),
        "anchor_original_path": anchor_remote.as_posix(),
        "review": {
            "source_caption": row["source_caption"],
            "initial_state": initial_state,
            "source_trajectory": source_action,
            "source_action_normalized": source_action_normalized,
            "source_instruction_provenance": row["prompt"],
            "source_edited_caption_provenance": row["edited_caption"],
            "target_action": target_action,
            "target_action_verb": target_verb,
            "rewritten_edit_instruction": instruction,
            "causal_bridge": causal_bridge,
            "causal_bridge_description": bridge_description,
            "causal_stages": causal_stages,
            "absolute_target_prompt": absolute_prompt,
            "target_support_evidence": dict(target_support),
        },
    }


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    context: str,
) -> int:
    source = _regular_file(source, context=context)
    open_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        open_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    descriptor = os.open(source, open_flags)
    hasher = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "rb") as input_handle:
            descriptor = -1
            before = os.fstat(input_handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ReviewPacketError(
                    f"{context} is no longer a regular file"
                )
            with destination.open("xb") as output:
                while block := input_handle.read(1024 * 1024):
                    output.write(block)
                    hasher.update(block)
                    total += len(block)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(input_handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path_after = os.lstat(source)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        identity_before != identity_after
        or not stat.S_ISREG(path_after.st_mode)
        or path_after.st_dev != after.st_dev
        or path_after.st_ino != after.st_ino
    ):
        raise ReviewPacketError(f"{context} changed while it was copied")
    if hasher.hexdigest() != expected_sha256:
        raise ReviewPacketError(f"{context} SHA-256 changed while it was copied")
    return total


def _index_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'self'; img-src 'self'; media-src 'self';
                 script-src 'self'; style-src 'self'; object-src 'none';
                 base-uri 'none'; form-action 'none'">
  <title>Goku action-edit proposal review draft</title>
  <link rel="stylesheet" href="styles.css">
  <script src="app.js" defer></script>
</head>
<body>
  <header>
    <p class="eyebrow">Human review packet / draft only</p>
    <h1>Goku action-edit proposal review</h1>
    <p class="warning">
      This packet cannot authorize generation. Exported files are review
      drafts and are intentionally incompatible with the production approval
      schema.
    </p>
    <dl class="packet-meta">
      <div><dt>Proposal SHA-256</dt><dd id="proposal-sha"></dd></div>
      <div><dt>Packet manifest SHA-256</dt><dd id="manifest-sha"></dd></div>
    </dl>
    <p id="review-status" aria-live="polite"></p>
    <button id="export-draft" type="button" disabled>
      Export complete review draft
    </button>
  </header>
  <main id="samples"></main>
</body>
</html>
"""


def _styles_css() -> bytes:
    return b""":root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #0d1117;
  color: #e6edf3;
}
body { margin: 0; }
header {
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 1.25rem clamp(1rem, 4vw, 3rem);
  border-bottom: 1px solid #30363d;
  background: rgba(13, 17, 23, 0.97);
}
h1 { margin: 0.15rem 0 0.75rem; }
.eyebrow { margin: 0; color: #79c0ff; text-transform: uppercase; }
.warning {
  max-width: 75rem;
  padding: 0.75rem;
  border: 1px solid #d29922;
  background: #2d2408;
  color: #f2cc60;
}
.packet-meta { display: grid; gap: 0.35rem; }
.packet-meta div { display: grid; grid-template-columns: 14rem 1fr; }
dt { color: #8b949e; }
dd { margin: 0; overflow-wrap: anywhere; font-family: ui-monospace, monospace; }
button {
  border: 1px solid #58a6ff;
  border-radius: 0.4rem;
  padding: 0.55rem 0.85rem;
  background: #1f6feb;
  color: white;
  font-weight: 650;
  cursor: pointer;
}
main {
  display: grid;
  gap: 1.5rem;
  padding: 1.5rem clamp(1rem, 4vw, 3rem) 4rem;
}
.sample {
  border: 1px solid #30363d;
  border-radius: 0.75rem;
  padding: 1rem;
  background: #161b22;
}
.sample h2 { margin-top: 0; overflow-wrap: anywhere; }
.media-grid, .field-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr));
  gap: 1rem;
}
video, img {
  display: block;
  width: 100%;
  max-height: 34rem;
  object-fit: contain;
  background: black;
}
.field {
  border-top: 1px solid #30363d;
  padding-top: 0.65rem;
}
.field h3 { color: #79c0ff; margin: 0 0 0.35rem; font-size: 0.9rem; }
.field p, .field pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.field pre { font-family: ui-monospace, monospace; }
.decision {
  margin-top: 1rem;
  padding: 0.85rem;
  border: 1px solid #484f58;
  border-radius: 0.5rem;
}
.choices { display: flex; flex-wrap: wrap; gap: 1rem; }
.notes-label { display: grid; gap: 0.35rem; margin-top: 0.75rem; }
textarea {
  min-height: 5rem;
  padding: 0.5rem;
  color: inherit;
  background: #0d1117;
  border: 1px solid #484f58;
}
"""


def _app_javascript(
    *,
    packet_payload: Mapping[str, Any],
    manifest_sha256: str,
) -> bytes:
    embedded = json.dumps(
        {
            "proposal_sha256": packet_payload["proposal"]["sha256"],
            "packet_manifest_sha256": manifest_sha256,
            "rows": packet_payload["rows"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    source = r"""
"use strict";

const PACKET = Object.freeze(__PACKET_DATA__);
const DRAFT_SCHEMA = "motive-goku-action-review-draft-v1";
const OFFICIAL_APPROVAL_SCHEMA = "motive-goku-action-anchor-approval-v1";
const decisions = PACKET.rows.map((row) => ({
  iid: row.iid,
  decision: "undecided",
  notes: "",
}));

function appendText(parent, tag, value, className) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  node.textContent = String(value);
  parent.appendChild(node);
  return node;
}

function addField(parent, label, value, asJson) {
  const field = document.createElement("section");
  field.className = "field";
  appendText(field, "h3", label);
  const body = document.createElement(asJson ? "pre" : "p");
  body.textContent = asJson ? JSON.stringify(value, null, 2) : String(value);
  field.appendChild(body);
  parent.appendChild(field);
}

function updateStatus() {
  const counts = { undecided: 0, approve_candidate: 0, reject_candidate: 0 };
  decisions.forEach((entry) => {
    counts[entry.decision] += 1;
  });
  const complete = decisions.every(
    (entry) =>
      entry.decision !== "undecided" && entry.notes.trim().length > 0,
  );
  const status = document.getElementById("review-status");
  status.textContent =
    `Undecided: ${counts.undecided} / ` +
    `Approve candidates: ${counts.approve_candidate} / ` +
    `Reject candidates: ${counts.reject_candidate}. ` +
    (complete
      ? "Draft is complete and can be exported."
      : "Decide every row and enter non-empty notes for every decision.");
  document.getElementById("export-draft").disabled = !complete;
}

function addChoice(parent, index, value, labelText) {
  const label = document.createElement("label");
  const input = document.createElement("input");
  input.type = "radio";
  input.name = `decision-${index}`;
  input.value = value;
  input.checked = value === "undecided";
  input.addEventListener("change", () => {
    if (input.checked) {
      decisions[index].decision = value;
      updateStatus();
    }
  });
  label.appendChild(input);
  label.appendChild(document.createTextNode(` ${labelText}`));
  parent.appendChild(label);
}

function renderSample(row, index) {
  const card = document.createElement("article");
  card.className = "sample";
  appendText(card, "h2", `${index + 1}. ${row.iid}`);

  const media = document.createElement("div");
  media.className = "media-grid";
  const videoBox = document.createElement("section");
  appendText(videoBox, "h3", "Exact full source video");
  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  video.src = row.assets.source_video.packet_path;
  videoBox.appendChild(video);
  addField(
    videoBox,
    "Source SHA-256",
    row.assets.source_video.sha256,
    false,
  );
  const imageBox = document.createElement("section");
  appendText(imageBox, "h3", "Exact I0 anchor");
  const image = document.createElement("img");
  image.loading = "lazy";
  image.alt = `I0 anchor for ${row.iid}`;
  image.src = row.assets.anchor_image.packet_path;
  imageBox.appendChild(image);
  addField(
    imageBox,
    "Anchor SHA-256",
    row.assets.anchor_image.sha256,
    false,
  );
  media.appendChild(videoBox);
  media.appendChild(imageBox);
  card.appendChild(media);

  const fields = document.createElement("div");
  fields.className = "field-grid";
  addField(fields, "Source caption", row.review.source_caption, false);
  addField(fields, "Initial state at I0", row.review.initial_state, false);
  addField(fields, "Source trajectory", row.review.source_trajectory, false);
  addField(
    fields,
    "Normalized source action",
    row.review.source_action_normalized,
    false,
  );
  addField(
    fields,
    "Original instruction provenance",
    row.review.source_instruction_provenance,
    false,
  );
  addField(
    fields,
    "Original edited_caption provenance",
    row.review.source_edited_caption_provenance,
    false,
  );
  addField(fields, "Target action", row.review.target_action, false);
  addField(fields, "Target action verb", row.review.target_action_verb, false);
  addField(
    fields,
    "Rewritten edit instruction",
    row.review.rewritten_edit_instruction,
    false,
  );
  addField(fields, "Causal bridge type", row.review.causal_bridge, false);
  addField(
    fields,
    "Causal bridge description",
    row.review.causal_bridge_description,
    false,
  );
  addField(fields, "Causal stages", row.review.causal_stages, true);
  addField(
    fields,
    "Absolute target prompt",
    row.review.absolute_target_prompt,
    false,
  );
  addField(
    fields,
    "Target-support evidence",
    row.review.target_support_evidence,
    true,
  );
  card.appendChild(fields);

  const review = document.createElement("fieldset");
  review.className = "decision";
  appendText(review, "legend", "Draft review decision");
  const choices = document.createElement("div");
  choices.className = "choices";
  addChoice(choices, index, "undecided", "Undecided");
  addChoice(choices, index, "approve_candidate", "Approve candidate");
  addChoice(choices, index, "reject_candidate", "Reject candidate");
  review.appendChild(choices);
  const notesLabel = document.createElement("label");
  notesLabel.className = "notes-label";
  appendText(notesLabel, "span", "Review notes");
  const notes = document.createElement("textarea");
  notes.addEventListener("input", () => {
    decisions[index].notes = notes.value;
    updateStatus();
  });
  notesLabel.appendChild(notes);
  review.appendChild(notesLabel);
  card.appendChild(review);
  return card;
}

function exportDraft() {
  const complete = decisions.every(
    (entry) =>
      entry.decision !== "undecided" && entry.notes.trim().length > 0,
  );
  if (!complete) {
    updateStatus();
    throw new Error(
      "Refusing incomplete draft export: every row needs a decision and notes.",
    );
  }
  const draft = {
    schema_version: DRAFT_SCHEMA,
    draft_only: true,
    production_authorization: false,
    incompatible_official_approval_schema: OFFICIAL_APPROVAL_SCHEMA,
    proposal_sha256: PACKET.proposal_sha256,
    packet_manifest_sha256: PACKET.packet_manifest_sha256,
    exported_at_utc: new Date().toISOString(),
    decisions: decisions.map((entry) => ({
      iid: entry.iid,
      decision: entry.decision,
      notes: entry.notes.trim(),
    })),
  };
  const blob = new Blob(
    [JSON.stringify(draft, null, 2) + "\n"],
    { type: "application/json" },
  );
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download =
    `goku-action-review-draft-${PACKET.proposal_sha256.slice(0, 12)}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

document.getElementById("proposal-sha").textContent = PACKET.proposal_sha256;
document.getElementById("manifest-sha").textContent =
  PACKET.packet_manifest_sha256;
const samples = document.getElementById("samples");
PACKET.rows.forEach((row, index) => {
  samples.appendChild(renderSample(row, index));
});
document.getElementById("export-draft").addEventListener("click", exportDraft);
updateStatus();
"""
    return source.replace("__PACKET_DATA__", embedded).lstrip().encode("utf-8")


def _checksum_payload(output_dir: Path, names: Iterable[str]) -> bytes:
    lines = []
    for name in sorted(names):
        path = output_dir / name
        lines.append(f"{_sha256_file(path)}  {name}\n")
    return "".join(lines).encode("utf-8")


def _verify_checksum_payload(output_dir: Path, payload: bytes) -> None:
    for raw_line in payload.decode("utf-8").splitlines():
        digest, separator, name = raw_line.partition("  ")
        if not separator or not name or "/" == name:
            raise AssertionError("invalid generated SHA256SUMS line")
        if _sha256_file(output_dir / name) != digest:
            raise AssertionError(f"generated checksum differs for {name}")


def _freeze_packet_tree(output_dir: Path) -> None:
    directories: list[Path] = []
    for current_text, directory_names, file_names in os.walk(
        output_dir,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_text)
        directories.append(current)
        for name in directory_names:
            directory = current / name
            if directory.is_symlink():
                raise ReviewPacketError(
                    f"refusing to freeze symlink directory: {directory}"
                )
            if not directory.is_dir():
                raise ReviewPacketError(
                    f"packet directory changed before freeze: {directory}"
                )
        for name in file_names:
            file_path = current / name
            metadata = os.lstat(file_path)
            if not stat.S_ISREG(metadata.st_mode):
                raise ReviewPacketError(
                    f"refusing to freeze non-regular packet file: {file_path}"
                )
            os.chmod(file_path, 0o444, follow_symlinks=False)
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        os.chmod(directory, 0o555, follow_symlinks=False)


def _thaw_owned_packet_tree(output_dir: Path) -> None:
    """Make only this invocation's fresh output removable after a failure."""

    if not _lexists(output_dir) or output_dir.is_symlink():
        return
    os.chmod(output_dir, 0o755, follow_symlinks=False)
    for current_text, directory_names, file_names in os.walk(
        output_dir,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_text)
        os.chmod(current, 0o755, follow_symlinks=False)
        for name in directory_names:
            directory = current / name
            if not directory.is_symlink():
                os.chmod(directory, 0o755, follow_symlinks=False)
        for name in file_names:
            file_path = current / name
            if not file_path.is_symlink():
                os.chmod(file_path, 0o644, follow_symlinks=False)


def build_review_packet(
    *,
    proposal_path: str | Path,
    output_dir: str | Path,
    path_map_specs: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate inputs and publish one new, draft-only static review packet."""

    proposal = _absolute_lexical_path(proposal_path, context="proposal input")
    output = _absolute_lexical_path(output_dir, context="output directory")
    output_parent = _regular_directory(
        output.parent,
        context="output directory parent",
    )
    if output.parent != output_parent:
        raise AssertionError("output parent canonicalization changed")
    if _lexists(output):
        raise ReviewPacketError(
            f"output directory already exists; refusing overwrite: {output}"
        )
    path_maps = parse_path_maps(path_map_specs)
    builder_path, builder_sha256 = _builder_implementation()
    rows, line_payloads, proposal_raw = _load_proposal(proposal)

    validated_rows: list[dict[str, Any]] = []
    seen_iids: set[str] = set()
    seen_source_original_paths: set[str] = set()
    seen_source_local_paths: set[str] = set()
    seen_source_digests: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        validated = _validate_review_row(
            row,
            line_number=line_number,
            path_maps=path_maps,
        )
        iid = str(validated["iid"])
        if iid in seen_iids:
            raise ReviewPacketError(f"duplicate proposal IID: {iid}")
        seen_iids.add(iid)
        source_original_path = str(validated["source_original_path"])
        source_local_path = str(validated["source_local"])
        source_digest = str(validated["source_sha256"])
        if (
            source_original_path in seen_source_original_paths
            or source_local_path in seen_source_local_paths
            or source_digest in seen_source_digests
        ):
            raise ReviewPacketError(
                f"duplicate proposal source video for iid={iid}"
            )
        seen_source_original_paths.add(source_original_path)
        seen_source_local_paths.add(source_local_path)
        seen_source_digests.add(source_digest)
        validated["proposal_line"] = line_number
        validated["proposal_row_sha256"] = _sha256_bytes(
            line_payloads[line_number - 1]
        )
        validated_rows.append(validated)

    proposal_sha = _sha256_bytes(proposal_raw)
    output.mkdir(mode=0o755)
    try:
        written_names: list[str] = []
        _write_new(output / PROPOSAL_COPY_NAME, proposal_raw)
        written_names.append(PROPOSAL_COPY_NAME)

        manifest_rows: list[dict[str, Any]] = []
        assets_root = output / "assets"
        assets_root.mkdir(mode=0o755)
        for index, validated in enumerate(validated_rows, start=1):
            iid = str(validated["iid"])
            iid_token = hashlib.sha256(iid.encode("utf-8")).hexdigest()[:12]
            relative_dir = Path("assets") / f"{index:04d}-{iid_token}"
            asset_dir = output / relative_dir
            asset_dir.mkdir(mode=0o755)
            source_relative = relative_dir / "source.mp4"
            anchor_relative = relative_dir / "i0.png"
            source_size = _copy_verified(
                validated["source_local"],
                output / source_relative,
                expected_sha256=str(validated["source_sha256"]),
                context=f"proposal iid={iid} source video",
            )
            anchor_size = _copy_verified(
                validated["anchor_local"],
                output / anchor_relative,
                expected_sha256=str(validated["anchor_sha256"]),
                context=f"proposal iid={iid} anchor image",
            )
            source_name = source_relative.as_posix()
            anchor_name = anchor_relative.as_posix()
            written_names.extend((source_name, anchor_name))
            manifest_rows.append(
                {
                    "iid": iid,
                    "proposal_line": validated["proposal_line"],
                    "proposal_row_sha256": validated[
                        "proposal_row_sha256"
                    ],
                    "assets": {
                        "source_video": {
                            "original_path": validated[
                                "source_original_path"
                            ],
                            "mapped_local_input_path": str(
                                validated["source_local"]
                            ),
                            "packet_path": source_name,
                            "sha256": validated["source_sha256"],
                            "bytes": source_size,
                        },
                        "anchor_image": {
                            "original_path": validated[
                                "anchor_original_path"
                            ],
                            "mapped_local_input_path": str(
                                validated["anchor_local"]
                            ),
                            "packet_path": anchor_name,
                            "sha256": validated["anchor_sha256"],
                            "bytes": anchor_size,
                        },
                    },
                    "review": validated["review"],
                    "draft_decision_default": "undecided",
                }
            )

        manifest: dict[str, Any] = {
            "schema_version": PACKET_SCHEMA,
            "review_scope": "draft_only",
            "production_authorization": False,
            "draft_schema_version": DRAFT_SCHEMA,
            "incompatible_official_approval_schema": OFFICIAL_APPROVAL_SCHEMA,
            "builder": {
                "implementation_path": str(builder_path),
                "implementation_sha256": builder_sha256,
            },
            "proposal": {
                "original_path": str(proposal),
                "packet_path": PROPOSAL_COPY_NAME,
                "sha256": proposal_sha,
                "bytes": len(proposal_raw),
                "rows": len(manifest_rows),
            },
            "path_maps": [
                mapping.manifest_record() for mapping in path_maps
            ],
            "rows": manifest_rows,
        }
        manifest_payload = _pretty_bytes(manifest)
        _write_new(output / MANIFEST_NAME, manifest_payload)
        written_names.append(MANIFEST_NAME)
        manifest_sha = _sha256_bytes(manifest_payload)

        _write_new(output / INDEX_NAME, _index_html())
        _write_new(output / STYLES_NAME, _styles_css())
        _write_new(
            output / APP_NAME,
            _app_javascript(
                packet_payload=manifest,
                manifest_sha256=manifest_sha,
            ),
        )
        written_names.extend((INDEX_NAME, STYLES_NAME, APP_NAME))

        sums_payload = _checksum_payload(output, written_names)
        _verify_checksum_payload(output, sums_payload)
        _write_new(output / SUMS_NAME, sums_payload)
        ready = {
            "schema_version": READY_SCHEMA,
            "review_scope": "draft_only",
            "production_authorization": False,
            "proposal_sha256": proposal_sha,
            "packet_manifest_sha256": manifest_sha,
            "sha256sums_sha256": _sha256_bytes(sums_payload),
            "builder_implementation_sha256": builder_sha256,
            "row_count": len(manifest_rows),
            "draft_schema_version": DRAFT_SCHEMA,
            "incompatible_official_approval_schema": (
                OFFICIAL_APPROVAL_SCHEMA
            ),
        }
        _write_new(output / READY_NAME, _pretty_bytes(ready))

        if (output / PROPOSAL_COPY_NAME).read_bytes() != proposal_raw:
            raise AssertionError("published proposal copy is not byte-exact")
        if _sha256_file(output / MANIFEST_NAME) != manifest_sha:
            raise AssertionError("published manifest binding differs")
        if _sha256_file(output / SUMS_NAME) != ready["sha256sums_sha256"]:
            raise AssertionError("published checksum binding differs")
        _freeze_packet_tree(output)
        return ready
    except BaseException:
        # This directory was created by this invocation with mkdir(exist_ok=False);
        # removing it cannot overwrite or delete a pre-existing packet.
        _thaw_owned_packet_tree(output)
        shutil.rmtree(output)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal",
        required=True,
        help="Absolute path to the immutable proposed_128.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Absolute path for a new review packet directory.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="REMOTE_PREFIX=LOCAL_PREFIX",
        help=(
            "Map proposal AUH paths to a local mirror. May be repeated; the "
            "longest matching path-component prefix wins."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        ready = build_review_packet(
            proposal_path=args.proposal,
            output_dir=args.output_dir,
            path_map_specs=args.path_map,
        )
    except (OSError, ReviewPacketError) as error:
        print(f"goku-action-review-packet: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(ready, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
