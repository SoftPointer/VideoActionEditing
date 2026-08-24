"""Immutable R6 UMT5/CLIP text-feature extraction.

The extractor consumes the committed R5 final manifest in its existing order
and produces four *raw* frozen-encoder embeddings:

``umt5_prompt`` / ``clip_prompt``
    Derived only from the original edit instruction in ``row["prompt"]``.
    These are the only semantic features that may enter an R6 delta predictor.

``umt5_observed_target`` / ``clip_observed_target``
    Derived only from the original, schema-validated Qwen
    ``observation.target_action``.  They are explicitly target-derived,
    diagnostic-only features for prompt/observed-motion compatibility and
    must never enter the semantic-only predictor.

No PCA, standardizer, centroid, pairing, label-derived text, or other
data-dependent fit happens here.  The output matrices are float32 and L2
normalized per row.  ``done.json`` is the final commit marker; validation
rechecks the input text contract, every artifact digest, IID order, array
shape/dtype/norm, and registered model files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .qwen_filter import _parse_object, _validate_observation


R6_SEMANTIC_SCHEMA = "motive-r6-semantic-features-v1"
R6_SEMANTIC_MANIFEST_SCHEMA = "motive-r6-semantic-row-v1"
R6_SEMANTIC_METADATA_SCHEMA = "motive-r6-semantic-metadata-v1"
R6_SEMANTIC_SUMMARY_SCHEMA = "motive-r6-semantic-summary-v1"
R6_SEMANTIC_DONE_SCHEMA = "motive-r6-semantic-done-v1"
R6_TEXT_CONTRACT = "r5-manifest-original-qwen-observation-v1"
R6_PROMPT_TEMPLATE_VERSION = "raw-manifest-text-no-template-v1"
R6_SOURCE_SNAPSHOT_SCHEMA = "motive-r6-source-snapshot-binding-v1"
R6_SOURCE_SNAPSHOT_SENTINEL = "synthetic-source-snapshot-not-applicable-v1"
SOURCE_FILES_NAME = "SOURCE_FILES.jsonl"
SOURCE_PROVENANCE_NAME = "SOURCE_PROVENANCE.json"
SOURCE_SNAPSHOT_UPSTREAM_SCHEMA = "motive-action-source-snapshot-v1"

UMT5_ENCODER_ID = "google/umt5-xxl"
UMT5_REVISION = "f12c3ab18266dcc1eb97f26b9102af42dfd327c5"
UMT5_DIM = 4096
UMT5_POOLING = "attention-mask-mean-last-hidden-state-v1"
UMT5_DEFAULT_MAX_LENGTH = 256

CLIP_ENCODER_ID = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
CLIP_DIM = 768
CLIP_POOLING = "clip-eos-pooler-text-projection-v1"
CLIP_DEFAULT_MAX_LENGTH = 77

UMT5_REQUIRED_FILES_SHA256 = {
    "text_encoder/config.json":
        "a2bcb24699f6c009a2427432bdd483ef8b2b42a712abc9503759cdc77d171f07",
    "text_encoder/model.safetensors.index.json":
        "31c4c7bcce679eaa0dd4667462394ddb013dc2f748e0bffc893dc9146a320dab",
    "text_encoder/model-00001-of-00003.safetensors":
        "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58",
    "text_encoder/model-00002-of-00003.safetensors":
        "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7",
    "text_encoder/model-00003-of-00003.safetensors":
        "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f",
    "tokenizer/special_tokens_map.json":
        "456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270",
    "tokenizer/spiece.model":
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
    "tokenizer/tokenizer.json":
        "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef",
    "tokenizer/tokenizer_config.json":
        "1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd",
}
CLIP_REQUIRED_FILES_SHA256 = {
    "config.json":
        "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
    "model.safetensors":
        "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
    "merges.txt":
        "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer.json":
        "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4",
    "tokenizer_config.json":
        "deef455e52fa5e8151e339add0582e4235f066009601360999d3a9cda83b1129",
    "vocab.json":
        "3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69",
}

ARCHIVE_NAME = "semantic_features.npz"
ROW_MANIFEST_NAME = "manifest.jsonl"
METADATA_NAME = "metadata.json"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SemanticTextRow:
    input_index: int
    iid: str
    prompt: str
    observed_target: str
    input_row_sha256: str
    prompt_text_sha256: str
    observed_target_text_sha256: str
    qwen_observation_sha256: str

    def manifest_record(self) -> dict[str, Any]:
        return {
            "schema_version": R6_SEMANTIC_MANIFEST_SCHEMA,
            "input_index": self.input_index,
            "iid": self.iid,
            "input_row_sha256": self.input_row_sha256,
            "prompt_text_sha256": self.prompt_text_sha256,
            "observed_target_text_sha256":
                self.observed_target_text_sha256,
            "qwen_observation_sha256": self.qwen_observation_sha256,
            "prompt_source_field": "prompt",
            "observed_target_source_field":
                "qwen_evidence.visual.observation.target_action",
            "observed_target_diagnostic_only": True,
            "observed_target_target_derived": True,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _object_digest(
        {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not a JSON object"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _expect_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _extract_text_rows(input_manifest: Path) -> list[SemanticTextRow]:
    input_manifest = input_manifest.expanduser().resolve(strict=True)
    if not input_manifest.is_file():
        raise FileNotFoundError(input_manifest)
    raw_rows = _load_jsonl(input_manifest)
    output: list[SemanticTextRow] = []
    seen_iids: set[str] = set()
    for input_index, row in enumerate(raw_rows):
        iid_value = row.get("iid")
        if not isinstance(iid_value, str) or not iid_value.strip():
            raise ValueError(f"row {input_index} has no non-empty iid")
        iid = iid_value.strip()
        if iid != iid_value:
            raise ValueError(f"row {input_index} iid has surrounding whitespace")
        if iid in seen_iids:
            raise ValueError(f"row {input_index} duplicates iid={iid}")
        seen_iids.add(iid)
        if "feature_index" in row and row["feature_index"] != input_index:
            raise ValueError(
                f"row {input_index} feature_index breaks manifest order"
            )

        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"row {input_index} prompt is empty/non-string")

        qwen_evidence = _expect_mapping(
            row.get("qwen_evidence"),
            name=f"row {input_index} qwen_evidence",
        )
        visual = _expect_mapping(
            qwen_evidence.get("visual"),
            name=f"row {input_index} qwen_evidence.visual",
        )
        if visual.get("status") != "ok":
            raise ValueError(f"row {input_index} Qwen visual status is not ok")
        if visual.get("observation_validated_from") != "original":
            raise ValueError(
                f"row {input_index} Qwen observation is not original"
            )
        for repair_field in ("observation_repairs", "alignment_repairs"):
            repairs = visual.get(repair_field)
            if not isinstance(repairs, list) or repairs:
                raise ValueError(
                    f"row {input_index} {repair_field} must be an empty list"
                )
        observation_value = visual.get("observation")
        if not isinstance(observation_value, dict):
            raise ValueError(f"row {input_index} observation is not an object")
        observation = dict(observation_value)
        try:
            validated = _validate_observation(observation)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"row {input_index} current observation validation failed"
            ) from error
        if validated != observation:
            raise ValueError(f"row {input_index} observation mutated on validate")
        observation_digest = _object_digest(observation)
        if visual.get("observation_digest") != observation_digest:
            raise ValueError(
                f"row {input_index} observation digest does not match"
            )
        observation_raw = visual.get("observation_raw")
        if not isinstance(observation_raw, str) or not observation_raw.strip():
            raise ValueError(f"row {input_index} observation_raw is missing")
        try:
            # Use the exact parser which originally established
            # observation_validated_from="original".  It accepts a fenced
            # object or surrounding non-object text, but performs no semantic
            # repair; the parsed object must still equal the committed one.
            raw_observation = _parse_object(observation_raw)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                f"row {input_index} observation_raw is invalid JSON"
            ) from error
        if raw_observation != observation:
            raise ValueError(
                f"row {input_index} original observation_raw differs from "
                "validated observation"
            )
        observed_target = observation.get("target_action")
        if (
            not isinstance(observed_target, str)
            or not observed_target.strip()
        ):
            raise ValueError(
                f"row {input_index} observed target_action is empty/non-string"
            )

        output.append(
            SemanticTextRow(
                input_index=input_index,
                iid=iid,
                prompt=prompt,
                observed_target=observed_target,
                input_row_sha256=_object_digest(row),
                prompt_text_sha256=_text_digest(prompt),
                observed_target_text_sha256=_text_digest(observed_target),
                qwen_observation_sha256=observation_digest,
            )
        )
    return output


def _checked_revision(value: str, *, expected: str, name: str) -> str:
    revision = str(value).strip().lower()
    if _IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"{name} revision must be a full 40-hex git commit")
    if revision != expected:
        raise ValueError(
            f"{name} revision is unregistered: expected {expected}, got "
            f"{revision}"
        )
    return revision


def _checked_sha256(value: str, *, name: str) -> str:
    digest = str(value).strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a full 64-hex SHA-256")
    return digest


def _validate_source_snapshot_binding(
    *,
    source_snapshot: Path,
    source_tree_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Fully revalidate and bind the immutable executable source snapshot."""

    tree_digest = _checked_sha256(
        source_tree_sha256,
        name="source_tree_sha256",
    )
    manifest_digest = _checked_sha256(
        source_manifest_sha256,
        name="source_manifest_sha256",
    )
    snapshot = source_snapshot.expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise NotADirectoryError(snapshot)
    manifest_path = snapshot / SOURCE_FILES_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(
            f"source snapshot manifest is missing/non-regular: "
            f"{manifest_path}"
        )
    if _file_digest(manifest_path) != manifest_digest:
        raise ValueError("source snapshot SOURCE_FILES.jsonl SHA-256 mismatch")
    actual_manifest_text = manifest_path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        actual_manifest_text.splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(
                f"source snapshot manifest line {line_number} is blank"
            )
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"source snapshot manifest line {line_number} is not an "
                "object"
            )
        rows.append(value)
    if not rows:
        raise ValueError("source snapshot manifest contains no files")
    canonical_manifest_text = "".join(
        _canonical_json(row) + "\n" for row in rows
    )
    if actual_manifest_text != canonical_manifest_text:
        raise ValueError("source snapshot manifest is not canonical")
    actual_tree_digest = hashlib.sha256(
        canonical_manifest_text.encode("utf-8")
    ).hexdigest()
    if actual_tree_digest != tree_digest:
        raise ValueError("source snapshot tree SHA-256 mismatch")

    seen: set[str] = set()
    manifest_file_digests: dict[str, str] = {}
    required_keys = {"mode", "path", "sha256", "size", "type"}
    for index, row in enumerate(rows):
        if set(row) != required_keys or row.get("type") != "file":
            raise ValueError(
                f"source snapshot row {index} has an invalid file contract"
            )
        relative = str(row.get("path") or "")
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
        ):
            raise ValueError(
                f"invalid/duplicate source snapshot path: {relative!r}"
            )
        seen.add(relative)
        expected_file_digest = _checked_sha256(
            str(row.get("sha256")),
            name=f"source snapshot {relative} sha256",
        )
        path = snapshot / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"source snapshot file is missing/non-regular: {relative}"
            )
        if path.stat().st_size != int(row.get("size", -1)):
            raise ValueError(f"source snapshot size mismatch: {relative}")
        actual_mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
        if actual_mode != row.get("mode"):
            raise ValueError(f"source snapshot mode mismatch: {relative}")
        if _file_digest(path) != expected_file_digest:
            raise ValueError(f"source snapshot file SHA mismatch: {relative}")
        manifest_file_digests[relative] = expected_file_digest

    provenance_path = snapshot / SOURCE_PROVENANCE_NAME
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise FileNotFoundError(provenance_path)
    provenance = _load_json(provenance_path)
    if (
        provenance.get("schema") != SOURCE_SNAPSHOT_UPSTREAM_SCHEMA
        or provenance.get("source_tree_sha256") != tree_digest
        or provenance.get("source_manifest_sha256") != manifest_digest
        or provenance.get("source_file_count") != len(rows)
    ):
        raise ValueError("source snapshot upstream provenance differs")

    package = Path(__file__).resolve().parent
    current_implementation = {
        "methods/motive/motive/r6_semantic_features.py":
            _file_digest(Path(__file__).resolve()),
        "methods/motive/motive/qwen_filter.py":
            _file_digest(package / "qwen_filter.py"),
    }
    for relative, current_digest in current_implementation.items():
        if manifest_file_digests.get(relative) != current_digest:
            raise ValueError(
                f"executed implementation is not bound to source snapshot: "
                f"{relative}"
            )
    return {
        "schema_version": R6_SOURCE_SNAPSHOT_SCHEMA,
        "synthetic_test_sentinel": False,
        "resolved_path": str(snapshot),
        "source_tree_sha256": tree_digest,
        "source_manifest_sha256": manifest_digest,
        "source_files_manifest": SOURCE_FILES_NAME,
        "source_file_count": len(rows),
        "executed_implementation_files_sha256": current_implementation,
    }


def _synthetic_source_snapshot_binding() -> dict[str, Any]:
    return {
        "schema_version": R6_SOURCE_SNAPSHOT_SCHEMA,
        "synthetic_test_sentinel": True,
        "sentinel": R6_SOURCE_SNAPSHOT_SENTINEL,
        "resolved_path": None,
        "source_tree_sha256": "0" * 64,
        "source_manifest_sha256": "0" * 64,
        "source_files_manifest": SOURCE_FILES_NAME,
        "source_file_count": 0,
        "executed_implementation_files_sha256": {},
    }


def _verify_registered_files(
    root: Path,
    expected: Mapping[str, str],
    *,
    name: str,
) -> tuple[Path, dict[str, str]]:
    resolved = root.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    actual: dict[str, str] = {}
    for relative, expected_digest in sorted(expected.items()):
        path = resolved / relative
        if not path.is_file():
            raise FileNotFoundError(f"{name} required file missing: {path}")
        digest = _file_digest(path)
        if digest != expected_digest:
            raise ValueError(
                f"{name} registered checksum mismatch for {relative}: "
                f"expected {expected_digest}, got {digest}"
            )
        actual[relative] = digest
    return resolved, actual


def _model_file_digest(
    files: Mapping[str, str],
    names: Sequence[str],
) -> str:
    selected = {name: files[name] for name in sorted(names)}
    return _object_digest(selected)


def _load_model_config(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    return value


def _registered_encoder_provenance(
    *,
    umt5_root: Path,
    umt5_revision: str,
    umt5_max_length: int,
    clip_root: Path,
    clip_revision: str,
    clip_max_length: int,
) -> dict[str, dict[str, Any]]:
    if umt5_max_length < 2:
        raise ValueError("UMT5 max length must be >= 2")
    if clip_max_length != CLIP_DEFAULT_MAX_LENGTH:
        raise ValueError("registered CLIP contract requires max_length=77")
    umt5_revision = _checked_revision(
        umt5_revision,
        expected=UMT5_REVISION,
        name="UMT5",
    )
    clip_revision = _checked_revision(
        clip_revision,
        expected=CLIP_REVISION,
        name="CLIP",
    )
    umt5_resolved, umt5_files = _verify_registered_files(
        umt5_root,
        UMT5_REQUIRED_FILES_SHA256,
        name="UMT5",
    )
    clip_resolved, clip_files = _verify_registered_files(
        clip_root,
        CLIP_REQUIRED_FILES_SHA256,
        name="CLIP",
    )
    umt5_config = _load_model_config(
        umt5_resolved / "text_encoder" / "config.json"
    )
    if (
        umt5_config.get("model_type") != "umt5"
        or umt5_config.get("d_model") != UMT5_DIM
        or umt5_config.get("architectures") != ["UMT5EncoderModel"]
    ):
        raise ValueError("registered UMT5 config contract changed")
    clip_config = _load_model_config(clip_resolved / "config.json")
    text_config = _expect_mapping(
        clip_config.get("text_config"),
        name="CLIP text_config",
    )
    if (
        clip_config.get("model_type") != "clip"
        or clip_config.get("projection_dim") != CLIP_DIM
        or text_config.get("hidden_size") != CLIP_DIM
    ):
        raise ValueError("registered CLIP config contract changed")

    umt5_weights = [
        name for name in umt5_files
        if name.startswith("text_encoder/model-")
        and name.endswith(".safetensors")
    ]
    umt5_tokenizer = [
        name for name in umt5_files if name.startswith("tokenizer/")
    ]
    return {
        "umt5": {
            "encoder_id": UMT5_ENCODER_ID,
            "resolved_path": str(umt5_resolved),
            "model_subdir": "text_encoder",
            "tokenizer_subdir": "tokenizer",
            "revision": umt5_revision,
            "registered_model": True,
            "required_files_sha256": umt5_files,
            "weights_sha256": _model_file_digest(
                umt5_files,
                umt5_weights,
            ),
            "weight_files_sha256": {
                name: umt5_files[name] for name in sorted(umt5_weights)
            },
            "tokenizer_sha256": _model_file_digest(
                umt5_files,
                umt5_tokenizer,
            ),
            "tokenizer_files_sha256": {
                name: umt5_files[name] for name in sorted(umt5_tokenizer)
            },
            "config_sha256": umt5_files[
                "text_encoder/config.json"
            ],
            "index_sha256": umt5_files[
                "text_encoder/model.safetensors.index.json"
            ],
            "pooling": UMT5_POOLING,
            "embedding_dim": UMT5_DIM,
            "model_compute_dtype": "bfloat16",
            "output_dtype": "float32",
            "normalization": "l2_per_row",
            "max_length": int(umt5_max_length),
            "prompt_template_version": R6_PROMPT_TEMPLATE_VERSION,
            "frozen_encoder": True,
        },
        "clip": {
            "encoder_id": CLIP_ENCODER_ID,
            "resolved_path": str(clip_resolved),
            "model_subdir": ".",
            "tokenizer_subdir": ".",
            "revision": clip_revision,
            "registered_model": True,
            "required_files_sha256": clip_files,
            "weights_sha256": clip_files["model.safetensors"],
            "weight_files_sha256": {
                "model.safetensors": clip_files["model.safetensors"],
            },
            "tokenizer_sha256": _model_file_digest(
                clip_files,
                [
                    "merges.txt",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ],
            ),
            "tokenizer_files_sha256": {
                name: clip_files[name]
                for name in (
                    "merges.txt",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                )
            },
            "config_sha256": clip_files["config.json"],
            "pooling": CLIP_POOLING,
            "embedding_dim": CLIP_DIM,
            "model_compute_dtype": "float32",
            "output_dtype": "float32",
            "normalization": "l2_per_row",
            "max_length": int(clip_max_length),
            "prompt_template_version": R6_PROMPT_TEMPLATE_VERSION,
            "frozen_encoder": True,
        },
    }


def _implementation_provenance() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    files = {
        "r6_semantic_features.py": _file_digest(Path(__file__).resolve()),
        "qwen_filter.py": _file_digest(package / "qwen_filter.py"),
    }
    return {
        "files_sha256": files,
        "aggregate_sha256": _object_digest(files),
    }


def _gpu_preflight() -> tuple[Any, dict[str, Any]]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for R6 extraction") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "R6 semantic extraction requires a real CUDA/ROCm-visible GPU"
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("torch reports zero CUDA/ROCm devices")
    device = torch.device("cuda:0")
    try:
        left = torch.arange(
            16,
            dtype=torch.float32,
            device=device,
        ).reshape(4, 4)
        result = left @ left.T
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("GPU preflight produced non-finite values")
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
    except Exception as error:
        raise RuntimeError("real GPU tensor preflight failed") from error
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "transformers": None,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": int(properties.total_memory),
        "gpu_tensor_preflight": True,
    }
    return torch, runtime


def _tokenize_lengths(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_lengths: list[int] = []
    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise RuntimeError("tokenizer produced invalid input_ids")
        if input_ids and isinstance(input_ids[0], list):
            raise RuntimeError("single-text tokenizer unexpectedly batched")
        raw_lengths.append(len(input_ids))
    raw = np.asarray(raw_lengths, dtype=np.int32)
    actual = np.minimum(raw, int(max_length)).astype(np.int32)
    truncated = (raw > int(max_length)).astype(np.bool_)
    if bool((actual <= 0).any()):
        raise RuntimeError("tokenizer produced an empty sequence")
    return actual, raw, truncated


def _l2_float32(matrix: Any, *, rows: int, dim: int, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.shape != (rows, dim):
        raise RuntimeError(
            f"{name} shape mismatch: expected {(rows, dim)}, got "
            f"{value.shape}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values")
    norms = np.linalg.norm(value.astype(np.float64), axis=1, keepdims=True)
    if bool((norms <= 1e-12).any()):
        raise RuntimeError(f"{name} contains a zero vector")
    return np.ascontiguousarray(
        value / norms.astype(np.float32),
        dtype=np.float32,
    )


def _encode_umt5(
    *,
    torch: Any,
    root: Path,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from transformers import AutoTokenizer, UMT5EncoderModel
    except ImportError as error:
        raise RuntimeError(
            "transformers with UMT5EncoderModel is required"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    lengths, raw_lengths, truncated = _tokenize_lengths(
        tokenizer,
        texts,
        max_length=max_length,
    )
    model = UMT5EncoderModel.from_pretrained(
        str(root / "text_encoder"),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(torch.device("cuda:0"))
    pieces: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to("cuda:0", non_blocking=False)
                    for key, value in tokens.items()
                    if key in {"input_ids", "attention_mask"}
                }
                output = model(**tokens, return_dict=True).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(
                    dtype=torch.float32
                )
                pooled = (
                    output.to(dtype=torch.float32) * mask
                ).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.nn.functional.normalize(
                    pooled,
                    p=2.0,
                    dim=1,
                    eps=1e-12,
                )
                pieces.append(pooled.cpu().numpy().astype(np.float32))
        torch.cuda.synchronize()
    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    matrix = _l2_float32(
        np.concatenate(pieces, axis=0),
        rows=len(texts),
        dim=UMT5_DIM,
        name="UMT5 embeddings",
    )
    return matrix, lengths, raw_lengths, truncated


def _encode_clip(
    *,
    torch: Any,
    root: Path,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from transformers import CLIPModel, CLIPTokenizerFast
    except ImportError as error:
        raise RuntimeError("transformers with CLIPModel is required") from error
    tokenizer = CLIPTokenizerFast.from_pretrained(
        str(root),
        local_files_only=True,
    )
    lengths, raw_lengths, truncated = _tokenize_lengths(
        tokenizer,
        texts,
        max_length=max_length,
    )
    model = CLIPModel.from_pretrained(
        str(root),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(torch.device("cuda:0"))
    pieces: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to("cuda:0", non_blocking=False)
                    for key, value in tokens.items()
                    if key in {"input_ids", "attention_mask"}
                }
                text_output = model.text_model(**tokens, return_dict=True)
                pooled = model.text_projection(
                    text_output.pooler_output
                ).to(dtype=torch.float32)
                pooled = torch.nn.functional.normalize(
                    pooled,
                    p=2.0,
                    dim=1,
                    eps=1e-12,
                )
                pieces.append(pooled.cpu().numpy().astype(np.float32))
        torch.cuda.synchronize()
    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    matrix = _l2_float32(
        np.concatenate(pieces, axis=0),
        rows=len(texts),
        dim=CLIP_DIM,
        name="CLIP embeddings",
    )
    return matrix, lengths, raw_lengths, truncated


def _string_array(values: Sequence[str]) -> np.ndarray:
    width = max(1, max(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


def _token_stats(
    arrays: Mapping[str, np.ndarray],
    *,
    encoder: str,
    field: str,
) -> dict[str, Any]:
    prefix = f"{encoder}_{field}"
    actual = np.asarray(arrays[f"{prefix}_token_length"], dtype=np.int32)
    raw = np.asarray(arrays[f"{prefix}_raw_token_length"], dtype=np.int32)
    truncated = np.asarray(arrays[f"{prefix}_truncated"], dtype=np.bool_)
    return {
        "minimum": int(actual.min()),
        "maximum": int(actual.max()),
        "mean": float(actual.astype(np.float64).mean()),
        "raw_maximum": int(raw.max()),
        "truncated_rows": int(truncated.sum()),
    }


def _array_contract(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": _array_digest(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }


def _synthetic_encoder_provenance() -> dict[str, dict[str, Any]]:
    fake = "0" * 64
    common = {
        "resolved_path": "synthetic-test-artifact",
        "revision": fake,
        "registered_model": False,
        "required_files_sha256": {},
        "weights_sha256": fake,
        "weight_files_sha256": {},
        "tokenizer_sha256": fake,
        "tokenizer_files_sha256": {},
        "config_sha256": fake,
        "output_dtype": "float32",
        "normalization": "l2_per_row",
        "prompt_template_version": R6_PROMPT_TEMPLATE_VERSION,
        "frozen_encoder": True,
    }
    return {
        "umt5": {
            **common,
            "encoder_id": "synthetic/umt5-test-double",
            "model_subdir": ".",
            "tokenizer_subdir": ".",
            "pooling": UMT5_POOLING,
            "embedding_dim": UMT5_DIM,
            "model_compute_dtype": "synthetic",
            "max_length": UMT5_DEFAULT_MAX_LENGTH,
        },
        "clip": {
            **common,
            "encoder_id": "synthetic/clip-test-double",
            "model_subdir": ".",
            "tokenizer_subdir": ".",
            "pooling": CLIP_POOLING,
            "embedding_dim": CLIP_DIM,
            "model_compute_dtype": "synthetic",
            "max_length": CLIP_DEFAULT_MAX_LENGTH,
        },
    }


def _build_metadata(
    *,
    input_manifest: Path,
    text_rows: Sequence[SemanticTextRow],
    arrays_without_metadata: Mapping[str, np.ndarray],
    encoders: Mapping[str, Mapping[str, Any]],
    source_snapshot_binding: Mapping[str, Any],
    runtime: Mapping[str, Any],
    synthetic: bool,
) -> dict[str, Any]:
    row_records = [row.manifest_record() for row in text_rows]
    return {
        "schema_version": R6_SEMANTIC_METADATA_SCHEMA,
        "feature_schema_version": R6_SEMANTIC_SCHEMA,
        "synthetic_test_artifact": bool(synthetic),
        "source_snapshot": dict(source_snapshot_binding),
        "rows": len(text_rows),
        "input_manifest": {
            "resolved_path": str(input_manifest.resolve(strict=True)),
            "sha256": _file_digest(input_manifest),
            "rows": len(text_rows),
            "iid_order_sha256": _object_digest(
                [row.iid for row in text_rows]
            ),
        },
        "row_contract_sha256": _object_digest(row_records),
        "prompt_text_order_sha256": _object_digest(
            [row.prompt_text_sha256 for row in text_rows]
        ),
        "observed_target_text_order_sha256": _object_digest(
            [row.observed_target_text_sha256 for row in text_rows]
        ),
        "source_fields": {
            "prompt": {
                "path": "prompt",
                "instruction_only": True,
                "target_derived": False,
                "label_derived": False,
                "allowed_predictor_input": True,
            },
            "observed_target": {
                "path":
                    "qwen_evidence.visual.observation.target_action",
                "requires_status": "ok",
                "requires_validated_from": "original",
                "requires_no_repairs": True,
                "requires_raw_object_equality": True,
                "target_derived": True,
                "diagnostic_only": True,
                "allowed_predictor_input": False,
            },
        },
        "fit_contract": {
            "pca_fitted": False,
            "standardizer_fitted": False,
            "centroid_fitted": False,
            "pairing_performed": False,
            "raw_frozen_l2_embeddings": True,
            "future_fit_scope": "trainer-train-iids-only",
        },
        "encoders": {
            name: dict(value) for name, value in sorted(encoders.items())
        },
        "arrays": _array_contract(arrays_without_metadata),
        "token_length_stats": {
            f"{encoder}_{field}": _token_stats(
                arrays_without_metadata,
                encoder=encoder,
                field=field,
            )
            for encoder in ("umt5", "clip")
            for field in ("prompt", "observed_target")
        },
        "runtime": dict(runtime),
        "implementation": _implementation_provenance(),
    }


def _commit(
    *,
    input_manifest: Path,
    output_dir: Path,
    text_rows: Sequence[SemanticTextRow],
    arrays: Mapping[str, np.ndarray],
    encoders: Mapping[str, Mapping[str, Any]],
    source_snapshot_binding: Mapping[str, Any],
    runtime: Mapping[str, Any],
    synthetic: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "archive": output_dir / ARCHIVE_NAME,
        "manifest": output_dir / ROW_MANIFEST_NAME,
        "metadata": output_dir / METADATA_NAME,
        "summary": output_dir / SUMMARY_NAME,
        "done": output_dir / DONE_NAME,
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing/partial R6 artifacts: "
            + ", ".join(existing)
        )
    rows = len(text_rows)
    required_matrix_shapes = {
        "umt5_prompt": (rows, UMT5_DIM),
        "umt5_observed_target": (rows, UMT5_DIM),
        "clip_prompt": (rows, CLIP_DIM),
        "clip_observed_target": (rows, CLIP_DIM),
    }
    committed_arrays: dict[str, np.ndarray] = {
        name: np.asarray(value) for name, value in arrays.items()
    }
    for name, shape in required_matrix_shapes.items():
        matrix = committed_arrays.get(name)
        if matrix is None:
            raise ValueError(f"missing required matrix {name}")
        committed_arrays[name] = _l2_float32(
            matrix,
            rows=rows,
            dim=shape[1],
            name=name,
        )
    committed_arrays["iids"] = _string_array([row.iid for row in text_rows])
    committed_arrays["prompt_text_sha256"] = _string_array(
        [row.prompt_text_sha256 for row in text_rows]
    )
    committed_arrays["observed_target_text_sha256"] = _string_array(
        [row.observed_target_text_sha256 for row in text_rows]
    )
    token_names = [
        f"{encoder}_{field}_{suffix}"
        for encoder in ("umt5", "clip")
        for field in ("prompt", "observed_target")
        for suffix in ("token_length", "raw_token_length", "truncated")
    ]
    for name in token_names:
        if name not in committed_arrays:
            raise ValueError(f"missing token provenance array {name}")
        expected_dtype = np.bool_ if name.endswith("_truncated") else np.int32
        value = np.asarray(committed_arrays[name], dtype=expected_dtype)
        if value.shape != (rows,):
            raise ValueError(f"{name} must have shape {(rows,)}")
        committed_arrays[name] = value

    metadata = _build_metadata(
        input_manifest=input_manifest,
        text_rows=text_rows,
        arrays_without_metadata=committed_arrays,
        encoders=encoders,
        source_snapshot_binding=source_snapshot_binding,
        runtime=runtime,
        synthetic=synthetic,
    )
    archive_arrays = dict(committed_arrays)
    archive_arrays["metadata_json"] = np.asarray(_canonical_json(metadata))
    row_records = [row.manifest_record() for row in text_rows]
    _atomic_npz(paths["archive"], archive_arrays)
    _atomic_jsonl(paths["manifest"], row_records)
    _atomic_json(paths["metadata"], metadata)
    summary = {
        "schema_version": R6_SEMANTIC_SUMMARY_SCHEMA,
        "status": "complete",
        "synthetic_test_artifact": bool(synthetic),
        "source_snapshot": metadata["source_snapshot"],
        "rows": rows,
        "input_manifest_sha256": metadata["input_manifest"]["sha256"],
        "iid_order_sha256": metadata["input_manifest"][
            "iid_order_sha256"
        ],
        "row_contract_sha256": metadata["row_contract_sha256"],
        "archive_sha256": _file_digest(paths["archive"]),
        "row_manifest_sha256": _file_digest(paths["manifest"]),
        "encoders": {
            name: {
                "encoder_id": value["encoder_id"],
                "revision": value["revision"],
                "resolved_path": value["resolved_path"],
                "weights_sha256": value["weights_sha256"],
                "tokenizer_sha256": value["tokenizer_sha256"],
                "pooling": value["pooling"],
                "embedding_dim": value["embedding_dim"],
                "output_dtype": value["output_dtype"],
                "normalization": value["normalization"],
                "max_length": value["max_length"],
            }
            for name, value in sorted(encoders.items())
        },
        "token_length_stats": metadata["token_length_stats"],
        "fit_contract": metadata["fit_contract"],
    }
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": R6_SEMANTIC_DONE_SCHEMA,
        "status": "complete",
        "rows": rows,
        "synthetic_test_artifact": bool(synthetic),
        "source_snapshot": metadata["source_snapshot"],
        "input_manifest_sha256": metadata["input_manifest"]["sha256"],
        "iid_order_sha256": metadata["input_manifest"][
            "iid_order_sha256"
        ],
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": _file_digest(path),
            }
            for name, path in paths.items()
            if name != "done"
        },
    }
    _atomic_json(paths["done"], done)
    return done


def commit_synthetic_artifact(
    *,
    input_manifest: Path,
    output_dir: Path,
    seed: int = 260108828,
    source_snapshot_sentinel: str = R6_SOURCE_SNAPSHOT_SENTINEL,
) -> dict[str, Any]:
    """Build a fully validated test artifact without loading real models."""

    if source_snapshot_sentinel != R6_SOURCE_SNAPSHOT_SENTINEL:
        raise ValueError("unsupported synthetic source snapshot sentinel")
    input_manifest = input_manifest.expanduser().resolve(strict=True)
    text_rows = _extract_text_rows(input_manifest)
    rows = len(text_rows)
    rng = np.random.default_rng(int(seed))
    arrays: dict[str, np.ndarray] = {}
    for encoder, dim in (("umt5", UMT5_DIM), ("clip", CLIP_DIM)):
        max_length = (
            UMT5_DEFAULT_MAX_LENGTH
            if encoder == "umt5"
            else CLIP_DEFAULT_MAX_LENGTH
        )
        for field in ("prompt", "observed_target"):
            arrays[f"{encoder}_{field}"] = _l2_float32(
                rng.normal(size=(rows, dim)).astype(np.float32),
                rows=rows,
                dim=dim,
                name=f"synthetic {encoder}_{field}",
            )
            raw = np.asarray(
                [
                    max(
                        2,
                        len(
                            (
                                row.prompt
                                if field == "prompt"
                                else row.observed_target
                            ).split()
                        ) + 1,
                    )
                    for row in text_rows
                ],
                dtype=np.int32,
            )
            arrays[f"{encoder}_{field}_raw_token_length"] = raw
            arrays[f"{encoder}_{field}_token_length"] = np.minimum(
                raw,
                max_length,
            ).astype(np.int32)
            arrays[f"{encoder}_{field}_truncated"] = (
                raw > max_length
            ).astype(np.bool_)
    return _commit(
        input_manifest=input_manifest,
        output_dir=output_dir,
        text_rows=text_rows,
        arrays=arrays,
        encoders=_synthetic_encoder_provenance(),
        source_snapshot_binding=_synthetic_source_snapshot_binding(),
        runtime={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": None,
            "transformers": None,
            "torch_hip": None,
            "torch_cuda": None,
            "device_index": None,
            "device_name": "synthetic-test-double",
            "device_total_memory_bytes": 0,
            "gpu_tensor_preflight": False,
        },
        synthetic=True,
    )


def _expected_array_names() -> set[str]:
    names = {
        "iids",
        "prompt_text_sha256",
        "observed_target_text_sha256",
        "metadata_json",
        "umt5_prompt",
        "umt5_observed_target",
        "clip_prompt",
        "clip_observed_target",
    }
    names.update(
        f"{encoder}_{field}_{suffix}"
        for encoder in ("umt5", "clip")
        for field in ("prompt", "observed_target")
        for suffix in ("token_length", "raw_token_length", "truncated")
    )
    return names


def _validate_encoder_metadata(
    metadata: Mapping[str, Any],
    *,
    synthetic: bool,
) -> None:
    encoders = _expect_mapping(metadata.get("encoders"), name="encoders")
    if set(encoders) != {"umt5", "clip"}:
        raise ValueError("metadata must contain exactly UMT5 and CLIP")
    if synthetic:
        for name, value in encoders.items():
            encoder = _expect_mapping(value, name=f"encoders.{name}")
            if encoder.get("registered_model") is not False:
                raise ValueError("synthetic encoder marked registered")
        return
    umt5 = _expect_mapping(encoders["umt5"], name="encoders.umt5")
    clip = _expect_mapping(encoders["clip"], name="encoders.clip")
    rebuilt = _registered_encoder_provenance(
        umt5_root=Path(str(umt5["resolved_path"])),
        umt5_revision=str(umt5["revision"]),
        umt5_max_length=int(umt5["max_length"]),
        clip_root=Path(str(clip["resolved_path"])),
        clip_revision=str(clip["revision"]),
        clip_max_length=int(clip["max_length"]),
    )
    if rebuilt != dict(encoders):
        raise ValueError("registered encoder provenance changed")


def validate_artifact(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser()
    paths = {
        "archive": output_dir / ARCHIVE_NAME,
        "manifest": output_dir / ROW_MANIFEST_NAME,
        "metadata": output_dir / METADATA_NAME,
        "summary": output_dir / SUMMARY_NAME,
        "done": output_dir / DONE_NAME,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    if (
        done.get("schema_version") != R6_SEMANTIC_DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise ValueError("invalid R6 done marker")
    artifacts = _expect_mapping(done.get("artifacts"), name="done.artifacts")
    if set(artifacts) != {"archive", "manifest", "metadata", "summary"}:
        raise ValueError("done artifact set differs")
    for name in sorted(artifacts):
        contract = _expect_mapping(
            artifacts[name],
            name=f"done.artifacts.{name}",
        )
        if contract.get("filename") != paths[name].name:
            raise ValueError(f"done filename mismatch for {name}")
        if contract.get("sha256") != _file_digest(paths[name]):
            raise ValueError(f"done SHA-256 mismatch for {name}")

    metadata = _load_json(paths["metadata"])
    if metadata.get("schema_version") != R6_SEMANTIC_METADATA_SCHEMA:
        raise ValueError("metadata schema mismatch")
    if metadata.get("feature_schema_version") != R6_SEMANTIC_SCHEMA:
        raise ValueError("feature schema mismatch")
    synthetic = metadata.get("synthetic_test_artifact")
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic_test_artifact must be boolean")
    if done.get("synthetic_test_artifact") is not synthetic:
        raise ValueError("done/metadata synthetic flag mismatch")
    if metadata.get("implementation") != _implementation_provenance():
        raise ValueError("implementation provenance changed")
    source_snapshot_contract = _expect_mapping(
        metadata.get("source_snapshot"),
        name="metadata.source_snapshot",
    )
    if synthetic:
        rebuilt_source_snapshot = _synthetic_source_snapshot_binding()
    else:
        if source_snapshot_contract.get("synthetic_test_sentinel") is not False:
            raise ValueError("real artifact has a synthetic snapshot sentinel")
        rebuilt_source_snapshot = _validate_source_snapshot_binding(
            source_snapshot=Path(
                str(source_snapshot_contract.get("resolved_path"))
            ),
            source_tree_sha256=str(
                source_snapshot_contract.get("source_tree_sha256")
            ),
            source_manifest_sha256=str(
                source_snapshot_contract.get("source_manifest_sha256")
            ),
        )
    if dict(source_snapshot_contract) != rebuilt_source_snapshot:
        raise ValueError("source snapshot provenance changed")
    if done.get("source_snapshot") != rebuilt_source_snapshot:
        raise ValueError("done/metadata source snapshot provenance differs")

    input_contract = _expect_mapping(
        metadata.get("input_manifest"),
        name="metadata.input_manifest",
    )
    input_manifest = Path(str(input_contract.get("resolved_path")))
    if not input_manifest.is_file():
        raise FileNotFoundError(input_manifest)
    if _file_digest(input_manifest) != input_contract.get("sha256"):
        raise ValueError("input R5 manifest SHA-256 changed")
    text_rows = _extract_text_rows(input_manifest)
    rows = len(text_rows)
    if (
        metadata.get("rows") != rows
        or input_contract.get("rows") != rows
        or done.get("rows") != rows
    ):
        raise ValueError("row counts disagree")
    iids = [row.iid for row in text_rows]
    iid_order_sha256 = _object_digest(iids)
    if (
        input_contract.get("iid_order_sha256") != iid_order_sha256
        or done.get("iid_order_sha256") != iid_order_sha256
    ):
        raise ValueError("IID order digest differs")
    row_records = [row.manifest_record() for row in text_rows]
    if _object_digest(row_records) != metadata.get("row_contract_sha256"):
        raise ValueError("metadata row contract digest differs")
    if _load_jsonl(paths["manifest"]) != row_records:
        raise ValueError("semantic row manifest differs from R5 order/text")

    _validate_encoder_metadata(metadata, synthetic=synthetic)
    source_fields = _expect_mapping(
        metadata.get("source_fields"),
        name="metadata.source_fields",
    )
    if (
        source_fields.get("prompt", {}).get("allowed_predictor_input")
        is not True
        or source_fields.get("observed_target", {}).get(
            "allowed_predictor_input"
        )
        is not False
        or source_fields.get("observed_target", {}).get("target_derived")
        is not True
        or source_fields.get("observed_target", {}).get("diagnostic_only")
        is not True
    ):
        raise ValueError("semantic source-field safety contract changed")
    fit_contract = _expect_mapping(
        metadata.get("fit_contract"),
        name="metadata.fit_contract",
    )
    if (
        fit_contract.get("pca_fitted") is not False
        or fit_contract.get("standardizer_fitted") is not False
        or fit_contract.get("raw_frozen_l2_embeddings") is not True
    ):
        raise ValueError("extraction contains a forbidden fitted transform")

    with np.load(paths["archive"], allow_pickle=False) as loaded:
        if set(loaded.files) != _expected_array_names():
            raise ValueError(
                "archive keys differ: "
                f"{sorted(set(loaded.files) ^ _expected_array_names())}"
            )
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    metadata_json = arrays.pop("metadata_json")
    if metadata_json.ndim != 0:
        raise ValueError("metadata_json must be scalar")
    if str(metadata_json.item()) != _canonical_json(metadata):
        raise ValueError("archive/external metadata differs")
    if arrays["iids"].tolist() != iids:
        raise ValueError("archive IID order differs")
    prompt_digests = [row.prompt_text_sha256 for row in text_rows]
    observed_digests = [
        row.observed_target_text_sha256 for row in text_rows
    ]
    if arrays["prompt_text_sha256"].tolist() != prompt_digests:
        raise ValueError("archive prompt text digests differ")
    if arrays["observed_target_text_sha256"].tolist() != observed_digests:
        raise ValueError("archive observed-target text digests differ")
    if metadata.get("prompt_text_order_sha256") != _object_digest(
        prompt_digests
    ):
        raise ValueError("prompt digest order provenance differs")
    if metadata.get("observed_target_text_order_sha256") != _object_digest(
        observed_digests
    ):
        raise ValueError("observed-target digest order provenance differs")

    expected_shapes = {
        "umt5_prompt": (rows, UMT5_DIM),
        "umt5_observed_target": (rows, UMT5_DIM),
        "clip_prompt": (rows, CLIP_DIM),
        "clip_observed_target": (rows, CLIP_DIM),
    }
    for name, shape in expected_shapes.items():
        matrix = arrays[name]
        if matrix.shape != shape or matrix.dtype != np.float32:
            raise ValueError(
                f"{name} expected float32 {shape}, got "
                f"{matrix.dtype} {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
            raise ValueError(f"{name} is not per-row L2 normalized")
    encoders = metadata["encoders"]
    for encoder in ("umt5", "clip"):
        max_length = int(encoders[encoder]["max_length"])
        for field in ("prompt", "observed_target"):
            prefix = f"{encoder}_{field}"
            actual = arrays[f"{prefix}_token_length"]
            raw = arrays[f"{prefix}_raw_token_length"]
            truncated = arrays[f"{prefix}_truncated"]
            if (
                actual.shape != (rows,)
                or raw.shape != (rows,)
                or truncated.shape != (rows,)
                or actual.dtype != np.int32
                or raw.dtype != np.int32
                or truncated.dtype != np.bool_
            ):
                raise ValueError(f"{prefix} token provenance contract differs")
            if bool((actual <= 0).any()) or bool((raw < actual).any()):
                raise ValueError(f"{prefix} token lengths are invalid")
            if bool((actual > max_length).any()):
                raise ValueError(f"{prefix} exceeds registered max_length")
            if not np.array_equal(truncated, raw > max_length):
                raise ValueError(f"{prefix} truncation flags differ")
            if not np.array_equal(actual, np.minimum(raw, max_length)):
                raise ValueError(f"{prefix} actual/raw lengths disagree")

    recorded_arrays = _expect_mapping(
        metadata.get("arrays"),
        name="metadata.arrays",
    )
    if set(recorded_arrays) != set(arrays):
        raise ValueError("metadata array contract keys differ")
    for name, value in arrays.items():
        expected = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_digest(value),
        }
        if recorded_arrays.get(name) != expected:
            raise ValueError(f"metadata array contract differs for {name}")
    rebuilt_stats = {
        f"{encoder}_{field}": _token_stats(
            arrays,
            encoder=encoder,
            field=field,
        )
        for encoder in ("umt5", "clip")
        for field in ("prompt", "observed_target")
    }
    if metadata.get("token_length_stats") != rebuilt_stats:
        raise ValueError("token length summary differs")

    summary = _load_json(paths["summary"])
    if (
        summary.get("schema_version") != R6_SEMANTIC_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("rows") != rows
        or summary.get("archive_sha256") != _file_digest(paths["archive"])
        or summary.get("row_manifest_sha256")
        != _file_digest(paths["manifest"])
        or summary.get("row_contract_sha256")
        != metadata["row_contract_sha256"]
        or summary.get("source_snapshot") != rebuilt_source_snapshot
        or summary.get("encoders")
        != {
            name: {
                "encoder_id": value["encoder_id"],
                "revision": value["revision"],
                "resolved_path": value["resolved_path"],
                "weights_sha256": value["weights_sha256"],
                "tokenizer_sha256": value["tokenizer_sha256"],
                "pooling": value["pooling"],
                "embedding_dim": value["embedding_dim"],
                "output_dtype": value["output_dtype"],
                "normalization": value["normalization"],
                "max_length": value["max_length"],
            }
            for name, value in sorted(metadata["encoders"].items())
        }
    ):
        raise ValueError("summary contract differs")
    return {
        "rows": rows,
        "synthetic_test_artifact": synthetic,
        "metadata": metadata,
        "summary": summary,
        "done": done,
    }


def _split_combined(
    matrix: np.ndarray,
    actual: np.ndarray,
    raw: np.ndarray,
    truncated: np.ndarray,
    *,
    rows: int,
    encoder: str,
) -> dict[str, np.ndarray]:
    if (
        len(matrix) != rows * 2
        or len(actual) != rows * 2
        or len(raw) != rows * 2
        or len(truncated) != rows * 2
    ):
        raise RuntimeError(f"{encoder} combined encoder row count differs")
    output: dict[str, np.ndarray] = {}
    for field, selection in (
        ("prompt", slice(0, rows)),
        ("observed_target", slice(rows, rows * 2)),
    ):
        output[f"{encoder}_{field}"] = np.ascontiguousarray(
            matrix[selection],
            dtype=np.float32,
        )
        output[f"{encoder}_{field}_token_length"] = np.asarray(
            actual[selection],
            dtype=np.int32,
        )
        output[f"{encoder}_{field}_raw_token_length"] = np.asarray(
            raw[selection],
            dtype=np.int32,
        )
        output[f"{encoder}_{field}_truncated"] = np.asarray(
            truncated[selection],
            dtype=np.bool_,
        )
    return output


def extract(args: argparse.Namespace) -> int:
    input_manifest = args.input.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser()
    source_snapshot_binding = _validate_source_snapshot_binding(
        source_snapshot=args.source_snapshot,
        source_tree_sha256=args.source_tree_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
    )
    if (output_dir / DONE_NAME).exists():
        if not args.resume:
            raise FileExistsError(output_dir / DONE_NAME)
        result = validate_artifact(output_dir)
        metadata = result["metadata"]
        if (
            metadata["input_manifest"]["resolved_path"] != str(input_manifest)
            or metadata["encoders"]["umt5"]["resolved_path"]
            != str(args.umt5_root.expanduser().resolve(strict=True))
            or metadata["encoders"]["clip"]["resolved_path"]
            != str(args.clip_root.expanduser().resolve(strict=True))
            or metadata["encoders"]["umt5"]["revision"]
            != args.umt5_revision.lower()
            or metadata["encoders"]["clip"]["revision"]
            != args.clip_revision.lower()
            or metadata["source_snapshot"] != source_snapshot_binding
        ):
            raise ValueError("resume arguments differ from committed artifact")
        print(
            f"[r6-semantic] resume validated rows={result['rows']} "
            f"output={output_dir}"
        )
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"{output_dir} contains an incomplete/conflicting R6 artifact"
        )

    text_rows = _extract_text_rows(input_manifest)
    encoders = _registered_encoder_provenance(
        umt5_root=args.umt5_root,
        umt5_revision=args.umt5_revision,
        umt5_max_length=int(args.umt5_max_length),
        clip_root=args.clip_root,
        clip_revision=args.clip_revision,
        clip_max_length=int(args.clip_max_length),
    )
    torch, runtime = _gpu_preflight()
    try:
        import transformers
    except ImportError as error:
        raise RuntimeError("transformers is required") from error
    runtime["transformers"] = str(transformers.__version__)
    texts = (
        [row.prompt for row in text_rows]
        + [row.observed_target for row in text_rows]
    )
    umt5_values = _encode_umt5(
        torch=torch,
        root=Path(encoders["umt5"]["resolved_path"]),
        texts=texts,
        max_length=int(args.umt5_max_length),
        batch_size=int(args.umt5_batch_size),
    )
    arrays = _split_combined(
        *umt5_values,
        rows=len(text_rows),
        encoder="umt5",
    )
    clip_values = _encode_clip(
        torch=torch,
        root=Path(encoders["clip"]["resolved_path"]),
        texts=texts,
        max_length=int(args.clip_max_length),
        batch_size=int(args.clip_batch_size),
    )
    arrays.update(
        _split_combined(
            *clip_values,
            rows=len(text_rows),
            encoder="clip",
        )
    )
    _commit(
        input_manifest=input_manifest,
        output_dir=output_dir,
        text_rows=text_rows,
        arrays=arrays,
        encoders=encoders,
        source_snapshot_binding=source_snapshot_binding,
        runtime=runtime,
        synthetic=False,
    )
    result = validate_artifact(output_dir)
    print(
        f"[r6-semantic] committed rows={result['rows']} "
        f"output={output_dir}"
    )
    return 0


def validate_command(args: argparse.Namespace) -> int:
    result = validate_artifact(args.output_dir)
    print(
        f"[r6-semantic-validate] rows={result['rows']} "
        f"synthetic={result['synthetic_test_artifact']} "
        f"output={args.output_dir}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract immutable R6 UMT5/CLIP semantic features."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--input", required=True, type=Path)
    extract_parser.add_argument("--output-dir", required=True, type=Path)
    extract_parser.add_argument(
        "--source-snapshot",
        required=True,
        type=Path,
    )
    extract_parser.add_argument("--source-tree-sha256", required=True)
    extract_parser.add_argument("--source-manifest-sha256", required=True)
    extract_parser.add_argument("--umt5-root", required=True, type=Path)
    extract_parser.add_argument("--umt5-revision", required=True)
    extract_parser.add_argument(
        "--umt5-max-length",
        type=int,
        default=UMT5_DEFAULT_MAX_LENGTH,
    )
    extract_parser.add_argument("--umt5-batch-size", type=int, default=2)
    extract_parser.add_argument("--clip-root", required=True, type=Path)
    extract_parser.add_argument("--clip-revision", required=True)
    extract_parser.add_argument(
        "--clip-max-length",
        type=int,
        default=CLIP_DEFAULT_MAX_LENGTH,
    )
    extract_parser.add_argument("--clip-batch-size", type=int, default=32)
    extract_parser.add_argument("--resume", action="store_true")
    extract_parser.set_defaults(handler=extract)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", required=True, type=Path)
    validate_parser.set_defaults(handler=validate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "umt5_batch_size", 1) < 1:
        parser.error("--umt5-batch-size must be positive")
    if getattr(args, "clip_batch_size", 1) < 1:
        parser.error("--clip-batch-size must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
