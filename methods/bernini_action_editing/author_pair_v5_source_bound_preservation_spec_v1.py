#!/usr/bin/env python3
"""Author a sealed PAIR-v5 source-bound evaluator spec for one AUH runtime.

The resulting spec is environment-specific by design: it binds the current
eight-candidate rollout spec, evaluator/contract source bytes, complete frozen
visual checkpoint manifest and config, exact Python/Torch/ROCm/Transformers/
Safetensors/PyAV/NumPy/Pillow versions, official processor golden output,
preprocessing, metric formulas, probe order, native-generation provenance,
and method source archive.  Authoring does not evaluate media or allocate a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_source_bound_preservation_evaluator_v1 as contract  # noqa: E402
import score_pair_v5_source_bound_preservation_v1 as scorer  # noqa: E402


_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")


def _plain_file(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise contract.PairV5SourceBoundEvaluationError(
            f"{label} must be an absolute plain file"
        )
    return value.resolve(strict=True)


def _plain_directory(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_dir() or value.is_symlink():
        raise contract.PairV5SourceBoundEvaluationError(
            f"{label} must be an absolute non-symlink directory"
        )
    return value.resolve(strict=True)


def inspect_checkpoint(
    checkpoint_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = _plain_directory(checkpoint_root, label="checkpoint root")
    manifest = _plain_file(manifest_path, label="checkpoint manifest")
    manifest_sha = contract.file_sha256(manifest)
    if manifest_sha != expected_manifest_sha256:
        raise contract.PairV5SourceBoundEvaluationError(
            "checkpoint manifest expected SHA-256 differs"
        )
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise contract.PairV5SourceBoundEvaluationError("checkpoint manifest is empty")
    expected: dict[str, str] = {}
    for line in lines:
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise contract.PairV5SourceBoundEvaluationError(
                "checkpoint manifest syntax differs"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise contract.PairV5SourceBoundEvaluationError(
                "checkpoint manifest path escapes root"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise contract.PairV5SourceBoundEvaluationError(
                "checkpoint manifest path is empty/duplicate"
            )
        expected[normalized] = digest
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise contract.PairV5SourceBoundEvaluationError(
                "checkpoint tree contains a symlink"
            )
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise contract.PairV5SourceBoundEvaluationError(
                "checkpoint tree contains a non-regular entry"
            )
    if actual != set(expected):
        raise contract.PairV5SourceBoundEvaluationError(
            "checkpoint tree/manifest file closure differs"
        )
    for relative in sorted(expected):
        if contract.file_sha256(root / relative) != expected[relative]:
            raise contract.PairV5SourceBoundEvaluationError(
                f"checkpoint content differs: {relative}"
            )
    config_path = _plain_file(root / "config.json", label="checkpoint config")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise contract.PairV5SourceBoundEvaluationError(
            "checkpoint config is invalid JSON"
        ) from error
    if not isinstance(config, dict):
        raise contract.PairV5SourceBoundEvaluationError("checkpoint config root differs")
    architecture = config.get("model_type")
    register_tokens = config.get("num_register_tokens", 0)
    image_size = config.get("image_size")
    patch_size = config.get("patch_size")
    if not isinstance(architecture, str) or not architecture:
        raise contract.PairV5SourceBoundEvaluationError("checkpoint model_type differs")
    if type(register_tokens) is not int or register_tokens < 0:
        raise contract.PairV5SourceBoundEvaluationError(
            "checkpoint num_register_tokens differs"
        )
    if type(image_size) is not int or type(patch_size) is not int:
        raise contract.PairV5SourceBoundEvaluationError(
            "checkpoint image_size/patch_size differs"
        )
    preprocessor_path = _plain_file(
        root / "preprocessor_config.json", label="checkpoint preprocessor config"
    )
    processor = scorer.inspect_official_processor(root)
    return {
        "architecture_id": architecture,
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint_config_sha256": contract.file_sha256(config_path),
        "preprocessor_config_sha256": contract.file_sha256(preprocessor_path),
        "checkpoint_file_count": len(expected),
        "num_register_tokens": register_tokens,
        "image_size": image_size,
        "patch_size": patch_size,
        "preprocessor_golden_input_sha256": processor[
            "preprocessor_golden_input_sha256"
        ],
        "preprocessor_golden_output_sha256": processor[
            "preprocessor_golden_output_sha256"
        ],
        "preprocessor_golden_output_shape": processor[
            "preprocessor_golden_output_shape"
        ],
    }


def author_spec(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    rollout_path = _plain_file(args.root_spec, label="current-family rollout spec")
    normalized, rollout_sha = contract.load_current_family_rollout_spec(
        rollout_path, args.expected_root_spec_sha256
    )
    raw_rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    checkpoint = inspect_checkpoint(
        args.checkpoint,
        args.checkpoint_content_manifest,
        expected_manifest_sha256=args.expected_checkpoint_manifest_sha256,
    )
    native_path = _plain_file(
        args.reference_native_receipt, label="reference native receipt"
    )
    native, native_file_sha = scorer._strict_json_file(
        native_path, label="reference native receipt"
    )
    if native_file_sha != args.expected_reference_native_receipt_sha256:
        raise contract.PairV5SourceBoundEvaluationError(
            "reference native receipt file SHA-256 differs"
        )
    scorer._verify_embedded_digest(
        native,
        field="receipt_digest",
        label="reference native receipt",
        ensure_ascii=False,
    )
    generation_provenance = contract.generation_provenance_from_native_receipt(
        native, reference_file_sha256=native_file_sha
    )
    spec = contract.make_evaluator_spec(
        raw_rollout,
        rollout_spec_raw_sha256=rollout_sha,
        implementation_sha256=contract.file_sha256(Path(scorer.__file__).resolve()),
        contract_sha256=contract.file_sha256(Path(contract.__file__).resolve()),
        method_source_revision=args.method_source_revision,
        method_source_archive_sha256=args.method_source_archive_sha256,
        runtime_versions=scorer.runtime_versions(),
        generation_provenance=generation_provenance,
        **checkpoint,
    )
    if spec["candidate_order"] != normalized["candidate_order"]:
        raise contract.PairV5SourceBoundEvaluationError(
            "authored evaluator candidate order differs"
        )
    output = Path(args.output)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise contract.PairV5SourceBoundEvaluationError(
            "output must be a fresh absolute non-root path"
        )
    raw = contract.canonical_json_bytes(spec) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(output, 0o400)
    return spec, hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--reference-native-receipt", required=True)
    parser.add_argument("--expected-reference-native-receipt-sha256", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    spec, raw_sha = author_spec(build_parser().parse_args(argv))
    print(
        contract.canonical_json_bytes(
            {
                "schema_version": "bernini-pair-v5-source-bound-spec-authoring-v1",
                "spec_digest": spec["spec_digest"],
                "spec_raw_sha256": raw_sha,
            }
        ).decode("ascii"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
