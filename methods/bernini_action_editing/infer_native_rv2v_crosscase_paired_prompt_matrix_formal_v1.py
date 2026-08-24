#!/usr/bin/env python3
"""Generic sealed-authority adapter for the formal native RV2V prompt matrix.

The sampling implementation is intentionally inherited without modification
from ``infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1``.  This adapter
only replaces its case-specific authority, CLI, Slurm, and receipt bindings.
Consequently every run still materializes the source condition once and calls
the frozen native sampler in one WORLD4 process in the exact order
P0a -> P1 -> P2 -> P0b.  The core continues to require one bit-exact native
Gaussian across the four calls and a bit-exact P0a/P0b generated-latent replay.

The generator CLI deliberately has no target-video or target-action argument.
The only semantic inputs admitted at generation time are the source video and
the three positive prompts sealed by the case authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any, Mapping, Optional, Sequence

METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_mev840_native_rv2v_paired_prompt_matrix_formal_v1 as _core


AUTHORITY_SCHEMA = "native-rv2v-crosscase-same-process-formal-v1"
PROMPT_MATRIX_SCHEMA = "native-rv2v-crosscase-prompt-matrix-v1"
DEFAULT_RECEIPT_SCHEMA = (
    "native-rv2v-crosscase-paired-prompt-matrix-formal-v1"
)
DEFAULT_PAIRED_CONTRACT_SCHEMA = (
    "native-rv2v-crosscase-paired-same-process-contract-v1"
)
METHOD = "frozen-bernini-native-rv2v-crosscase-paired-prompt-matrix-formal"
RUN_CLASSES = {"target_oracle_diagnostic", "formal_scientific_candidate"}

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SAFE_SCHEMA = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

NativeIdentityCanaryError = _core.NativeIdentityCanaryError
_ORIGINAL_BUILD_PAIRED_RECEIPT = _core._build_paired_receipt


def _fail(message: str) -> None:
    raise NativeIdentityCanaryError(message)


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        _fail(f"{label} must be absolute")
    if requested.is_symlink():
        _fail(f"{label} must not be a symlink")
    resolved = requested.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{label} must be a plain file")
    return resolved


def _read_ascii_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeIdentityCanaryError(
            f"{label} is not canonical ASCII JSON"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} root must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--case-authority", required=True)
    parser.add_argument("--expected-case-authority-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=_core.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=_core.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=_core.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def _validate_pairing(authority: Mapping[str, Any]) -> None:
    pairing = authority.get("same_process_pairing")
    expected = {
        "execution_order": list(_core.EXECUTION_ORDER),
        "prompt_label_by_execution_cell": dict(_core.PROMPT_LABEL_BY_CELL),
        "source_decode_count": 1,
        "full_source_vae_encode_count": 1,
        "reference_frame_indices": list(_core.RV2V_REFERENCE_INDICES),
        "each_reference_vae_encode_count": 1,
        "rank_zero_condition_broadcast_before_first_sample": True,
        "same_condition_tensor_objects_for_all_four_calls": True,
        "same_seed_for_all_four_calls": True,
        "official_fresh_gaussian_per_call": True,
        "external_gaussian_injection": False,
        "custom_sampler_or_scheduler": False,
        "same_scheduler_object_all_calls": True,
        "no_manual_model_or_scheduler_state_reset_between_calls": True,
        "rope_unregistered_state_observed_not_mutated": True,
        "p0a_p0b_generated_latent_bit_exact_required": True,
    }
    if not isinstance(pairing, Mapping):
        _fail("same-process pairing authority is absent")
    for key, value in expected.items():
        if pairing.get(key) != value:
            _fail(f"same-process authority differs: {key}")


def _validate_generator_contract(authority: Mapping[str, Any]) -> None:
    generator = authority.get("generator_contract")
    if not isinstance(generator, Mapping):
        _fail("generator contract is absent")
    required = {
        "guidance_mode": "rv2v",
        "frame_count": _core.FRAME_COUNT,
        "fps": _core.FPS,
        "zero_update": True,
        "accepted_external_conditions": ["source_video", "positive_prompt_matrix"],
        "target_video_read": False,
        "target_action_json_read": False,
        "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian_read": False,
        "anchor_rgb_kv_latent_gaussian_read": False,
        "legacy_activity25_qk_read": False,
    }
    for key, value in required.items():
        if generator.get(key) != value:
            _fail(f"generator input authority differs: {key}")


def _validate_runtime_authority(
    authority: Mapping[str, Any], *, seed: int
) -> tuple[dict[int, dict[str, str]], int, int]:
    runtime = authority.get("runtime_authority")
    if not isinstance(runtime, Mapping):
        _fail("runtime authority is absent")
    if runtime.get("unipc_source") != {
        "path": _core.UNIPC_SOURCE_PATH,
        "sha256": _core.UNIPC_SOURCE_SHA256,
    }:
        _fail("UniPC runtime authority differs")
    sealed_rows = runtime.get("formal_slurm_by_seed")
    if not isinstance(sealed_rows, Mapping) or not sealed_rows:
        _fail("formal Slurm seed authority is absent")
    rows: dict[int, dict[str, str]] = {}
    for seed_text, row in sealed_rows.items():
        if (
            not isinstance(seed_text, str)
            or not seed_text.isdigit()
            or str(int(seed_text)) != seed_text
            or not isinstance(row, Mapping)
            or not isinstance(row.get("job_id"), str)
            or not row["job_id"].isdigit()
            or not isinstance(row.get("node"), str)
            or _NODE.fullmatch(row["node"]) is None
            or row.get("world_size") != _core.ULYSSES_SIZE
        ):
            _fail("formal Slurm seed row differs")
        rows[int(seed_text)] = {
            "job_id": row["job_id"],
            "node": row["node"],
        }
    if seed not in rows:
        _fail("requested seed is not sealed by the case authority")
    limit = runtime.get("nearest_finite_cgroup_limit_bytes")
    headroom = runtime.get("minimum_cgroup_headroom_bytes")
    if (
        type(limit) is not int
        or type(headroom) is not int
        or limit <= 0
        or headroom <= 0
        or headroom >= limit
    ):
        _fail("cgroup memory authority differs")
    return rows, limit, headroom


def _validate_execution_mode(
    authority: Mapping[str, Any], *, available_seeds: set[int], seed: int
) -> tuple[str, bool]:
    execution = authority.get("execution_mode")
    if not isinstance(execution, Mapping):
        _fail("execution mode is absent")
    run_class = execution.get("run_class")
    scientific = execution.get("scientific_candidate")
    required = {
        "seeds": sorted(available_seeds),
        "num_inference_steps": _core.NUM_INFERENCE_STEPS,
        "decode_cells": ["p0a", "p1", "p2"],
        "latent_only_replay_cells": ["p0b"],
    }
    for key, value in required.items():
        if execution.get(key) != value:
            _fail(f"execution mode differs: {key}")
    if run_class not in RUN_CLASSES or type(scientific) is not bool:
        _fail("execution run class differs")
    if run_class == "target_oracle_diagnostic" and scientific is not False:
        _fail("target-oracle diagnostic cannot be a scientific candidate")
    if run_class == "formal_scientific_candidate" and scientific is not True:
        _fail("formal run must be declared a scientific candidate")
    if seed not in available_seeds:
        _fail("seed differs from execution authority")
    return str(run_class), bool(scientific)


def _validate_mechanical_gate(
    authority: Mapping[str, Any],
    *,
    run_class: str,
    case_id: str,
    source_sha256: str,
    prompt_matrix_sha256: str,
) -> Optional[dict[str, Any]]:
    pins = authority.get("receipt_pins")
    if not isinstance(pins, Mapping) or set(pins) != {"mechanical_gate"}:
        _fail("receipt pins differ")
    pin = pins.get("mechanical_gate")
    if run_class == "target_oracle_diagnostic":
        if pin is not None:
            _fail("diagnostic authority must not claim a mechanical gate")
        return None
    if not isinstance(pin, Mapping):
        _fail("formal scientific authority requires a mechanical gate")
    required_strings = (
        "path",
        "sha256",
        "receipt_digest",
        "schema_version",
        "case_id",
        "source_video_sha256",
        "prompt_matrix_sha256",
    )
    if any(not isinstance(pin.get(key), str) for key in required_strings):
        _fail("mechanical gate pins differ")
    if any(
        _SHA256.fullmatch(pin[key]) is None
        for key in ("sha256", "receipt_digest", "source_video_sha256", "prompt_matrix_sha256")
    ):
        _fail("mechanical gate SHA-256 pin differs")
    if (
        pin["case_id"] != case_id
        or pin["source_video_sha256"] != source_sha256
        or pin["prompt_matrix_sha256"] != prompt_matrix_sha256
    ):
        _fail("mechanical gate case binding differs")
    path = _plain_file(pin["path"], label="mechanical gate receipt")
    if _sha256_file(path) != pin["sha256"]:
        _fail("mechanical gate receipt SHA-256 differs")
    receipt = _read_ascii_json(path, label="mechanical gate receipt")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if declared != pin["receipt_digest"] or _canonical_digest(unsigned) != declared:
        _fail("mechanical gate canonical receipt digest differs")
    if (
        receipt.get("schema_version") != pin["schema_version"]
        or receipt.get("case_id") != case_id
        or receipt.get("input", {}).get("source_video_sha256") != source_sha256
        or receipt.get("input", {}).get("prompt_matrix_sha256")
        != prompt_matrix_sha256
        or receipt.get("paired_same_process_contract", {})
        .get("p0_replay", {})
        .get("generated_latent_bit_exact")
        is not True
        or receipt.get("paired_same_process_contract", {}).get(
            "target_media_or_action_json_read"
        )
        is not False
    ):
        _fail("mechanical gate runtime binding differs")
    return dict(pin)


def load_case_authority(
    value: str | Path, *, expected_sha256: str, seed: int
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        _fail("case authority SHA-256 is invalid")
    path = _plain_file(value, label="case authority")
    if _sha256_file(path) != expected_sha256:
        _fail("case authority SHA-256 differs")
    authority = _read_ascii_json(path, label="case authority")
    if authority.get("schema") != AUTHORITY_SCHEMA:
        _fail("case authority schema differs")
    if authority.get("launch_authorized") is not True:
        _fail("case authority does not authorize launch")
    case = authority.get("case")
    if not isinstance(case, Mapping):
        _fail("case binding is absent")
    case_id = case.get("case_id")
    source = case.get("source_video")
    if (
        not isinstance(case_id, str)
        or _SAFE_ID.fullmatch(case_id) is None
        or not isinstance(source, Mapping)
        or not isinstance(source.get("path"), str)
        or not isinstance(source.get("sha256"), str)
        or _SHA256.fullmatch(source["sha256"]) is None
    ):
        _fail("case/source binding differs")
    source_path = _plain_file(source["path"], label="sealed source video")
    if _sha256_file(source_path) != source["sha256"]:
        _fail("sealed source video SHA-256 differs")

    matrix_ref = authority.get("prompt_matrix")
    if not isinstance(matrix_ref, Mapping):
        _fail("prompt matrix reference is absent")
    basename = matrix_ref.get("basename")
    matrix_sha = matrix_ref.get("sha256")
    if (
        not isinstance(basename, str)
        or _core._SAFE_BASENAME.fullmatch(basename) is None
        or not isinstance(matrix_sha, str)
        or _SHA256.fullmatch(matrix_sha) is None
        or matrix_ref.get("schema") != PROMPT_MATRIX_SCHEMA
        or matrix_ref.get("labels") != ["P0", "P1", "P2"]
        or matrix_ref.get("only_registered_design_variable")
        != "positive_prompt_utf8"
    ):
        _fail("prompt matrix reference differs")
    matrix_path = _plain_file(path.parent / basename, label="prompt matrix")
    if matrix_path.parent != path.parent or _sha256_file(matrix_path) != matrix_sha:
        _fail("prompt matrix file identity differs")
    matrix = _read_ascii_json(matrix_path, label="prompt matrix")
    if matrix.get("schema") != PROMPT_MATRIX_SCHEMA or matrix.get("case_id") != case_id:
        _fail("prompt matrix schema/case binding differs")
    rows = matrix.get("prompts")
    sealed_rows = authority.get("prompts")
    if (
        not isinstance(rows, Mapping)
        or set(rows) != {"P0", "P1", "P2"}
        or not isinstance(sealed_rows, Mapping)
        or set(sealed_rows) != {"P0", "P1", "P2"}
    ):
        _fail("prompt rows differ")
    prompts: dict[str, str] = {}
    prompt_rows: dict[str, Any] = {}
    for label in ("P0", "P1", "P2"):
        row, sealed = rows[label], sealed_rows[label]
        if not isinstance(row, Mapping) or not isinstance(sealed, Mapping):
            _fail(f"prompt row differs: {label}")
        prompt = row.get("full_prompt_utf8")
        if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt:
            _fail(f"prompt text differs: {label}")
        payload = prompt.encode("utf-8")
        if (
            len(payload) != row.get("full_prompt_utf8_bytes")
            or hashlib.sha256(payload).hexdigest()
            != row.get("full_prompt_utf8_sha256")
        ):
            _fail(f"prompt identity differs: {label}")
        for key in ("full_prompt_utf8", "full_prompt_utf8_bytes", "full_prompt_utf8_sha256"):
            if sealed.get(key) != row.get(key):
                _fail(f"authority/matrix prompt differs: {label} {key}")
        for key in (
            "final_task_prompt_utf8_bytes",
            "final_task_prompt_utf8_sha256",
            "untruncated_token_count",
            "terminal_token_id",
        ):
            if key in row and sealed.get(key) != row.get(key):
                _fail(f"authority/matrix task prompt differs: {label} {key}")
        if (
            type(sealed.get("final_task_prompt_utf8_bytes")) is not int
            or not isinstance(sealed.get("final_task_prompt_utf8_sha256"), str)
            or _SHA256.fullmatch(sealed["final_task_prompt_utf8_sha256"]) is None
            or type(sealed.get("untruncated_token_count")) is not int
            or not 1 <= sealed["untruncated_token_count"] <= 512
            or sealed.get("terminal_token_id") != 1
        ):
            _fail(f"sealed task prompt contract differs: {label}")
        prompts[label] = prompt
        prompt_rows[label] = dict(sealed)

    _validate_pairing(authority)
    _validate_generator_contract(authority)
    slurm_rows, cgroup_limit, cgroup_headroom = _validate_runtime_authority(
        authority, seed=seed
    )
    run_class, scientific = _validate_execution_mode(
        authority, available_seeds=set(slurm_rows), seed=seed
    )
    schemas = authority.get("receipt_schemas")
    if (
        not isinstance(schemas, Mapping)
        or set(schemas) != {"formal_receipt", "paired_contract"}
        or any(
            not isinstance(value, str) or _SAFE_SCHEMA.fullmatch(value) is None
            for value in schemas.values()
        )
    ):
        _fail("receipt schemas differ")
    mechanical_gate = _validate_mechanical_gate(
        authority,
        run_class=run_class,
        case_id=case_id,
        source_sha256=source["sha256"],
        prompt_matrix_sha256=matrix_sha,
    )
    return {
        "authority": authority,
        "authority_path": str(path),
        "authority_sha256": expected_sha256,
        "prompt_matrix": matrix,
        "prompt_matrix_path": str(matrix_path),
        "prompt_matrix_sha256": matrix_sha,
        "prompts": prompts,
        "prompt_rows": prompt_rows,
        "case_id": case_id,
        "source_path": str(source_path),
        "source_sha256": source["sha256"],
        "slurm_by_seed": slurm_rows,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_headroom_bytes": cgroup_headroom,
        "run_class": run_class,
        "scientific_candidate": scientific,
        "receipt_schema": schemas["formal_receipt"],
        "paired_contract_schema": schemas["paired_contract"],
        "mechanical_gate": mechanical_gate,
    }


def validate_cli(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
            _fail(f"{name} must be a full lowercase SHA-1")
    for name in ("expected_checkpoint_tree_sha256", "expected_case_authority_sha256", "method_source_archive_sha256"):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            _fail(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit != _core.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        _fail("unsupported Bernini source revision")
    if args.expected_veomni_commit != _core.legacy.trainer.VEOMNI_TESTED_COMMIT:
        _fail("unsupported VeOmni source revision")
    if args.expected_checkpoint_tree_sha256 != _core.legacy.trainer.CHECKPOINT_TREE_SHA256:
        _fail("unsupported Bernini-R checkpoint tree")
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output == Path("/") or _core._SAFE_BASENAME.fullmatch(output.name) is None:
        _fail("output-dir must be an absolute non-root path with a safe basename")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        _fail("seed must be in [0,2^63)")
    bundle = load_case_authority(
        args.case_authority,
        expected_sha256=args.expected_case_authority_sha256,
        seed=args.seed,
    )
    # Populate the private compatibility namespace consumed by the unchanged
    # formal sampling core.  None of these values come from an unsealed CLI.
    args.source_video = bundle["source_path"]
    args.expected_source_sha256 = bundle["source_sha256"]
    args.prompt_matrix_authority = bundle["authority_path"]
    args.expected_prompt_matrix_authority_sha256 = bundle["authority_sha256"]
    args.num_inference_steps = _core.NUM_INFERENCE_STEPS
    args.skip_video_decode = False
    args.case_id = bundle["case_id"]
    _core.FORMAL_SLURM_BY_SEED = dict(bundle["slurm_by_seed"])
    _core.FORMAL_CGROUP_LIMIT_BYTES = bundle["cgroup_limit_bytes"]
    _core.FORMAL_CGROUP_MIN_HEADROOM_BYTES = bundle["cgroup_headroom_bytes"]
    _core.SCHEMA_VERSION = bundle["receipt_schema"]
    _core.METHOD = METHOD
    return bundle


def _formal_slurm_context(
    args: argparse.Namespace, authority_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    expected = authority_bundle["slurm_by_seed"].get(args.seed)
    if not isinstance(expected, Mapping):
        _fail("formal Slurm seed authority differs")
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    node = socket.gethostname().split(".", 1)[0]
    world_size = os.environ.get("WORLD_SIZE")
    if (
        job_id != expected["job_id"]
        or not isinstance(step_id, str)
        or not step_id.isdigit()
        or node != expected["node"]
        or world_size != str(_core.ULYSSES_SIZE)
    ):
        _fail("formal Slurm job/node/step/WORLD4 differs")
    return {
        "job_id": job_id,
        "step_id": step_id,
        "job_step_id": f"{job_id}.{step_id}",
        "node": node,
        "world_size": int(world_size),
    }


def _build_paired_receipt(**kwargs: Any) -> dict[str, Any]:
    bundle = kwargs["authority_bundle"]
    receipt = _ORIGINAL_BUILD_PAIRED_RECEIPT(**kwargs)
    receipt["schema_version"] = bundle["receipt_schema"]
    receipt["method"] = METHOD
    receipt["case_id"] = bundle["case_id"]
    receipt["authority_schema"] = AUTHORITY_SCHEMA
    receipt["run_class"] = bundle["run_class"]
    receipt["execution_mode"]["scientific_candidate"] = bundle[
        "scientific_candidate"
    ]
    receipt["execution_mode"]["run_class"] = bundle["run_class"]
    receipt["execution_mode"]["formal_generation"] = (
        bundle["run_class"] == "formal_scientific_candidate"
    )
    receipt["execution_mode"]["target_oracle_diagnostic"] = (
        bundle["run_class"] == "target_oracle_diagnostic"
    )
    receipt["paired_same_process_contract"]["schema"] = bundle[
        "paired_contract_schema"
    ]
    receipt["paired_same_process_contract"]["case_id"] = bundle["case_id"]
    receipt["paired_same_process_contract"][
        "current_authorized_overlay_runner"
    ] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256_file(Path(__file__).resolve()),
        "sampling_core_path": str(Path(_core.__file__).resolve()),
        "sampling_core_sha256": _sha256_file(Path(_core.__file__).resolve()),
        "upstream_release_entrypoint_authorized": False,
    }
    receipt["input"]["case_authority_path"] = bundle["authority_path"]
    receipt["input"]["case_authority_sha256"] = bundle["authority_sha256"]
    receipt["input"]["target_video"] = False
    receipt["input"]["target_action_json"] = False
    receipt["mechanical_gate"] = bundle["mechanical_gate"]
    receipt["interpretation"]["target_oracle_action_language"] = True
    receipt["interpretation"]["target_media_or_action_json_read_by_generator"] = False
    receipt["scientific_claim_authorized"] = False
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = _core.legacy.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Install the sealed generic bindings, then call the unchanged core."""

    originals = {
        "build_parser": _core.build_parser,
        "validate_cli": _core.validate_cli,
        "_formal_slurm_context": _core._formal_slurm_context,
        "_build_paired_receipt": _core._build_paired_receipt,
        "FORMAL_SLURM_BY_SEED": _core.FORMAL_SLURM_BY_SEED,
        "FORMAL_CGROUP_LIMIT_BYTES": _core.FORMAL_CGROUP_LIMIT_BYTES,
        "FORMAL_CGROUP_MIN_HEADROOM_BYTES": _core.FORMAL_CGROUP_MIN_HEADROOM_BYTES,
        "SCHEMA_VERSION": _core.SCHEMA_VERSION,
        "METHOD": _core.METHOD,
    }
    try:
        _core.build_parser = build_parser
        _core.validate_cli = validate_cli
        _core._formal_slurm_context = _formal_slurm_context
        _core._build_paired_receipt = _build_paired_receipt
        return _core.main(argv)
    finally:
        for name, value in originals.items():
            setattr(_core, name, value)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "DEFAULT_PAIRED_CONTRACT_SCHEMA",
    "DEFAULT_RECEIPT_SCHEMA",
    "METHOD",
    "PROMPT_MATRIX_SCHEMA",
    "RUN_CLASSES",
    "build_parser",
    "load_case_authority",
    "main",
    "validate_cli",
]
