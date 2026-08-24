#!/usr/bin/env python3
"""Build a create-only, synthetic PACT fixture for integration smoke tests.

This command exercises the real signed post-generation release, atomization,
and latent-payload binding contracts.  Its media, masks, latents, and encoder
contexts are synthetic and MUST NOT be used for training, evaluation, or a
scientific result.  The single authorized atom exists only so an end-to-end
checkpoint integration test traverses the same fail-closed gates as real data.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pact.dataset import (  # noqa: E402
    ENCODER_CONTRACT_FORMAT,
    PAYLOAD_FORMAT,
    PAYLOAD_PROVENANCE_BINDINGS,
    UMT5_EMBEDDING_DIM,
    UMT5_MAX_SEQUENCE_LENGTH,
    UMT5_PADDING_POLICY,
    UMT5_SEGMENT_ORDER,
    VLM_EMBEDDING_DIM,
    VLM_FEATURE_TENSOR,
    VLM_TOKEN_SELECTION,
    WAN21_VAE_CHANNEL_MEAN,
    WAN21_VAE_CHANNEL_STD,
    WAN21_VAE_INPUT_PIXEL_RANGE,
    WAN21_VAE_POSTERIOR_MODE,
    WAN21_VAE_STRIDE,
    encoder_contract_sha256,
)
from pact.manifest import (  # noqa: E402
    TRACK_SCHEMA,
    AtomizeOptions,
    ManifestError,
    atomize_global_row,
    canonical_json_bytes,
    file_sha256,
    sign_post_generation_release,
    verify_post_generation_release,
)
from pact.training import validate_training_config  # noqa: E402
from tools.bind_latent_payloads import bind_latent_payloads  # noqa: E402


FIXTURE_SCHEMA = "pact-synthetic-integration-fixture-v1"
GLOBAL_ROW_SCHEMA = "pact-synthetic-smoke-global-row-v1"
FIXTURE_IID = "synthetic_smoke_001"
COMPONENT_ID = "actor_component"
RELEASE_ID = "synthetic_smoke_release_v1"
DEFAULT_ISSUED_AT_UTC = "2026-08-03T00:00:00Z"
DEFAULT_SEED = 20260803
SYNTHETIC_VAE_CHECKPOINT_SHA256 = "1" * 64
SYNTHETIC_UMT5_MANIFEST_SHA256 = "2" * 64
SYNTHETIC_VLM_MANIFEST_SHA256 = "3" * 64
SYNTHETIC_VAE_PREPROCESSING_SHA256 = "4" * 64
SYNTHETIC_UMT5_PREPROCESSING_SHA256 = "5" * 64
SYNTHETIC_VLM_FEATURE_CONTRACT_SHA256 = "6" * 64


class SyntheticFixtureError(RuntimeError):
    """Raised when a synthetic integration fixture cannot be built safely."""


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_canonical_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")


def _generate_ephemeral_signer(trust_root: Path) -> tuple[Path, Path, str]:
    private_key = trust_root / "synthetic_release_ed25519"
    result = subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(private_key),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise SyntheticFixtureError(
            "cannot generate the synthetic SSHSIG signer: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    public_key = Path(str(private_key) + ".pub")
    fingerprint_result = subprocess.run(
        ["ssh-keygen", "-lf", str(public_key), "-E", "sha256"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    tokens = fingerprint_result.stdout.split()
    if (
        fingerprint_result.returncode != 0
        or len(tokens) < 2
        or not tokens[1].startswith("SHA256:")
    ):
        raise SyntheticFixtureError(
            "cannot fingerprint the synthetic SSHSIG signer: "
            + fingerprint_result.stderr.strip()
        )
    return private_key, public_key, tokens[1]


def _synthetic_masks() -> tuple[torch.Tensor, torch.Tensor]:
    source_mask = torch.zeros(1, 3, 8, 8, dtype=torch.float32)
    target_mask = torch.zeros_like(source_mask)
    source_mask[:, :, 2:4, 2:4] = 1.0
    target_mask[:, :, 4:6, 4:6] = 1.0
    return source_mask, target_mask


def _synthetic_encoder_contract() -> dict[str, Any]:
    """Return deterministic fake provenance for a reporting-forbidden fixture."""

    return {
        "format": ENCODER_CONTRACT_FORMAT,
        "vae": {
            "checkpoint_sha256": SYNTHETIC_VAE_CHECKPOINT_SHA256,
            "preprocessing_contract_sha256": SYNTHETIC_VAE_PREPROCESSING_SHA256,
            "input_pixel_range": list(WAN21_VAE_INPUT_PIXEL_RANGE),
            "posterior_mode": WAN21_VAE_POSTERIOR_MODE,
            "channel_mean": list(WAN21_VAE_CHANNEL_MEAN),
            "channel_std": list(WAN21_VAE_CHANNEL_STD),
            "stride": list(WAN21_VAE_STRIDE),
        },
        "umt5": {
            "checkpoint_manifest_sha256": SYNTHETIC_UMT5_MANIFEST_SHA256,
            "preprocessing_contract_sha256": SYNTHETIC_UMT5_PREPROCESSING_SHA256,
            "embedding_dim": UMT5_EMBEDDING_DIM,
            "max_sequence_length_per_segment": UMT5_MAX_SEQUENCE_LENGTH,
            "segment_order": list(UMT5_SEGMENT_ORDER),
            "padding_policy": UMT5_PADDING_POLICY,
        },
        "vlm": {
            "checkpoint_manifest_sha256": SYNTHETIC_VLM_MANIFEST_SHA256,
            "feature_extraction_contract_sha256": SYNTHETIC_VLM_FEATURE_CONTRACT_SHA256,
            "embedding_dim": VLM_EMBEDDING_DIM,
            "feature_tensor": VLM_FEATURE_TENSOR,
            "token_selection": VLM_TOKEN_SELECTION,
        },
    }


def _parent_row(source_path: Path, target_path: Path) -> dict[str, Any]:
    return {
        "schema_version": GLOBAL_ROW_SCHEMA,
        "iid": FIXTURE_IID,
        "synthetic_integration_only": True,
        "benchmark_reporting_forbidden": True,
        "source_video_path": str(source_path.resolve()),
        "source_video_sha256": file_sha256(source_path),
        "target_video_path": str(target_path.resolve()),
        "target_video_sha256": file_sha256(target_path),
        # These two fields intentionally traverse the production authorization
        # path.  The signed row remains synthetic and reporting-forbidden.
        "production_eligible": True,
        "human_review_status": "accepted",
        "source_census": {
            "dynamic_subjects": [
                {
                    "subject_id": "actor_01",
                    "stable_reference": "the centered synthetic actor",
                    "i0_bbox_xyxy_1000": [250, 100, 750, 950],
                    "source_action_signature": "synthetic_wave",
                    "source_motion": "waves one hand while standing still",
                }
            ],
            "camera": {
                "motion_class": "locked_off",
                "source_motion": "camera remains locked off",
            },
        },
        "target_plan": {
            "dynamic_subject_targets": [
                {
                    "subject_id": "actor_01",
                    "target_action_signature": "synthetic_crouch",
                    "target_motion": "crouches once and returns to standing",
                    "substantive_change": True,
                }
            ],
            "camera_target": {
                "relation": "preserve_static",
                "motion_class": "locked_off",
                "target_motion": "camera remains locked off",
            },
        },
    }


def _component_track(
    source_mask_path: Path, target_mask_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": TRACK_SCHEMA,
        "iid": FIXTURE_IID,
        "component_id": COMPONENT_ID,
        "subject_ids": ["actor_01"],
        "source_mask_path": str(source_mask_path.resolve()),
        "source_mask_sha256": file_sha256(source_mask_path),
        "target_mask_path": str(target_mask_path.resolve()),
        "target_mask_sha256": file_sha256(target_mask_path),
        "interaction_safe": True,
        "review_status": "accepted",
        "confidence": 1.0,
        "synthetic_integration_only": True,
    }


def _payload(
    atom: dict[str, Any],
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (16, 3, 8, 8)
    source = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
    target = source.clone()
    target_support = torch.maximum(source_mask, target_mask).expand_as(target)
    target = target + target_support * 0.25
    value: dict[str, Any] = {
        "format": PAYLOAD_FORMAT,
        "atom_id": atom["atom_id"],
        "encoder_contract": _synthetic_encoder_contract(),
        "source_latent": source,
        "global_target_latent": target,
        "source_component_mask": source_mask.clone(),
        "target_component_mask": target_mask.clone(),
        "text_context": torch.randn(
            (2, 4096), generator=generator, dtype=torch.float32
        )
        * 0.01,
        "vlm_context": torch.randn(
            (3, 2048), generator=generator, dtype=torch.float32
        )
        * 0.01,
    }
    for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS:
        value[payload_field] = atom[atomic_field]
    return value


def _smoke_config(seed: int) -> dict[str, Any]:
    base_path = PROJECT_ROOT / "configs" / "pact_1_3b.json"
    with base_path.open("r", encoding="utf-8") as handle:
        config = copy.deepcopy(json.load(handle))
    config["seed"] = seed
    config["training"].update(
        {
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "max_steps": 1,
            "checkpoint_every": 1,
            "log_every": 1,
        }
    )
    return validate_training_config(config)


def build_synthetic_smoke_fixture(
    output_dir: os.PathLike[str] | str,
    *,
    seed: int = DEFAULT_SEED,
    issued_at_utc: str = DEFAULT_ISSUED_AT_UTC,
) -> dict[str, Any]:
    """Publish one synthetic, contract-authorized integration fixture.

    ``output_dir`` is create-only.  A missing top-level ``done.json`` means the
    directory is a partial failed build and must not be consumed.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SyntheticFixtureError("seed must be a non-negative integer")
    output_root = Path(os.path.abspath(Path(output_dir).expanduser()))
    if output_root.exists() or output_root.is_symlink():
        raise SyntheticFixtureError(
            f"output directory already exists; refusing overwrite: {output_root}"
        )
    if not output_root.parent.is_dir():
        raise SyntheticFixtureError(
            f"output parent directory does not exist: {output_root.parent}"
        )
    try:
        output_root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise SyntheticFixtureError(
            f"output directory already exists; refusing overwrite: {output_root}"
        ) from exc

    inputs_root = output_root / "synthetic_inputs"
    release_root = output_root / "signed_release"
    atomic_root = output_root / "atomic"
    payload_root = output_root / "payloads"
    config_root = output_root / "configs"
    trust_root = output_root / "synthetic_trust"
    for directory in (
        inputs_root,
        release_root,
        atomic_root,
        payload_root,
        config_root,
        trust_root,
    ):
        directory.mkdir()

    source_path = inputs_root / "source.synthetic.bin"
    target_path = inputs_root / "global_target.synthetic.bin"
    source_path.write_bytes(b"PACT SYNTHETIC SOURCE; NOT A VIDEO\n")
    target_path.write_bytes(b"PACT SYNTHETIC GLOBAL TARGET; NOT A VIDEO\n")
    source_mask, target_mask = _synthetic_masks()
    source_mask_path = inputs_root / "source_component_mask.pt"
    target_mask_path = inputs_root / "target_component_mask.pt"
    torch.save(source_mask, source_mask_path)
    torch.save(target_mask, target_mask_path)

    parent = _parent_row(source_path, target_path)
    track = _component_track(source_mask_path, target_mask_path)
    global_manifest_path = release_root / "global_manifest.jsonl"
    track_manifest_path = release_root / "component_tracks.jsonl"
    _write_canonical_jsonl(global_manifest_path, [parent])
    _write_canonical_jsonl(track_manifest_path, [track])

    private_key, public_key, fingerprint = _generate_ephemeral_signer(trust_root)
    receipt_path = release_root / "post_generation_release.json"
    try:
        sign_post_generation_release(
            global_manifest_path=global_manifest_path,
            output_path=receipt_path,
            signing_key_path=private_key,
            public_key_path=public_key,
            expected_signer_fingerprint=fingerprint,
            release_id=RELEASE_ID,
            issued_at_utc=issued_at_utc,
            row_schema_version=GLOBAL_ROW_SCHEMA,
        )
    finally:
        # This generated signer is not a trust anchor.  Retain only the public
        # key needed to independently verify the synthetic receipt.
        if private_key.is_file() and not private_key.is_symlink():
            private_key.unlink()

    verified_release = verify_post_generation_release(
        global_manifest_path=global_manifest_path,
        release_receipt_path=receipt_path,
        public_key_path=public_key,
        expected_signer_fingerprint=fingerprint,
        row_schema_version=GLOBAL_ROW_SCHEMA,
    )
    atoms = atomize_global_row(
        parent,
        [track],
        options=AtomizeOptions(
            verify_mask_files=True,
            verify_media_files=True,
            verified_release=verified_release,
        ),
    )
    if len(atoms) != 1 or atoms[0]["training_authorized"] is not True:
        raise ManifestError("synthetic fixture did not produce one authorized atom")
    atom = atoms[0]
    atomic_manifest_path = atomic_root / "atomic_manifest.jsonl"
    _write_canonical_jsonl(atomic_manifest_path, atoms)

    payload_path = payload_root / f"{atom['atom_id']}.pt"
    torch.save(
        _payload(atom, source_mask, target_mask, seed=seed),
        payload_path,
    )
    training_root = output_root / "training"
    binding_summary = bind_latent_payloads(
        atomic_manifest_path, payload_root, training_root
    )

    smoke_config = _smoke_config(seed)
    config_path = config_root / "pact_1_3b_one_step_smoke.json"
    _write_canonical_json(config_path, smoke_config)
    marker = {
        "schema_version": FIXTURE_SCHEMA,
        "synthetic_integration_only": True,
        "production_training_forbidden": True,
        "benchmark_reporting_forbidden": True,
        "contains_real_video": False,
        "contains_real_encoder_outputs": False,
        "purpose": (
            "one-step official-checkpoint forward/backward and adapter-save smoke test"
        ),
        "warning": (
            "Contract authorization only tests gates; it is not scientific data "
            "authorization. Never report or train a model from this fixture."
        ),
    }
    _write_canonical_json(output_root / "SYNTHETIC_INTEGRATION_ONLY.json", marker)

    summary = {
        "schema_version": "pact-synthetic-integration-fixture-summary-v1",
        "synthetic_integration_only": True,
        "seed": seed,
        "atom_id": atom["atom_id"],
        "authorized_atoms": 1,
        "latent_shape": [16, 3, 8, 8],
        "text_context_shape": [2, 4096],
        "vlm_context_shape": [3, 2048],
        "encoder_contract_sha256": encoder_contract_sha256(
            _synthetic_encoder_contract()
        ),
        "release_id": verified_release.release_id,
        "release_signer_fingerprint": fingerprint,
        "ephemeral_private_key_destroyed": not private_key.exists(),
        "global_manifest": "signed_release/global_manifest.jsonl",
        "release_receipt": "signed_release/post_generation_release.json",
        "signer_public_key": "synthetic_trust/synthetic_release_ed25519.pub",
        "atomic_manifest": "atomic/atomic_manifest.jsonl",
        "payload_root": "payloads",
        "training_manifest": "training/training_manifest.jsonl",
        "smoke_config": "configs/pact_1_3b_one_step_smoke.json",
        "binding_summary_sha256": file_sha256(training_root / "summary.json"),
        "payload_sha256": file_sha256(payload_path),
    }
    summary_path = output_root / "summary.json"
    _write_canonical_json(summary_path, summary)
    done = {
        "schema_version": "pact-synthetic-integration-fixture-done-v1",
        "complete": True,
        "synthetic_integration_only": True,
        "summary_sha256": file_sha256(summary_path),
        "training_manifest_sha256": binding_summary["training_manifest_sha256"],
    }
    _write_canonical_json(output_root / "done.json", done)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--issued-at-utc", default=DEFAULT_ISSUED_AT_UTC)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_synthetic_smoke_fixture(
        args.output_dir,
        seed=args.seed,
        issued_at_utc=args.issued_at_utc,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
