#!/usr/bin/env python3
"""Build the independent deterministic v16r6-C two-sided debug32 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import build_v16r6_ab_debug32_release as core


BASE_ARCHIVE_SHA256 = (
    "f77d896e324b59698093847962d6834d7e31b52f0a7b74cfdfba060be61c9e7e"
)
BASE_MANIFEST_SHA256 = (
    "adc2d03e057ed433070d4cd19fec82aead8c29c542f81d20ab8e1ca8063986d8"
)
BASE_CLOSURE_SHA256 = (
    "52fcdd6d79dfccfdee0d2b599236386b9b82f6407aab4bc4910662eaf3eca3dd"
)
BASE_SCHEMA = "bernini-v16r6-ab-debug32-source-release-v1"
SOURCE_SCHEMA = "bernini-v16r6c-two-sided-debug32-source-release-v1"
RELEASE_SCHEMA = "bernini-v16r6c-two-sided-debug32-release-v1"
BASE_FILE_COUNT = 39
SOURCE_FILE_COUNT = 41
BASE_V1_SHA256 = (
    "5e152ebf4ccdb4211c96c7c66b1891b0b38750bbc30f62d61a9a93261a73c178"
)

V1_MEMBER = "methods/bernini_action_editing/train_online_anchor_attention_v1.py"
C_TRAINER_MEMBER = (
    "methods/bernini_action_editing/"
    "train_online_anchor_attention_full644_dynamic_static_"
    "v16r6c_two_sided_delta32.py"
)
C_TEST_MEMBER = (
    "methods/bernini_action_editing/tests/"
    "test_train_online_anchor_attention_v16r6c_two_sided_delta.py"
)
REGRESSION_TEST_MEMBER = (
    "methods/bernini_action_editing/tests/"
    "test_train_online_anchor_attention_full644_dynamic_static_v16r5.py"
)
SOURCE_MEMBERS = {
    V1_MEMBER: (
        V1_MEMBER,
        "c5b80c3918aafd687169c3dd075fd1d78c1dd2bdfbafd134b3db44298bfa05b6",
    ),
    C_TRAINER_MEMBER: (
        C_TRAINER_MEMBER,
        "cb4d9047176bc11018fe2f97b49fd0334d193e7c8a382bec7e3b2732665417df",
    ),
    C_TEST_MEMBER: (
        C_TEST_MEMBER,
        "0af0d28f7e14b16cf41e8d643c93fe1ec75dfac2602d629c90fe6b68e4f23907",
    ),
}
CONTROL_RELATIVES = (
    "methods/bernini_action_editing/scripts/auh_run_v16r6_ab_debug32.sh",
    "methods/bernini_action_editing/scripts/auh_launch_v16r6_ab_debug32.sh",
    "methods/bernini_action_editing/tools/build_v16r6_ab_debug32_release.py",
)


def configure_core() -> None:
    core.BASE_ARCHIVE_SHA256 = BASE_ARCHIVE_SHA256
    core.BASE_MANIFEST_SHA256 = BASE_MANIFEST_SHA256
    core.BASE_CLOSURE_SHA256 = BASE_CLOSURE_SHA256
    core.BASE_SCHEMA = BASE_SCHEMA
    core.SOURCE_SCHEMA = SOURCE_SCHEMA
    core.BASE_FILE_COUNT = BASE_FILE_COUNT
    core.SOURCE_FILE_COUNT = SOURCE_FILE_COUNT
    core.ORIGINAL_V1_SHA256 = BASE_V1_SHA256
    core.SOURCE_MEMBERS = dict(SOURCE_MEMBERS)


def build(
    workspace_root: Path,
    base_archive: Path,
    base_manifest: Path,
    output: Path,
) -> Mapping[str, Any]:
    configure_core()
    workspace_root = workspace_root.resolve(strict=True)
    archive_raw, manifest_raw, source = core.build_source(
        workspace_root, base_archive, base_manifest
    )
    if output.exists() or output.is_symlink():
        raise core.ReleaseError("release output must be fresh")

    controls = {
        Path(relative).name: core.regular_file(
            workspace_root / relative, relative
        ).read_bytes()
        for relative in CONTROL_RELATIVES
    }
    builder_raw = Path(__file__).resolve(strict=True).read_bytes()
    controls[Path(__file__).name] = builder_raw
    control_hashes = {
        name: core.sha256(raw) for name, raw in sorted(controls.items())
    }
    archive_sha = core.sha256(archive_raw)
    manifest_sha = core.sha256(manifest_raw)
    release = {
        "schema_version": RELEASE_SCHEMA,
        "created_date": "2026-08-24",
        "source_release": {
            "archive": "v16r6ab-source.tar",
            "manifest": "v16r6ab-source.manifest.json",
            "builder": Path(__file__).name,
            "archive_sha256": archive_sha,
            "manifest_sha256": manifest_sha,
            "content_closure_sha256": source["content_closure_sha256"],
            "content_closure_sha1": source["content_closure_sha1"],
            "file_count": SOURCE_FILE_COUNT,
            "base_archive_sha256": BASE_ARCHIVE_SHA256,
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "base_content_closure_sha256": BASE_CLOSURE_SHA256,
        },
        "control_file_sha256": control_hashes,
        "debug_contract": {
            "sealed_manifest_row_count": 644,
            "optimizer_step_budget": 32,
            "schedule": "sealed_full644_family_round_robin_prefix32_exact_once_debug",
            "exact644_training_complete": False,
            "terminal_full644_checkpoint": False,
            "scientific_claim_authorized": False,
            "training_complete_filename_is_only_process_completion": True,
            "resume_allowed": False,
            "seed": 2026082302,
        },
        "variants": {
            "c": {
                "trainer_member": C_TRAINER_MEMBER,
                "method": (
                    "bernini-online-anchor-v16r6c-two-sided-"
                    "student-delta-prefix32"
                ),
                "sole_changed_variable": (
                    "same_action_student_delta_gradient_estimator_only"
                ),
                "gradient_mode": (
                    "two_sided_sequential_j_on_minus_j_off_v16r6c"
                ),
                "student_delta_jacobian": "J_route_on_minus_J_route_off",
                "learning_rate": 1e-6,
                "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
                "target_module_count": 240,
                "trainable_tensor_count": 480,
                "trainable_parameter_count": 188_743_680,
                "target_modules_sha256": (
                    "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a"
                ),
            }
        },
        "training_inputs": {
            "full644_manifest_sha256": core.FULL644_MANIFEST_SHA256,
            "heldout8_manifest_sha256": core.HELDOUT8_SHA256,
            "data_teacher_scalar_loss_lr_lora_optimizer_unchanged_from_v16r5": True,
        },
        "preflight": {
            "new_contract_test_member": C_TEST_MEMBER,
            "new_contract_expected_test_count": 6,
            "v16r5_regression_test_member": REGRESSION_TEST_MEMBER,
            "v16r5_regression_expected_test_count": 8,
            "runs_before_any_gpu_step": True,
        },
    }
    release_raw = json.dumps(
        release, indent=2, sort_keys=True, ensure_ascii=True
    ).encode("ascii") + b"\n"

    output.mkdir(parents=True, mode=0o755)
    core.write_new(output / "v16r6ab-source.tar", archive_raw, 0o444)
    core.write_new(
        output / "v16r6ab-source.manifest.json", manifest_raw, 0o444
    )
    for name, raw in controls.items():
        core.write_new(
            output / name, raw, 0o555 if name.endswith(".sh") else 0o444
        )
    core.write_new(output / "v16r6ab-release.json", release_raw, 0o444)
    sums = []
    for name in sorted(
        (
            "v16r6ab-source.tar",
            "v16r6ab-source.manifest.json",
            "v16r6ab-release.json",
            *controls.keys(),
        )
    ):
        sums.append(f"{core.sha256((output / name).read_bytes())}  {name}")
    sums_raw = ("\n".join(sums) + "\n").encode("ascii")
    core.write_new(output / "SHA256SUMS", sums_raw, 0o444)
    verified = verify(
        output / "v16r6ab-source.tar",
        output / "v16r6ab-source.manifest.json",
        archive_sha,
        manifest_sha,
    )
    return {
        "output": str(output.resolve()),
        **verified,
        "release_manifest_sha256": core.sha256(release_raw),
        "control_file_sha256": control_hashes,
        "sha256sums_sha256": core.sha256(sums_raw),
    }


def verify(
    archive: Path,
    manifest: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    configure_core()
    return core.verify_source(
        archive,
        manifest,
        expected_archive_sha256,
        expected_manifest_sha256,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--workspace-root", type=Path, required=True)
    build_parser.add_argument("--base-source-archive", type=Path, required=True)
    build_parser.add_argument("--base-source-manifest", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-archive-sha256", required=True)
    verify_parser.add_argument("--expected-manifest-sha256", required=True)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build":
        result = build(
            args.workspace_root,
            args.base_source_archive,
            args.base_source_manifest,
            args.output,
        )
    else:
        result = verify(
            args.archive,
            args.manifest,
            args.expected_archive_sha256,
            args.expected_manifest_sha256,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
