#!/usr/bin/env python3
"""Build deterministic v16r6-A debug32 release with LR-scaled audit epsilon.

This release starts from the authenticated v16r6-C source closure and replaces
only the shared v1 audit plumbing, the A wrapper, and the A/B contract test.
The A training data, objectives, gradients, optimizer, learning rate, and LoRA
scope are unchanged; only the absolute diagnostic epsilon follows the 0.1 LR
ratio (1e-12 -> 1e-13) while the required support remains 480/480.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import build_v16r6_ab_debug32_release as core


BASE_ARCHIVE_SHA256 = (
    "32fbfbb637542eec4477e388b409d111e93db8e34ba685ee671905ef2ade0857"
)
BASE_MANIFEST_SHA256 = (
    "5b368991b399727f0ded933f08ae0df6f7467755bbf9f6b164fa572b67da6f22"
)
BASE_CLOSURE_SHA256 = (
    "d38967351d355d60752b7e88d2546856685a7bc60e39c08e48c1d7a688222348"
)
BASE_SCHEMA = "bernini-v16r6c-two-sided-debug32-source-release-v1"
SOURCE_SCHEMA = "bernini-v16r6a-lr1e7-scaled-audit-debug32-source-release-v1"
RELEASE_SCHEMA = "bernini-v16r6-ab-debug32-release-v1"
BASE_FILE_COUNT = 41
SOURCE_FILE_COUNT = 41
BASE_V1_SHA256 = (
    "c5b80c3918aafd687169c3dd075fd1d78c1dd2bdfbafd134b3db44298bfa05b6"
)

V1_MEMBER = "methods/bernini_action_editing/train_online_anchor_attention_v1.py"
A_TRAINER_MEMBER = (
    "methods/bernini_action_editing/"
    "train_online_anchor_attention_full644_dynamic_static_v16r6a_lr1e7_32.py"
)
AB_TEST_MEMBER = (
    "methods/bernini_action_editing/tests/"
    "test_train_online_anchor_attention_v16r6_ab.py"
)
REGRESSION_TEST_MEMBER = (
    "methods/bernini_action_editing/tests/"
    "test_train_online_anchor_attention_full644_dynamic_static_v16r5.py"
)
SOURCE_MEMBERS = {
    V1_MEMBER: (
        V1_MEMBER,
        "224fabe68adea729c46700e7469c6f784c070b046daa9015b21747e91f654c2f",
    ),
    A_TRAINER_MEMBER: (
        A_TRAINER_MEMBER,
        "ade6975edcfb137e9c8db7350a36edbebb48f5e0473b18197077b018ec94d939",
    ),
    AB_TEST_MEMBER: (
        AB_TEST_MEMBER,
        "6161debdb2642334e0678c508473cde281a025d9c02649e8fc57a74dc6383555",
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
    controls[Path(__file__).name] = Path(__file__).resolve(strict=True).read_bytes()
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
            "a": {
                "trainer_member": A_TRAINER_MEMBER,
                "method": "bernini-online-anchor-v16r6a-full-lora-lr1e7-prefix32",
                "sole_changed_training_variable": (
                    "active_coordinate_rms_learning_rate_only"
                ),
                "learning_rate": 1e-7,
                "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
                "target_module_count": 240,
                "trainable_tensor_count": 480,
                "trainable_parameter_count": 188_743_680,
                "target_modules_sha256": (
                    "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a"
                ),
                "audit_only_change_from_failed_a_attempt": True,
                "component_gradient_audit": {
                    "v16r5_absolute_epsilon": 1e-12,
                    "learning_rate_scale_factor": 0.1,
                    "effective_absolute_epsilon": 1e-13,
                    "required_action_tensor_count_step2_plus": 480,
                    "required_raw_replay_tensor_count_step2_plus": 480,
                    "training_gradient_loss_optimizer_or_data_changed": False,
                },
            }
        },
        "training_inputs": {
            "full644_manifest_sha256": core.FULL644_MANIFEST_SHA256,
            "heldout8_manifest_sha256": core.HELDOUT8_SHA256,
            "data_teacher_scalar_loss_lr_lora_optimizer_unchanged_from_failed_a_attempt": True,
        },
        "preflight": {
            "new_contract_test_member": AB_TEST_MEMBER,
            "new_contract_expected_test_count": 12,
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
    sum_rows = []
    for name in sorted(
        (
            "v16r6ab-source.tar",
            "v16r6ab-source.manifest.json",
            "v16r6ab-release.json",
            *controls.keys(),
        )
    ):
        sum_rows.append(f"{core.sha256((output / name).read_bytes())}  {name}")
    sums_raw = ("\n".join(sum_rows) + "\n").encode("ascii")
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
