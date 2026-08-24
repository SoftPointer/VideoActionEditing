from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive import r10b_bernini_prompt_variants as variants
from motive.r10b_bernini_pilot_manifest import (
    FINAL_DONE_NAME,
    FINAL_DONE_SCHEMA,
    FINAL_MANIFEST_NAME,
    FINAL_QUOTAS,
    FINAL_SHORTFALL_NAME,
    FINAL_SUMMARY_NAME,
    FINAL_SUMMARY_SCHEMA,
    SHORTFALL_SCHEMA,
)
from motive.r10b_tangent_core import SMOKE_ROW_SCHEMA, canonical_json


FALSE_AUTHORIZATION = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}
DATA_ROOT = "/vast/test/goku/subject_movement/extracted"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pretty(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _specifications() -> list[tuple[str, str, str, str]]:
    values = [
        ("positive", "wave", "positive:wave:adult_human", "adult_human"),
        ("positive", "wave", "positive:wave:child_human", "child_human"),
        (
            "positive",
            "wave",
            "positive:wave:character_or_nonhuman",
            "character_or_nonhuman",
        ),
        (
            "positive",
            "wave",
            "positive:wave:additional_direct_nonreflection",
            "adult_human",
        ),
        (
            "positive",
            "quadruped_lie_down",
            "positive:quadruped_lie_down:dog_or_bulldog",
            "dog",
        ),
        (
            "positive",
            "quadruped_lie_down",
            "positive:quadruped_lie_down:dog_or_bulldog",
            "bulldog",
        ),
        (
            "positive",
            "quadruped_lie_down",
            "positive:quadruped_lie_down:cat",
            "cat",
        ),
        (
            "positive",
            "quadruped_lie_down",
            "positive:quadruped_lie_down:other_quadruped",
            "other_quadruped",
        ),
    ]
    for index in range(4):
        family = "wave" if index % 2 == 0 else "quadruped_lie_down"
        morphology = "adult_human" if family == "wave" else "dog"
        values.append(("static", family, "control:static:global", morphology))
    for role in ("camera", "effect"):
        for index in range(4):
            family = "wave" if index % 2 == 0 else "quadruped_lie_down"
            morphology = "adult_human" if family == "wave" else "dog"
            values.append(
                (role, family, f"control:{role}:global", morphology)
            )
    return values


def _pilot_rows() -> list[dict]:
    rows: list[dict] = []
    wave_prompt = "Make the subject wave one forelimb toward the viewer."
    lie_prompt = "Make the quadruped lie down."
    for index, (role, family, quota_cell, morphology) in enumerate(
        _specifications()
    ):
        canonical = wave_prompt if family == "wave" else lie_prompt
        cross_family = (
            "quadruped_lie_down" if family == "wave" else "wave"
        )
        cross_prompt = (
            lie_prompt if cross_family == "quadruped_lie_down" else wave_prompt
        )
        iid = f"pilot-{index:03d}"
        rows.append(
            {
                "schema_version": SMOKE_ROW_SCHEMA,
                "iid": iid,
                "family": family,
                "primary_family": family,
                "prompt": canonical,
                "canonical_prompt": canonical,
                "original_prompt": f"Original instruction for {iid}.",
                "noop_prompt": "Keep the video unchanged.",
                "cross_family_shuffle_prompt": cross_prompt,
                "cross_family_shuffle_family": cross_family,
                "component_id": f"component-{index:03d}",
                "source_split": "train",
                "fresh": True,
                "data_root": DATA_ROOT,
                "src_video": f"{iid}/source.mp4",
                "tgt_video": f"{iid}/target.mp4",
                "src_video_sha256": f"{index + 1:064x}",
                "tgt_video_sha256": f"{index + 101:064x}",
                "candidate_input_digest": f"{index + 201:064x}",
                "track_input_index": index,
                "track_cache_index": index,
                "pilot_role": role,
                "quota_cell": quota_cell,
                "pilot_rank": index + 1,
                "qwen_audit_binding": {
                    "audit": {
                        "subject_morphology": morphology,
                        "identity_appearance_change": "none",
                    }
                },
                "formal_evidence": False,
                "representation_promoted": False,
                "renderer_probe_authorized": False,
                "generation_authorized": False,
                "training_authorized": False,
                "authorization": copy.deepcopy(FALSE_AUTHORIZATION),
            }
        )
    return rows


def _write_pilot(
    root: Path,
    rows: list[dict],
    *,
    balanced: bool,
) -> Path:
    pilot = root / "pilot"
    pilot.mkdir()
    manifest_raw = _jsonl(rows)
    shortfall_values = (
        {}
        if balanced
        else {
            "control:effect:global": {
                "required": 4,
                "selected": 3,
                "eligible_before_component_dedup": 3,
            }
        }
    )
    shortfalls = {
        "schema_version": SHORTFALL_SCHEMA,
        "balanced_pilot_ready": balanced,
        "shortfalls": shortfall_values,
        "no_control_rows_fabricated": True,
        "row_reuse_allowed": False,
        "component_reuse_allowed": False,
    }
    shortfall_raw = _pretty(shortfalls)
    quota_counts: dict[str, int] = {}
    for row in rows:
        cell = str(row["quota_cell"])
        quota_counts[cell] = quota_counts.get(cell, 0) + 1
    summary = {
        "schema_version": FINAL_SUMMARY_SCHEMA,
        "balanced_pilot_ready": balanced,
        "rows": len(rows),
        "unique_iids": len(rows),
        "unique_components": len(rows),
        "component_disjoint": True,
        "quota_targets": copy.deepcopy(FINAL_QUOTAS),
        "quota_selected": dict(sorted(quota_counts.items())),
        "shortfalls": shortfall_values,
        "qwen_audit": {
            "qwen_model_id": "Qwen-test",
            "qwen_prompt_sha256": "a" * 64,
        },
        "outputs": {
            FINAL_MANIFEST_NAME: {
                "rows": len(rows),
                "sha256": _digest(manifest_raw),
            },
            FINAL_SHORTFALL_NAME: {"sha256": _digest(shortfall_raw)},
        },
        "video_bytes_copied": False,
        "controls_fabricated": False,
        "human_labels": False,
        "authorization": copy.deepcopy(FALSE_AUTHORIZATION),
    }
    summary_raw = _pretty(summary)
    done = {
        "schema_version": FINAL_DONE_SCHEMA,
        "rows": len(rows),
        "balanced_pilot_ready": balanced,
        "files": {
            FINAL_MANIFEST_NAME: _digest(manifest_raw),
            FINAL_SHORTFALL_NAME: _digest(shortfall_raw),
            FINAL_SUMMARY_NAME: _digest(summary_raw),
        },
        "authorization": copy.deepcopy(FALSE_AUTHORIZATION),
    }
    (pilot / FINAL_MANIFEST_NAME).write_bytes(manifest_raw)
    (pilot / FINAL_SHORTFALL_NAME).write_bytes(shortfall_raw)
    (pilot / FINAL_SUMMARY_NAME).write_bytes(summary_raw)
    (pilot / FINAL_DONE_NAME).write_bytes(_pretty(done))
    return pilot


class R10BBerniniPromptVariantTests(unittest.TestCase):
    def test_build_is_exact_prompt_only_transform_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot_rows = _pilot_rows()
            pilot = _write_pilot(root, pilot_rows, balanced=True)
            output = root / "variants"

            report = variants.build_prompt_variants(
                pilot_dir=pilot,
                output_dir=output,
            )
            self.assertEqual(report["status"], "VALID")
            self.assertEqual(set(path.name for path in output.iterdir()), set(
                variants.OUTPUT_NAMES
            ))
            self.assertFalse((output / FINAL_MANIFEST_NAME).exists())

            original = [
                json.loads(line)
                for line in (output / variants.ORIGINAL_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            cross = [
                json.loads(line)
                for line in (output / variants.CROSS_FAMILY_SHUFFLE_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["iid"] for row in original],
                [row["iid"] for row in pilot_rows],
            )
            self.assertEqual(
                [row["component_id"] for row in cross],
                [row["component_id"] for row in pilot_rows],
            )
            for pilot_row, original_row, cross_row in zip(
                pilot_rows,
                original,
                cross,
                strict=True,
            ):
                expected_original = copy.deepcopy(pilot_row)
                expected_original["prompt"] = pilot_row["original_prompt"]
                expected_cross = copy.deepcopy(pilot_row)
                expected_cross["prompt"] = pilot_row[
                    "cross_family_shuffle_prompt"
                ]
                self.assertEqual(original_row, expected_original)
                self.assertEqual(cross_row, expected_cross)

            summary = json.loads(
                (output / variants.SUMMARY_NAME).read_text(encoding="utf-8")
            )
            self.assertFalse(summary["canonical"]["copied"])
            self.assertFalse(summary["video_bytes_copied"])
            self.assertFalse(summary["rendering_performed"])
            self.assertFalse(summary["training_performed"])
            self.assertEqual(summary["authorization"], FALSE_AUTHORIZATION)
            self.assertEqual(
                variants.validate_prompt_variants(
                    variant_dir=output,
                    pilot_dir=pilot,
                )["commit_digest"],
                report["commit_digest"],
            )

    def test_tampered_non_prompt_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = _write_pilot(root, _pilot_rows(), balanced=True)
            output = root / "variants"
            variants.build_prompt_variants(
                pilot_dir=pilot,
                output_dir=output,
            )
            path = output / variants.ORIGINAL_NAME
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["component_id"] = "tampered-component"
            path.write_bytes(_jsonl(rows))

            with self.assertRaisesRegex(
                variants.R10BBerniniPromptVariantError,
                "exact prompt-only transform",
            ):
                variants.validate_prompt_variants(
                    variant_dir=output,
                    pilot_dir=pilot,
                )

    def test_unbalanced_pilot_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _pilot_rows()[:-1]
            pilot = _write_pilot(root, rows, balanced=False)
            output = root / "variants"

            with self.assertRaisesRegex(
                variants.R10BBerniniPromptVariantError,
                "balanced_pilot_ready=true",
            ):
                variants.build_prompt_variants(
                    pilot_dir=pilot,
                    output_dir=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
