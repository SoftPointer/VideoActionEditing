from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from motive import r10b_bernini_retrieval_audit as retrieval
from motive import r10b_bernini_tangent_extract as bernini
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
from motive.r10b_tangent_core import (
    SMOKE_ROW_SCHEMA,
    canonical_json,
    object_digest,
)


FALSE_AUTHORIZATION = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}
DATA_ROOT = "/vast/test/goku/subject_movement/extracted"
SEEDS = (101, 202)
DIMENSION = 64


def _pretty(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _digest(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


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
        values.append(
            (
                "static",
                family,
                "control:static:global",
                "adult_human" if family == "wave" else "dog",
            )
        )
    for role in ("camera", "effect"):
        for index in range(4):
            family = "wave" if index % 2 == 0 else "quadruped_lie_down"
            values.append(
                (
                    role,
                    family,
                    f"control:{role}:global",
                    "adult_human" if family == "wave" else "dog",
                )
            )
    return values


def _pilot_rows() -> list[dict]:
    rows = []
    for index, (role, family, quota_cell, morphology) in enumerate(
        _specifications()
    ):
        canonical = (
            "Make the subject wave one forelimb toward the viewer."
            if family == "wave"
            else "Make the quadruped lie down."
        )
        cross_family = (
            "quadruped_lie_down" if family == "wave" else "wave"
        )
        cross_prompt = (
            "Make the quadruped lie down."
            if cross_family == "quadruped_lie_down"
            else "Make the subject wave one forelimb toward the viewer."
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
        quota_counts[row["quota_cell"]] = (
            quota_counts.get(row["quota_cell"], 0) + 1
        )
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


def _feature_vectors(rows: list[dict], seed: int) -> np.ndarray:
    values = np.zeros((len(rows), DIMENSION), dtype=np.float64)
    base_by_role = {
        ("positive", "wave"): 0,
        ("positive", "quadruped_lie_down"): 1,
        ("static", "*"): 2,
        ("camera", "*"): 3,
        ("effect", "*"): 4,
    }
    for index, row in enumerate(rows):
        role = row["pilot_role"]
        key = (role, row["family"]) if role == "positive" else (role, "*")
        values[index, base_by_role[key]] = 1.0
        values[index, 6 + index] = 0.01 + (seed % 7) * 1e-5
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _write_artifact(
    root: Path,
    *,
    tag: str,
    pilot_rows: list[dict],
) -> Path:
    artifact = root / f"artifact_{tag}"
    artifact.mkdir()
    variant_rows = copy.deepcopy(pilot_rows)
    prompt_field = retrieval.PROMPT_FIELD_BY_TAG[tag]
    for row in variant_rows:
        row["prompt"] = row[prompt_field]
    variant_path = root / f"manifest_{tag}.jsonl"
    variant_raw = _jsonl(variant_rows)
    variant_path.write_bytes(variant_raw)
    summary = {
        "schema_version": bernini.EXTRACT_SCHEMA,
        "model": {
            "id": "bernini_r_1_3b",
            "huggingface_repo": "ByteDance/Bernini-R-1.3B-Diffusers",
            "huggingface_revision": "f" * 40,
            "checkpoint_manifest": {"tree_sha256": "1" * 64},
            "selected_weight_sha256_before": "2" * 64,
            "selected_weight_sha256_after": "2" * 64,
            "selected_weights_unchanged": True,
        },
        "measurement": {
            "projection_seeds": list(SEEDS),
            "projection_dimension_per_role": DIMENSION,
            "noise_mode": "iid_spatiotemporal",
            "diffusion_noise_seed": 991,
            "scheduler_sigma": 0.55,
            "resize_policy": {"mode": "aspect_preserving_center_crop"},
            "source_condition": "source-id-1",
        },
        "parameter_manifest": [{"name": "block.attn1.to_q.weight"}],
        "parameter_manifest_sha256": "3" * 64,
        "data": {
            "manifest": str(variant_path.resolve()),
            "manifest_sha256": _digest(variant_raw),
            "track_cache_sha256": "4" * 64,
            "rows": len(pilot_rows),
            "families": ["quadruped_lie_down", "wave"],
            "unique_components": len(pilot_rows),
        },
        "runtime": {
            "dtype": "torch.bfloat16",
            "width": 256,
            "height": 256,
            "num_frames": 17,
        },
        "implementation": {"tree": "5" * 64},
        "official_bernini_source": {
            "commit": "6" * 40,
            "bundle_sha256": "7" * 64,
        },
        "source_tree_sha256": "8" * 64,
    }
    (artifact / bernini.SUMMARY_NAME).write_bytes(_pretty(summary))
    output_rows = []
    for index, row in enumerate(variant_rows):
        prompt = row["prompt"]
        noop = row["noop_prompt"]
        output_rows.append(
            {
                "schema_version": bernini.ROW_SCHEMA,
                "iid": row["iid"],
                "family": row["family"],
                "component_id": row["component_id"],
                "case_index": index,
                "source_split": row["source_split"],
                "projection_seed_coordinates_comparable": False,
                "prompt_conditioning": {
                    "raw_prompt_sha256": _digest(prompt.encode()),
                    "raw_noop_prompt_sha256": _digest(noop.encode()),
                    "effective_prompt_sha256": (
                        retrieval._effective_prompt_digest(prompt)
                    ),
                    "effective_noop_prompt_sha256": (
                        retrieval._effective_prompt_digest(noop)
                    ),
                },
                "formal_evidence": False,
                "representation_gate_passed": False,
                "renderer_probe_authorized": False,
                "editor_training_authorized": False,
            }
        )
    (artifact / bernini.ROWS_NAME).write_bytes(_jsonl(output_rows))
    arrays: dict[str, np.ndarray] = {
        "ids": np.asarray([row["iid"] for row in pilot_rows]),
        "metadata_json": np.asarray("{}"),
    }
    for seed in SEEDS:
        arrays[f"{retrieval.PRIMARY_FEATURE}__p{seed}"] = _feature_vectors(
            pilot_rows,
            seed,
        ).astype(np.float32)
    np.savez_compressed(artifact / bernini.FEATURES_NAME, **arrays)
    return artifact


def _validation(artifact: str | Path) -> dict:
    tag = Path(artifact).name.removeprefix("artifact_")
    return {
        "status": "VALID",
        "output_dir": str(Path(artifact).resolve()),
        "artifact_digest": _digest(tag.encode()),
        "rows": 20,
        "feature_arrays": 2,
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }


class R10BBerniniRetrievalAuditTests(unittest.TestCase):
    def test_balanced_retrieval_is_seed_local_and_leakage_readout_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _pilot_rows()
            pilot = _write_pilot(root, rows, balanced=True)
            artifacts = {
                tag: _write_artifact(root, tag=tag, pilot_rows=rows)
                for tag in retrieval.ARTIFACT_TAGS
            }
            with mock.patch.object(
                bernini,
                "validate_published_extract",
                side_effect=lambda path: _validation(path),
            ):
                result = retrieval.evaluate_retrieval(
                    pilot_dir=pilot,
                    artifacts=artifacts,
                )

            self.assertTrue(result["support"]["passed"])
            for seed in SEEDS:
                macro = result["per_projection_seed"][str(seed)][
                    "retrieval"
                ]["macro"]
                self.assertEqual(macro["recall_at_1"], 1.0)
                self.assertEqual(macro["recall_at_3"], 1.0)
                self.assertEqual(macro["same_family_pair_auroc"], 1.0)
                self.assertGreater(macro["similarity_margin"], 0.9)
                cross = result["cross_prompt_text_leakage"][
                    "per_projection_seed"
                ][str(seed)]["macro"]
                self.assertEqual(cross["actual_family_accuracy"], 1.0)
                self.assertGreater(cross["actual_minus_shuffled_margin"], 0.9)
            rank_policy = result["projection_seed_rank_agreement"]["policy"]
            self.assertFalse(rank_policy["cross_seed_vector_dot_products_computed"])
            self.assertFalse(rank_policy["cross_seed_vector_cosines_computed"])
            self.assertTrue(
                rank_policy[
                    "only_scalar_metrics_and_candidate_rankings_compared"
                ]
            )
            self.assertFalse(
                result["leakage_readouts"]["appearance"]["sufficient_for_gate"]
            )
            self.assertFalse(
                result["leakage_readouts"]["morphology"]["sufficient_for_gate"]
            )
            self.assertFalse(
                result["decision"][
                    "development_signal_requires_sigma_noise_dimension_holdout"
                ]
            )
            self.assertFalse(result["representation_gate_passed"])
            self.assertFalse(result["editor_training_authorized"])
            self.assertEqual(result["media_io"]["video_files_read"], 0)

    def test_unbalanced_control_commit_is_diagnostics_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _pilot_rows()[:-1]
            pilot = _write_pilot(root, rows, balanced=False)
            artifact = _write_artifact(
                root,
                tag="canonical",
                pilot_rows=rows,
            )

            def validate(path: str | Path) -> dict:
                value = _validation(path)
                value["rows"] = len(rows)
                return value

            with mock.patch.object(
                bernini,
                "validate_published_extract",
                side_effect=validate,
            ):
                result = retrieval.evaluate_retrieval(
                    pilot_dir=pilot,
                    artifacts={"canonical": artifact},
                )
            self.assertFalse(result["support"]["passed"])
            self.assertFalse(
                result["support"]["checks"]["balanced_pilot_commit"]
            )
            self.assertIn(
                "balanced_support",
                result["development_checks"]["failed"],
            )
            self.assertFalse(
                result["decision"][
                    "development_signal_requires_sigma_noise_dimension_holdout"
                ]
            )

    def test_empty_unbalanced_pilot_commit_has_strict_valid_closure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pilot = _write_pilot(Path(temporary), [], balanced=False)
            validation = retrieval.validate_controlled_pilot_commit(pilot)
            self.assertEqual(validation["status"], "VALID")
            self.assertEqual(validation["rows"], 0)
            self.assertFalse(validation["balanced_pilot_ready"])
            self.assertFalse(validation["representation_gate_passed"])

    def test_variant_manifest_may_change_only_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _pilot_rows()
            pilot = _write_pilot(root, rows, balanced=True)
            canonical = _write_artifact(
                root,
                tag="canonical",
                pilot_rows=rows,
            )
            original = _write_artifact(
                root,
                tag="original",
                pilot_rows=rows,
            )
            summary_path = original / bernini.SUMMARY_NAME
            summary = json.loads(summary_path.read_text())
            variant_path = Path(summary["data"]["manifest"])
            variant_rows = [
                json.loads(line) for line in variant_path.read_text().splitlines()
            ]
            variant_rows[0]["component_id"] = "tampered-component"
            variant_raw = _jsonl(variant_rows)
            variant_path.write_bytes(variant_raw)
            summary["data"]["manifest_sha256"] = _digest(variant_raw)
            summary_path.write_bytes(_pretty(summary))
            with mock.patch.object(
                bernini,
                "validate_published_extract",
                side_effect=lambda path: _validation(path),
            ):
                with self.assertRaisesRegex(
                    retrieval.R10BBerniniRetrievalAuditError,
                    "differs outside its bound prompt",
                ):
                    retrieval.evaluate_retrieval(
                        pilot_dir=pilot,
                        artifacts={
                            "canonical": canonical,
                            "original": original,
                        },
                    )

    def test_atomic_output_validator_detects_payload_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _pilot_rows()
            pilot = _write_pilot(root, rows, balanced=True)
            artifacts = {
                tag: _write_artifact(root, tag=tag, pilot_rows=rows)
                for tag in retrieval.ARTIFACT_TAGS
            }
            output = root / "audit"
            with mock.patch.object(
                bernini,
                "validate_published_extract",
                side_effect=lambda path: _validation(path),
            ):
                retrieval.publish_retrieval_audit(
                    pilot_dir=pilot,
                    artifacts=artifacts,
                    output_dir=output,
                )
                validation = retrieval.validate_published_retrieval_audit(
                    output
                )
            self.assertEqual(validation["status"], "VALID")
            self.assertFalse(validation["representation_gate_passed"])

            audit_path = output / retrieval.AUDIT_NAME
            audit_path.chmod(0o644)
            raw = audit_path.read_text()
            audit_path.write_text(
                raw.replace(
                    '"video_files_read": 0',
                    '"video_files_read": 1',
                    1,
                )
            )
            with self.assertRaisesRegex(
                retrieval.R10BBerniniRetrievalAuditError,
                "done binding differs",
            ):
                retrieval.validate_published_retrieval_audit(
                    output,
                    revalidate_sources=False,
                )
            os.chmod(output, 0o755)


if __name__ == "__main__":
    unittest.main()
