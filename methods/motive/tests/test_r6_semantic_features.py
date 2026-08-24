from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import motive.r6_semantic_features as semantic_module

from motive.r6_semantic_features import (
    CLIP_DIM,
    CLIP_REVISION,
    DONE_NAME,
    METADATA_NAME,
    R6_SEMANTIC_METADATA_SCHEMA,
    R6_SOURCE_SNAPSHOT_SENTINEL,
    SOURCE_FILES_NAME,
    SOURCE_PROVENANCE_NAME,
    UMT5_DIM,
    UMT5_REVISION,
    _checked_revision,
    _file_digest,
    _object_digest,
    _validate_source_snapshot_binding,
    commit_synthetic_artifact,
    validate_artifact,
)


def _observation(index: int) -> dict[str, object]:
    return {
        "schema_version": "qwen-motion-observation-v2",
        "source_action": f"walking before edit {index}",
        "target_action": f"running toward the camera after edit {index}",
        "source_actor_motion": "clear",
        "target_actor_motion": "clear",
        "camera_dominance": "low",
        "background_dominance": "low",
        "artifact_level": "low",
        "preservation_quality": "acceptable",
        "temporal_evidence": [
            f"The actor changes position from T0 through T5 in row {index}."
        ],
        "uncertainty_codes": [],
    }


def _row(index: int) -> dict[str, object]:
    observation = _observation(index)
    return {
        "feature_index": index,
        "iid": f"iid-{index:03d}",
        "prompt": f"Make the foreground actor run quickly, example {index}.",
        "qwen_evidence": {
            "visual": {
                "status": "ok",
                "observation_validated_from": "original",
                "observation_repairs": [],
                "alignment_repairs": [],
                "observation": observation,
                "observation_raw": json.dumps(
                    observation,
                    ensure_ascii=False,
                    indent=2,
                ),
                "observation_digest": _object_digest(observation),
            }
        },
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _write_source_snapshot(root: Path) -> tuple[Path, str, str]:
    snapshot = root / "source_snapshot"
    package = Path(semantic_module.__file__).resolve().parent
    sources = {
        "methods/motive/motive/r6_semantic_features.py":
            Path(semantic_module.__file__).resolve(),
        "methods/motive/motive/qwen_filter.py": package / "qwen_filter.py",
    }
    manifest_rows: list[dict[str, object]] = []
    for relative, source in sorted(sources.items()):
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o444)
        manifest_rows.append(
            {
                "mode": f"{target.stat().st_mode & 0o7777:04o}",
                "path": relative,
                "sha256": _file_digest(target),
                "size": target.stat().st_size,
                "type": "file",
            }
        )
    manifest_text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in manifest_rows
    )
    manifest_path = snapshot / SOURCE_FILES_NAME
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha256 = _file_digest(manifest_path)
    tree_sha256 = hashlib.sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()
    (snapshot / SOURCE_PROVENANCE_NAME).write_text(
        json.dumps(
            {
                "schema": "motive-action-source-snapshot-v1",
                "source_file_count": len(manifest_rows),
                "source_tree_sha256": tree_sha256,
                "source_manifest_sha256": manifest_sha256,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot, tree_sha256, manifest_sha256


class R6SemanticFeatureTests(unittest.TestCase):
    def test_registered_revisions_are_full_immutable_git_commits(self) -> None:
        self.assertEqual(
            _checked_revision(
                UMT5_REVISION,
                expected=UMT5_REVISION,
                name="UMT5",
            ),
            UMT5_REVISION,
        )
        self.assertEqual(
            _checked_revision(
                CLIP_REVISION,
                expected=CLIP_REVISION,
                name="CLIP",
            ),
            CLIP_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "40-hex"):
            _checked_revision(
                UMT5_REVISION[:12],
                expected=UMT5_REVISION,
                name="UMT5",
            )

    def test_synthetic_commit_and_validate_full_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_manifest = root / "r5_manifest.jsonl"
            output = root / "semantic"
            rows = [_row(index) for index in range(3)]
            _write_manifest(input_manifest, rows)

            done = commit_synthetic_artifact(
                input_manifest=input_manifest,
                output_dir=output,
                seed=17,
            )
            self.assertEqual(done["status"], "complete")
            result = validate_artifact(output)
            self.assertEqual(result["rows"], 3)
            self.assertTrue(result["synthetic_test_artifact"])

            metadata = json.loads(
                (output / METADATA_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["schema_version"],
                R6_SEMANTIC_METADATA_SCHEMA,
            )
            self.assertFalse(metadata["fit_contract"]["pca_fitted"])
            self.assertFalse(
                metadata["fit_contract"]["standardizer_fitted"]
            )
            self.assertTrue(
                metadata["fit_contract"]["raw_frozen_l2_embeddings"]
            )
            source_snapshot = metadata["source_snapshot"]
            self.assertTrue(source_snapshot["synthetic_test_sentinel"])
            self.assertEqual(
                source_snapshot["sentinel"],
                R6_SOURCE_SNAPSHOT_SENTINEL,
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            committed_done = json.loads(
                (output / DONE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["source_snapshot"], source_snapshot)
            self.assertEqual(
                committed_done["source_snapshot"],
                source_snapshot,
            )
            self.assertTrue(
                metadata["source_fields"]["prompt"][
                    "allowed_predictor_input"
                ]
            )
            observed_contract = metadata["source_fields"][
                "observed_target"
            ]
            self.assertTrue(observed_contract["target_derived"])
            self.assertTrue(observed_contract["diagnostic_only"])
            self.assertFalse(observed_contract["allowed_predictor_input"])

            with np.load(
                output / "semantic_features.npz",
                allow_pickle=False,
            ) as archive:
                expected = {
                    "umt5_prompt": (3, UMT5_DIM),
                    "umt5_observed_target": (3, UMT5_DIM),
                    "clip_prompt": (3, CLIP_DIM),
                    "clip_observed_target": (3, CLIP_DIM),
                }
                for name, shape in expected.items():
                    matrix = archive[name]
                    self.assertEqual(matrix.shape, shape)
                    self.assertEqual(matrix.dtype, np.float32)
                    np.testing.assert_allclose(
                        np.linalg.norm(matrix, axis=1),
                        np.ones(3),
                        rtol=0.0,
                        atol=2e-5,
                    )
                self.assertEqual(
                    archive["iids"].tolist(),
                    [f"iid-{index:03d}" for index in range(3)],
                )
                self.assertEqual(
                    archive["prompt_text_sha256"].tolist(),
                    [
                        __import__("hashlib").sha256(
                            str(row["prompt"]).encode("utf-8")
                        ).hexdigest()
                        for row in rows
                    ],
                )

    def test_source_snapshot_binding_and_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, tree_sha256, manifest_sha256 = (
                _write_source_snapshot(root)
            )
            contract = _validate_source_snapshot_binding(
                source_snapshot=snapshot,
                source_tree_sha256=tree_sha256,
                source_manifest_sha256=manifest_sha256,
            )
            self.assertFalse(contract["synthetic_test_sentinel"])
            self.assertEqual(
                contract["resolved_path"],
                str(snapshot.resolve()),
            )
            self.assertEqual(
                contract["source_tree_sha256"],
                tree_sha256,
            )
            self.assertEqual(
                contract["source_manifest_sha256"],
                manifest_sha256,
            )

            with self.assertRaisesRegex(ValueError, "64-hex"):
                _validate_source_snapshot_binding(
                    source_snapshot=snapshot,
                    source_tree_sha256=tree_sha256[:40],
                    source_manifest_sha256=manifest_sha256,
                )
            with self.assertRaisesRegex(ValueError, "tree SHA-256"):
                _validate_source_snapshot_binding(
                    source_snapshot=snapshot,
                    source_tree_sha256="1" * 64,
                    source_manifest_sha256=manifest_sha256,
                )
            with self.assertRaisesRegex(
                ValueError,
                "SOURCE_FILES.jsonl SHA-256",
            ):
                _validate_source_snapshot_binding(
                    source_snapshot=snapshot,
                    source_tree_sha256=tree_sha256,
                    source_manifest_sha256="2" * 64,
                )

            implementation = (
                snapshot
                / "methods/motive/motive/r6_semantic_features.py"
            )
            implementation.chmod(0o644)
            with implementation.open("ab") as handle:
                handle.write(b"\n# tampered\n")
            implementation.chmod(0o444)
            with self.assertRaisesRegex(
                ValueError,
                "size mismatch|file SHA mismatch",
            ):
                _validate_source_snapshot_binding(
                    source_snapshot=snapshot,
                    source_tree_sha256=tree_sha256,
                    source_manifest_sha256=manifest_sha256,
                )

    def test_synthetic_snapshot_sentinel_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            _write_manifest(manifest, [_row(0)])
            with self.assertRaisesRegex(
                ValueError,
                "unsupported synthetic",
            ):
                commit_synthetic_artifact(
                    input_manifest=manifest,
                    output_dir=root / "out",
                    source_snapshot_sentinel="not-the-registered-sentinel",
                )

    def test_original_qwen_observation_contract_fails_closed(self) -> None:
        mutations = {
            "not_original": lambda row: row["qwen_evidence"]["visual"].update(
                {"observation_validated_from": "repair_1"}
            ),
            "has_repairs": lambda row: row["qwen_evidence"]["visual"].update(
                {"observation_repairs": [{"attempt": 1}]}
            ),
            "digest_mismatch": lambda row: row["qwen_evidence"][
                "visual"
            ].update({"observation_digest": "0" * 64}),
            "raw_mismatch": lambda row: row["qwen_evidence"]["visual"].update(
                {"observation_raw": json.dumps(_observation(999))}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = root / "input.jsonl"
                row = copy.deepcopy(_row(0))
                mutate(row)
                _write_manifest(manifest, [row])
                with self.assertRaises(ValueError):
                    commit_synthetic_artifact(
                        input_manifest=manifest,
                        output_dir=root / "out",
                    )

    def test_original_fenced_qwen_json_is_accepted_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            row = _row(0)
            visual = row["qwen_evidence"]["visual"]
            visual["observation_raw"] = (
                "```json\n"
                + json.dumps(visual["observation"], ensure_ascii=False)
                + "\n```"
            )
            _write_manifest(manifest, [row])
            commit_synthetic_artifact(
                input_manifest=manifest,
                output_dir=root / "out",
            )
            self.assertEqual(validate_artifact(root / "out")["rows"], 1)

    def test_archive_tamper_is_detected_even_if_done_sha_is_rewritten(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            output = root / "out"
            _write_manifest(manifest, [_row(0), _row(1)])
            commit_synthetic_artifact(
                input_manifest=manifest,
                output_dir=output,
            )

            archive_path = output / "semantic_features.npz"
            with np.load(archive_path, allow_pickle=False) as loaded:
                arrays = {
                    name: np.asarray(loaded[name])
                    for name in loaded.files
                }
            arrays["umt5_prompt"] = arrays["umt5_prompt"].copy()
            arrays["umt5_prompt"][0] *= np.float32(2.0)
            with archive_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)

            done_path = output / DONE_NAME
            done = json.loads(done_path.read_text(encoding="utf-8"))
            done["artifacts"]["archive"]["sha256"] = _file_digest(
                archive_path
            )
            done_path.write_text(
                json.dumps(done, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "L2 normalized|array contract",
            ):
                validate_artifact(output)

    def test_input_manifest_tamper_and_partial_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            output = root / "out"
            rows = [_row(0), _row(1)]
            _write_manifest(manifest, rows)
            commit_synthetic_artifact(
                input_manifest=manifest,
                output_dir=output,
            )
            rows.reverse()
            rows[0]["feature_index"] = 0
            rows[1]["feature_index"] = 1
            _write_manifest(manifest, rows)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                validate_artifact(output)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            output = root / "out"
            _write_manifest(manifest, [_row(0)])
            output.mkdir()
            (output / "metadata.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                commit_synthetic_artifact(
                    input_manifest=manifest,
                    output_dir=output,
                )


if __name__ == "__main__":
    unittest.main()
