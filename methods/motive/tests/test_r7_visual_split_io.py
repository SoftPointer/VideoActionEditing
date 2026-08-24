from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motive.r7_preflight_extract import (
    FINAL_WORLD_SIZE,
    R7_ROW_SCHEMA,
    _array_digest,
    _commit_shard,
    _empty_arrays,
    _file_digest,
    _object_digest,
    finalize_shards,
    rank_directory,
)
from motive.r7_visual_split import R7VisualSplitConfig
from motive.r7_visual_split_io import (
    ASSIGNMENTS_NAME,
    COMPONENTS_NAME,
    DONE_NAME,
    SUMMARY_NAME,
    build_visual_split_artifacts,
    load_preflight_visual_pairs,
    main,
    read_prior_iid_ledger,
    validate_visual_split_artifacts,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _dino_provenance(*, revision: str = "a" * 64) -> dict[str, object]:
    return {
        "encoder_id": "facebook/dinov2-base",
        "encoder_revision": revision,
        "resolved_path": "/models/dinov2-base",
        "model_tree_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "model_file_count": 3,
        "frame_sampling_version": "uniform-6-from-uniform-32-v1",
        "preprocessing_version":
            "transformers-auto-image-processor-local-v1",
        "pooling": "last-hidden-state-cls-token-v1",
        "embedding_dim": 768,
        "dtype": "float32",
        "normalization": "l2-per-frame",
        "frozen_encoder": True,
        "local_files_only": True,
    }


def _input_row(index: int) -> dict[str, object]:
    iid = f"iid-{index:03d}"
    return {
        "iid": iid,
        "src_video": f"{iid}/source.mp4",
        "tgt_video": f"{iid}/target.mp4",
        "prompt": f"action {index}",
        "input_digest": _sha(f"input:{index}"),
        "r5_pilot_label": {
            "class": "positive",
            "action_signature": f"action-{index}",
        },
    }


def _side_fixture(
    *,
    index: int,
    side: str,
    video_sha256: str,
    hashes: list[str],
    dino_valid: bool,
) -> dict[str, object]:
    return {
        "status": "failed",
        "usable": False,
        "failure_stage": "temporal_teacher",
        "failure_reason": "unit_test_teacher_rejection",
        "failure_message": "unit test",
        "resolved_path": f"/data/iid-{index:03d}/{side}.mp4",
        "video_sha256": video_sha256,
        "decode": {
            "sampling_version": "uniform-32-decoded-frames-v1",
            "decoded_frames": 32,
            "source_frame_indices": list(range(0, 64, 2)),
            "source_fps": 24.0,
            "source_frame_count": 64,
            "source_size": [256, 256],
            "resized_size": [256, 256],
            "dino_frame_offsets": [0, 6, 12, 19, 25, 31],
            "dino_source_frame_indices": [0, 12, 24, 38, 50, 62],
            "perceptual_hashes": hashes,
        },
        "dino_valid": dino_valid,
    }


def _write_complete_preflight(
    root: Path,
    *,
    inconsistent_dino_rank: int | None = None,
    invalid_dino: tuple[int, str] | None = None,
) -> tuple[Path, Path]:
    manifest = root / "input.jsonl"
    input_rows = [_input_row(index) for index in range(FINAL_WORLD_SIZE)]
    with manifest.open("w", encoding="utf-8") as handle:
        for row in input_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    output = root / "preflight"
    shared_source_sha = _sha("shared-source-iid-000-iid-001")
    for rank, input_row in enumerate(input_rows):
        arrays = _empty_arrays(1)
        arrays["input_indices"][0] = rank
        arrays["positive"][0] = True
        side_records: dict[str, dict[str, object]] = {}
        for side_offset, side in enumerate(("source", "target")):
            invalid = invalid_dino == (rank, side)
            arrays[f"{side}_dino_valid"][0] = not invalid
            basis = rank * 2 + side_offset
            matrix = np.zeros((6, 768), dtype=np.float32)
            matrix[:, basis] = 1.0
            arrays[f"{side}_dino_cls"][0] = matrix
            hashes = [
                _sha(f"phash:{rank}:{side}:{frame}")[:16]
                for frame in range(6)
            ]
            arrays[f"{side}_perceptual_hashes"][0] = hashes
            video_sha256 = _sha(f"video:{rank}:{side}")
            if side == "source" and rank in {0, 1}:
                video_sha256 = shared_source_sha
            side_records[side] = _side_fixture(
                index=rank,
                side=side,
                video_sha256=video_sha256,
                hashes=hashes,
                dino_valid=not invalid,
            )
        output_row = {
            "schema_version": R7_ROW_SCHEMA,
            "input_index": rank,
            "shard_array_index": 0,
            "shard_rank": rank,
            "world_size": FINAL_WORLD_SIZE,
            "iid": input_row["iid"],
            "input_row_sha256": _object_digest(input_row),
            "input_digest": input_row["input_digest"],
            "prompt": input_row["prompt"],
            "label_type": "positive",
            "negative_type": None,
            "positive": True,
            "action_signature": input_row["r5_pilot_label"][
                "action_signature"
            ],
            "source": side_records["source"],
            "target": side_records["target"],
            "paired_usable": False,
        }
        revision = (
            "d" * 64
            if inconsistent_dino_rank == rank
            else "a" * 64
        )
        contract = {
            "schema_version": "motive-r7-preflight-extract-v2",
            "input_manifest": str(manifest.resolve()),
            "input_manifest_sha256": _file_digest(manifest),
            "data_root": "/data",
            "rank": rank,
            "world_size": FINAL_WORLD_SIZE,
            "partition": "input-index-modulo-world-size-v1",
            "device": f"cuda:{rank}",
            "video_sampling": {
                "version": "uniform-32-decoded-frames-v1",
                "frames": 32,
                "maximum_side": 384,
            },
            "tracker": {"checkpoint_sha256": "e" * 64},
            "temporal_teacher_config": {"version": "unit-test"},
            "dino": _dino_provenance(revision=revision),
            "seed": 260108828,
            "implementation": {"unit-test": "f" * 64},
        }
        _commit_shard(
            directory=rank_directory(
                output, rank, FINAL_WORLD_SIZE
            ),
            rows=[output_row],
            arrays=arrays,
            contract=contract,
            input_rows=FINAL_WORLD_SIZE,
        )
    finalize_shards(input_manifest=manifest, output_root=output)
    return output, manifest


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class R7VisualSplitIoTests(unittest.TestCase):
    def test_build_forces_seen_visual_component_and_resume_only_validates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight, _manifest = _write_complete_preflight(root)
            ledger = root / "prior.jsonl"
            ledger.write_text(
                json.dumps({"iid": "iid-000"}) + "\n",
                encoding="utf-8",
            )
            output = root / "visual-split"
            built = build_visual_split_artifacts(
                preflight_output_root=preflight,
                prior_iid_ledger=ledger,
                output_dir=output,
                config=R7VisualSplitConfig(
                    maximum_phash_hamming_fraction=0.0,
                    minimum_dino_cosine=1.0,
                ),
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    ASSIGNMENTS_NAME,
                    COMPONENTS_NAME,
                    SUMMARY_NAME,
                    DONE_NAME,
                },
            )
            by_iid = {
                row["iid"]: row for row in built["assignments"]
            }
            self.assertEqual(
                by_iid["iid-000"]["component_id"],
                by_iid["iid-001"]["component_id"],
            )
            for iid in ("iid-000", "iid-001"):
                self.assertEqual(by_iid[iid]["split"], "train")
                self.assertTrue(
                    by_iid[iid][
                        "forced_train_by_seen_component"
                    ]
                )
                self.assertFalse(
                    by_iid[iid]["evaluation_fresh"]
                )
            interpretation = built["summary"]["interpretation"]
            self.assertEqual(
                interpretation["p0_old_181_allowed_use"],
                "io_contract_test_only_not_fresh_evaluation",
            )
            self.assertFalse(
                interpretation["fresh_evaluation_authorized"]
            )
            before = {
                path.name: path.read_bytes()
                for path in output.iterdir()
            }
            resumed = build_visual_split_artifacts(
                preflight_output_root=preflight,
                prior_iid_ledger=ledger,
                output_dir=output,
                config=R7VisualSplitConfig(
                    maximum_phash_hamming_fraction=0.0,
                    minimum_dino_cosine=1.0,
                ),
                resume=True,
            )
            self.assertEqual(resumed["done"], built["done"])
            self.assertEqual(
                before,
                {
                    path.name: path.read_bytes()
                    for path in output.iterdir()
                },
            )
            with self.assertRaises(FileExistsError):
                build_visual_split_artifacts(
                    preflight_output_root=preflight,
                    prior_iid_ledger=ledger,
                    output_dir=output,
                )

    def test_cli_validate_and_hash_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight, _manifest = _write_complete_preflight(root)
            ledger = root / "prior.jsonl"
            ledger.write_text('"unmatched-old-iid"\n', encoding="utf-8")
            output = root / "split"
            build_visual_split_artifacts(
                preflight_output_root=preflight,
                prior_iid_ledger=ledger,
                output_dir=output,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "validate",
                        "--preflight-output-root",
                        str(preflight),
                        "--prior-iid-ledger",
                        str(ledger),
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["status"],
                "complete",
            )
            with (output / ASSIGNMENTS_NAME).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                validate_visual_split_artifacts(
                    preflight_output_root=preflight,
                    prior_iid_ledger=ledger,
                    output_dir=output,
                )

    def test_cross_rank_dino_provenance_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight, _manifest = _write_complete_preflight(
                root,
                inconsistent_dino_rank=7,
            )
            with self.assertRaisesRegex(
                ValueError,
                "DINO provenance contracts differ",
            ):
                load_preflight_visual_pairs(preflight)

    def test_missing_side_dino_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight, _manifest = _write_complete_preflight(
                root,
                invalid_dino=(3, "target"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "lacks committed DINO evidence",
            ):
                load_preflight_visual_pairs(preflight)

    def test_final_must_equal_hash_valid_shard_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight, _manifest = _write_complete_preflight(root)
            shard = rank_directory(preflight, 0, FINAL_WORLD_SIZE)
            archive_path = shard / "features.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                arrays = {
                    name: archive[name].copy()
                    for name in archive.files
                }
            arrays["source_dino_cls"][0] = 0.0
            arrays["source_dino_cls"][0, :, 100] = 1.0
            np.savez_compressed(archive_path, **arrays)

            shard_summary_path = shard / "summary.json"
            shard_summary = json.loads(
                shard_summary_path.read_text(encoding="utf-8")
            )
            changed = arrays["source_dino_cls"]
            shard_summary["array_contract"]["source_dino_cls"] = {
                "shape": list(changed.shape),
                "dtype": str(changed.dtype),
                "sha256": _array_digest(changed),
            }
            shard_summary["archive_sha256"] = _file_digest(
                archive_path
            )
            _write_json(shard_summary_path, shard_summary)

            shard_done_path = shard / "done.json"
            shard_done = json.loads(
                shard_done_path.read_text(encoding="utf-8")
            )
            shard_done["artifacts"]["archive"]["sha256"] = (
                _file_digest(archive_path)
            )
            shard_done["artifacts"]["summary"]["sha256"] = (
                _file_digest(shard_summary_path)
            )
            _write_json(shard_done_path, shard_done)

            final_summary_path = preflight / "final" / "summary.json"
            final_summary = json.loads(
                final_summary_path.read_text(encoding="utf-8")
            )
            final_summary["shard_done_sha256"][0] = _file_digest(
                shard_done_path
            )
            _write_json(final_summary_path, final_summary)
            final_done_path = preflight / "final" / "done.json"
            final_done = json.loads(
                final_done_path.read_text(encoding="utf-8")
            )
            final_done["artifacts"]["summary"]["sha256"] = (
                _file_digest(final_summary_path)
            )
            _write_json(final_done_path, final_done)

            with self.assertRaisesRegex(
                ValueError,
                "final array source_dino_cls differs from shard 0",
            ):
                load_preflight_visual_pairs(preflight)

    def test_prior_ledger_is_exact_and_partial_resume_is_never_repaired(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                '{"iid":"old"}\n"old"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_prior_iid_ledger(duplicate)

            preflight, _manifest = _write_complete_preflight(root)
            ledger = root / "prior.jsonl"
            ledger.write_text('"old"\n', encoding="utf-8")
            partial = root / "partial"
            partial.mkdir()
            (partial / ASSIGNMENTS_NAME).write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                FileNotFoundError,
                "validation-only",
            ):
                build_visual_split_artifacts(
                    preflight_output_root=preflight,
                    prior_iid_ledger=ledger,
                    output_dir=partial,
                    resume=True,
                )
            with self.assertRaises(FileExistsError):
                build_visual_split_artifacts(
                    preflight_output_root=preflight,
                    prior_iid_ledger=ledger,
                    output_dir=partial,
                )


if __name__ == "__main__":
    unittest.main()
