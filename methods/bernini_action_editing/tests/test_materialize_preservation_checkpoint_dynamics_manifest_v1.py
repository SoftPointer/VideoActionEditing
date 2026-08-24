from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import materialize_preservation_checkpoint_dynamics_manifest_v1 as helper  # noqa: E402


def _download_tree(root: Path) -> None:
    for rank_name in helper.RANKS:
        receipt = root / "training" / rank_name / "dataset-receipt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}", encoding="ascii")
    cell_root = root / "cells" / "dog"
    cell_root.mkdir(parents=True)
    (cell_root / "cell.json").write_text(
        json.dumps(
            {
                "schema_version": helper.CELL_SCHEMA_VERSION,
                "cell_id": "dog",
                "source_iid": "heldout-dog",
                "source_action_caption": "A grey dog stands in an autumn park.",
                "full_instruction": "The dog lowers its hips and sits.",
                "seed": 2026081601,
            }
        ),
        encoding="utf-8",
    )
    (cell_root / "source.mp4").write_bytes(b"source")
    for rank_name in helper.RANKS:
        for step_name in helper.STEPS:
            step_root = cell_root / rank_name / step_name
            step_root.mkdir(parents=True)
            (step_root / "video.mp4").write_bytes(b"video")
            (step_root / "inference-receipt.json").write_text(
                "{}", encoding="ascii"
            )
            if step_name != "step0":
                (step_root / "paired-native.mp4").write_bytes(b"native")
                (step_root / "training-receipt.json").write_text(
                    "{}", encoding="ascii"
                )


class CheckpointDynamicsManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        _download_tree(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_materializes_exact_builder_schema_and_relative_layout(self) -> None:
        output = self.root / "manifest.json"
        self.assertEqual(
            helper.write_manifest(media_root=self.root, output=output), output
        )
        manifest = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], helper.SCHEMA_VERSION)
        self.assertEqual(manifest["authority"], helper.AUTHORITY)
        self.assertEqual(manifest["ranks"]["rank8"]["adapter_rank"], 8)
        cell = manifest["cells"][0]
        self.assertEqual(cell["source_video"], "cells/dog/source.mp4")
        self.assertEqual(
            cell["full_instruction"], "The dog lowers its hips and sits."
        )
        self.assertEqual(
            cell["variants"]["rank2"]["step0"],
            {
                "video": "cells/dog/rank2/step0/video.mp4",
                "inference_receipt": (
                    "cells/dog/rank2/step0/inference-receipt.json"
                ),
            },
        )
        self.assertEqual(
            cell["variants"]["rank8"]["step20"]["training_receipt"],
            "cells/dog/rank8/step20/training-receipt.json",
        )
        self.assertEqual(
            cell["variants"]["rank8"]["step40"]["paired_native_video"],
            "cells/dog/rank8/step40/paired-native.mp4",
        )
        self.assertTrue(
            all(
                not Path(path).is_absolute()
                for path in (
                    cell["source_video"],
                    cell["variants"]["rank8"]["step40"]["video"],
                )
            )
        )

    def test_missing_checkpoint_artifact_fails_before_manifest(self) -> None:
        (self.root / "cells/dog/rank2/step20/video.mp4").unlink()
        output = self.root / "manifest.json"
        with self.assertRaisesRegex(
            helper.CheckpointDynamicsManifestError, "missing"
        ):
            helper.write_manifest(media_root=self.root, output=output)
        self.assertFalse(output.exists())

    def test_missing_or_linked_paired_native_fails(self) -> None:
        paired = self.root / "cells/dog/rank8/step40/paired-native.mp4"
        paired.unlink()
        with self.assertRaisesRegex(
            helper.CheckpointDynamicsManifestError, "missing"
        ):
            helper.materialize(media_root=self.root)
        paired.symlink_to(self.root / "cells/dog/rank8/step20/paired-native.mp4")
        with self.assertRaises(helper.CheckpointDynamicsManifestError):
            helper.materialize(media_root=self.root)

    def test_manifest_output_parent_must_be_plain_media_root(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(
            helper.CheckpointDynamicsManifestError, "plain media root"
        ):
            helper.write_manifest(
                media_root=self.root,
                output=nested / "manifest.json",
            )

    def test_ancestor_symlink_alias_cannot_publish_manifest(self) -> None:
        alias = self.root.parent / f"{self.root.name}-alias-parent"
        alias.symlink_to(self.root.parent, target_is_directory=True)
        try:
            aliased_root = alias / self.root.name
            with self.assertRaisesRegex(
                helper.CheckpointDynamicsManifestError, "plain media root"
            ):
                helper.write_manifest(
                    media_root=self.root,
                    output=aliased_root / "alias-manifest.json",
                )
        finally:
            alias.unlink(missing_ok=True)

    def test_cell_directory_and_declared_identity_must_match(self) -> None:
        path = self.root / "cells/dog/cell.json"
        cell = json.loads(path.read_text(encoding="utf-8"))
        cell["cell_id"] = "human"
        path.write_text(json.dumps(cell), encoding="utf-8")
        with self.assertRaisesRegex(
            helper.CheckpointDynamicsManifestError, "directory"
        ):
            helper.materialize(media_root=self.root)


if __name__ == "__main__":
    unittest.main()
