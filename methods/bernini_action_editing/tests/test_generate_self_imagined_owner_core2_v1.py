from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import generate_self_imagined_owner_core2_v1 as owner  # noqa: E402
import self_imagined_motion_cotangent_v1 as contract  # noqa: E402


REGISTRY = METHOD_ROOT / "assets/self_imagined_motion_cotangent_core2_v1.json"


class SelfImaginedOwnerGenerationTests(unittest.TestCase):
    def test_registry_maps_to_source_free_native_t2v_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bernini = root / "bernini"
            veomni = root / "veomni"
            checkpoint = root / "checkpoint"
            for directory in (bernini, veomni, checkpoint):
                directory.mkdir()
            manifest = root / "manifest.sha256"
            manifest.write_text("manifest", encoding="ascii")
            source = root / "source.mp4"
            source.write_bytes(b"source")
            cell = SimpleNamespace(
                cell_id="dog",
                source_video=str(source),
                source_video_sha256=hashlib.sha256(b"source").hexdigest(),
                action_caption="A dog sits and holds the pose.",
                action_caption_utf8_sha256=hashlib.sha256(
                    b"A dog sits and holds the pose."
                ).hexdigest(),
                owner_generation_seed=2026081501,
            )
            args = SimpleNamespace(
                bernini_root=str(bernini),
                veomni_root=str(veomni),
                checkpoint=str(checkpoint),
                checkpoint_content_manifest=str(manifest),
                output_dir=str(root / "fresh-output"),
                method_source_revision="a" * 40,
                method_source_archive_sha256="b" * 64,
            )
            argv = owner.build_native_argv(args, cell)
            self.assertEqual(argv[argv.index("--arms") + 1], "t2v")
            self.assertEqual(argv[argv.index("--num-inference-steps") + 1], "40")
            self.assertEqual(argv[argv.index("--seed") + 1], "2026081501")
            for forbidden in (
                "--target-video",
                "--mask",
                "--flow",
                "--pose",
                "--track",
            ):
                self.assertNotIn(forbidden, argv)

    def test_actual_registry_has_two_fixed_owner_seeds(self) -> None:
        raw = REGISTRY.read_bytes()
        registry = contract.load_probe_registry(
            REGISTRY.resolve(), expected_file_sha256=hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(
            [registry.cell(cell_id).owner_generation_seed for cell_id in owner.CELL_IDS],
            [2026081501, 2026081504],
        )

    def test_master_keeps_semantic_authority_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry.json"
            registry.write_bytes(REGISTRY.read_bytes())
            registry_sha = owner.file_sha256(registry)
            for cell_id in owner.CELL_IDS:
                cell_root = root / cell_id
                cell_root.mkdir()
                media = cell_root / "owner.mp4"
                media.write_bytes((cell_id + "-media").encode("ascii"))
                artifact = {
                    "path": str(media.resolve()),
                    "sha256": owner.file_sha256(media),
                }
                unsigned = {
                    "schema_version": owner.SCHEMA_VERSION,
                    "cell_id": cell_id,
                    "registry_file_sha256": registry_sha,
                    "artifacts": {"mp4": artifact},
                    "runtime_topology": {
                        "world_size": 4,
                        "ulysses_size": 4,
                        "rocr_visible_devices": owner.VISIBLE_GPUS_BY_CELL[cell_id],
                    },
                    "owner_exact81_action_audit_status":
                    "pending_detached_full_video_review",
                    "owner_template_materialization_authorized": False,
                }
                receipt = {
                    **unsigned,
                    "receipt_digest": contract.object_sha256(unsigned),
                }
                (cell_root / owner.OWNER_RECEIPT_BASENAME).write_text(
                    json.dumps(receipt), encoding="ascii"
                )
            status = owner.audit_master(
                SimpleNamespace(
                    output_root=str(root),
                    registry=str(registry),
                    expected_registry_sha256=registry_sha,
                )
            )
            self.assertEqual(status, 0)
            master = json.loads(
                (root / owner.MASTER_RECEIPT_BASENAME).read_text(encoding="ascii")
            )
            self.assertFalse(master["semantic_action_audit_complete"])
            self.assertFalse(master["owner_template_materialization_authorized"])
            self.assertEqual(master["exact81_owner_count"], 2)
            self.assertTrue(master["all8_used"])


if __name__ == "__main__":
    unittest.main()
