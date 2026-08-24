from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_saic_source8_hidden_quotient_v1 as materializer


def _rows(actor_family: str = "dog") -> list[dict]:
    rows = []
    for ordinal, split in enumerate(("fit", "fit", "confirmation", "confirmation")):
        iid = f"iid{ordinal}"
        source = {
            "iid": iid,
            "actor_family": actor_family,
            "analysis_split": split,
        }
        for branch in materializer.BRANCH_ORDER:
            rows.append(
                {
                    "source": dict(source),
                    "candidate": {
                        "candidate_id": materializer._candidate_id(
                            iid, branch, 100 + ordinal
                        ),
                        "branch": branch,
                    },
                }
            )
    return rows


class Source8HiddenMaterializerTest(unittest.TestCase):
    def test_spatial_binding_map_accepts_canonical_key_sorting(self) -> None:
        source_order = ["z-source", "a-source", "y-source", "b-source"]
        bindings = {}
        for iid, grid in zip(source_order, materializer.SOURCE8_PATCH_GRIDS[:4]):
            positions = grid[0] * grid[1]
            digests = materializer.SOURCE8_SKETCH_DIGESTS_BY_PATCH_POSITIONS[
                positions
            ]
            bindings[iid] = {
                "patch_grid_height_width": list(grid),
                "patch_positions": positions,
                "matrix_shape": [materializer.starc.SKETCH_COORDINATES, positions],
                "matrix_raw_bytes_sha256": digests[0],
                "matrix_value_sha256": digests[1],
                "critic_tensor_sha256": digests[2],
                "data_dependent": False,
                "full_support_no_mask_or_localizer": True,
            }
        canonical_order = dict(sorted(bindings.items()))
        self.assertEqual(
            materializer.validate_group_spatial_bindings(
                canonical_order, source_order=source_order
            ),
            canonical_order,
        )
        canonical_order["a-source"]["critic_tensor_sha256"] = "0" * 64
        with self.assertRaises(materializer.Source8HiddenMaterializationError):
            materializer.validate_group_spatial_bindings(
                canonical_order, source_order=source_order
            )

    def test_clean_authentication_receives_source8_geometry_registry(self) -> None:
        signature = inspect.signature(
            materializer.starc.verify_authenticated_native_clean_tensor_identity
        )
        self.assertEqual(
            signature.parameters["allowed_latent_shapes"].default,
            materializer.starc.CORE4_LATENT_SHAPES,
        )
        source = inspect.getsource(materializer.materialize_group)
        self.assertIn("allowed_latent_shapes=SOURCE8_LATENT_SHAPES", source)
        self.assertIn("allowed_patch_grids=SOURCE8_PATCH_GRIDS", source)

    def test_source8_geometry_whitelist_and_sketch_digests_are_closed(self) -> None:
        observed_positions = []
        for shape in materializer.SOURCE8_LATENT_SHAPES:
            geometry = materializer.source8_latent_geometry(shape)
            observed_positions.append(geometry[3])
            self.assertEqual(
                materializer.starc.spatial_sketch_digests(
                    patch_height=geometry[1],
                    patch_width=geometry[2],
                    allowed_patch_grids=materializer.SOURCE8_PATCH_GRIDS,
                    geometry_label="registered source8",
                ),
                materializer.SOURCE8_SKETCH_DIGESTS_BY_PATCH_POSITIONS[geometry[3]],
            )
        self.assertEqual(observed_positions, [930, 928, 902, 918, 925])
        with self.assertRaises(materializer.starc.STARCMaterializationError):
            materializer.source8_latent_geometry((1, 16, 21, 62, 60))

    def test_candidate_id_and_branch_instruction_are_deterministic(self) -> None:
        source = {
            "forward_instruction": "forward",
            "noop_instruction": "noop",
            "inverse_instruction": "reverse",
        }
        self.assertEqual(
            materializer._candidate_id("abc", "forward", 7),
            "saic-abc-forward-s7",
        )
        self.assertEqual(
            [
                materializer._source_branch_instruction(source, branch)
                for branch in materializer.BRANCH_ORDER
            ],
            ["forward", "noop", "reverse"],
        )

    def test_family_group_closes_two_fit_two_confirmation(self) -> None:
        group = materializer.selected_group(_rows(), actor_family="dog")
        self.assertEqual(len(group), materializer.ARMS_PER_GROUP)
        self.assertEqual(
            [row["candidate"]["branch"] for row in group],
            list(materializer.BRANCH_ORDER) * materializer.SOURCES_PER_GROUP,
        )

    def test_family_group_rejects_branch_reordering(self) -> None:
        rows = _rows()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(materializer.Source8HiddenMaterializationError):
            materializer.selected_group(rows, actor_family="dog")

    def test_family_group_rejects_missing_confirmation(self) -> None:
        rows = _rows()
        for row in rows:
            row["source"]["analysis_split"] = "fit"
        with self.assertRaises(materializer.Source8HiddenMaterializationError):
            materializer.selected_group(rows, actor_family="dog")

    def test_parser_requires_explicit_authority_acknowledgements(self) -> None:
        parser = materializer.build_parser()
        args = parser.parse_args(
            [
                "materialize-group",
                "--actor-family",
                "human",
                "--source-manifest",
                "/x/source.json",
                "--attempts-root",
                "/x/attempts",
                "--bernini-root",
                "/x/bernini",
                "--veomni-root",
                "/x/veomni",
                "--checkpoint",
                "/x/checkpoint",
                "--checkpoint-content-manifest",
                "/x/checkpoint.json",
                "--output-root",
                "/x/output",
                "--expected-bernini-commit",
                "0" * 40,
                "--expected-veomni-commit",
                "1" * 40,
                "--method-source-revision",
                "2" * 40,
                "--method-source-archive-sha256",
                "3" * 64,
                "--expected-materializer-source-sha256",
                "4" * 64,
                "--expected-starc-source-sha256",
                "5" * 64,
                "--expected-source-contract-sha256",
                "6" * 64,
                "--expected-generation-contract-sha256",
                "7" * 64,
            ]
        )
        self.assertIsInstance(args, argparse.Namespace)
        self.assertFalse(args.ack_hidden_diagnostic_only)
        self.assertFalse(args.ack_no_generated_media_editor_use)
        self.assertFalse(args.ack_no_optimizer_or_editor_update)


if __name__ == "__main__":
    unittest.main()
