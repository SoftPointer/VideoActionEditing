from __future__ import annotations

import dataclasses
import hashlib
import unittest

import numpy as np

from motive.r7_visual_split import (
    R7DinoProvenance,
    R7VisualAsset,
    R7VisualAssignment,
    R7VisualPair,
    R7VisualSplitConfig,
    assignments_by_iid,
    audit_r7_visual_split,
    build_r7_visual_split,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(*, dimension: int = 4) -> R7DinoProvenance:
    return R7DinoProvenance(
        encoder_id="facebook/dinov2-large",
        encoder_revision="a" * 40,
        weights_sha256="b" * 64,
        frame_sampling_version="uniform-two-frame-v1",
        preprocessing_version="dinov2-518-center-crop-v1",
        pooling="cls-token",
        embedding_dim=dimension,
    )


def _asset(
    name: str,
    *,
    phash: str,
    vectors: list[list[float]],
    video_digest: str | None = None,
) -> R7VisualAsset:
    return R7VisualAsset.create(
        video_sha256=video_digest or _sha(name),
        frame_indices=[0, 8],
        perceptual_hashes=[phash, phash],
        dino_embeddings=np.asarray(vectors, dtype=np.float32),
    )


def _pair(
    iid: str,
    source: R7VisualAsset,
    target: R7VisualAsset,
) -> R7VisualPair:
    return R7VisualPair.create(iid=iid, source=source, target=target)


def _config(
    *,
    phash: float = 0.01,
    dino: float = 0.9999,
    seed: int = 17,
) -> R7VisualSplitConfig:
    return R7VisualSplitConfig(
        data_seed=seed,
        train_fraction=0.6,
        validation_fraction=0.2,
        maximum_phash_hamming_fraction=phash,
        minimum_dino_cosine=dino,
    )


class R7VisualAssetTests(unittest.TestCase):
    def test_asset_contract_is_fail_closed_and_frozen(self) -> None:
        asset = _asset(
            "valid",
            phash="0000000000000000",
            vectors=[[1, 0, 0, 0], [0.99, 0.01, 0, 0]],
        )
        self.assertFalse(asset.dino_embeddings.flags.writeable)
        self.assertEqual(asset.dino_embeddings.dtype, np.float32)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            R7VisualAsset.create(
                video_sha256=_sha("bad-order"),
                frame_indices=[2, 2],
                perceptual_hashes=["00", "ff"],
                dino_embeddings=np.ones((2, 4), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "floating dtype"):
            R7VisualAsset.create(
                video_sha256=_sha("integer"),
                frame_indices=[0],
                perceptual_hashes=["00"],
                dino_embeddings=np.ones((1, 4), dtype=np.int64),
            )
        with self.assertRaisesRegex(ValueError, "zero frame vector"):
            R7VisualAsset.create(
                video_sha256=_sha("zero"),
                frame_indices=[0],
                perceptual_hashes=["00"],
                dino_embeddings=np.zeros((1, 4), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            R7VisualAsset.create(
                video_sha256=_sha("unaligned"),
                frame_indices=[0, 1],
                perceptual_hashes=["00"],
                dino_embeddings=np.ones((2, 4), dtype=np.float32),
            )

    def test_provenance_and_config_reject_mutable_or_ambiguous_contracts(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "immutable"):
            dataclasses.replace(
                _provenance(), encoder_revision="main"
            ).validate()
        with self.assertRaisesRegex(ValueError, "frozen"):
            dataclasses.replace(
                _provenance(), frozen_encoder=False
            ).validate()
        with self.assertRaisesRegex(ValueError, "positive test"):
            R7VisualSplitConfig(
                train_fraction=0.8,
                validation_fraction=0.2,
            ).validate()
        with self.assertRaisesRegex(ValueError, r"\(0,1\]"):
            dataclasses.replace(
                _config(), minimum_dino_cosine=0.0
            ).validate()


class R7VisualSplitTests(unittest.TestCase):
    def _unrelated_pairs(self) -> list[R7VisualPair]:
        return [
            _pair(
                "iid-a",
                _asset(
                    "a-source",
                    phash="0000000000000000",
                    vectors=[[1, 0, 0, 0], [1, 0.01, 0, 0]],
                ),
                _asset(
                    "a-target",
                    phash="ffffffffffffffff",
                    vectors=[[0, 1, 0, 0], [0.01, 1, 0, 0]],
                ),
            ),
            _pair(
                "iid-b",
                _asset(
                    "b-source",
                    phash="aaaaaaaaaaaaaaaa",
                    vectors=[[0, 0, 1, 0], [0, 0.01, 1, 0]],
                ),
                _asset(
                    "b-target",
                    phash="5555555555555555",
                    vectors=[[0, 0, 0, 1], [0.01, 0, 0, 1]],
                ),
            ),
        ]

    def test_source_and_target_are_always_one_component(self) -> None:
        pair = self._unrelated_pairs()[0]
        result = build_r7_visual_split(
            [pair],
            config=_config(),
            dino_provenance=_provenance(),
            previously_seen_iids=[],
        )
        self.assertTrue(result.audit.passed)
        self.assertEqual(len(result.components), 1)
        self.assertEqual(
            result.components[0].member_nodes,
            (("iid-a", "source"), ("iid-a", "target")),
        )
        self.assertIn(
            "paired_sample",
            {edge.relation for edge in result.edges},
        )

    def test_exact_phash_and_dino_edges_join_transitively(self) -> None:
        common_digest = _sha("shared-video")
        pair_a = _pair(
            "a",
            _asset(
                "a-s",
                phash="0000000000000000",
                vectors=[[1, 0, 0, 0], [0.98, 0.02, 0, 0]],
            ),
            _asset(
                "a-t",
                phash="1111111111111111",
                vectors=[[0, 1, 0, 0], [0.02, 0.98, 0, 0]],
                video_digest=common_digest,
            ),
        )
        pair_b = _pair(
            "b",
            _asset(
                "b-s",
                phash="3333333333333333",
                vectors=[[0, 0, 1, 0], [0, 0.02, 0.98, 0]],
                video_digest=common_digest,
            ),
            _asset(
                "b-t",
                phash="aaaaaaaaaaaaaaaa",
                vectors=[[0, 0, 0, 1], [0.02, 0, 0, 0.98]],
            ),
        )
        # One-bit pHash distance joins b -> c.
        pair_c = _pair(
            "c",
            _asset(
                "c-s",
                phash="aaaaaaaaaaaaaaab",
                vectors=[[0.7, 0.7, 0, 0], [0.71, 0.69, 0, 0]],
            ),
            _asset(
                "c-t",
                phash="cccccccccccccccc",
                vectors=[[-1, 0, 0, 0], [-0.98, 0.02, 0, 0]],
            ),
        )
        # A single matching frame is sufficient for the documented DINO edge.
        pair_d = _pair(
            "d",
            _asset(
                "d-s",
                phash="f0f0f0f0f0f0f0f0",
                vectors=[[-0.7, -0.7, 0, 0], [-0.71, -0.69, 0, 0]],
            ),
            _asset(
                "d-t",
                phash="0f0f0f0f0f0f0f0f",
                vectors=[
                    [-0.2, 0.4, 0.8, 0.1],
                    [-0.98, 0.02, 0, 0],
                ],
            ),
        )
        result = build_r7_visual_split(
            [pair_a, pair_b, pair_c, pair_d],
            config=_config(phash=1 / 64, dino=0.999),
            dino_provenance=_provenance(),
            previously_seen_iids=[],
        )
        relations = {edge.relation for edge in result.edges}
        self.assertTrue(
            {"paired_sample", "exact_digest", "phash", "dino_cosine"}
            <= relations
        )
        self.assertEqual(len(result.components), 1)
        self.assertEqual(
            result.components[0].member_iids,
            ("a", "b", "c", "d"),
        )

    def test_result_is_invariant_to_input_order(self) -> None:
        pairs = self._unrelated_pairs()
        kwargs = {
            "config": _config(),
            "dino_provenance": _provenance(),
            "previously_seen_iids": ["old-z"],
        }
        forward = build_r7_visual_split(pairs, **kwargs)
        reverse = build_r7_visual_split(list(reversed(pairs)), **kwargs)
        self.assertEqual(forward.to_dict(), reverse.to_dict())

    def test_seen_iid_quarantines_entire_visual_component(self) -> None:
        pairs = self._unrelated_pairs()
        # Make iid-b source a DINO duplicate of iid-a target while retaining a
        # distinct video digest and pHash.
        pairs[1] = _pair(
            "iid-b",
            _asset(
                "b-source-replaced",
                phash="aaaaaaaaaaaaaaaa",
                vectors=[[0, 1, 0, 0], [0.01, 1, 0, 0]],
            ),
            pairs[1].target,
        )
        result = build_r7_visual_split(
            pairs,
            config=_config(dino=0.9999),
            dino_provenance=_provenance(),
            previously_seen_iids=["unmatched-old", "iid-a"],
        )
        by_iid = assignments_by_iid(result)
        for iid in ("iid-a", "iid-b"):
            self.assertEqual(by_iid[iid].split, "train")
            self.assertTrue(
                by_iid[iid].forced_train_by_seen_component
            )
            self.assertFalse(by_iid[iid].evaluation_fresh)
        self.assertEqual(
            result.provenance.matched_prior_seen_iids, ("iid-a",)
        )
        self.assertEqual(result.provenance.prior_seen_iid_count, 2)

    def test_cross_split_visual_relation_fails_closed(self) -> None:
        pairs = self._unrelated_pairs()
        pairs[1] = _pair(
            "iid-b",
            _asset(
                "b-source-replaced",
                phash="aaaaaaaaaaaaaaaa",
                vectors=[[0, 1, 0, 0], [0.01, 1, 0, 0]],
            ),
            pairs[1].target,
        )
        config = _config(dino=0.9999)
        result = build_r7_visual_split(
            pairs,
            config=config,
            dino_provenance=_provenance(),
            previously_seen_iids=[],
        )
        broken = list(result.assignments)
        original = broken[1]
        broken[1] = dataclasses.replace(
            original,
            split=(
                "test" if original.split != "test" else "validation"
            ),
        )
        diagnostic = audit_r7_visual_split(
            pairs,
            broken,
            config=config,
            dino_provenance=_provenance(),
            previously_seen_iids=[],
            raise_on_failure=False,
        )
        self.assertFalse(diagnostic.passed)
        self.assertTrue(diagnostic.cross_split_component_ids)
        self.assertTrue(diagnostic.cross_split_relation_edges)
        with self.assertRaisesRegex(
            ValueError, "R7 visual split audit failed"
        ):
            audit_r7_visual_split(
                pairs,
                broken,
                config=config,
                dino_provenance=_provenance(),
                previously_seen_iids=[],
            )

    def test_whole_component_reassignment_still_fails_stable_policy(
        self,
    ) -> None:
        pair = self._unrelated_pairs()[0]
        config = _config()
        result = build_r7_visual_split(
            [pair],
            config=config,
            dino_provenance=_provenance(),
            previously_seen_iids=[],
        )
        assignment = result.assignments[0]
        tampered = dataclasses.replace(
            assignment,
            split=(
                "validation"
                if assignment.split != "validation"
                else "test"
            ),
        )
        audit = audit_r7_visual_split(
            [pair],
            [tampered],
            config=config,
            dino_provenance=_provenance(),
            previously_seen_iids=[],
            raise_on_failure=False,
        )
        self.assertFalse(audit.passed)
        self.assertEqual(
            audit.stable_split_mismatches, ("iid-a",)
        )
        self.assertFalse(audit.cross_split_relation_edges)

    def test_seen_iid_cannot_be_marked_as_evaluation(self) -> None:
        pair = self._unrelated_pairs()[0]
        config = _config()
        result = build_r7_visual_split(
            [pair],
            config=config,
            dino_provenance=_provenance(),
            previously_seen_iids=["iid-a"],
        )
        bad = dataclasses.replace(
            result.assignments[0],
            split="test",
            evaluation_fresh=True,
            forced_train_by_seen_component=False,
        )
        audit = audit_r7_visual_split(
            [pair],
            [bad],
            config=config,
            dino_provenance=_provenance(),
            previously_seen_iids=["iid-a"],
            raise_on_failure=False,
        )
        self.assertFalse(audit.passed)
        self.assertEqual(
            audit.seen_component_evaluation_iids, ("iid-a",)
        )
        self.assertEqual(
            audit.assignment_component_mismatches, ("iid-a",)
        )

    def test_missing_duplicate_and_incompatible_features_fail_closed(
        self,
    ) -> None:
        pairs = self._unrelated_pairs()
        with self.assertRaisesRegex(ValueError, "unique"):
            build_r7_visual_split(
                [pairs[0], dataclasses.replace(pairs[1], iid="iid-a")],
                config=_config(),
                dino_provenance=_provenance(),
                previously_seen_iids=[],
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            audit_r7_visual_split(
                pairs,
                [],
                config=_config(),
                dino_provenance=_provenance(),
                previously_seen_iids=[],
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_r7_visual_split(
                pairs,
                config=_config(),
                dino_provenance=_provenance(),
                previously_seen_iids=["old", "old"],
            )
        with self.assertRaisesRegex(ValueError, "violates provenance"):
            build_r7_visual_split(
                pairs,
                config=_config(),
                dino_provenance=_provenance(dimension=5),
                previously_seen_iids=[],
            )

    def test_provenance_binds_inputs_thresholds_and_seen_ledger(self) -> None:
        pairs = self._unrelated_pairs()
        base = build_r7_visual_split(
            pairs,
            config=_config(dino=0.9999),
            dino_provenance=_provenance(),
            previously_seen_iids=[],
        )
        threshold = build_r7_visual_split(
            pairs,
            config=_config(dino=0.9998),
            dino_provenance=_provenance(),
            previously_seen_iids=[],
        )
        ledger = build_r7_visual_split(
            pairs,
            config=_config(dino=0.9999),
            dino_provenance=_provenance(),
            previously_seen_iids=["not-in-current-graph"],
        )
        self.assertNotEqual(
            base.provenance.provenance_digest,
            threshold.provenance.provenance_digest,
        )
        self.assertNotEqual(
            base.provenance.prior_seen_iid_ledger_digest,
            ledger.provenance.prior_seen_iid_ledger_digest,
        )
        self.assertNotEqual(
            base.provenance.provenance_digest,
            ledger.provenance.provenance_digest,
        )


if __name__ == "__main__":
    unittest.main()
