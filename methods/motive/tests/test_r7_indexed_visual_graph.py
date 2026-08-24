from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import random
import unittest

from motive.r7_indexed_visual_graph import (
    DINO_HARD_THRESHOLD,
    MAXIMUM_DHASH_HAMMING,
    R7IndexedDinoEdge,
    R7IndexedVisualAsset,
    R7IndexedVisualGraphConfig,
    R7IndexedVisualPair,
    assignments_by_iid,
    build_r7_indexed_visual_graph,
    verify_r7_indexed_visual_graph,
)


Node = tuple[str, str]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _hex(value: int) -> str:
    return f"{value & ((1 << 64) - 1):016x}"


def _asset(
    label: str,
    hashes: list[int] | tuple[int, ...],
    *,
    video_sha256: str | None = None,
) -> R7IndexedVisualAsset:
    if len(hashes) == 1:
        hashes = list(hashes) * 6
    return R7IndexedVisualAsset.create(
        video_sha256=video_sha256 or _sha(f"video:{label}"),
        dhashes=[_hex(value) for value in hashes],
    )


def _pair(
    iid: str,
    source_hashes: list[int] | tuple[int, ...],
    target_hashes: list[int] | tuple[int, ...],
    *,
    source_sha: str | None = None,
    target_sha: str | None = None,
) -> R7IndexedVisualPair:
    return R7IndexedVisualPair.create(
        iid=iid,
        source=_asset(
            f"{iid}:source",
            source_hashes,
            video_sha256=source_sha,
        ),
        target=_asset(
            f"{iid}:target",
            target_hashes,
            video_sha256=target_sha,
        ),
    )


def _edge(
    left: Node,
    right: Node,
    cosine: float,
) -> R7IndexedDinoEdge:
    return R7IndexedDinoEdge.create(
        left_iid=left[0],
        left_role=left[1],
        right_iid=right[0],
        right_role=right[1],
        cosine=cosine,
    )


class _BruteDsu:
    def __init__(self, nodes: list[Node]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: Node) -> Node:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: Node, right: Node) -> None:
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left


def _brute_components(
    pairs: list[R7IndexedVisualPair],
    dino_edges: list[R7IndexedDinoEdge],
    *,
    minimum_dino_cosine: float,
) -> set[frozenset[Node]]:
    assets = {
        (pair.iid, role): getattr(pair, role)
        for pair in pairs
        for role in ("source", "target")
    }
    nodes = sorted(assets)
    dsu = _BruteDsu(nodes)
    for pair in pairs:
        dsu.union(
            (pair.iid, "source"),
            (pair.iid, "target"),
        )
    for left, right in itertools.combinations(nodes, 2):
        left_asset = assets[left]
        right_asset = assets[right]
        if left_asset.video_sha256 == right_asset.video_sha256:
            dsu.union(left, right)
        minimum = min(
            (int(a, 16) ^ int(b, 16)).bit_count()
            for a in left_asset.dhashes
            for b in right_asset.dhashes
        )
        if minimum <= MAXIMUM_DHASH_HAMMING:
            dsu.union(left, right)
    for edge in dino_edges:
        if edge.cosine >= minimum_dino_cosine:
            dsu.union(edge.left_node, edge.right_node)
    groups: dict[Node, set[Node]] = {}
    for node in nodes:
        groups.setdefault(dsu.find(node), set()).add(node)
    return {frozenset(members) for members in groups.values()}


def _hard_relations(
    pairs: list[R7IndexedVisualPair],
    dino_edges: list[R7IndexedDinoEdge],
    *,
    minimum_dino_cosine: float,
) -> list[tuple[str, Node, Node]]:
    """Enumerate every brute-force hard relation for completeness checks."""

    assets = {
        (pair.iid, role): getattr(pair, role)
        for pair in pairs
        for role in ("source", "target")
    }
    relations: list[tuple[str, Node, Node]] = [
        (
            "paired_sample",
            (pair.iid, "source"),
            (pair.iid, "target"),
        )
        for pair in pairs
    ]
    for left, right in itertools.combinations(sorted(assets), 2):
        left_asset = assets[left]
        right_asset = assets[right]
        if left_asset.video_sha256 == right_asset.video_sha256:
            relations.append(("exact_sha256", left, right))
        if any(
            (int(a, 16) ^ int(b, 16)).bit_count()
            <= MAXIMUM_DHASH_HAMMING
            for a in left_asset.dhashes
            for b in right_asset.dhashes
        ):
            relations.append(("dhash_hamming", left, right))
    relations.extend(
        ("dino_cosine", edge.left_node, edge.right_node)
        for edge in dino_edges
        if edge.cosine >= minimum_dino_cosine
    )
    return relations


def _result_components(result: object) -> set[frozenset[Node]]:
    return {
        frozenset(component.member_assets)
        for component in result.components
    }


class R7IndexedVisualGraphTests(unittest.TestCase):
    def test_default_dino_threshold_is_locked_to_production_hard_edge(
        self,
    ) -> None:
        config = R7IndexedVisualGraphConfig()
        self.assertEqual(DINO_HARD_THRESHOLD, 0.96)
        self.assertEqual(config.minimum_dino_cosine, 0.96)
        self.assertEqual(
            config.thresholds_dict()["minimum_dino_cosine"],
            0.96,
        )

    def test_random_indexed_partition_matches_full_bruteforce(self) -> None:
        """Many small graphs prove exact BK-tree component semantics."""

        for trial in range(24):
            rng = random.Random(9000 + trial)
            count = 3 + trial % 14
            pairs: list[R7IndexedVisualPair] = []
            previous_source_first: int | None = None
            shared_sha = _sha(f"shared:{trial}")
            for index in range(count):
                source = [rng.getrandbits(64) for _ in range(6)]
                target = [rng.getrandbits(64) for _ in range(6)]
                # Inject boundary and transitive near-hash relations without
                # making every random trial one trivial component.
                if index % 4 == 1 and previous_source_first is not None:
                    source[0] = previous_source_first ^ ((1 << 6) - 1)
                source_sha = (
                    shared_sha if index in {2, 7} and count > 7 else None
                )
                pairs.append(
                    _pair(
                        f"iid-{index:03d}",
                        source,
                        target,
                        source_sha=source_sha,
                    )
                )
                previous_source_first = source[0]

            nodes = [
                (pair.iid, role)
                for pair in pairs
                for role in ("source", "target")
            ]
            dino_edges: list[R7IndexedDinoEdge] = []
            used: set[tuple[Node, Node]] = set()
            for edge_index in range(min(5, len(nodes) // 2)):
                left, right = rng.sample(nodes, 2)
                left, right = sorted((left, right))
                if (left, right) in used:
                    continue
                used.add((left, right))
                score = 0.97 if edge_index % 2 == 0 else 0.90
                dino_edges.append(_edge(left, right, score))

            config = R7IndexedVisualGraphConfig(
                data_seed=trial,
                minimum_dino_cosine=0.95,
            )
            indexed = build_r7_indexed_visual_graph(
                pairs,
                dino_edges=dino_edges,
                config=config,
            )
            brute = _brute_components(
                pairs,
                dino_edges,
                minimum_dino_cosine=0.95,
            )
            with self.subTest(trial=trial, count=count):
                self.assertEqual(_result_components(indexed), brute)
                component_by_iid = {
                    assignment.iid: assignment.component_id
                    for assignment in indexed.assignments
                }
                # The retained graph is only a spanning forest, so verify
                # completeness semantically: every pair/exact/dHash/DINO
                # hard edge from the full graph has co-component endpoints.
                for relation, left, right in _hard_relations(
                    pairs,
                    dino_edges,
                    minimum_dino_cosine=0.95,
                ):
                    with self.subTest(
                        trial=trial,
                        relation=relation,
                        left=left,
                        right=right,
                    ):
                        self.assertEqual(
                            component_by_iid[left[0]],
                            component_by_iid[right[0]],
                        )
                self.assertEqual(
                    len(indexed.spanning_edges),
                    2 * count - len(brute),
                )
                self.assertEqual(
                    indexed.statistics.dhash_index_queries,
                    12 * count,
                )
                self.assertTrue(
                    verify_r7_indexed_visual_graph(
                        indexed,
                        pairs,
                        dino_edges=dino_edges,
                        config=config,
                    )
                )

    def test_transitive_relations_anchor_and_seen_force_whole_component(
        self,
    ) -> None:
        shared_sha = _sha("b-target=c-source")
        pairs = [
            _pair(
                "a",
                [0x0000000000000000],
                [0x1111111111111111],
            ),
            # a.target -> b.source is exactly Hamming six.
            _pair(
                "b",
                [0x111111111111117E],
                [0x2222222222222222],
                target_sha=shared_sha,
            ),
            _pair(
                "c",
                [0x3333333333333333],
                [0x4444444444444444],
                source_sha=shared_sha,
            ),
            _pair(
                "d",
                [0x5555555555555555],
                [0x6666666666666666],
            ),
            _pair(
                "isolated",
                [0xAAAAAAAAAAAAAAAA],
                [0xBBBBBBBBBBBBBBBB],
            ),
        ]
        dino = [_edge(("c", "target"), ("d", "source"), 0.98)]
        result = build_r7_indexed_visual_graph(
            pairs,
            dino_edges=dino,
            anchor_iids=["a"],
            previously_seen_iids=["d", "not-in-current-input"],
        )
        by_iid = assignments_by_iid(result)
        connected = {by_iid[iid].component_id for iid in "abcd"}
        self.assertEqual(len(connected), 1)
        for iid in "abcd":
            self.assertEqual(by_iid[iid].split, "train")
            self.assertFalse(by_iid[iid].fresh)
            self.assertTrue(by_iid[iid].forced_train)
        self.assertTrue(by_iid["a"].forced_by_anchor)
        self.assertTrue(by_iid["d"].forced_by_previously_seen)
        self.assertTrue(by_iid["isolated"].fresh)

        component = next(
            value
            for value in result.components
            if value.component_id == by_iid["a"].component_id
        )
        self.assertEqual(component.anchor_iids, ("a",))
        self.assertEqual(component.previously_seen_iids, ("d",))
        self.assertEqual(
            {edge.relation for edge in result.spanning_edges},
            {
                "paired_sample",
                "dhash_hamming",
                "exact_sha256",
                "dino_cosine",
            },
        )

    def test_result_and_digests_are_invariant_to_all_input_ordering(
        self,
    ) -> None:
        pairs = [
            _pair("z", [0], [0xFFFFFFFFFFFFFFFF]),
            _pair("a", [0x1234], [0x5678]),
            _pair("m", [0xABCDEF], [0x1111222233334444]),
        ]
        dino = [
            _edge(("a", "target"), ("m", "source"), 0.98),
            _edge(("m", "target"), ("z", "target"), 0.97),
        ]
        kwargs = {
            "config": R7IndexedVisualGraphConfig(data_seed=17),
            "anchor_iids": ["z", "a"],
            "previously_seen_iids": ["old-z", "m"],
        }
        forward = build_r7_indexed_visual_graph(
            pairs,
            dino_edges=dino,
            **kwargs,
        )
        reverse = build_r7_indexed_visual_graph(
            list(reversed(pairs)),
            dino_edges=list(reversed(dino)),
            anchor_iids=list(reversed(kwargs["anchor_iids"])),
            previously_seen_iids=list(
                reversed(kwargs["previously_seen_iids"])
            ),
            config=kwargs["config"],
        )
        self.assertEqual(forward.to_dict(), reverse.to_dict())

        for name in (
            "input_digest",
            "threshold_digest",
            "edge_digest",
            "component_digest",
            "assignment_digest",
            "provenance_digest",
        ):
            value = getattr(forward.provenance, name)
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_threshold_filter_digests_and_giant_component_warning(
        self,
    ) -> None:
        pairs = [
            _pair("a", [0], [0x1111111111111111]),
            _pair("b", [0xAAAAAAAAAAAAAAAA], [0xFFFFFFFFFFFFFFFF]),
        ]
        dino = [_edge(("a", "source"), ("b", "source"), 0.94)]
        strict = build_r7_indexed_visual_graph(
            pairs,
            dino_edges=dino,
            config=R7IndexedVisualGraphConfig(
                minimum_dino_cosine=0.95
            ),
        )
        loose = build_r7_indexed_visual_graph(
            pairs,
            dino_edges=dino,
            config=R7IndexedVisualGraphConfig(
                minimum_dino_cosine=0.90
            ),
        )
        self.assertEqual(strict.statistics.component_count, 2)
        self.assertEqual(strict.statistics.dino_below_threshold_count, 1)
        self.assertEqual(loose.statistics.component_count, 1)
        self.assertEqual(loose.statistics.dino_above_threshold_count, 1)
        self.assertNotEqual(
            strict.provenance.threshold_digest,
            loose.provenance.threshold_digest,
        )
        self.assertNotEqual(
            strict.provenance.edge_digest,
            loose.provenance.edge_digest,
        )
        self.assertTrue(loose.statistics.giant_component_warning)
        self.assertEqual(loose.statistics.largest_component_fraction, 1.0)

    def test_tampering_and_illegal_dino_edges_fail_closed(self) -> None:
        pairs = [
            _pair("a", [0], [0x1111111111111111]),
            _pair("b", [0xAAAAAAAAAAAAAAAA], [0xFFFFFFFFFFFFFFFF]),
        ]
        valid = _edge(("a", "source"), ("b", "source"), 0.98)
        result = build_r7_indexed_visual_graph(
            pairs,
            dino_edges=[valid],
        )

        forged_provenance = dataclasses.replace(
            result.provenance,
            dino_input_edges_digest="0" * 64,
        )
        forged_result = dataclasses.replace(
            result,
            provenance=forged_provenance,
        )
        with self.assertRaisesRegex(ValueError, "recomputation"):
            verify_r7_indexed_visual_graph(
                forged_result,
                pairs,
                dino_edges=[valid],
            )

        illegal = [
            R7IndexedDinoEdge(
                "missing", "source", "b", "source", 0.98
            ),
            R7IndexedDinoEdge("a", "bad-role", "b", "source", 0.98),
            R7IndexedDinoEdge("a", "source", "a", "source", 0.98),
            R7IndexedDinoEdge("a", "source", "b", "source", math.nan),
            R7IndexedDinoEdge("a", "source", "b", "source", 1.01),
        ]
        for edge in illegal:
            with self.subTest(edge=edge):
                with self.assertRaises(ValueError):
                    build_r7_indexed_visual_graph(
                        pairs,
                        dino_edges=[edge],
                    )

        reverse_duplicate = R7IndexedDinoEdge(
            "b", "source", "a", "source", 0.98
        )
        with self.assertRaisesRegex(ValueError, "unordered asset pair"):
            build_r7_indexed_visual_graph(
                pairs,
                dino_edges=[valid, reverse_duplicate],
            )

    def test_asset_pair_and_fixed_policy_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            R7IndexedVisualAsset.create(
                video_sha256=_sha("short"),
                dhashes=["0" * 16] * 5,
            )
        with self.assertRaisesRegex(ValueError, "lowercase 64-bit"):
            R7IndexedVisualAsset.create(
                video_sha256=_sha("upper"),
                dhashes=["A" * 16] * 6,
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            R7IndexedVisualAsset.create(
                video_sha256="not-a-sha",
                dhashes=["0" * 16] * 6,
            )
        with self.assertRaisesRegex(ValueError, "maximum_dhash_hamming=6"):
            R7IndexedVisualGraphConfig(
                maximum_dhash_hamming=5
            ).validate()
        with self.assertRaisesRegex(ValueError, "80/10/10"):
            R7IndexedVisualGraphConfig(
                train_fraction=0.7
            ).validate()

        duplicate = _pair("same", [0], [1])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_r7_indexed_visual_graph([duplicate, duplicate])


if __name__ == "__main__":
    unittest.main()
