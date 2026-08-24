from __future__ import annotations

import copy
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motive.r10_dynamic_dino_representation_search import (
    DINO_DELTA_DCT,
    DONE_NAME,
    OUTPUT_NAMES,
    TRACK_DELTA_ACCELERATION,
    _DinoBasis,
    _R10Example,
    _appearance_groups,
    _array_records,
    _build_r10_examples,
    _dct_basis,
    _make_inner_folds,
    _make_folds,
    _publish,
    _raw_block,
    _signed_dct,
    _transform_arrays,
    search_examples,
    validate_published_search,
)


def _unit_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def _example(
    iid: str,
    *,
    family: str,
    split: str,
    component: str,
    content_index: int,
    negative: bool = False,
    fresh: bool = True,
) -> _R10Example:
    rng = np.random.default_rng(7000 + content_index)
    dimension = 12
    frames = 6
    transitions = 9
    content = rng.normal(size=dimension)
    content /= np.linalg.norm(content)
    source_dino = np.repeat(content[None, :], frames, axis=0)
    phase = np.asarray([-1.0, -0.4, 0.3, 1.0, 0.2, -0.7])
    target_dino = source_dino.copy()
    if not negative:
        action_dimension = 0 if family == "walk" else 1
        target_dino[:, action_dimension] += 0.8 * phase
        target_dino[:, 2 + action_dimension] += 0.35 * phase**2
    target_dino = _unit_rows(target_dino)
    source_dino = _unit_rows(source_dino)

    source_track = np.zeros((transitions, 15), dtype=np.float64)
    target_track = np.zeros_like(source_track)
    envelope = np.asarray(
        [0.0, 0.1, 0.5, 1.0, 0.7, 0.2, -0.3, -0.1, 0.0]
    )
    if not negative:
        column = 0 if family == "walk" else 1
        target_track[:, column] = envelope
        target_track[:, 10:15] = np.abs(envelope)[:, None]
    reverse_source = -source_track[::-1]
    reverse_target = -target_track[::-1]
    return _R10Example(
        iid=iid,
        label_class="negative" if negative else "positive",
        family=family,
        original_split=split,
        component_id=component,
        fresh=fresh,
        sampling_weight=1.0,
        motion_energy=0.0 if negative else 0.6,
        source_dino=source_dino,
        target_dino=target_dino,
        source_track=source_track,
        target_track=target_track,
        reverse_source_track=reverse_source,
        reverse_target_track=reverse_target,
        pooled_target_dino=np.mean(target_dino, axis=0),
        target_endpoint=np.mean(target_track, axis=0),
        target_orderless=np.concatenate(
            (
                np.mean(target_track, axis=0),
                np.std(target_track, axis=0),
            )
        ),
        camera_nuisance=np.asarray(
            [0.0, 0.0, 1.0, 1.0],
            dtype=np.float64,
        ),
    )


def _cohort(*, corrupt_legacy_test: bool = False) -> list[_R10Example]:
    rows: list[_R10Example] = []
    content_index = 0
    for family in ("walk", "jump"):
        for index in range(21):
            split = "train" if index < 14 else "validation"
            rows.append(
                _example(
                    f"dev-{family}-{index}",
                    family=family,
                    split=split,
                    component=f"dev-component-{family}-{index}",
                    content_index=content_index,
                    fresh=index >= 2,
                )
            )
            content_index += 1
        for index in range(5):
            actual_family = (
                ("jump" if family == "walk" else "walk")
                if corrupt_legacy_test
                else family
            )
            rows.append(
                _example(
                    f"test-{family}-{index}",
                    family=family,
                    split="test",
                    component=f"test-component-{family}-{index}",
                    content_index=content_index,
                )
            )
            if corrupt_legacy_test:
                replacement = _example(
                    f"test-{family}-{index}",
                    family=actual_family,
                    split="test",
                    component=f"test-component-{family}-{index}",
                    content_index=content_index,
                )
                rows[-1] = _R10Example(
                    **{
                        **replacement.__dict__,
                        "family": family,
                    }
                )
            content_index += 1
    for index in range(18):
        split = "train" if index < 12 else "validation"
        family = "walk" if index % 2 == 0 else "jump"
        rows.append(
            _example(
                f"dev-negative-{index}",
                family=family,
                split=split,
                component=f"dev-negative-component-{index}",
                content_index=content_index,
                negative=True,
            )
        )
        content_index += 1
    for index in range(6):
        family = "walk" if index % 2 == 0 else "jump"
        rows.append(
            _example(
                f"test-negative-{index}",
                family=family,
                split="test",
                component=f"test-negative-component-{index}",
                content_index=content_index,
                negative=True,
            )
        )
        content_index += 1
    return rows


class TemporalDescriptorTests(unittest.TestCase):
    def test_dct_is_orthonormal_and_reverse_has_signed_parity(self) -> None:
        basis = _dct_basis(6)
        np.testing.assert_allclose(
            basis @ basis.T,
            np.eye(6),
            atol=1e-12,
        )
        sequence = np.asarray(
            [[-1.0], [-0.4], [0.2], [0.8], [0.3], [-0.7]]
        )
        forward = _signed_dct(
            sequence,
            coefficients=5,
            include_dc=False,
        )
        reverse = _signed_dct(
            sequence[::-1],
            coefficients=5,
            include_dc=False,
        )
        parity = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(reverse, parity * forward, atol=1e-12)

    def test_delta_controls_are_not_clean_noops(self) -> None:
        example = _example(
            "control-example",
            family="walk",
            split="train",
            component="component",
            content_index=3,
        )
        basis = _DinoBasis(mean=np.zeros(12), basis=np.eye(12))
        values = {}
        for block in (TRACK_DELTA_ACCELERATION, DINO_DELTA_DCT):
            values[block] = {
                control: _raw_block(
                    example,
                    block=block,
                    control=control,
                    seed=123,
                    dino_basis=basis,
                    dino_dimension=12,
                )
                for control in ("clean", "shuffle", "reverse")
            }
            self.assertFalse(
                np.allclose(
                    values[block]["clean"],
                    values[block]["shuffle"],
                )
            )
            self.assertFalse(
                np.allclose(
                    values[block]["clean"],
                    values[block]["reverse"],
                )
            )


class GroupFoldTests(unittest.TestCase):
    def test_groups_are_disjoint_nonfresh_stays_train_and_seed_changes(self) -> None:
        rows = _cohort()
        groups, _summary = _appearance_groups(rows, maximum_groups=24)
        first, _records = _make_folds(
            rows,
            index_to_group=groups,
            seed=111,
            repeats=1,
            folds=3,
        )
        second, _records = _make_folds(
            rows,
            index_to_group=groups,
            seed=222,
            repeats=1,
            folds=3,
        )
        nonfresh = {
            index for index, example in enumerate(rows) if not example.fresh
        }
        for fold in first:
            self.assertTrue(nonfresh.isdisjoint(fold.query_indices))
            train_components = {
                rows[index].component_id for index in fold.train_indices
            }
            query_components = {
                rows[index].component_id for index in fold.query_indices
            }
            self.assertTrue(train_components.isdisjoint(query_components))
            self.assertTrue(
                set(fold.train_groups).isdisjoint(fold.query_groups)
            )
        self.assertNotEqual(
            [fold.query_groups for fold in first],
            [fold.query_groups for fold in second],
        )

    def test_nested_inner_folds_never_see_outer_query(self) -> None:
        rows = _cohort()
        groups, _summary = _appearance_groups(rows, maximum_groups=24)
        outer_folds, _records = _make_folds(
            rows,
            index_to_group=groups,
            seed=111,
            repeats=1,
            folds=3,
        )
        outer = outer_folds[0]
        inner_folds, inner_records = _make_inner_folds(
            rows,
            index_to_group=groups,
            outer_fold=outer,
            seed=111,
            folds=2,
        )
        self.assertEqual(len(inner_folds), 2)
        outer_query = set(outer.query_indices)
        outer_train = set(outer.train_indices)
        for inner, record in zip(inner_folds, inner_records):
            self.assertTrue(set(inner.train_indices).issubset(outer_train))
            self.assertTrue(set(inner.query_indices).issubset(outer_train))
            self.assertTrue(
                outer_query.isdisjoint(inner.train_indices)
            )
            self.assertTrue(
                outer_query.isdisjoint(inner.query_indices)
            )
            self.assertEqual(record["outer_query_rows_seen"], 0)


class SearchAndPublicationTests(unittest.TestCase):
    def test_legacy_test_change_cannot_change_frozen_spec(self) -> None:
        clean = search_examples(
            _cohort(corrupt_legacy_test=False),
            seed=333,
            repeats=1,
            folds=3,
            maximum_trials=6,
        )
        corrupt = search_examples(
            _cohort(corrupt_legacy_test=True),
            seed=333,
            repeats=1,
            folds=3,
            maximum_trials=6,
        )
        self.assertEqual(
            clean["summary"]["champion"]["frozen_spec"]["spec_digest"],
            corrupt["summary"]["champion"]["frozen_spec"]["spec_digest"],
        )
        self.assertFalse(
            clean["summary"]["decision"]["renderer_probe_authorized"]
        )
        self.assertFalse(
            clean["summary"]["decision"]["editor_training_authorized"]
        )
        self.assertTrue(
            clean["summary"]["champion"]["frozen_spec"][
                "champion_eligible"
            ]
        )
        self.assertEqual(
            clean["summary"]["champion"]["frozen_spec"]["head"],
            "identity",
        )
        nested = clean["summary"]["nested_outer_model_selection"]
        self.assertTrue(
            all(
                record["outer_query_seen_by_inner_fit"] is False
                for record in nested["records"]
            )
        )
        self.assertRegex(
            clean["summary"]["fold_protocol"][
                "development_fold_assignment_sha256"
            ],
            r"^[0-9a-f]{64}$",
        )
        self.assertTrue(
            all(
                set(record["assignment_commitment"])
                == {
                    "query_group_ids_sha256",
                    "query_iids_sha256",
                    "query_component_ids_sha256",
                }
                for record in clean["folds"]
            )
        )

    def test_publication_closes_transform_and_all_gates(self) -> None:
        result = search_examples(
            _cohort(),
            seed=444,
            repeats=1,
            folds=3,
            maximum_trials=4,
        )
        arrays = _transform_arrays(
            result["frozen_transform"],
            result["frozen_dino_basis"],
            result["frozen_appearance_basis"],
        )
        self.assertEqual(arrays["action_head"].shape[1], 0)
        self.assertEqual(arrays["action_families"].size, 0)
        summary = copy.deepcopy(result["summary"])
        summary["source_snapshot"] = {
            "tree_sha256": "4" * 64,
            "exact_tree_verified_by_controller_before_search": False,
        }
        summary["frozen_transform"] = {
            "array_records": _array_records(arrays)
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "commit"
            _publish(
                output,
                trials=result["trials"],
                folds=result["folds"],
                failures=result["failure_memory"],
                predictions=result["predictions"],
                summary=summary,
                transform_arrays=arrays,
            )
            validated = validate_published_search(output)
            self.assertGreater(validated["predictions"], 0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(OUTPUT_NAMES),
            )
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o555)
            for name in OUTPUT_NAMES:
                self.assertEqual(
                    stat.S_IMODE((output / name).stat().st_mode),
                    0o444,
                )
            self.assertFalse(
                validated["done"]["renderer_probe_authorized"]
            )
            self.assertTrue((output / DONE_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
