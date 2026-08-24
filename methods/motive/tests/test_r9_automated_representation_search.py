from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motive import r7_candidate_temporal_screen as r7
from motive.r9_automated_representation_search import (
    DONE_NAME,
    FAILURE_SCHEMA,
    FAILURES_NAME,
    OUTPUT_NAMES,
    SUMMARY_NAME,
    TRIAL_SCHEMA,
    TRIALS_NAME,
    _publish,
    search_examples,
    validate_published_search,
)


def _unit(values: list[float]) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def _features(
    family: str,
    *,
    negative: bool = False,
    corrupt_temporal: bool = False,
    content_index: int = 0,
) -> dict[str, np.ndarray]:
    walk_sequence = np.zeros((4, 15), dtype=np.float64)
    jump_sequence = np.zeros((4, 15), dtype=np.float64)
    envelope = np.asarray([0.1, 0.5, 1.0, 0.2])
    walk_sequence[:, 0] = envelope
    walk_sequence[:, 10:15] = envelope[:, None]
    jump_sequence[:, 1] = envelope
    jump_sequence[:, 10:15] = (
        np.asarray([0.2, 1.0, 0.5, 0.1])[:, None]
    )
    if family == "walk":
        action = _unit(walk_sequence.reshape(-1).tolist())
        wrong = _unit(jump_sequence.reshape(-1).tolist())
    else:
        action = _unit(jump_sequence.reshape(-1).tolist())
        wrong = _unit(walk_sequence.reshape(-1).tolist())
    if negative:
        action = -action
    if corrupt_temporal:
        action = wrong
    endpoint = _unit([1.0, 1.0, 0.0, 0.0])
    dino = _unit(
        [
            float((content_index * 17 + 3) % 11),
            float((content_index * 13 + 5) % 7),
            float((content_index * 19 + 1) % 5),
            1.0,
        ]
    )
    return {
        r7.TARGET_TEMPORAL: action,
        r7.DELTA_TEMPORAL: action,
        r7.TARGET_ENDPOINT: endpoint,
        r7.ORDERLESS_TEMPORAL: endpoint,
        r7.CAMERA_NUISANCE: _unit([0.0, 0.0, 1.0, 1.0]),
        r7.POOLED_DINO: dino,
        r7.SHUFFLED_QUERY: wrong,
        r7.REVERSED_QUERY: wrong,
    }


def _example(
    iid: str,
    *,
    family: str,
    split: str,
    component: str,
    content_index: int,
    negative: bool = False,
    corrupt_temporal: bool = False,
) -> r7._Example:
    return r7._Example(
        iid=iid,
        label_class="negative" if negative else "positive",
        family=family,
        split=split,
        component_id=component,
        fresh=True,
        sampling_weight=1.0,
        features=_features(
            family,
            negative=negative,
            corrupt_temporal=corrupt_temporal,
            content_index=content_index,
        ),
        motion_energy=0.1 if negative else 0.9,
    )


def _cohort(*, corrupt_test: bool = False) -> list[r7._Example]:
    rows: list[r7._Example] = []
    index = 0
    for family in ("walk", "jump"):
        for item in range(6):
            rows.append(
                _example(
                    f"train-{family}-{item}",
                    family=family,
                    split="train",
                    component=f"train-component-{family}-{item}",
                    content_index=index,
                )
            )
            index += 1
        for split in ("validation", "test"):
            for item in range(3):
                rows.append(
                    _example(
                        f"{split}-{family}-{item}",
                        family=family,
                        split=split,
                        component=f"{split}-component-{family}-{item}",
                        content_index=index,
                        corrupt_temporal=(
                            corrupt_test and split == "test"
                        ),
                    )
                )
                index += 1
    for split in ("validation", "test"):
        for item in range(4):
            family = "walk" if item % 2 == 0 else "jump"
            rows.append(
                _example(
                    f"{split}-negative-{item}",
                    family=family,
                    split=split,
                    component=f"{split}-negative-component-{item}",
                    content_index=index,
                    negative=True,
                )
            )
            index += 1
    return rows


class R9AutomatedRepresentationSearchTests(unittest.TestCase):
    def test_search_finds_temporal_representation_and_records_failures(
        self,
    ) -> None:
        result = search_examples(
            _cohort(),
            generations=2,
            beam_width=4,
            max_trials=24,
        )
        summary = result["summary"]
        self.assertGreater(len(result["trials"]), 4)
        self.assertTrue(result["failure_memory"])
        self.assertEqual(
            summary["selection_protocol"]["search_split"],
            "validation_only",
        )
        self.assertTrue(
            summary["selection_protocol"][
                "test_opened_after_spec_freeze"
            ]
        )
        champion = summary["champion"]
        self.assertIn(
            r7.TARGET_TEMPORAL,
            champion["frozen_spec"]["components"],
        )
        self.assertGreaterEqual(
            champion["validation_metrics"]["retrieval"][
                "macro_family_r_at_5"
            ],
            0.99,
        )
        self.assertTrue(
            all(
                trial["test_metrics_read"] is False
                for trial in result["trials"]
            )
        )
        self.assertFalse(summary["decision"]["editor_training_authorized"])

    def test_test_features_cannot_change_frozen_validation_champion(
        self,
    ) -> None:
        clean = search_examples(
            _cohort(corrupt_test=False),
            generations=2,
            beam_width=4,
            max_trials=20,
        )
        corrupt = search_examples(
            _cohort(corrupt_test=True),
            generations=2,
            beam_width=4,
            max_trials=20,
        )
        self.assertEqual(
            clean["summary"]["champion"]["frozen_spec"]["spec_digest"],
            corrupt["summary"]["champion"]["frozen_spec"]["spec_digest"],
        )
        self.assertNotEqual(
            clean["summary"]["champion"]["test_metrics"]["retrieval"][
                "macro_family_r_at_1"
            ],
            corrupt["summary"]["champion"]["test_metrics"]["retrieval"][
                "macro_family_r_at_1"
            ],
        )

    def test_publication_is_closed_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "commit"
            summary = {
                "decision": {
                    "representation_gate_passed": False,
                    "renderer_probe_authorized": False,
                    "editor_training_authorized": False,
                },
                "training_authorized": False,
            }
            _publish(
                output,
                trials=[
                    {
                        "schema_version": TRIAL_SCHEMA,
                        "trial": 1,
                    }
                ],
                failures=[
                    {
                        "schema_version": FAILURE_SCHEMA,
                        "failure": "x",
                    }
                ],
                summary=summary,
            )
            validated = validate_published_search(output)
            self.assertFalse(
                validated["done"]["representation_gate_passed"]
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(OUTPUT_NAMES),
            )
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o555)
            for name in (
                TRIALS_NAME,
                FAILURES_NAME,
                SUMMARY_NAME,
                DONE_NAME,
            ):
                self.assertEqual(
                    stat.S_IMODE((output / name).stat().st_mode),
                    0o444,
                )


if __name__ == "__main__":
    unittest.main()
