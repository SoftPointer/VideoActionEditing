from __future__ import annotations

import json
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from render_shared8_contact_sheets import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
    PRIMARY_LABELS,
    _blind_model_order,
    _primary_blind_rows,
    _primary_manifest,
)


def _sample() -> dict[str, object]:
    return {
        "index": 3,
        "iid": "81473c034c1b4839",
        "source_video": "/data/source.mp4",
        "models": {
            model_id: {"output_path": f"/data/{model_id}/output.mp4"}
            for model_id in MODEL_ORDER
        },
    }


def test_blind_permutation_is_deterministic_and_bijective() -> None:
    kwargs = {
        "audit_digest": "a" * 64,
        "index": 3,
        "iid": "81473c034c1b4839",
    }
    first = _blind_model_order(**kwargs)
    assert first == _blind_model_order(**kwargs)
    assert first == (
        "lucy_official_base",
        "omnivideo2_official_base",
        "bernini_full644_lora_step644",
    )
    assert len(first) == 3
    assert set(first) == set(MODEL_ORDER)


def test_primary_rows_and_manifest_do_not_expose_target_or_model_labels() -> None:
    rows, mapping = _primary_blind_rows(_sample(), audit_digest="b" * 64)
    assert tuple(row.label for row in rows) == PRIMARY_LABELS
    assert set(mapping) == {"A", "B", "C"}
    assert all("target" not in row.label.lower() for row in rows)
    assert all(row.label not in MODEL_LABELS.values() for row in rows)

    manifest = _primary_manifest(
        audit_sha256="c" * 64,
        rendered=[{"index": 3, "iid": "81473c034c1b4839", "path": "sample.jpg"}],
    )
    serialized = json.dumps(manifest, sort_keys=True)
    assert "target" not in serialized.lower()
    assert "blind_key" not in serialized
    assert all(model_id not in serialized for model_id in MODEL_ORDER)
    assert all(model_label not in serialized for model_label in MODEL_LABELS.values())
