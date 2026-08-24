import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_preservation_only_review_html_v1.py"
)
SPEC = importlib.util.spec_from_file_location("build_preservation_review", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _receipt(cell_id: str, rank: int) -> dict:
    return {
        "cell_id": cell_id,
        "training_bundle": {"adapter_rank": rank},
        "action_reward_consumed": False,
        "feature_reward_consumed": False,
        "vlm_reward_consumed": False,
        "synthetic_target_consumed": False,
        "scientific_or_action_editing_claim_authorized": False,
        "rank_zero_only_vae": True,
        "sampling": {
            "same_official_gaussian_all_arms": True,
            "num_inference_steps": 40,
            "frame_count": 81,
        },
        "preservation_residual": {
            "preservation-residual": {
                "composition": "v_native_action+(v_adapted_noop-v_frozen_noop)",
                "adapter_action_text_input": False,
                "unit_gain": True,
                "noop_forwards": 80,
                "scheduler_steps": 40,
                "trace": [{} for _ in range(40)],
            }
        },
        "outputs": {
            arm: {"frame_count": 81, "fps": 25}
            for arm in ("native-rv2v", "preservation-residual")
        },
        "freeze_certificate": {"all_ranks_sampling_model_unchanged": True},
    }


def _packet(root: Path) -> Path:
    cell_root = root / "dog"
    cell_root.mkdir(parents=True)
    for basename in module.CELL_FILES:
        path = cell_root / basename
        if basename == "rank8-receipt.json":
            path.write_text(json.dumps(_receipt("dog", 8)), encoding="utf-8")
        elif basename == "rank2-receipt.json":
            path.write_text(json.dumps(_receipt("dog", 2)), encoding="utf-8")
        else:
            path.write_bytes(b"media")
    diagnostics = {
        "semantic_success_assessed": False,
        "wrong_source_gap_positive": 1,
        "wrong_source_gap_total": 2,
        "wrong_source_gap_mean": 0.1,
        "loss_min": 0.01,
        "loss_max": 0.2,
        "grad_norm_min": 0.03,
        "grad_norm_max": 1.2,
    }
    manifest = {
        "schema_version": module.SCHEMA,
        "authority": {
            "automatic_semantic_score_present": False,
            "reward_used": False,
            "synthetic_target_used": False,
            "manual_review_completed": False,
            "method_success_claimed": False,
        },
        "experiments": {
            variant: {
                "adapter_rank": rank,
                "optimizer_steps": 40,
                "training_target": "real_source_exact_noop",
                "dataset_rows": 2,
                "training_source_iids": ["train-a", "train-b"],
                "holder_job": "135407" if rank == 8 else "135411",
                "node": "gpu-260" if rank == 8 else "gpu-214",
                "diagnostics": dict(diagnostics),
            }
            for variant, rank in (("rank8", 8), ("rank2", 2))
        },
        "cells": [
            {
                "cell_id": "dog",
                "source_iid": "heldout-iid",
                "requested_action": "stand to sit",
                "source_action_caption": "source caption",
                "target_action_caption": "exact target caption",
                "seed": 7,
                "cross_variant_native_mp4_byte_exact": True,
                "cross_variant_source_condition_raw_byte_exact": True,
                "cross_variant_direct_comparison_authorized": True,
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_exposes_source_prompt_and_five_video_arms(tmp_path: Path) -> None:
    manifest = _packet(tmp_path)
    output = tmp_path / "index.html"
    result = module.build(manifest_path=manifest, media_root=tmp_path, output=output)
    page = output.read_text(encoding="utf-8")
    assert result["cell_count"] == 1
    assert "Source state caption" in page
    assert "Exact RV2V target caption" in page
    assert "2 real-source training clips" in page
    assert "未参与训练的 IID" in page
    assert "dog/rank8-native-rv2v.mp4" in page
    assert "dog/rank2-native-rv2v.mp4" in page
    assert "dog/rank8-preservation-residual.mp4" in page
    assert "dog/review_5x5.jpg" in page
    assert "frames 0 / 20 / 40 / 60 / 80" in page
    assert "没有自动“成功 value”" in page
    assert "Cross-rank control exact" in page


def test_rejects_reward_or_success_authority(tmp_path: Path) -> None:
    manifest = _packet(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["authority"]["reward_used"] = True
    manifest.write_text(json.dumps(value), encoding="utf-8")
    try:
        module.build(
            manifest_path=manifest,
            media_root=tmp_path,
            output=tmp_path / "index.html",
        )
    except module.PreservationReviewHTMLError as error:
        assert "fail-closed" in str(error)
    else:
        raise AssertionError("reward-bearing review must be rejected")


def test_rejects_training_iid_used_as_review_source(tmp_path: Path) -> None:
    manifest = _packet(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["cells"][0]["source_iid"] = "train-a"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    try:
        module.build(
            manifest_path=manifest,
            media_root=tmp_path,
            output=tmp_path / "index.html",
        )
    except module.PreservationReviewHTMLError as error:
        assert "held out" in str(error)
    else:
        raise AssertionError("training IID must not be presented as held-out review")
