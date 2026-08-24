#!/usr/bin/env python3
"""Frozen target-video middle-layer object/interaction-graph pilot (v5 R1).

This executable is a zero-training representation probe.  It reads the 12
pre-registered MEV source/target pairs, makes five independent calls per pair
to a frozen V-JEPA2 encoder, and immediately reduces the selected early,
middle, and late patch tokens into partial persistent slots, node trajectories,
a sparse interaction graph, and ordered lifecycle events.

No generator, optimizer, target pixel/latent regression, global pooled feature,
or dataset writer exists in this module.  Appearance-bearing patch tokens are
used only inside deterministic slot correspondence and are discarded after
each view.  The JSON receipt contains hashes and scalar controls, never RGB,
hidden tokens, slot coordinates, or descriptor values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np


MANIFEST_SCHEMA = "bernini-target-middle-object-graph-teacher-pilot-manifest-v5"
PREREG_SCHEMA = "bernini-target-middle-object-graph-teacher-pilot-prereg-v5"
RECEIPT_SCHEMA = "bernini-target-middle-object-graph-teacher-pilot-receipt-v5"
EXPERIMENT_ID = "target_middle_object_graph_teacher_pilot_v5_r1"
MANIFEST_FILE_SHA256 = "b333fc210501a4b5df23a9aa944c3b6da76a5a86d6fc4285587d873cad8c85c7"
MANIFEST_SELF_SHA256 = "caa533be2fc7f8cea2782bd7b8ca8f657440a8d036e9061ae02dfe9c33852ba2"
PREREG_FILE_SHA256 = "293e9b9371070814a68ffd89d4de757416fd3ef1240a03666fef39f8920a4b97"
PREREG_SELF_SHA256 = "c2d1b8405b4ce8eced3d377ab4a07f0c9e562288010ebf3afc77a28819d73ade"
CATALOG_SHA256 = "0c96e808114154e2d069da6ca698debfb6c9f824e0e780f6f39ec70612207ca8"
MODEL_REPO = "facebook/vjepa2-vitl-fpc64-256"
MODEL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
MODEL_FILES = {
    "config.json": ("3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9", 785),
    "model.safetensors": ("25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b", 1_303_947_864),
    "video_preprocessor_config.json": ("d2fab4418fc0390b62c4cd72ade56908a7929f80c62288adbe10dd8d23421227", 1_298),
}
DEFAULT_MODEL_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v3/"
    "vendor/vjepa2-vitl-fpc64-256-b3c1679"
)
HIDDEN_INDICES = (6, 12, 24)
VIEW_ORDER = (
    "target_forward_reference",
    "target_forward_eval",
    "target_reverse",
    "target_deterministic_shuffle",
    "source_noop",
)
SEED = 20260823
FRAMES = 64
TOKEN_TIME = 32
GRID = 16
PATCHES = GRID * GRID
HIDDEN = 1024
CHANNELS_PER_LAYER = 48
PROJECTED = len(HIDDEN_INDICES) * CHANNELS_PER_LAYER
MAX_SLOTS = 6
MAX_EDGES = 6
MAX_EVENTS = 16
PHASES = 8
NODE_PHASE_WIDTH = 4
EDGE_PHASE_WIDTH = 5
NODE_TAIL_WIDTH = 5
EDGE_TAIL_WIDTH = 5
NODE_DIM = PHASES * NODE_PHASE_WIDTH + NODE_TAIL_WIDTH
EDGE_BASE_DIM = PHASES * EDGE_PHASE_WIDTH + EDGE_TAIL_WIDTH
EDGE_ENDPOINT_WIDTH = 2
EDGE_DIM = EDGE_ENDPOINT_WIDTH + EDGE_BASE_DIM
EVENT_DIM = 4
DESCRIPTOR_DIM = 2 + MAX_SLOTS * NODE_DIM + MAX_EDGES * EDGE_DIM + MAX_EVENTS * EVENT_DIM
CHANNEL_PLAN_SHA256 = "7b0a36c522e0df4fd4825448845c636d17491cce6864ca8ac0094d77ee58fc78"
V4C_EXTRACTOR_SHA256 = "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc"
LAUNCH_WRAPPER_SHA256 = "26642a6d745671852c849587453b64a5a61523fec3092b3b54256b6d0d9cdc66"

TRAIN_IDS = (
    "6ae4b656b36ae36950140af78ff96c6038de540b5a72cbf0d68ca1c995f321fc",
    "9951e48b614307f9a91ac4d48b617dbd6801fcba99e3936e87b9de875e5bd7ac",
    "750534151b0221f867cb48d9fd8a9e7bdad001729cf0d50b39e03fafeff5df01",
    "41d5ab043ecfdf5ce32cc6ad58c2c210db8938f4948096477acaceeaaf015e6e",
    "c5efe4f7dcd084395891550185a73ecf67c9340cbc14d748b45592f0980828f0",
    "7b3db60f5d9b181f3c2fdca881eca180c96d45064c60370b90e628111e3daf4e",
    "ef46896149a4386d74f8c66caadfc459eded00f4dda5115452d46702d98907cc",
    "b6217e1595627ee516247a71fe47a44bf610473c3a66afa2bae7ed704a3623f8",
)
VALIDATION_IDS = (
    "ab48476cbf04260695a0188fe6c3faa75983636aa95d5e6a51fcc53e8696c235",
    "a40e955ecb46888c19f77d1e11cea76ce0d225f440e753a1665d628986150d30",
    "867c2b1dbc09238dfca5c4b5a22a608ca0b831b6568e3b118bba73bc4294fb67",
    "efdd171c5b8b6a256c10fe80ac011c9784f68b8f6bef27dcfb9aef1f1a354fd0",
)

DECODED_FRAME_COUNTS = {
    "6ae4b656b36ae36950140af78ff96c6038de540b5a72cbf0d68ca1c995f321fc": (39, 100),
    "9951e48b614307f9a91ac4d48b617dbd6801fcba99e3936e87b9de875e5bd7ac": (44, 63),
    "750534151b0221f867cb48d9fd8a9e7bdad001729cf0d50b39e03fafeff5df01": (175, 322),
    "41d5ab043ecfdf5ce32cc6ad58c2c210db8938f4948096477acaceeaaf015e6e": (126, 259),
    "c5efe4f7dcd084395891550185a73ecf67c9340cbc14d748b45592f0980828f0": (39, 324),
    "7b3db60f5d9b181f3c2fdca881eca180c96d45064c60370b90e628111e3daf4e": (101, 316),
    "ef46896149a4386d74f8c66caadfc459eded00f4dda5115452d46702d98907cc": (79, 91),
    "b6217e1595627ee516247a71fe47a44bf610473c3a66afa2bae7ed704a3623f8": (161, 199),
    "ab48476cbf04260695a0188fe6c3faa75983636aa95d5e6a51fcc53e8696c235": (25, 91),
    "a40e955ecb46888c19f77d1e11cea76ce0d225f440e753a1665d628986150d30": (75, 138),
    "867c2b1dbc09238dfca5c4b5a22a608ca0b831b6568e3b118bba73bc4294fb67": (46, 240),
    "efdd171c5b8b6a256c10fe80ac011c9784f68b8f6bef27dcfb9aef1f1a354fd0": (45, 87),
}

THRESHOLDS = {
    "target_forward_reference_eval_cosine_min": 0.90,
    "target_forward_reference_eval_distance_max": 0.10,
    "minimum_margin_above_hardest_negative": 0.08,
    "target_reverse_cosine_max": 0.92,
    "target_shuffle_cosine_max": 0.90,
    "source_noop_cosine_max": 0.85,
    "minimum_non_dustbin_slots_per_target_forward": 2,
    "minimum_dynamic_sparse_edges_per_target_forward": 1,
    "train_pair_pass_min": 6,
    "validation_pair_pass_min": 3,
    "all_twelve_forward_above_each_control_min": 9,
}


class PilotError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise PilotError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PilotError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_regular_file(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
        fail(f"{label} file SHA differs")
    return {
        "path": str(path), "realpath": str(resolved), "sha256": expected_sha256,
        "size_bytes": before.st_size, "device": before.st_dev, "inode": before.st_ino,
    }


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)}))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, expected_file_sha256: str) -> Mapping[str, Any]:
    requested = path
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"authority path must be absolute and non-symlink: {requested}")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not resolved.is_file():
        fail(f"authority path must be one regular canonical file: {requested}")
    before = requested.stat()
    with requested.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        raw = handle.read()
        handle.seek(0)
        reread = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = requested.stat()
    identity = lambda row: (
        row.st_dev, row.st_ino, row.st_size, row.st_mode, row.st_nlink,
        row.st_mtime_ns, row.st_ctime_ns,
    )
    if (
        raw != reread or identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        fail(f"authority changed or file SHA differs: {requested}")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_json_pairs,
            parse_constant=lambda token: fail(f"nonfinite JSON constant: {token}"),
        )
    except PilotError:
        raise
    except Exception as error:
        raise PilotError(f"cannot parse authority: {requested}") from error
    if type(value) is not dict:
        fail("authority root must be an object")
    return value


def verify_self_hash(value: Mapping[str, Any], key: str, expected: str) -> None:
    if value.get(key) != expected:
        fail(f"{key} pin differs")
    payload = dict(value)
    payload.pop(key, None)
    if object_sha256(payload) != expected:
        fail(f"{key} self hash differs")


@dataclass(frozen=True)
class PairRow:
    ordinal: int
    pair_id: str
    split: str
    uuid: str
    instruction: str
    source_path: Path
    source_sha256: str
    source_size: int
    target_path: Path
    target_sha256: str
    target_size: int


def validate_manifest(value: Mapping[str, Any]) -> list[PairRow]:
    if value.get("schema_version") != MANIFEST_SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        fail("manifest schema/experiment differs")
    verify_self_hash(value, "manifest_sha256", MANIFEST_SELF_SHA256)
    if value.get("catalog_source") != {
        "path": "/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction/runs/full_v5_20260817T193449Z/final_metadata_annotation_v2/paired_training_candidates.jsonl",
        "sha256": CATALOG_SHA256,
        "row_count": 3749,
    }:
        fail("catalog authority differs")
    authority = value.get("authority")
    if type(authority) is not dict or authority != {
        "formal_sft_authorized": False,
        "exploratory_representation_only": True,
        "human_qualification_complete": False,
        "dataset_materialization_authorized": False,
        "generator_training_authorized": False,
        "generator_connection_authorized": False,
        "validation_threshold_tuning_authorized": False,
        "raw_target_pixels_or_hidden_states_export_authorized": False,
    }:
        fail("manifest representation-only authority differs")
    pairs = value.get("pairs")
    if type(pairs) is not list or len(pairs) != 12:
        fail("manifest must contain exact12 pairs")
    result: list[PairRow] = []
    seen_paths: set[str] = set()
    seen_uuid: dict[str, str] = {}
    for ordinal, row in enumerate(pairs):
        if type(row) is not dict or row.get("ordinal") != ordinal:
            fail("pair ordinal differs")
        pair_id = row.get("pair_id")
        split = row.get("locked_split")
        expected_ids = TRAIN_IDS if split == "train" else VALIDATION_IDS if split == "validation" else ()
        expected_ordinal = ordinal if split == "train" else ordinal - len(TRAIN_IDS)
        if (
            expected_ordinal < 0 or expected_ordinal >= len(expected_ids)
            or pair_id != expected_ids[expected_ordinal]
            or row.get("catalog_split") != split
            or row.get("formal_sft_authorized") is not False
            or row.get("qualification_status")
            != "qwen-visual-accepted-annotation-instruction-pending-human"
        ):
            fail(f"pair authority differs at ordinal {ordinal}")
        uuid = row.get("uuid")
        if type(uuid) is not str or uuid in seen_uuid:
            fail("UUID split leakage or duplicate")
        seen_uuid[uuid] = split
        paths = (row.get("source_video_path"), row.get("target_video_path"))
        if any(type(path) is not str or not path.startswith("/vast/") or path in seen_paths for path in paths):
            fail("pair media path differs or aliases")
        seen_paths.update(paths)
        hashes = (row.get("source_video_sha256"), row.get("target_video_sha256"))
        sizes = (row.get("source_video_size_bytes"), row.get("target_video_size_bytes"))
        if any(type(item) is not str or len(item) != 64 for item in hashes):
            fail("media SHA differs")
        if any(type(item) is not int or item <= 0 for item in sizes):
            fail("media size differs")
        instruction = row.get("instruction")
        if type(instruction) is not str or not instruction.startswith("Edit the action so that "):
            fail("instruction authority differs")
        result.append(PairRow(
            ordinal=ordinal, pair_id=pair_id, split=split, uuid=uuid,
            instruction=instruction,
            source_path=Path(paths[0]), source_sha256=hashes[0], source_size=sizes[0],
            target_path=Path(paths[1]), target_sha256=hashes[1], target_size=sizes[1],
        ))
    if tuple(row.pair_id for row in result[:8]) != TRAIN_IDS or tuple(
        row.pair_id for row in result[8:]
    ) != VALIDATION_IDS:
        fail("exact 8/4 split order differs")
    return result


def validate_prereg(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != PREREG_SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        fail("prereg schema/experiment differs")
    verify_self_hash(value, "preregistration_sha256", PREREG_SELF_SHA256)
    if value.get("manifest_pin") != {
        "relative_path": "methods/bernini_action_editing/assets/target_middle_object_graph_teacher_pilot_manifest_v5.json",
        "file_sha256": MANIFEST_FILE_SHA256,
        "manifest_sha256": MANIFEST_SELF_SHA256,
        "pair_count": 12, "train_count": 8, "validation_count": 4,
    }:
        fail("prereg manifest pin differs")
    frozen = value.get("frozen_backbone")
    if type(frozen) is not dict or any((
        frozen.get("repo") != MODEL_REPO,
        frozen.get("revision") != MODEL_REVISION,
        frozen.get("parameters_frozen") is not True,
        frozen.get("output_hidden_states") is not True,
        frozen.get("selected_hidden_state_indices") != list(HIDDEN_INDICES),
        frozen.get("expected_each_hidden_shape") != [1, 8192, 1024],
        frozen.get("patch_geometry") != [32, 16, 16],
        frozen.get("reused_extractor_file_sha256") != V4C_EXTRACTOR_SHA256,
    )):
        fail("frozen backbone or reused extractor binding differs")
    boundary = value.get("optimization_boundary")
    if type(boundary) is not dict or any((
        boundary.get("optimizer_created") is not False,
        boundary.get("loss_backward_calls") != 0,
        boundary.get("parameter_updates") != 0,
        boundary.get("train_split_parameter_fitting") is not False,
        boundary.get("validation_parameter_fitting") is not False,
        boundary.get("validation_threshold_selection") is not False,
        boundary.get("generator_loaded") is not False,
        boundary.get("generator_forward_calls") != 0,
        boundary.get("generator_updates") != 0,
        boundary.get("dataset_written") is not False,
    )):
        fail("zero-training boundary differs")
    gates = value.get("fixed_admission_thresholds")
    if type(gates) is not dict or any(gates.get(key) != expected for key, expected in {
        **THRESHOLDS,
        "reference_above_each_negative_control": True,
        "train_pair_total": 8,
        "validation_pair_total": 4,
        "descriptor_finite_required": True,
        "exact_pair_and_forward_count_required": True,
    }.items()):
        fail("fixed admission thresholds differ")
    algorithm = value.get("sealed_algorithm")
    if type(algorithm) is not dict or algorithm.get("channel_plan_sha256") != CHANNEL_PLAN_SHA256:
        fail("sealed algorithm binding differs")
    expected_permutations = {row["pair_id"]: row["permutation"] for row in algorithm.get("shuffle_permutations", []) if type(row) is dict and "pair_id" in row and "permutation" in row}
    if set(expected_permutations) != set(TRAIN_IDS + VALIDATION_IDS):
        fail("shuffle permutation registry differs")
    for pair_id, expected in expected_permutations.items():
        if shuffle_permutation(pair_id) != tuple(expected):
            fail(f"shuffle permutation differs for {pair_id}")
    if algorithm.get("sampling_geometry_registry") != sampling_geometry_registry():
        fail("sampling geometry registry differs")
    exact_algorithm_values = {
        "target_decoded_frame_count_min": 48,
        "target_reference_unique_index_count_min": 48,
        "target_eval_unique_index_count_min": 48,
        "source_decoded_frame_count_min": 16,
        "source_unique_index_count_min": 16,
        "seed_spatial_nms_min_distance": 0.32,
        "slot_max": MAX_SLOTS,
        "track_cosine_min": 0.35,
        "track_cost_max": 0.72,
        "track_consecutive_gap_max": 4,
        "local_support_radius": 0.32,
        "local_support_cosine_min": 0.55,
        "slot_min_present_tubelets": 8,
        "slot_min_median_confidence_margin": 0.01,
        "node_motion_onset_speed": 0.08,
        "node_phase_bins": PHASES,
        "node_phase_frames": 4,
        "edge_joint_presence_min": 8,
        "edge_max": MAX_EDGES,
        "edge_degree_cap": 2,
        "edge_approach_or_recede_delta_min": 0.06,
        "descriptor_dimension": DESCRIPTOR_DIM,
        "descriptor_layout": "2 counts + 6*37 padded node values + 6*47 padded edge-topology/lifecycle values + 16*4 padded ordered events = 570",
        "edge_canonical_endpoint_encoding": "prepend ((canonical_left+1)/6,(canonical_right+1)/6) to each 45-value lifecycle descriptor; require 0<=left<right<6",
        "empty_or_one_slot_or_zero_edge_policy": "mechanically invalid; similarity is exactly 0.0 and pair hard-fails regardless of another invalid graph",
    }
    if any(algorithm.get(key) != expected for key, expected in exact_algorithm_values.items()):
        fail("sealed slot/edge/descriptor constants differ")
    if algorithm.get("descriptor_group_weights") != {
        "counts": 1.0, "nodes": 1.0, "edges": 1.25, "events": 0.75,
    }:
        fail("descriptor group weights differ")
    object_contract = value.get("object_centric_representation")
    if type(object_contract) is not dict or any((
        object_contract.get("max_slots") != MAX_SLOTS,
        object_contract.get("max_serialized_nodes") != MAX_SLOTS,
        object_contract.get("max_serialized_edges") != MAX_EDGES,
        object_contract.get("max_serialized_events") != MAX_EVENTS,
        object_contract.get("dustbin_required") is not True,
        object_contract.get("forced_equal_mass_assignment") is not False,
        object_contract.get("forced_slot_presence") is not False,
    )):
        fail("object-centric cardinality/dustbin contract differs")
    runtime_closure = value.get("runtime_source_closure")
    if runtime_closure != {
        "launch_wrapper_relative_path": "methods/bernini_action_editing/scripts/auh_target_middle_object_graph_teacher_pilot_rank_wrapper_v5.sh",
        "launch_wrapper_file_sha256": LAUNCH_WRAPPER_SHA256,
        "reused_v4c_extractor_relative_path": "methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py",
        "reused_v4c_extractor_file_sha256": V4C_EXTRACTOR_SHA256,
        "runner_is_recorded_dynamically_in_contract_and_receipt": True,
    }:
        fail("runtime source closure differs")


def reference_indices(frame_count: int) -> np.ndarray:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be positive")
    return np.asarray([(i * (frame_count - 1)) // 63 for i in range(64)], dtype=np.int64)


def eval_indices(frame_count: int) -> np.ndarray:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be positive")
    return np.asarray(
        [i * (126 + i) * (frame_count - 1) // (63 * 189) for i in range(64)],
        dtype=np.int64,
    )


def sampling_geometry_registry() -> list[Mapping[str, Any]]:
    rows = []
    for pair_id in TRAIN_IDS + VALIDATION_IDS:
        source_count, target_count = DECODED_FRAME_COUNTS[pair_id]
        source = reference_indices(source_count)
        reference = reference_indices(target_count)
        evaluation = eval_indices(target_count)
        rows.append({
            "pair_id": pair_id,
            "source_decoded_frames": source_count,
            "target_decoded_frames": target_count,
            "source_unique_indices": len(set(source.tolist())),
            "target_reference_unique_indices": len(set(reference.tolist())),
            "target_eval_unique_indices": len(set(evaluation.tolist())),
            "source_indices_sha256": array_sha256(source),
            "target_reference_indices_sha256": array_sha256(reference),
            "target_eval_indices_sha256": array_sha256(evaluation),
        })
    return rows


def shuffle_permutation(pair_id: str) -> tuple[int, ...]:
    if type(pair_id) is not str or len(pair_id) != 64:
        raise ValueError("pair_id differs")
    blocks = sorted(
        range(8),
        key=lambda block: (
            hashlib.sha256(f"v5-target-shuffle:{SEED}:{pair_id}:{block}".encode("ascii")).digest(),
            block,
        ),
    )
    if blocks in (list(range(8)), list(reversed(range(8)))):
        blocks = blocks[3:] + blocks[:3]
    if blocks in (list(range(8)), list(reversed(range(8)))) or set(blocks) != set(range(8)):
        raise RuntimeError("shuffle permutation is forbidden")
    return tuple(blocks)


def shuffle64_indices(pair_id: str) -> np.ndarray:
    return np.asarray(
        [8 * block + offset for block in shuffle_permutation(pair_id) for offset in range(8)],
        dtype=np.int64,
    )


def channel_plan() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    serial: dict[str, Any] = {}
    for layer in HIDDEN_INDICES:
        channels = sorted(
            range(HIDDEN),
            key=lambda channel: (
                hashlib.sha256(f"v5-channel:{SEED}:{layer}:{channel}".encode("ascii")).digest(),
                channel,
            ),
        )[:CHANNELS_PER_LAYER]
        signs = [
            1 if hashlib.sha256(
                f"v5-sign:{SEED}:{layer}:{channel}".encode("ascii")
            ).digest()[0] & 1 else -1
            for channel in channels
        ]
        result[layer] = (np.asarray(channels, dtype=np.int64), np.asarray(signs, dtype=np.float32))
        serial[str(layer)] = {"indices": channels, "signs": signs}
    if object_sha256(serial) != CHANNEL_PLAN_SHA256:
        raise RuntimeError("fixed channel plan differs")
    return result


def project_hidden_layers(hidden_layers: np.ndarray) -> np.ndarray:
    value = np.asarray(hidden_layers)
    if value.shape != (3, TOKEN_TIME, PATCHES, HIDDEN) or value.dtype != np.float32:
        raise ValueError("selected hidden patch geometry/dtype differs")
    if not np.isfinite(value).all():
        raise ValueError("selected hidden patches contain nonfinite values")
    pieces = []
    plan = channel_plan()
    for index, layer in enumerate(HIDDEN_INDICES):
        channels, signs = plan[layer]
        selected = np.take(value[index], channels, axis=-1) * signs.reshape(1, 1, -1)
        norms = np.linalg.norm(selected, axis=-1, keepdims=True)
        pieces.append(selected / np.maximum(norms, np.float32(1e-8)))
    projected = np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)
    projected /= np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), np.float32(1e-8))
    if projected.shape != (TOKEN_TIME, PATCHES, PROJECTED) or not np.isfinite(projected).all():
        raise RuntimeError("projected patch token closure differs")
    return np.ascontiguousarray(projected)


def grid_coordinates() -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, GRID, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)


@dataclass
class SlotTrack:
    seed_score: float
    presence: np.ndarray
    patch_ids: np.ndarray
    position: np.ndarray
    extent: np.ndarray
    confidence: np.ndarray


def _unit(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _track_one(projected: np.ndarray, seed_patch: int, seed_time: int, seed_score: float) -> SlotTrack:
    coordinates = grid_coordinates()
    presence = np.zeros(TOKEN_TIME, dtype=bool)
    patch_ids = np.full(TOKEN_TIME, -1, dtype=np.int64)
    confidence = np.zeros(TOKEN_TIME, dtype=np.float32)
    prototypes: dict[int, np.ndarray] = {seed_time: projected[seed_time, seed_patch].copy()}
    presence[seed_time] = True
    patch_ids[seed_time] = seed_patch
    confidence[seed_time] = 1.0
    for direction in (-1, 1):
        previous_patch = seed_patch
        prototype = projected[seed_time, seed_patch].copy()
        gap = 0
        for current_time in range(seed_time + direction, TOKEN_TIME if direction > 0 else -1, direction):
            radius = min(1.50, 0.30 + 0.12 * gap)
            displacement = np.linalg.norm(coordinates - coordinates[previous_patch], axis=1)
            similarity = projected[current_time] @ prototype
            cost = 0.80 * (1.0 - similarity) + 0.20 * (displacement / radius)
            valid = (displacement <= radius) & (similarity >= 0.35) & (cost <= 0.72)
            if not np.any(valid):
                gap += 1
                if gap > 4:
                    break
                continue
            candidates = np.flatnonzero(valid)
            chosen = int(candidates[np.lexsort((candidates, cost[candidates]))[0]])
            ordered_similarity = np.sort(similarity[valid])
            runner_up = float(ordered_similarity[-2]) if ordered_similarity.size > 1 else 0.0
            presence[current_time] = True
            patch_ids[current_time] = chosen
            confidence[current_time] = np.float32(max(0.0, float(similarity[chosen]) - runner_up))
            prototype = _unit(0.85 * prototype + 0.15 * projected[current_time, chosen]).astype(np.float32)
            prototypes[current_time] = prototype.copy()
            previous_patch = chosen
            gap = 0
    position = np.full((TOKEN_TIME, 2), np.nan, dtype=np.float32)
    extent = np.zeros(TOKEN_TIME, dtype=np.float32)
    for t in np.flatnonzero(presence):
        center_patch = int(patch_ids[t])
        center_token = projected[t, center_patch]
        distance = np.linalg.norm(coordinates - coordinates[center_patch], axis=1)
        similarity = projected[t] @ center_token
        support = (distance <= 0.32) & (similarity >= 0.55)
        members = np.flatnonzero(support)
        weights = np.maximum(similarity[members] - 0.55, 1e-4)
        centroid = np.sum(coordinates[members] * weights[:, None], axis=0) / np.sum(weights)
        radius = math.sqrt(float(np.sum(weights * np.sum((coordinates[members] - centroid) ** 2, axis=1)) / np.sum(weights)))
        position[t] = centroid.astype(np.float32)
        extent[t] = np.float32(max(2.0 / 15.0, radius))
    return SlotTrack(seed_score, presence, patch_ids, position, extent, confidence)


def build_partial_slots(projected: np.ndarray) -> tuple[list[SlotTrack], dict[str, Any]]:
    value = np.asarray(projected)
    if value.shape != (TOKEN_TIME, PATCHES, PROJECTED) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("projected patch tensor differs")
    change = np.maximum(0.0, 1.0 - np.sum(value[1:] * value[:-1], axis=-1))
    patch_score = np.maximum(
        np.quantile(change, 0.90, axis=0),
        0.25 * np.max(change, axis=0),
    )
    median = float(np.median(patch_score))
    mad = float(np.median(np.abs(patch_score - median)))
    cutoff = max(0.02, median + 2.0 * mad)
    coordinates = grid_coordinates()
    order = np.lexsort((np.arange(PATCHES), -patch_score))
    seeds: list[int] = []
    for patch in order:
        if float(patch_score[patch]) < cutoff:
            break
        if all(float(np.linalg.norm(coordinates[patch] - coordinates[other])) >= 0.32 for other in seeds):
            seeds.append(int(patch))
        if len(seeds) == MAX_SLOTS:
            break
    tracks = [
        _track_one(value, patch, int(np.argmax(change[:, patch])) + 1, float(patch_score[patch]))
        for patch in seeds
    ]
    # Resolve same-patch collisions without forcing every slot to receive mass.
    for t in range(TOKEN_TIME):
        owners: dict[int, list[int]] = {}
        for slot_index, track in enumerate(tracks):
            if track.presence[t]:
                owners.setdefault(int(track.patch_ids[t]), []).append(slot_index)
        for colliders in owners.values():
            if len(colliders) <= 1:
                continue
            keep = min(colliders, key=lambda index: (-float(tracks[index].confidence[t]), index))
            for index in colliders:
                if index != keep:
                    tracks[index].presence[t] = False
                    tracks[index].patch_ids[t] = -1
                    tracks[index].position[t] = np.nan
                    tracks[index].extent[t] = 0.0
                    tracks[index].confidence[t] = 0.0
    tracks = [
        track for track in tracks
        if int(track.presence.sum()) >= 8 and float(np.median(track.confidence[track.presence])) >= 0.01
    ]
    tracks.sort(key=lambda row: (-row.seed_score, int(np.flatnonzero(row.presence)[0]), int(row.patch_ids[row.presence][0])))
    occupied = [len({int(track.patch_ids[t]) for track in tracks if track.presence[t]}) for t in range(TOKEN_TIME)]
    diagnostics = {
        "motion_score_median": median,
        "motion_score_mad": mad,
        "motion_seed_cutoff": cutoff,
        "candidate_seed_count": len(seeds),
        "retained_slot_count": len(tracks),
        "dustbin_patch_fraction_median": float(np.median(1.0 - np.asarray(occupied) / PATCHES)),
        "forced_slot_presence": False,
        "forced_equal_mass": False,
    }
    return tracks, diagnostics


def _support_scale(slots: Sequence[SlotTrack]) -> float:
    distances: list[float] = []
    for t in range(TOKEN_TIME):
        active = [slot.position[t] for slot in slots if slot.presence[t]]
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                distance = float(np.linalg.norm(active[left] - active[right]))
                if distance > 1e-6:
                    distances.append(distance)
    return max(float(np.median(distances)) if distances else 0.0, 2.0 / 15.0)


def _phase_stat(values: np.ndarray, valid: np.ndarray, phase: int, quantile: float = 0.75) -> float:
    start = phase * (TOKEN_TIME // PHASES)
    stop = (phase + 1) * (TOKEN_TIME // PHASES)
    selected = values[start:stop][valid[start:stop]]
    return float(np.quantile(selected, quantile)) if selected.size else 0.0


def _squash(value: float) -> float:
    value = max(0.0, float(value))
    return value / (1.0 + value)


def node_descriptor(slot: SlotTrack, scale: float) -> tuple[np.ndarray, tuple[Any, ...], list[tuple[float, int, int, int]]]:
    pair_valid = slot.presence[1:] & slot.presence[:-1]
    speed31 = np.zeros(TOKEN_TIME - 1, dtype=np.float32)
    speed31[pair_valid] = np.linalg.norm(slot.position[1:][pair_valid] - slot.position[:-1][pair_valid], axis=1) / scale
    speed = np.zeros(TOKEN_TIME, dtype=np.float32)
    speed[1:] = speed31
    speed_valid = np.zeros(TOKEN_TIME, dtype=bool)
    speed_valid[1:] = pair_valid
    acceleration = np.zeros(TOKEN_TIME, dtype=np.float32)
    acceleration[2:] = np.abs(speed[2:] - speed[1:-1])
    acceleration_valid = np.zeros(TOKEN_TIME, dtype=bool)
    acceleration_valid[2:] = speed_valid[2:] & speed_valid[1:-1]
    extent_change = np.zeros(TOKEN_TIME, dtype=np.float32)
    extent_change[1:][pair_valid] = np.abs(slot.extent[1:][pair_valid] - slot.extent[:-1][pair_valid]) / scale
    values: list[float] = []
    for phase in range(PHASES):
        start, stop = phase * 4, (phase + 1) * 4
        values.extend((
            float(slot.presence[start:stop].sum()) / 4.0,
            _squash(_phase_stat(speed, speed_valid, phase)),
            _squash(_phase_stat(acceleration, acceleration_valid, phase)),
            _squash(_phase_stat(extent_change, speed_valid, phase)),
        ))
    moving = np.flatnonzero(speed_valid & (speed >= 0.08))
    onset = float(moving[0]) / 31.0 if moving.size else 1.0
    offset = float(moving[-1]) / 31.0 if moving.size else 0.0
    peak = float(np.argmax(speed)) / 31.0 if np.any(speed_valid) else 0.0
    path = float(speed.sum())
    present = np.flatnonzero(slot.presence)
    net = float(np.linalg.norm(slot.position[present[-1]] - slot.position[present[0]]) / scale) if present.size >= 2 else 0.0
    efficiency = min(1.0, net / max(path, 1e-8))
    values.extend((onset, peak, offset, _squash(path / 31.0), efficiency))
    descriptor = np.asarray(values, dtype=np.float32)
    key = (round(onset, 6), -round(_squash(path / 31.0), 6), round(peak, 6), round(offset, 6), array_sha256(descriptor))
    events: list[tuple[float, int, int, int]] = []
    if moving.size:
        events.extend(((onset, 1, -1, -1), (peak, 2, -1, -1), (offset, 3, -1, -1)))
    return descriptor, key, events


def _edge_candidates(slots: Sequence[SlotTrack], scale: float) -> list[tuple[float, int, int, np.ndarray, np.ndarray]]:
    candidates = []
    for left in range(len(slots)):
        for right in range(left + 1, len(slots)):
            valid = slots[left].presence & slots[right].presence
            if int(valid.sum()) < 8:
                continue
            distance = np.zeros(TOKEN_TIME, dtype=np.float32)
            distance[valid] = np.linalg.norm(slots[left].position[valid] - slots[right].position[valid], axis=1) / scale
            delta_valid = valid[1:] & valid[:-1]
            delta = np.zeros(TOKEN_TIME, dtype=np.float32)
            delta[1:][delta_valid] = distance[1:][delta_valid] - distance[:-1][delta_valid]
            priority = float(np.quantile(distance[valid], 0.20) - 0.5 * np.quantile(np.abs(delta[1:][delta_valid]), 0.75))
            candidates.append((priority, left, right, distance, valid))
    return candidates


def edge_descriptor(
    left: SlotTrack, right: SlotTrack, distance: np.ndarray, valid: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, tuple[Any, ...], list[tuple[float, int, int, int]]]:
    delta = np.zeros(TOKEN_TIME, dtype=np.float32)
    delta_valid = np.zeros(TOKEN_TIME, dtype=bool)
    delta_valid[1:] = valid[1:] & valid[:-1]
    delta[1:][delta_valid[1:]] = distance[1:][delta_valid[1:]] - distance[:-1][delta_valid[1:]]
    contact_threshold = np.maximum(0.12, 1.5 * (left.extent + right.extent) / scale)
    contact = valid & (distance <= contact_threshold)
    values: list[float] = []
    for phase in range(PHASES):
        start, stop = phase * 4, (phase + 1) * 4
        phase_valid = valid[start:stop]
        phase_distance = distance[start:stop][phase_valid]
        approach = np.maximum(-delta[start:stop][delta_valid[start:stop]], 0.0)
        recede = np.maximum(delta[start:stop][delta_valid[start:stop]], 0.0)
        values.extend((
            float(phase_valid.sum()) / 4.0,
            1.0 / (1.0 + float(np.quantile(phase_distance, 0.50))) if phase_distance.size else 0.0,
            _squash(float(np.quantile(approach, 0.75))) if approach.size else 0.0,
            _squash(float(np.quantile(recede, 0.75))) if recede.size else 0.0,
            float(contact[start:stop].sum()) / 4.0,
        ))
    closest = int(np.argmin(np.where(valid, distance, np.inf))) if np.any(valid) else 0
    approaches = np.flatnonzero(delta_valid & (-delta >= 0.06))
    recedes = np.flatnonzero(delta_valid & (delta >= 0.06))
    contacts = np.flatnonzero(contact)
    contact_onset = float(contacts[0]) / 31.0 if contacts.size else 1.0
    contact_release = float(contacts[-1]) / 31.0 if contacts.size else 0.0
    terminal_hold = float(contact[-4:].sum()) / 4.0
    values.extend((
        float(closest) / 31.0,
        float(approaches[0]) / 31.0 if approaches.size else 1.0,
        contact_onset,
        contact_release,
        terminal_hold,
    ))
    descriptor = np.asarray(values, dtype=np.float32)
    key = (round(contact_onset, 6), -round(terminal_hold, 6), round(float(closest) / 31.0, 6), array_sha256(descriptor))
    events: list[tuple[float, int, int, int]] = []
    if approaches.size:
        events.append((float(approaches[0]) / 31.0, 4, -1, -1))
    if contacts.size:
        events.extend(((contact_onset, 5, -1, -1), (contact_release, 6, -1, -1)))
    if recedes.size:
        events.append((float(recedes[0]) / 31.0, 7, -1, -1))
    return descriptor, key, events


@dataclass(frozen=True)
class GraphRepresentation:
    descriptor: np.ndarray
    descriptor_sha256: str
    slot_count: int
    edge_count: int
    event_count: int
    mechanically_valid: bool
    diagnostics: Mapping[str, Any]


def build_graph_representation(projected: np.ndarray) -> GraphRepresentation:
    slots, slot_diagnostics = build_partial_slots(projected)
    scale = _support_scale(slots)
    nodes = []
    node_events = []
    for original_index, slot in enumerate(slots):
        descriptor, key, events = node_descriptor(slot, scale)
        nodes.append((key, original_index, descriptor))
        node_events.append(events)
    nodes.sort(key=lambda row: row[0])
    canonical_index = {original: new for new, (_, original, _) in enumerate(nodes)}
    degrees = [0] * len(slots)
    selected_edges = []
    for priority, left, right, distance, valid in sorted(_edge_candidates(slots, scale), key=lambda row: (row[0], row[1], row[2])):
        if degrees[left] >= 2 or degrees[right] >= 2:
            continue
        descriptor, key, events = edge_descriptor(slots[left], slots[right], distance, valid, scale)
        selected_edges.append((key, left, right, descriptor, events))
        degrees[left] += 1
        degrees[right] += 1
        if len(selected_edges) == MAX_EDGES:
            break
    canonical_edges = []
    for key, left, right, descriptor, local_events in selected_edges:
        a, b = sorted((canonical_index[left], canonical_index[right]))
        canonical_edges.append((key, a, b, descriptor, local_events))
    canonical_edges.sort(key=lambda row: (row[0], row[1], row[2]))
    events: list[tuple[float, int, int, int]] = []
    for (_, original, _), local_events in zip(nodes, [node_events[row[1]] for row in nodes]):
        role = canonical_index[original]
        events.extend((phase, kind, role, -1) for phase, kind, _, _ in local_events)
    for _, a, b, _, local_events in canonical_edges:
        events.extend((phase, kind, a, b) for phase, kind, _, _ in local_events)
    events.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    events = events[:MAX_EVENTS]
    descriptor = assemble_action_descriptor(
        [row[2] for row in nodes[:MAX_SLOTS]],
        [(row[1], row[2], row[3]) for row in canonical_edges[:MAX_EDGES]],
        events,
    )
    norm = float(np.linalg.norm(descriptor))
    mechanically_valid = len(nodes) >= 2 and len(canonical_edges) >= 1 and bool(np.isfinite(descriptor).all()) and norm > 1e-8
    diagnostics = {
        **slot_diagnostics,
        "support_scale": scale,
        "node_count": len(nodes),
        "edge_count": len(canonical_edges),
        "event_count": len(events),
        "edge_degree_cap": 2,
        "descriptor_dimension": DESCRIPTOR_DIM,
        "descriptor_l2_norm": float(np.linalg.norm(descriptor)),
        "global_feature_mean_used": False,
        "absolute_appearance_serialized": False,
        "absolute_coordinates_serialized": False,
    }
    return GraphRepresentation(
        descriptor=np.ascontiguousarray(descriptor),
        descriptor_sha256=array_sha256(descriptor),
        slot_count=len(nodes), edge_count=len(canonical_edges), event_count=len(events),
        mechanically_valid=mechanically_valid, diagnostics=diagnostics,
    )


def graph_cosine(left: GraphRepresentation, right: GraphRepresentation) -> float:
    # Empty or one-slot graphs are not allowed to look spuriously identical.
    if not left.mechanically_valid or not right.mechanically_valid:
        return 0.0
    score = float(np.dot(left.descriptor, right.descriptor))
    return max(-1.0, min(1.0, score))


def require_distinct_processor_inputs(digests: Mapping[str, str]) -> None:
    if tuple(digests) != VIEW_ORDER or any(
        type(value) is not str or len(value) != 64 for value in digests.values()
    ) or len(set(digests.values())) != len(VIEW_ORDER):
        fail("five per-pair processor input tensors must have distinct SHA256 digests")


def assemble_action_descriptor(
    node_values: Sequence[np.ndarray],
    edge_values: Sequence[tuple[int, int, np.ndarray]],
    events: Sequence[tuple[float, int, int, int]],
) -> np.ndarray:
    if len(node_values) > MAX_SLOTS or len(edge_values) > MAX_EDGES or len(events) > MAX_EVENTS:
        raise ValueError("structured descriptor cardinality exceeds preregistration")
    raw = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    raw[0] = len(node_values) / MAX_SLOTS
    raw[1] = len(edge_values) / MAX_EDGES
    cursor = 2
    for descriptor in node_values:
        value = np.asarray(descriptor, dtype=np.float32)
        if value.shape != (NODE_DIM,) or not np.isfinite(value).all():
            raise ValueError("node descriptor differs")
        raw[cursor:cursor + NODE_DIM] = value
        cursor += NODE_DIM
    cursor = 2 + MAX_SLOTS * NODE_DIM
    for left, right, descriptor in edge_values:
        value = np.asarray(descriptor, dtype=np.float32)
        if not (0 <= left < right < MAX_SLOTS) or value.shape != (EDGE_BASE_DIM,) or not np.isfinite(value).all():
            raise ValueError("edge topology/descriptor differs")
        topology_and_lifecycle = np.concatenate((
            np.asarray(((left + 1) / MAX_SLOTS, (right + 1) / MAX_SLOTS), dtype=np.float32),
            value,
        ))
        raw[cursor:cursor + EDGE_DIM] = 1.25 * topology_and_lifecycle
        cursor += EDGE_DIM
    cursor = 2 + MAX_SLOTS * NODE_DIM + MAX_EDGES * EDGE_DIM
    for phase, kind, actor, patient in events:
        raw[cursor:cursor + EVENT_DIM] = 0.75 * np.asarray(
            (phase, kind / 7.0, (actor + 1) / 6.0, (patient + 1) / 6.0),
            dtype=np.float32,
        )
        cursor += EVENT_DIM
    norm = float(np.linalg.norm(raw))
    result = raw / norm if norm > 1e-8 else raw
    if result.shape != (DESCRIPTOR_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("final structured descriptor differs")
    return np.ascontiguousarray(result)


def evaluate_pair(graphs: Mapping[str, GraphRepresentation]) -> Mapping[str, Any]:
    if tuple(graphs) != VIEW_ORDER:
        fail("pair graph view order differs")
    reference = graphs["target_forward_reference"]
    positive = graph_cosine(reference, graphs["target_forward_eval"])
    reverse = graph_cosine(reference, graphs["target_reverse"])
    shuffle = graph_cosine(reference, graphs["target_deterministic_shuffle"])
    noop = graph_cosine(reference, graphs["source_noop"])
    negatives = {"target_reverse": reverse, "target_shuffle": shuffle, "source_noop": noop}
    hardest = max(negatives.values())
    margin = positive - hardest
    exact5_valid = all(graphs[name].mechanically_valid for name in VIEW_ORDER)
    gates = {
        "all_five_graphs_mechanically_valid": exact5_valid,
        "positive_cosine": positive >= THRESHOLDS["target_forward_reference_eval_cosine_min"],
        "positive_distance": 1.0 - positive <= THRESHOLDS["target_forward_reference_eval_distance_max"],
        "reference_above_each_control": all(positive > value for value in negatives.values()),
        "hardest_negative_margin": margin >= THRESHOLDS["minimum_margin_above_hardest_negative"],
        "reverse": reverse <= THRESHOLDS["target_reverse_cosine_max"],
        "shuffle": shuffle <= THRESHOLDS["target_shuffle_cosine_max"],
        "source_noop": noop <= THRESHOLDS["source_noop_cosine_max"],
        "reference_slots": reference.slot_count >= 2,
        "reference_edges": reference.edge_count >= 1,
        "eval_slots": graphs["target_forward_eval"].slot_count >= 2,
        "eval_edges": graphs["target_forward_eval"].edge_count >= 1,
    }
    return {
        "target_forward_reference_eval_cosine": positive,
        "target_forward_reference_eval_distance": 1.0 - positive,
        "negative_cosines": negatives,
        "hardest_negative_cosine": hardest,
        "margin_above_hardest_negative": margin,
        "gates": gates,
        "pair_pass": all(gates.values()),
    }


def validate_media_file(path: Path, expected_sha256: str, expected_size: int) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        fail(f"media path must be absolute and non-symlink: {path}")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file() or before.st_size != expected_size or file_sha256(resolved) != expected_sha256:
        fail(f"media envelope differs: {path}")
    return {
        "path": str(path), "realpath": str(resolved), "sha256": expected_sha256,
        "size_bytes": expected_size, "mode": stat.S_IMODE(before.st_mode),
        "device": before.st_dev, "inode": before.st_ino,
    }


def decode_rgb(path: Path) -> tuple[np.ndarray, Mapping[str, Any]]:
    import av

    frames = []
    pts = []
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            fail(f"media must contain one video stream: {path}")
        for frame in container.decode(streams[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
            pts.append(frame.pts)
    if not frames or any(frame.shape != frames[0].shape or frame.dtype != np.uint8 for frame in frames):
        fail(f"decoded RGB closure differs: {path}")
    if any(value is None for value in pts) or any(int(pts[i]) >= int(pts[i + 1]) for i in range(len(pts) - 1)):
        fail(f"decoded PTS order differs: {path}")
    array = np.ascontiguousarray(np.stack(frames))
    return array, {
        "decoded_frame_count": len(frames), "height": int(array.shape[1]),
        "width": int(array.shape[2]), "pts_sha256": object_sha256([int(value) for value in pts]),
        "decoded_rgb_sha256": array_sha256(array), "pyav_version": av.__version__,
    }


def required_model_closure(root: Path, v4c: Any) -> Mapping[str, Any]:
    resolved = root.resolve(strict=True)
    rows = []
    for name, (expected_sha, expected_size) in MODEL_FILES.items():
        path = resolved / name
        if path.is_symlink() or not path.is_file():
            fail(f"required V-JEPA2 file differs: {name}")
        row = {"name": name, "sha256": v4c.file_sha256(path), "size_bytes": path.stat().st_size}
        if row["sha256"] != expected_sha or row["size_bytes"] != expected_size:
            fail(f"required V-JEPA2 pin differs: {name}")
        rows.append(row)
    return {"root": str(resolved), "required_files": rows, "required_files_sha256": object_sha256(rows)}


class FrozenPatchBackbone:
    def __init__(self, model_root: Path, device: str):
        torch = importlib.import_module("torch")
        if device != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            fail("pilot requires exactly one logical GPU at cuda:0")
        if torch.cuda.get_device_name(0) != "AMD Instinct MI210":
            fail("pilot requires the audited AMD Instinct MI210 device")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            fail("backbone must load in explicit offline mode")
        if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
            fail("CUDA_VISIBLE_DEVICES must bind one explicit physical GPU")
        v4c = importlib.import_module(
            "methods.bernini_action_editing.extract_vjepa2_ordered_contextual_features_v4c"
        )
        if v4c.MODEL_REPO != MODEL_REPO or v4c.MODEL_REVISION != MODEL_REVISION:
            fail("v4c V-JEPA2 authority differs")
        self.v4c_binding_before = bind_regular_file(
            Path(v4c.__file__).resolve(strict=True), V4C_EXTRACTOR_SHA256,
            "reused v4c extractor",
        )
        self.model_closure_before = required_model_closure(model_root, v4c)
        self.module_closure_before = v4c.transformers_module_closure()
        transformers = importlib.import_module("transformers")
        self.processor = transformers.AutoVideoProcessor.from_pretrained(
            str(model_root), local_files_only=True,
        )
        self.model = transformers.AutoModel.from_pretrained(
            str(model_root), local_files_only=True, dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(torch.device(device)).requires_grad_(False).eval()
        self.torch = torch
        self.device = torch.device(device)
        config = self.model.config
        if (
            config.frames_per_clip != 64 or config.hidden_size != 1024
            or config.image_size != 256 or config.patch_size != 16
            or config.tubelet_size != 2 or config.num_hidden_layers != 24
            or any(parameter.requires_grad for parameter in self.model.parameters())
        ):
            fail("frozen V-JEPA2 configuration differs")
        self.forward_calls = 0
        self.processor_calls = 0
        self.v4c = v4c

    def process(self, frames: np.ndarray) -> tuple[Any, str]:
        from PIL import Image

        if frames.shape[0] != 64 or frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            fail("model RGB view differs")
        images = [Image.fromarray(frame, mode="RGB") for frame in frames]
        self.processor_calls += 1
        batch = self.processor(videos=images, return_tensors="pt")
        if set(batch) != {"pixel_values_videos"}:
            fail("processor output key closure differs")
        pixels = batch["pixel_values_videos"]
        if tuple(pixels.shape) != (1, 64, 3, 256, 256) or pixels.dtype != self.torch.float32:
            fail("processor pixel tensor geometry differs")
        return pixels, self.v4c.tensor_sha256(pixels)

    def forward(self, pixels: Any) -> np.ndarray:
        self.forward_calls += 1
        with self.torch.inference_mode():
            output = self.model(
                pixel_values_videos=pixels.to(self.device), skip_predictor=True,
                output_hidden_states=True, return_dict=True,
            )
        hidden_states = output.hidden_states
        if type(hidden_states) not in (tuple, list) or len(hidden_states) != 25:
            fail("V-JEPA2 output_hidden_states closure differs")
        selected = []
        for index in HIDDEN_INDICES:
            hidden = hidden_states[index]
            if tuple(hidden.shape) != (1, 8192, 1024):
                fail(f"V-JEPA2 hidden state {index} geometry differs")
            selected.append(hidden.detach().to(dtype=self.torch.float32, device="cpu")[0].reshape(32, 256, 1024).numpy())
        value = np.ascontiguousarray(np.stack(selected).astype(np.float32, copy=False))
        del hidden_states, output
        return value

    def final_closure(self) -> Mapping[str, Any]:
        after = required_model_closure(Path(self.model_closure_before["root"]), self.v4c)
        modules = self.v4c.transformers_module_closure()
        v4c_binding_after = bind_regular_file(
            Path(self.v4c.__file__).resolve(strict=True), V4C_EXTRACTOR_SHA256,
            "reused v4c extractor",
        )
        if (
            after != self.model_closure_before or modules != self.module_closure_before
            or v4c_binding_after != self.v4c_binding_before
        ):
            fail("frozen backbone closure changed during run")
        return {
            "model": after, "transformers": modules,
            "reused_v4c_extractor": v4c_binding_after, "unchanged": True,
        }


def pair_rgb_views(pair: PairRow) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    source, source_decode = decode_rgb(pair.source_path)
    target, target_decode = decode_rgb(pair.target_path)
    if source.shape[0] < 16 or target.shape[0] < 48:
        fail(f"decoded frame minimum failed for {pair.pair_id}")
    reference = reference_indices(int(target.shape[0]))
    evaluation = eval_indices(int(target.shape[0]))
    source_index = reference_indices(int(source.shape[0]))
    expected_geometry = {
        row["pair_id"]: row for row in sampling_geometry_registry()
    }[pair.pair_id]
    observed_geometry = {
        "pair_id": pair.pair_id,
        "source_decoded_frames": int(source.shape[0]),
        "target_decoded_frames": int(target.shape[0]),
        "source_unique_indices": len(set(source_index.tolist())),
        "target_reference_unique_indices": len(set(reference.tolist())),
        "target_eval_unique_indices": len(set(evaluation.tolist())),
        "source_indices_sha256": array_sha256(source_index),
        "target_reference_indices_sha256": array_sha256(reference),
        "target_eval_indices_sha256": array_sha256(evaluation),
    }
    if observed_geometry != expected_geometry:
        fail(f"pre-registered sampling geometry differs: {pair.pair_id}")
    if len(set(reference.tolist())) < 48 or len(set(evaluation.tolist())) < 48:
        fail(f"target views require at least48 unique decoded indices: {pair.pair_id}")
    if reference[0] != 0 or reference[-1] != target.shape[0] - 1 or evaluation[0] != 0 or evaluation[-1] != target.shape[0] - 1:
        fail(f"target sampling must cover both decoded endpoints: {pair.pair_id}")
    if len(set(source_index.tolist())) < 16:
        fail(f"source view requires at least 16 unique decoded indices: {pair.pair_id}")
    if np.array_equal(reference, evaluation):
        fail(f"target reference/eval index sequences alias: {pair.pair_id}")
    target_reference = np.ascontiguousarray(target[reference])
    target_eval = np.ascontiguousarray(target[evaluation])
    if array_sha256(target_reference) == array_sha256(target_eval):
        fail(f"target reference/eval RGB inputs alias: {pair.pair_id}")
    shuffle_index = shuffle64_indices(pair.pair_id)
    views = {
        "target_forward_reference": target_reference,
        "target_forward_eval": target_eval,
        "target_reverse": np.ascontiguousarray(target_reference[::-1]),
        "target_deterministic_shuffle": np.ascontiguousarray(target_reference[shuffle_index]),
        "source_noop": np.ascontiguousarray(source[source_index]),
    }
    if tuple(views) != VIEW_ORDER or len({array_sha256(view) for view in views.values()}) != 5:
        fail(f"five RGB control inputs must differ: {pair.pair_id}")
    receipt = {
        "source_decode": source_decode, "target_decode": target_decode,
        "source_unique_sample_indices": len(set(source_index.tolist())),
        "target_reference_indices_sha256": array_sha256(reference),
        "target_eval_indices_sha256": array_sha256(evaluation),
        "source_indices_sha256": array_sha256(source_index),
        "shuffle_permutation": list(shuffle_permutation(pair.pair_id)),
        "shuffle_permutation_sha256": object_sha256(list(shuffle_permutation(pair.pair_id))),
        "shuffle64_indices_sha256": array_sha256(shuffle_index),
        "view_rgb_sha256": {name: array_sha256(view) for name, view in views.items()},
        "all_five_rgb_inputs_distinct": True,
    }
    return views, receipt


def run(args: argparse.Namespace) -> int:
    start = time.time()
    manifest_value = load_json(Path(args.manifest), MANIFEST_FILE_SHA256)
    prereg_value = load_json(Path(args.prereg), PREREG_FILE_SHA256)
    pairs = validate_manifest(manifest_value)
    validate_prereg(prereg_value)
    wrapper_binding = bind_regular_file(Path(args.wrapper), LAUNCH_WRAPPER_SHA256, "launch wrapper")
    media = []
    for pair in pairs:
        media.extend((
            validate_media_file(pair.source_path, pair.source_sha256, pair.source_size),
            validate_media_file(pair.target_path, pair.target_sha256, pair.target_size),
        ))
    backbone = FrozenPatchBackbone(Path(args.model_root), args.device)
    pair_receipts = []
    validation_started_after_train_report = False
    train_reports = []
    validation_reports = []
    for pair in pairs:
        if pair.split == "validation" and not validation_started_after_train_report:
            validation_started_after_train_report = True
        views, input_receipt = pair_rgb_views(pair)
        processor_before_pair = backbone.processor_calls
        forward_before_pair = backbone.forward_calls
        graphs: dict[str, GraphRepresentation] = {}
        view_receipts = {}
        processor_digests: dict[str, str] = {}
        for name, frames in views.items():
            pixels, processor_digest = backbone.process(frames)
            processor_digests[name] = processor_digest
            hidden = backbone.forward(pixels)
            projected = project_hidden_layers(hidden)
            graph = build_graph_representation(projected)
            graphs[name] = graph
            view_receipts[name] = {
                "processor_input_sha256": processor_digest,
                "selected_hidden_shape": list(hidden.shape),
                "projected_patch_shape": list(projected.shape),
                "projected_patch_sha256": array_sha256(projected),
                "graph_descriptor_sha256": graph.descriptor_sha256,
                "graph_mechanically_valid": graph.mechanically_valid,
                "graph_diagnostics": graph.diagnostics,
                "raw_rgb_hidden_or_descriptor_values_exported": False,
            }
            hidden.fill(np.float32(0.0))
            projected.fill(np.float32(0.0))
            del hidden, projected, pixels
        require_distinct_processor_inputs(processor_digests)
        if (
            backbone.processor_calls - processor_before_pair != 5
            or backbone.forward_calls - forward_before_pair != 5
        ):
            fail(f"pair must execute exact five processor/model forwards: {pair.pair_id}")
        input_receipt["processor_input_sha256"] = processor_digests
        input_receipt["all_five_processor_inputs_distinct"] = True
        input_receipt["processor_calls"] = 5
        input_receipt["model_forward_calls"] = 5
        metrics = evaluate_pair(graphs)
        row = {
            "ordinal": pair.ordinal, "pair_id": pair.pair_id, "split": pair.split,
            "input": input_receipt, "views": view_receipts, "control_metrics": metrics,
        }
        pair_receipts.append(row)
        (train_reports if pair.split == "train" else validation_reports).append(metrics)
    if backbone.forward_calls != 60 or backbone.processor_calls != 60 or len(pair_receipts) != 12:
        fail("exact pair/forward count differs")
    train_pass = sum(bool(row["pair_pass"]) for row in train_reports)
    validation_pass = sum(bool(row["pair_pass"]) for row in validation_reports)
    forward_above = sum(bool(row["gates"]["reference_above_each_control"]) for row in train_reports + validation_reports)
    component_admitted = (
        train_pass >= THRESHOLDS["train_pair_pass_min"]
        and validation_pass >= THRESHOLDS["validation_pair_pass_min"]
        and forward_above >= THRESHOLDS["all_twelve_forward_above_each_control_min"]
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "V5_TARGET_MIDDLE_GRAPH_COMPONENT_ADMITTED" if component_admitted else "V5_TARGET_MIDDLE_GRAPH_COMPONENT_REJECTED",
        "component_pilot_admitted": component_admitted,
        "stable_transferable_action_representation_established": False,
        "self_generated_anchor_alignment_established": False,
        "generator_quality_preserved_established": False,
        "generator_connection_authorized": False,
        "manifest": {"path": args.manifest, "file_sha256": MANIFEST_FILE_SHA256, "self_sha256": MANIFEST_SELF_SHA256},
        "preregistration": {"path": args.prereg, "file_sha256": PREREG_FILE_SHA256, "self_sha256": PREREG_SELF_SHA256},
        "implementation": {
            "runner": {"path": str(Path(__file__).resolve(strict=True)), "sha256": file_sha256(Path(__file__).resolve(strict=True))},
            "launch_wrapper": wrapper_binding,
            "reused_v4c_extractor_sha256": V4C_EXTRACTOR_SHA256,
        },
        "media_closure_sha256": object_sha256(media),
        "media_file_count": len(media),
        "fixed_thresholds": THRESHOLDS,
        "summary": {
            "train_pair_pass": train_pass, "train_pair_total": 8,
            "validation_pair_pass": validation_pass, "validation_pair_total": 4,
            "all_twelve_forward_above_each_control": forward_above,
            "model_forward_calls": backbone.forward_calls,
            "processor_calls": backbone.processor_calls,
        },
        "optimization_boundary": {
            "optimizer_created": False, "loss_backward_calls": 0,
            "parameter_updates": 0, "train_split_parameter_fitting": False,
            "validation_parameter_fitting": False,
            "validation_threshold_selection": False,
            "validation_started_after_train_report_without_state_change": validation_started_after_train_report,
            "generator_loaded": False, "generator_forward_calls": 0,
            "generator_updates": 0, "dataset_written": False,
        },
        "backbone_final_closure": backbone.final_closure(),
        "pairs": pair_receipts,
        "elapsed_seconds": time.time() - start,
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("receipt path must be new, absolute, and non-symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(output, 0o444)
    print(json.dumps({"status": receipt["status"], "receipt": str(output), "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


def contract(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = load_json(Path(args.manifest), MANIFEST_FILE_SHA256)
    prereg = load_json(Path(args.prereg), PREREG_FILE_SHA256)
    pairs = validate_manifest(manifest)
    validate_prereg(prereg)
    wrapper_binding = bind_regular_file(Path(args.wrapper), LAUNCH_WRAPPER_SHA256, "launch wrapper")
    return {
        "schema_version": "bernini-target-middle-object-graph-teacher-pilot-contract-v5",
        "experiment_id": EXPERIMENT_ID,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "manifest_self_sha256": MANIFEST_SELF_SHA256,
        "prereg_file_sha256": PREREG_FILE_SHA256,
        "prereg_self_sha256": PREREG_SELF_SHA256,
        "implementation_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "launch_wrapper": wrapper_binding,
        "reused_v4c_extractor_file_sha256": V4C_EXTRACTOR_SHA256,
        "pair_ids": [row.pair_id for row in pairs],
        "splits": [row.split for row in pairs],
        "view_order": list(VIEW_ORDER),
        "model_forward_calls": 60,
        "hidden_indices": list(HIDDEN_INDICES),
        "channel_plan_sha256": CHANNEL_PLAN_SHA256,
        "descriptor_dimension": DESCRIPTOR_DIM,
        "thresholds": THRESHOLDS,
        "zero_training": True,
        "generator_connection_authorized": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--print-contract", action="store_true")
    args = parser.parse_args(argv)
    if args.print_contract == (args.output is not None):
        parser.error("choose exactly one of --print-contract or --output")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_contract:
        print(json.dumps(contract(args), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
