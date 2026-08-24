#!/usr/bin/env python3
"""Frozen-DINO temporal representations for the terminal r8 exact60 bank.

The only registered labels are ``incomplete``, ``camera_only``, and
``appearance_only``.  Fit rows define fixed label centroids and fixed
same-cell pairwise-difference prototypes; confirmation rows are evaluated
without extending them.  The diagnostic deliberately has no action/noop
interpretation, threshold, selection, ranking, event, identity, training, or
optimizer authority.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
ANCHOR_SOURCE_NAME = "diagnose_saic_anchor_dinov2_temporal_representation_v1.py"
ANCHOR_SOURCE_SHA256 = "3a1705dfb44522357bd9f136a40e65eefa468b70c771d1207edf4ff9a47ed2d1"
CYCLIC_SOURCE_NAME = "diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1.py"
CYCLIC_SOURCE_SHA256 = "2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e"
for _name, _sha in (
    (ANCHOR_SOURCE_NAME, ANCHOR_SOURCE_SHA256),
    (CYCLIC_SOURCE_NAME, CYCLIC_SOURCE_SHA256),
):
    _path = METHOD_ROOT / _name
    if (
        not _path.is_file()
        or _path.is_symlink()
        or hashlib.sha256(_path.read_bytes()).hexdigest() != _sha
    ):
        raise RuntimeError(f"pinned temporal-representation dependency differs: {_name}")

import diagnose_saic_anchor_dinov2_temporal_representation_v1 as anchor  # noqa: E402
import diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1 as cyclic  # noqa: E402


SCHEMA_VERSION = "bernini-saic-r8-exact60-anchor-dinov2-temporal-representation-v1"
BRANCHES = ("incomplete", "camera_only", "appearance_only")
PAIRWISE_CONTRASTS = (
    ("incomplete", "camera_only"),
    ("incomplete", "appearance_only"),
    ("camera_only", "appearance_only"),
)
FAMILIES = ("dog", "human")
REPRESENTATION_NAMES = (
    "appearance_mean",
    "centered_trajectory",
    "dense_lag_profile",
    "dense_speed_profile",
    "endpoint_arrow",
    "speed_profile",
    "temporal_self_similarity",
    "velocity_trajectory",
)
EXPECTED_CANDIDATE_COUNT = 60
EXPECTED_CELL_COUNT = 20
EXPECTED_BRANCH_COUNT = 20
EXPECTED_FIT_COUNT = 24
EXPECTED_CONFIRMATION_COUNT = 36
EXPECTED_INPUT_MANIFEST_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/diagnostics/"
    "allocation-134939-r8-sourcebound60-dinov2-full-28396417-r1/"
    "input-manifest.json"
)
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "28ff1e40f4dd314548616050013afdfb5e2a2a768aba9f0cbd4f00c9f6718c62"
)
EXPECTED_INPUT_RECEIPT_DIGEST = (
    "a4ad17a09e46d549089356ecf86e21d5d8a6da2f41aaa92d218b058a9e28f378"
)
EXPECTED_TERMINAL_EVIDENCE_SHA256 = (
    "07a6ec7ccbe165d89aa8757985537ef18d62eea5d08e245e452b607dee5bd29a"
)
EXPECTED_TERMINAL_EVIDENCE_RECEIPT_DIGEST = (
    "a8fe672840d597445a2164660a38bdfeb4fa51ccfbc3b822c3af8adb6d6519e5"
)
EXPECTED_MASTER_RECEIPT_SHA256 = (
    "c5528a08fa976c0dbfb16984a35df3169c2d013a73fabd982ad45f45d5defc61"
)
EXPECTED_MASTER_RECEIPT_DIGEST = (
    "8d28c170f5c8fdc5e76bdfb55bb89a5a819f02beb483c005f87d6898c5d8ae33"
)
EXPECTED_VISUAL_SCORER_SHA256 = cyclic.core.EXPECTED_VISUAL_SCORER_SHA256
EXPECTED_VISUAL_CONTRACT_SHA256 = cyclic.core.EXPECTED_VISUAL_CONTRACT_SHA256
EXPECTED_EVALUATOR_SPEC_SHA256 = cyclic.core.EXPECTED_EVALUATOR_SPEC_SHA256

AUTHORITY = {
    "diagnostic_only": True,
    "raw_proxy_evidence_only": True,
    "identity_authority": False,
    "identity_preservation_verified": False,
    "event_authority": False,
    "event_verified": False,
    "scientific_claim_authorized": False,
    "representation_selection_authorized": False,
    "selection_authorized": False,
    "ranking_authorized": False,
    "training_target_authorized": False,
    "optimizer_or_parameter_update_authorized": False,
}


class R8TemporalRepresentationError(RuntimeError):
    """Raised before incomplete or authority-expanding evidence is written."""


def fail(message: str) -> NoReturn:
    raise R8TemporalRepresentationError(message)


def file_sha256(path: str | Path) -> str:
    return cyclic.core.core.file_sha256(path)


def object_sha256(value: Any) -> str:
    return cyclic.core.core.object_sha256(value)


def _strict_number(value: Any, *, label: str) -> float:
    if type(value) not in {int, float}:
        fail(f"{label} is not a strict numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} is non-finite")
    return result


def _verify_self(expected_sha256: str) -> str:
    actual = file_sha256(Path(__file__).resolve())
    if actual != cyclic.core.core._sha256(
        expected_sha256, label="r8 temporal-representation source SHA-256"
    ):
        fail("r8 temporal-representation source SHA-256 differs")
    return actual


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    if (
        str(Path(args.input_manifest)) != EXPECTED_INPUT_MANIFEST_PATH
        or args.expected_input_manifest_sha256 != EXPECTED_INPUT_MANIFEST_SHA256
        or str(Path(args.terminal_evidence)) != cyclic.EXPECTED_TERMINAL_EVIDENCE_PATH
    ):
        fail("r8 cyclic manifest/terminal lexical identity differs")
    cyclic._install_specialization()
    terminal = cyclic._validate_terminal_evidence(args.terminal_evidence)
    terminal_value, terminal_sha = cyclic._load_canonical_receipt(
        args.terminal_evidence, label="r8 temporal terminal evidence"
    )
    if (
        terminal_sha != EXPECTED_TERMINAL_EVIDENCE_SHA256
        or terminal_value.get("receipt_digest")
        != EXPECTED_TERMINAL_EVIDENCE_RECEIPT_DIGEST
    ):
        fail("r8 terminal evidence raw/digest hard pin differs")
    manifest, manifest_sha = cyclic.load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=CYCLIC_SOURCE_SHA256,
    )
    if (
        manifest_sha != EXPECTED_INPUT_MANIFEST_SHA256
        or manifest.get("receipt_digest") != EXPECTED_INPUT_RECEIPT_DIGEST
        or manifest.get("attempt_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("world_size") != 8
        or manifest.get("authority") != cyclic.AUTHORITY_CLOSURE
    ):
        fail("r8 cyclic exact60 input closure differs")
    master, master_sha = cyclic._load_canonical_receipt(
        cyclic.EXPECTED_MASTER_RECEIPT_PATH, label="r8 temporal master receipt"
    )
    if (
        master_sha != EXPECTED_MASTER_RECEIPT_SHA256
        or master.get("receipt_digest") != EXPECTED_MASTER_RECEIPT_DIGEST
        or master_sha != terminal["master_receipt_sha256"]
    ):
        fail("r8 terminal/master raw/digest binding differs")
    return manifest, manifest_sha, terminal, _validate_master(master, master_sha, manifest)


def _validate_master(
    master: Mapping[str, Any], master_sha: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    proofs = master.get("same_seed_official_gaussian_proofs")
    attempts = master.get("attempts")
    manifest_attempts = manifest.get("attempts")
    if (
        master.get("schema_version") != cyclic.MASTER_SCHEMA
        or master.get("receipt_digest") is None
        or master.get("attempt_count") != EXPECTED_CANDIDATE_COUNT
        or master.get("seed_cell_count") != EXPECTED_CELL_COUNT
        or master.get("branch_order") != list(BRANCHES)
        or not isinstance(proofs, list)
        or len(proofs) != EXPECTED_CELL_COUNT
        or not isinstance(attempts, list)
        or len(attempts) != EXPECTED_CANDIDATE_COUNT
        or not isinstance(manifest_attempts, list)
        or len(manifest_attempts) != EXPECTED_CANDIDATE_COUNT
    ):
        fail("r8 master branch/Gaussian closure differs")
    proof_cells: set[tuple[str, int]] = set()
    proof_digests: list[dict[str, Any]] = []
    for proof in proofs:
        if not isinstance(proof, Mapping) or set(proof) != {
            "iid", "seed", "branch_order",
            "official_gaussian_tensor_values_byte_equal",
            "official_gaussian_identity_digest",
        }:
            fail("r8 same-cell Gaussian proof fields differ")
        iid, seed = proof.get("iid"), proof.get("seed")
        digest = proof.get("official_gaussian_identity_digest")
        if (
            not isinstance(iid, str)
            or type(seed) is not int
            or proof.get("branch_order") != list(BRANCHES)
            or proof.get("official_gaussian_tensor_values_byte_equal") is not True
            or cyclic.core.core._sha256(digest, label="Gaussian identity digest") == "0" * 64
        ):
            fail("r8 same-cell Gaussian proof differs")
        proof_cells.add((iid, seed))
        proof_digests.append({"iid": iid, "seed": seed, "identity_digest": digest})
    manifest_cells = {
        (row.get("iid"), row.get("seed"))
        for row in manifest_attempts if isinstance(row, Mapping)
    }
    master_by_id = {row.get("candidate_id"): row for row in attempts if isinstance(row, Mapping)}
    if len(proof_cells) != EXPECTED_CELL_COUNT or proof_cells != manifest_cells:
        fail("r8 Gaussian proof/candidate cell join differs")
    for row in manifest_attempts:
        old = master_by_id.get(row.get("candidate_id"))
        if (
            not isinstance(old, Mapping)
            or old.get("iid") != row.get("iid")
            or old.get("seed") != row.get("seed")
            or old.get("branch") != row.get("branch")
            or old.get("receipt_sha256") != row.get("receipt_sha256")
            or old.get("mp4_sha256") != row.get("mp4_sha256")
        ):
            fail("r8 master/cyclic candidate-media join differs")
    return {
        "path": cyclic.EXPECTED_MASTER_RECEIPT_PATH,
        "raw_sha256": master_sha,
        "receipt_digest": master["receipt_digest"],
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "cell_count": EXPECTED_CELL_COUNT,
        "branch_order": list(BRANCHES),
        "all20_same_cell_official_gaussians_byte_equal": True,
        "gaussian_proof_projection_sha256": object_sha256(proof_digests),
    }


def _validate_manifest_semantics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != EXPECTED_CANDIDATE_COUNT:
        fail("r8 attempt inventory differs")
    branch_counts = Counter(row.get("branch") for row in attempts if isinstance(row, Mapping))
    split_counts = Counter(row.get("analysis_split") for row in attempts if isinstance(row, Mapping))
    cells: dict[tuple[str, int], list[str]] = {}
    source_bindings: dict[str, Mapping[str, Any]] = {}
    for row in attempts:
        if not isinstance(row, Mapping):
            fail("r8 attempt row differs")
        cell = (row.get("iid"), row.get("seed"))
        if not isinstance(cell[0], str) or type(cell[1]) is not int:
            fail("r8 attempt cell identity differs")
        cells.setdefault(cell, []).append(row.get("branch"))
        correct = row.get("correct_source")
        if not isinstance(correct, Mapping) or correct.get("iid") != cell[0]:
            fail("r8 candidate/source IID binding differs")
        previous = source_bindings.setdefault(cell[0], correct)
        if previous != correct:
            fail("r8 source binding differs within IID")
    if (
        branch_counts != Counter({branch: EXPECTED_BRANCH_COUNT for branch in BRANCHES})
        or split_counts != Counter({"fit": EXPECTED_FIT_COUNT, "confirmation": EXPECTED_CONFIRMATION_COUNT})
        or len(cells) != EXPECTED_CELL_COUNT
        or any(Counter(branches) != Counter(BRANCHES) for branches in cells.values())
        or len(source_bindings) != 8
    ):
        fail("r8 exact60 branch/split/same-cell closure differs")
    return {
        "branch_counts": dict(branch_counts),
        "split_counts": dict(split_counts),
        "same_iid_seed_cell_count": len(cells),
        "each_cell_contains_registered_three_branch_labels_exactly_once": True,
        "source_count": len(source_bindings),
    }


def _rehash_live_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    candidate_hashes: list[dict[str, str]] = []
    for row in manifest["attempts"]:
        for path_key, sha_key, label in (
            ("receipt_path", "receipt_sha256", "generation receipt"),
            ("native_receipt_path", "native_receipt_sha256", "native receipt"),
            ("candidate_envelope_path", "candidate_envelope_sha256", "candidate envelope"),
            ("mp4_path", "mp4_sha256", "candidate MP4"),
        ):
            path = cyclic.core.core._plain_file(row[path_key], label=label)
            if file_sha256(path) != row[sha_key]:
                fail(f"live {label} SHA-256 differs")
        source = row["correct_source"]
        iid = source["iid"]
        path = cyclic.core.core._plain_file(source["source_video"], label="bound source video")
        digest = file_sha256(path)
        if digest != source["source_video_sha256"]:
            fail("live source video SHA-256 differs")
        if iid in source_hashes and source_hashes[iid] != digest:
            fail("source video hash differs across candidate bindings")
        source_hashes[iid] = digest
        candidate_hashes.append({
            "candidate_id": row["candidate_id"],
            "receipt_sha256": row["receipt_sha256"],
            "mp4_sha256": row["mp4_sha256"],
        })
    return {
        "candidate_count": len(candidate_hashes),
        "source_count": len(source_hashes),
        "all_candidate_receipt_native_envelope_mp4_files_rehashed": True,
        "all8_bound_source_videos_rehashed": True,
        "candidate_binding_projection_sha256": object_sha256(candidate_hashes),
        "source_sha256_by_iid": dict(sorted(source_hashes.items())),
    }


def _load_visual(args: argparse.Namespace) -> tuple[Mapping[str, Any], Any, Any, Any, dict[str, Any]]:
    if (
        str(Path(args.visual_checkpoint)) != cyclic.EXPECTED_CHECKPOINT_ROOT
        or str(Path(args.visual_checkpoint_manifest)) != cyclic.EXPECTED_CHECKPOINT_MANIFEST_PATH
        or str(Path(args.evaluator_spec)) != cyclic.EXPECTED_EVALUATOR_SPEC_PATH
        or str(Path(args.visual_scorer_source)) != cyclic.EXPECTED_VISUAL_SCORER_PATH
        or str(Path(args.visual_contract_source)) != cyclic.EXPECTED_VISUAL_CONTRACT_PATH
        or args.expected_evaluator_spec_sha256 != EXPECTED_EVALUATOR_SPEC_SHA256
        or args.expected_visual_scorer_sha256 != EXPECTED_VISUAL_SCORER_SHA256
        or args.expected_visual_contract_sha256 != EXPECTED_VISUAL_CONTRACT_SHA256
        or file_sha256(args.visual_checkpoint_manifest) != cyclic.EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        fail("frozen visual evaluator path/SHA closure differs")
    evaluator, checkpoint = cyclic.core.core._load_evaluator(args)
    device = cyclic.core.core._configure_device()
    try:
        model, loading_counts = evaluator["scorer"].load_frozen_model(checkpoint, device=device)
    except Exception as error:
        raise R8TemporalRepresentationError("cannot load frozen DINO model") from error
    checkpoint["root"] = str(checkpoint["root"])
    checkpoint["loading_counts"] = loading_counts
    checkpoint["frozen_eval"] = True
    checkpoint["trainable_parameter_tensors"] = 0
    checked = cyclic._validate_visual_evaluator(checkpoint, rank=0)
    return evaluator, model, device, evaluator["spec"], checked


def _measure(
    row: Mapping[str, Any], *, evaluator: Mapping[str, Any], model: Any,
    device: Any, spec: Mapping[str, Any]
) -> dict[str, Any]:
    if file_sha256(row["receipt_path"]) != row["receipt_sha256"]:
        fail("generation receipt changed after live binding audit")
    scorer = evaluator["scorer"]
    frames, decode = scorer.decode_exact81_rgb(
        row["mp4_path"], expected_sha256=row["mp4_sha256"]
    )
    _, pixels = scorer.preprocess_selected_rgb(frames, evaluator["processor"])
    global_feature, dense_feature, features = scorer.extract_features(
        model, pixels, device=device,
        num_register_tokens=spec["model"]["num_register_tokens"],
        evaluation_image_size=spec["model"]["preprocessor_golden_output_shape"][-1],
        patch_size=spec["model"]["patch_size"],
    )
    cyclic._validate_decode_evidence(
        decode, expected_artifact_sha256=row["mp4_sha256"], label="temporal candidate"
    )
    cyclic._validate_feature_evidence(features, label="temporal candidate")
    representations = anchor.temporal_representations(
        global_feature.numpy(), dense_feature.numpy()
    )
    if tuple(sorted(representations)) != REPRESENTATION_NAMES:
        fail("frozen temporal representation name closure differs")
    return {
        "candidate_id": row["candidate_id"],
        "iid": row["iid"],
        "seed": row["seed"],
        "branch": row["branch"],
        "actor_family": row["actor_family"],
        "analysis_split": row["analysis_split"],
        "candidate_binding": dict(row),
        "decode": decode,
        "features": features,
        "representation_sha256": {
            name: anchor.array_sha256(value)
            for name, value in representations.items()
        },
        "representations": representations,
        "authority": dict(AUTHORITY),
    }


def branch_label_centroid_diagnostic(
    rows: Sequence[Mapping[str, Any]], representation: str
) -> dict[str, Any]:
    fit = [row for row in rows if row["analysis_split"] == "fit"]
    confirmation = [row for row in rows if row["analysis_split"] == "confirmation"]
    prototypes = {
        (family, branch): anchor.centroid([
            row["representations"][representation]
            for row in fit
            if row["actor_family"] == family and row["branch"] == branch
        ])
        for family in FAMILIES for branch in BRANCHES
    }
    predictions = []
    confusion = {truth: Counter() for truth in BRANCHES}
    for row in confirmation:
        scores = {
            branch: anchor.cosine(
                row["representations"][representation],
                prototypes[(row["actor_family"], branch)],
            )
            for branch in BRANCHES
        }
        predicted = max(BRANCHES, key=lambda branch: scores[branch])
        confusion[row["branch"]][predicted] += 1
        predictions.append({
            "candidate_id": row["candidate_id"],
            "truth_label": row["branch"],
            "predicted_label": predicted,
            "label_match": predicted == row["branch"],
            "raw_cosines": scores,
        })
    return {
        "label_definition": "registered_r8_negative_branch_label_only",
        "fit_row_count": len(fit),
        "confirmation_row_count": len(confirmation),
        "actor_family_conditioned": True,
        "confirmation_rows_never_extend_centroids": True,
        "ties_resolved_by_registered_branch_order": True,
        "descriptive_label_match_count": sum(row["label_match"] for row in predictions),
        "descriptive_label_match_rate": sum(row["label_match"] for row in predictions) / len(predictions),
        "confusion": {
            truth: {branch: confusion[truth][branch] for branch in BRANCHES}
            for truth in BRANCHES
        },
        "prototype_sha256": {
            f"{family}:{branch}": anchor.array_sha256(prototypes[(family, branch)])
            for family in FAMILIES for branch in BRANCHES
        },
        "predictions": predictions,
    }


def nearest_neighbor_diagnostic(
    rows: Sequence[Mapping[str, Any]], representation: str
) -> dict[str, Any]:
    results = []
    for row in rows:
        others = [other for other in rows if other["candidate_id"] != row["candidate_id"]]
        best = max(
            others,
            key=lambda other: anchor.cosine(
                row["representations"][representation],
                other["representations"][representation],
            ),
        )
        results.append({
            "candidate_id": row["candidate_id"],
            "neighbor_id": best["candidate_id"],
            "raw_cosine": anchor.cosine(
                row["representations"][representation],
                best["representations"][representation],
            ),
            "same_iid": row["iid"] == best["iid"],
            "same_seed_cell": (row["iid"], row["seed"]) == (best["iid"], best["seed"]),
            "same_branch_label": row["branch"] == best["branch"],
            "same_actor_family": row["actor_family"] == best["actor_family"],
        })
    return {
        "row_count": len(results),
        "same_iid_top1_rate": sum(row["same_iid"] for row in results) / len(results),
        "same_seed_cell_top1_rate": sum(row["same_seed_cell"] for row in results) / len(results),
        "same_branch_label_top1_rate": sum(row["same_branch_label"] for row in results) / len(results),
        "same_actor_family_top1_rate": sum(row["same_actor_family"] for row in results) / len(results),
        "neighbors": results,
    }


def pairwise_branch_label_difference_diagnostic(
    rows: Sequence[Mapping[str, Any]], representation: str
) -> dict[str, Any]:
    indexed = {(row["iid"], row["seed"], row["branch"]): row for row in rows}
    if len(indexed) != EXPECTED_CANDIDATE_COUNT:
        fail("pairwise branch-label index differs")
    cells = sorted({(row["iid"], row["seed"]) for row in rows})
    output: dict[str, Any] = {}
    for left, right in PAIRWISE_CONTRASTS:
        vectors = []
        for iid, seed in cells:
            left_row, right_row = indexed[(iid, seed, left)], indexed[(iid, seed, right)]
            if (
                left_row["analysis_split"] != right_row["analysis_split"]
                or left_row["actor_family"] != right_row["actor_family"]
            ):
                fail("same-cell branch-label metadata differs")
            vectors.append({
                "iid": iid,
                "seed": seed,
                "actor_family": left_row["actor_family"],
                "analysis_split": left_row["analysis_split"],
                "left_candidate_id": left_row["candidate_id"],
                "right_candidate_id": right_row["candidate_id"],
                "vector": anchor.unit(
                    left_row["representations"][representation]
                    - right_row["representations"][representation]
                ),
            })
        prototypes = {
            family: anchor.centroid([
                row["vector"] for row in vectors
                if row["analysis_split"] == "fit" and row["actor_family"] == family
            ])
            for family in FAMILIES
        }
        confirmation = []
        for row in vectors:
            if row["analysis_split"] != "confirmation":
                continue
            value = anchor.cosine(row["vector"], prototypes[row["actor_family"]])
            confirmation.append({
                "iid": row["iid"],
                "seed": row["seed"],
                "actor_family": row["actor_family"],
                "left_candidate_id": row["left_candidate_id"],
                "right_candidate_id": row["right_candidate_id"],
                "raw_cosine_to_fit_prototype": value,
            })
        key = f"{left}_minus_{right}"
        values = [row["raw_cosine_to_fit_prototype"] for row in confirmation]
        output[key] = {
            "left_branch_label": left,
            "right_branch_label": right,
            "meaning": "unit_difference_of_registered_branch_label_representations_only",
            "same_iid_seed_gaussian_cell_required": True,
            "fit_vector_count": sum(row["analysis_split"] == "fit" for row in vectors),
            "confirmation_vector_count": len(confirmation),
            "confirmation_vectors_never_extend_prototype": True,
            "actor_family_conditioned": True,
            "prototype_sha256": {
                family: anchor.array_sha256(prototypes[family]) for family in FAMILIES
            },
            "confirmation_raw_cosine_mean": float(np.mean(values)),
            "confirmation_raw_cosine_minimum": min(values),
            "confirmation": confirmation,
        }
    return {
        "registered_pair_order": [list(pair) for pair in PAIRWISE_CONTRASTS],
        "no_action_or_noop_semantics": True,
        "contrasts": output,
    }


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "representations"}


def run(args: argparse.Namespace) -> int:
    source_sha = _verify_self(args.expected_source_sha256)
    output_root = Path(args.output_root)
    if (
        not output_root.is_absolute()
        or output_root == Path("/")
        or output_root.exists()
        or output_root.is_symlink()
    ):
        fail("output root must be fresh, absolute, and non-root")
    manifest, manifest_sha, terminal, master = _load_inputs(args)
    semantic = _validate_manifest_semantics(manifest)
    live = _rehash_live_bindings(manifest)
    evaluator, model, device, spec, visual = _load_visual(args)
    runtime_rows = []
    for ordinal, binding in enumerate(manifest["attempts"]):
        row = _measure(
            binding, evaluator=evaluator, model=model, device=device, spec=spec
        )
        runtime_rows.append(row)
        print({"ordinal": ordinal, "candidate_id": row["candidate_id"]}, flush=True)
    if len(runtime_rows) != EXPECTED_CANDIDATE_COUNT:
        fail("runtime exact60 coverage differs")
    diagnostics = {
        name: {
            "dimension": int(runtime_rows[0]["representations"][name].size),
            "fit_only_branch_label_centroids": branch_label_centroid_diagnostic(runtime_rows, name),
            "all_row_nearest_neighbor": nearest_neighbor_diagnostic(runtime_rows, name),
            "fit_only_same_cell_pairwise_branch_label_differences":
                pairwise_branch_label_difference_diagnostic(runtime_rows, name),
        }
        for name in REPRESENTATION_NAMES
    }
    public_rows = [_public_row(row) for row in runtime_rows]
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_r8_temporal_representation_diagnostic_no_authority",
        "diagnostic_source_sha256": source_sha,
        "anchor_algorithm_source": {
            "path": str((METHOD_ROOT / ANCHOR_SOURCE_NAME).resolve(strict=True)),
            "sha256": ANCHOR_SOURCE_SHA256,
            "reused_function": "temporal_representations",
        },
        "cyclic_r8_validator_source": {
            "path": str((METHOD_ROOT / CYCLIC_SOURCE_NAME).resolve(strict=True)),
            "sha256": CYCLIC_SOURCE_SHA256,
        },
        "input_manifest": {
            "path": EXPECTED_INPUT_MANIFEST_PATH,
            "raw_sha256": manifest_sha,
            "receipt_digest": EXPECTED_INPUT_RECEIPT_DIGEST,
        },
        "terminal_evidence": terminal,
        "master_semantic_closure": master,
        "live_artifact_binding_closure": live,
        "registered_branch_semantics": {
            "branch_order": list(BRANCHES),
            "branch_meanings": {
                "incomplete": "desired_action_progress_without_terminal_state",
                "camera_only": "camera_motion_without_desired_actor_transition",
                "appearance_only": "identity_appearance_change_without_desired_actor_transition",
            },
            "forward_reverse_noop_mapping_performed": False,
            "action_minus_noop_computation_performed": False,
            "typed_action_subspace_computation_performed": False,
        },
        "bank_semantic_closure": semantic,
        "candidate_count": len(public_rows),
        "candidate_rows": public_rows,
        "candidate_rows_digest": object_sha256(public_rows),
        "representation_names": list(REPRESENTATION_NAMES),
        "diagnostics": diagnostics,
        "frozen_visual_model": visual,
        "split_policy": {
            "fit_rows_define_actor_family_conditioned_label_centroids": True,
            "fit_rows_define_actor_family_conditioned_pairwise_difference_prototypes": True,
            "confirmation_rows_contribute_to_fit": False,
            "all_registered_exact60_rows_consumed": True,
            "row_or_seed_selection_performed": False,
        },
        "limitations": {
            "dinov2_is_image_not_video_pretraining": True,
            "global_and_dense_features_are_proxy_representations": True,
            "branch_labels_are_generation_contract_labels_not_verified_events": True,
            "descriptive_top1_or_cosine_values_grant_no_ranking_authority": True,
            "no_human_event_labels_consumed": True,
            "no_threshold_registered": True,
            "decoded_video_review_still_required": True,
        },
        "authority": dict(AUTHORITY),
    }
    output_root.mkdir(mode=0o700)
    cyclic.core.core._write_create_only(
        output_root / "aggregate-receipt.json",
        {**unsigned, "receipt_digest": object_sha256(unsigned)},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--terminal-evidence", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--visual-checkpoint", required=True)
    parser.add_argument("--visual-checkpoint-manifest", required=True)
    parser.add_argument("--evaluator-spec", required=True)
    parser.add_argument("--expected-evaluator-spec-sha256", required=True)
    parser.add_argument("--visual-scorer-source", required=True)
    parser.add_argument("--expected-visual-scorer-sha256", required=True)
    parser.add_argument("--visual-contract-source", required=True)
    parser.add_argument("--expected-visual-contract-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
