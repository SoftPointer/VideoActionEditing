from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import validate_pair_v5_t2v_calibration_mainline_v3 as mainline  # noqa: E402


T2V_SPEC = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    return mainline.file_sha256(path)


class PairV5T2VCalibrationMainlineV3Tests(unittest.TestCase):
    def test_formal_mainline_has_no_v3_compatibility_entrypoint(self) -> None:
        self.assertFalse(hasattr(mainline, "score_v3_compat"))
        self.assertTrue(mainline.scorer.SCORE_RECEIPT_SCHEMA.endswith("-v4"))
        self.assertTrue(
            mainline._score_path(Path("/scores"), "sp4-a", "candidate")
            .name.endswith("-v4.json")
        )
        option_strings = {
            option
            for action in mainline.build_parser()._actions
            for option in action.option_strings
        }
        self.assertFalse(
            any(option.startswith("--formal-v3") for option in option_strings)
        )

    def test_bank_manifest_validation_does_not_require_media_files(self) -> None:
        spec = json.loads(T2V_SPEC.read_bytes())
        spec_sha = mainline.file_sha256(T2V_SPEC)
        flattened = [
            (group["group_id"], candidate)
            for group in spec["groups"]
            for candidate in group["candidates"]
        ]
        candidate_rows = []
        cells = {}
        for _, candidate in flattened:
            candidate_id = candidate["candidate_id"]
            gaussian_sha = _digest(f"gaussian-file-{candidate_id}")
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "analysis_split": candidate["analysis_split"],
                    "action_family_id": candidate["action_family_id"],
                    "calibration_group_id": candidate["calibration_group_id"],
                    "semantic_branch": candidate["semantic_branch"],
                    # Deliberately nonexistent: scalar mainline validation must
                    # never open generation receipts or media artifacts.
                    "receipt_path": f"/nonexistent/formal-bank/{candidate_id}.json",
                    "receipt_sha256": _digest(f"receipt-file-{candidate_id}"),
                    "receipt_digest": _digest(f"receipt-{candidate_id}"),
                    "mp4_sha256": _digest(f"mp4-{candidate_id}"),
                    "predecode_clean_latent_sha256": _digest(
                        f"clean-{candidate_id}"
                    ),
                    "official_initial_gaussian_sha256": gaussian_sha,
                }
            )
            cells.setdefault(candidate["calibration_group_id"], []).append(
                (candidate, gaussian_sha)
            )
        proofs = []
        for cell_id, rows in cells.items():
            first = rows[0][0]
            proofs.append(
                {
                    "analysis_split": first["analysis_split"],
                    "action_family_id": first["action_family_id"],
                    "calibration_group_id": cell_id,
                    "semantic_branch_count": 10,
                    "semantic_branch_order": list(mainline.calibration.BRANCH_ORDER),
                    "all_ten_official_gaussian_tensor_values_byte_equal": True,
                    "all_container_files_individually_sha256_verified": True,
                    "official_gaussian_file_sha256_by_branch": {
                        candidate["semantic_branch"]: file_sha
                        for candidate, file_sha in rows
                    },
                    "official_gaussian_raw_value_sha256": _digest(
                        f"gaussian-raw-{cell_id}"
                    ),
                    "official_gaussian_content_sha256": _digest(
                        f"gaussian-content-{cell_id}"
                    ),
                    "seed": first["seed"],
                }
            )
        membership = {
            split: {
                axis: sorted(
                    {
                        candidate[axis]
                        for _, candidate in flattened
                        if candidate["analysis_split"] == split
                    }
                )
                for axis in mainline.bank_contract.SPLIT_GROUP_AXES
            }
            for split in mainline.bank_contract.ANALYSIS_SPLITS
        }
        unsigned = {
            "schema_version": mainline.bank_contract.BANK_RECEIPT_SCHEMA_VERSION,
            "root_spec_raw_sha256": spec_sha,
            "candidate_count": 40,
            "cell_count": 4,
            "mace_branch_order": list(mainline.calibration.BRANCH_ORDER),
            "sampling_contract": mainline.bank_contract.SAMPLING_CONTRACT,
            "semantic_input_closure": mainline.bank_contract.SEMANTIC_INPUT_CLOSURE,
            "artifact_use_contract": mainline.bank_contract.ARTIFACT_USE_CONTRACT,
            "split_contract": mainline.bank_contract.SPLIT_CONTRACT,
            "split_group_membership": membership,
            "fit_confirmation_all_registered_axes_disjoint": True,
            "same_cell_gaussian_proofs": proofs,
            "candidate_receipts": candidate_rows,
            "interpretation": {
                "calibration_evidence_only": True,
                "event_qualification_performed": False,
                "action_success_not_implied": True,
                "training_performed": False,
                "parameter_update_performed": False,
                "optimizer_authorized": False,
                "t2v_negative_media_are_rv2v_policy_candidates": False,
                "t2v_media_as_condition_target_donor_or_noise_forbidden": True,
            },
        }
        bank = {
            **unsigned,
            "receipt_digest": mainline.bank_contract.sha256_bytes(
                mainline.bank_contract.canonical_json_bytes(unsigned)
            ),
        }
        with tempfile.TemporaryDirectory() as root_text:
            path = Path(root_text).resolve() / "bank-receipt.json"
            bank_sha = _write(path, bank)
            checked_spec, checked_bank, bound = mainline._load_formal_bank_manifest(
                root_spec=T2V_SPEC.resolve(),
                root_spec_sha256=spec_sha,
                bank_receipt=path,
                bank_receipt_sha256=bank_sha,
            )
        self.assertEqual(checked_spec, spec)
        self.assertEqual(checked_bank["receipt_digest"], bank["receipt_digest"])
        self.assertEqual(len(bound), 40)

    def _run(self, *, optimizer_authorized: bool = True, mutate_score: bool = False):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text).resolve()
            bank_root = root / "bank"
            score_root = root / "scores"
            scalar_root = root / "calibration"
            bank_root.mkdir()
            score_root.mkdir()
            scalar_root.mkdir()
            spec_path = root / "bank-spec.json"
            bank_path = bank_root / "bank-receipt.json"
            spec_sha = _write(spec_path, {"fixture": "formal-core4"})
            bank_sha = _write(bank_path, {"fixture": "rendered-bank"})
            bank_digest = _digest("rendered-bank-object")
            checkpoint_tree = _digest("checkpoint-tree")

            branches = list(mainline.calibration.BRANCH_ORDER)
            cells = (
                ("sp4-a", "fit", "dog", "cell-fit-dog"),
                ("sp4-a", "fit", "human", "cell-fit-human"),
                ("sp4-b", "confirmation", "dog", "cell-confirm-dog"),
                ("sp4-b", "confirmation", "human", "cell-confirm-human"),
            )
            group_rows = {"sp4-a": [], "sp4-b": []}
            for group_id, split, family, cell_id in cells:
                for branch in branches:
                    candidate_id = f"{cell_id}-{branch}"
                    candidate = {
                        "candidate_id": candidate_id,
                        "analysis_split": split,
                        "action_family_id": family,
                        "calibration_group_id": cell_id,
                        "actor_group_id": f"actor-{split}-{family}",
                        "scene_group_id": f"scene-{split}-{family}",
                        "action_group_id": f"action-{split}-{family}",
                        "semantic_branch": branch,
                        "geometry_source_video_sha256": _digest(
                            f"geometry-{cell_id}"
                        ),
                        "full_t2v_caption": f"caption {candidate_id}",
                    }
                    group_rows[group_id].append(
                        {
                            "candidate": candidate,
                            "candidate_envelope_sha256": _digest(
                                f"envelope-{candidate_id}"
                            ),
                            "generation_receipt_digest": _digest(
                                f"generation-{candidate_id}"
                            ),
                            "generation_receipt_file_sha256": _digest(
                                f"generation-file-{candidate_id}"
                            ),
                            "native_rollout_receipt_digest": _digest(
                                f"native-{candidate_id}"
                            ),
                            "native_rollout_receipt_file_sha256": _digest(
                                f"native-file-{candidate_id}"
                            ),
                            "generation_runtime_binding": {
                                "candidate_id": candidate_id,
                                "semantic_branch": branch,
                                "checkpoint_tree_sha256": checkpoint_tree,
                            },
                            "artifacts": {
                                "mp4": {"sha256": _digest(f"mp4-{candidate_id}")},
                                "predecode_clean_latent": {
                                    "sha256": _digest(f"clean-{candidate_id}")
                                },
                                "official_initial_gaussian": {
                                    "sha256": _digest(f"noise-{candidate_id}")
                                },
                            },
                        }
                    )

            registry_by_cell = {}
            for rows in group_rows.values():
                for bound in rows:
                    candidate = bound["candidate"]
                    registry_by_cell.setdefault(
                        candidate["calibration_group_id"], {}
                    )[candidate["semantic_branch"]] = bound[
                        "generation_runtime_binding"
                    ]

            scalar_rows = []
            audits = []
            for group_id, rows in group_rows.items():
                for ordinal, bound in enumerate(rows):
                    candidate = bound["candidate"]
                    candidate_id = candidate["candidate_id"]
                    raw = float(ordinal + (0 if group_id == "sp4-a" else 20))
                    score = {
                        **{
                            field: candidate[field]
                            for field in (
                                "candidate_id",
                                "analysis_split",
                                "action_family_id",
                                "calibration_group_id",
                                "actor_group_id",
                                "scene_group_id",
                                "action_group_id",
                                "semantic_branch",
                            )
                        },
                        "root_spec_raw_sha256": spec_sha,
                        "bank_receipt_digest": bank_digest,
                        "candidate_envelope_sha256": bound[
                            "candidate_envelope_sha256"
                        ],
                        "generation_receipt_digest": bound[
                            "generation_receipt_digest"
                        ],
                        "generation_receipt_file_sha256": bound[
                            "generation_receipt_file_sha256"
                        ],
                        "native_rollout_receipt_digest": bound[
                            "native_rollout_receipt_digest"
                        ],
                        "native_rollout_receipt_file_sha256": bound[
                            "native_rollout_receipt_file_sha256"
                        ],
                        "generated_mp4_sha256": bound["artifacts"]["mp4"][
                            "sha256"
                        ],
                        "clean_latent_artifact_sha256": bound["artifacts"][
                            "predecode_clean_latent"
                        ]["sha256"],
                        "official_gaussian_artifact_sha256": bound["artifacts"][
                            "official_initial_gaussian"
                        ]["sha256"],
                        "geometry_source_video_sha256": candidate[
                            "geometry_source_video_sha256"
                        ],
                        "full_t2v_caption_utf8_sha256": hashlib.sha256(
                            candidate["full_t2v_caption"].encode("utf-8")
                        ).hexdigest(),
                        "generation_runtime_binding_by_branch": registry_by_cell[
                            candidate["calibration_group_id"]
                        ],
                        "frozen_scorer_receipt_digest": _digest(
                            f"frozen-{candidate_id}"
                        ),
                        "raw_global_action_energy_score": raw,
                        "receipt_digest": _digest(f"score-{candidate_id}"),
                    }
                    if mutate_score and candidate_id == "cell-fit-dog-action":
                        score["raw_global_action_energy_score"] += 0.25
                    _write(
                        score_root
                        / group_id
                        / candidate_id
                        / mainline.scorer.SCORE_RECEIPT_FILENAME,
                        score,
                    )
                    audit = {
                        **{
                            field: candidate[field]
                            for field in (
                                "candidate_id",
                                "analysis_split",
                                "action_family_id",
                                "calibration_group_id",
                                "actor_group_id",
                                "scene_group_id",
                                "action_group_id",
                                "semantic_branch",
                            )
                        },
                        "generation_receipt_digest": bound[
                            "generation_receipt_digest"
                        ],
                        "receipt_digest": _digest(f"audit-{candidate_id}"),
                    }
                    scalar = {
                        **{
                            field: candidate[field]
                            for field in (
                                "candidate_id",
                                "analysis_split",
                                "action_family_id",
                                "calibration_group_id",
                                "actor_group_id",
                                "scene_group_id",
                                "action_group_id",
                                "semantic_branch",
                            )
                        },
                        "generation_receipt_digest": bound[
                            "generation_receipt_digest"
                        ],
                        "event_audit_receipt_digest": audit["receipt_digest"],
                        "frozen_scorer_receipt_digest": score[
                            "frozen_scorer_receipt_digest"
                        ],
                        "raw_global_action_energy_score": raw,
                        "row_digest": _digest(f"row-{candidate_id}"),
                    }
                    audits.append(audit)
                    scalar_rows.append(scalar)
                    _write(
                        scalar_root / "event-audits" / f"{candidate_id}.json",
                        audit,
                    )
                    _write(
                        scalar_root / "score-rows" / f"{candidate_id}.json",
                        scalar,
                    )

            prereg = {"preregistration_digest": _digest("preregistration")}
            prereg_sha = _write(
                scalar_root / "preregistration-v3.json", prereg
            )
            mappings = {
                family: {"mapping_digest": _digest(f"map-{family}")}
                for family in ("dog", "human")
            }
            stored = {
                "action_family_order": ["dog", "human"],
                "mapping_by_family": mappings,
                "decision_threshold": 0.5,
                "optimizer_authorized": optimizer_authorized,
                "failure_reasons": [] if optimizer_authorized else ["fixture:no-go"],
                "t2v_media_as_rv2v_target_donor_input_or_noise": False,
                "confirmation_rows_consumed_by_optimizer": False,
                "scientific_action_editing_claim": False,
                "receipt_digest": _digest("calibration-receipt"),
            }
            calibration_sha = _write(
                scalar_root / "calibration-receipt-v3.json", stored
            )
            bank_value = {
                "receipt_digest": bank_digest,
                "file_sha256": bank_sha,
            }
            manifest_rows = []
            for group_id in ("sp4-a", "sp4-b"):
                for bound in group_rows[group_id]:
                    manifest_rows.append(
                        (
                            group_id,
                            bound["candidate"],
                            {
                                "receipt_digest": bound[
                                    "generation_receipt_digest"
                                ],
                                "receipt_sha256": bound[
                                    "generation_receipt_file_sha256"
                                ],
                                "mp4_sha256": bound["artifacts"]["mp4"][
                                    "sha256"
                                ],
                                "predecode_clean_latent_sha256": bound[
                                    "artifacts"
                                ]["predecode_clean_latent"]["sha256"],
                                "official_initial_gaussian_sha256": bound[
                                    "artifacts"
                                ]["official_initial_gaussian"]["sha256"],
                            },
                        )
                    )

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        mainline,
                        "_load_formal_bank_manifest",
                        return_value=(
                            {"fixture": "formal-core4"},
                            bank_value,
                            manifest_rows,
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        mainline.scorer,
                        "validate_score_receipt",
                        side_effect=lambda value: value,
                    )
                )
                for name in (
                    "validate_event_audit_receipt",
                    "validate_score_row",
                    "validate_preregistration",
                ):
                    stack.enter_context(
                        patch.object(
                            mainline.calibration,
                            name,
                            side_effect=lambda value: value,
                        )
                    )
                stack.enter_context(
                    patch.object(
                        mainline.calibration,
                        "calibrate_global_action_energy",
                        return_value=stored,
                    )
                )
                return mainline.load_mainline_calibration_bundle(
                    root_spec=spec_path,
                    root_spec_sha256=spec_sha,
                    bank_receipt=bank_path,
                    bank_receipt_sha256=bank_sha,
                    score_root=score_root,
                    calibration_root=scalar_root,
                    calibration_receipt_sha256=calibration_sha,
                    preregistration_sha256=prereg_sha,
                    checkpoint_tree_sha256=checkpoint_tree,
                )

    def test_recomputes_exact40_formal_scalar_provenance(self) -> None:
        bundle = self._run()
        authorization = bundle["authorization"]
        self.assertEqual(authorization["score_count"], 40)
        self.assertEqual(
            authorization["formal_score_schema"],
            mainline.scorer.SCORE_RECEIPT_SCHEMA,
        )
        self.assertTrue(
            authorization["formal_receipts_validated_by_active_v4_canonical_code"]
        )
        self.assertFalse(authorization["legacy_v3_compatibility_score_consumed"])
        self.assertTrue(authorization["calibration_maps_authorized"])
        self.assertFalse(authorization["native_rv2v_optimizer_authorized"])
        self.assertFalse(
            authorization[
                "t2v_media_latent_gaussian_or_proposal_exported_to_native_scorer"
            ]
        )
        self.assertEqual(len(bundle["formal_score_bindings"]), 40)

    def test_rejects_formal_score_scalar_mutation(self) -> None:
        with self.assertRaisesRegex(
            mainline.PairV5MainlineCalibrationError,
            "formal scalar provenance join differs",
        ):
            self._run(mutate_score=True)

    def test_rejects_non_go_calibration(self) -> None:
        with self.assertRaisesRegex(
            mainline.PairV5MainlineCalibrationError,
            "scalar calibration is not GO",
        ):
            self._run(optimizer_authorized=False)

    def test_mainline_source_has_no_initialization_evidence_import(self) -> None:
        source = Path(mainline.__file__).read_text(encoding="utf-8").lower()
        prohibited = "validate_pair_v5_" + "ca" + "gd_evidence_v3"
        self.assertNotIn(prohibited, source)
        self.assertNotIn("guidance_distill", source)
        self.assertNotIn("load_group_bank", source)


if __name__ == "__main__":
    unittest.main()
