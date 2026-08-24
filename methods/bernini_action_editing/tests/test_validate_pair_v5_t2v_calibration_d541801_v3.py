from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_score_d541801_v3_compat as compat  # noqa: E402
import validate_pair_v5_t2v_calibration_d541801_v3 as mainline  # noqa: E402


T2V_SPEC = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mainline.canonical_json_bytes(value) + b"\n")
    return mainline.file_sha256(path)


class PairV5T2VD541801CalibrationTests(unittest.TestCase):
    def test_native_packet_uses_exact_d541801_expression_only(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        branch = torch.tensor(
            [[0.125], [0.25], [0.5], [0.75], [1.0], [1.25], [1.5], [1.75], [2.0], [3.0]],
            dtype=torch.float32,
        )
        # These active-experiment-shaped attributes are intentionally bogus;
        # the compatibility boundary is allowed to consume branch energies only.
        energy = SimpleNamespace(
            branch_energies=branch,
            reward=torch.tensor([-999.0], dtype=torch.float32),
            negative_log_energy_ratios=torch.full((9, 1), 999.0),
        )
        packet = compat.make_native_v3_energy_packet(energy)
        expected_ratios = torch.log(branch[1:] + 1.0e-8) - torch.log(
            branch[:1] + 1.0e-8
        )
        expected_reward, expected_index = expected_ratios.min(dim=0)
        self.assertEqual(
            packet["raw_global_action_energy_score"],
            float(expected_reward.item()),
        )
        self.assertEqual(
            packet["global_hardest_negative_branch"],
            compat.HARD_NEGATIVE_BRANCHES[int(expected_index.item())],
        )
        self.assertEqual(packet["definition"], compat.V3_SCALAR_DEFINITION)
        self.assertEqual(
            packet["compatibility_source_revision"], compat.PINNED_SOURCE_REVISION
        )

        resigned = json.loads(compat.canonical_json_bytes(packet))
        resigned["raw_global_action_energy_score"] = float(
            torch.nextafter(
                torch.tensor(resigned["raw_global_action_energy_score"]),
                torch.tensor(float("inf")),
            ).item()
        )
        unsigned = dict(resigned)
        unsigned.pop("packet_digest")
        resigned["packet_digest"] = compat.object_sha256(unsigned)
        with self.assertRaisesRegex(
            compat.PairV5T2VScoreV3CompatibilityError,
            "reward/first-argmin closure differs",
        ):
            compat.validate_native_v3_energy_packet(resigned)

    def test_isolated_validator_accepts_only_v3_receipts(self) -> None:
        shim = textwrap.dedent(
            """
            import json

            SCORE_RECEIPT_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-score-v3"

            class PairV5T2VEnergyScoringError(RuntimeError):
                pass

            def canonical_json_bytes(value):
                return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

            def validate_score_receipt(value):
                if not isinstance(value, dict) or set(value) != {"schema_version", "candidate_id"}:
                    raise PairV5T2VEnergyScoringError("shim field closure differs")
                if value["schema_version"] != SCORE_RECEIPT_SCHEMA:
                    raise PairV5T2VEnergyScoringError("shim schema differs")
                return dict(value)
            """
        ).lstrip()
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text).resolve()
            method_root = root / "methods"
            method_root.mkdir()
            scorer_path = method_root / "score_pair_v5_t2v_energy_bank_v3.py"
            mace_path = method_root / "mace_candidate_action_energy.py"
            scorer_path.write_text(shim, encoding="utf-8")
            mace_path.write_text("# pinned-shim\n", encoding="utf-8")
            score_paths = []
            for index in range(40):
                path = root / "scores" / f"score-{index:02d}.json"
                _write(
                    path,
                    {
                        "schema_version": compat.FORMAL_SCORE_SCHEMA,
                        "candidate_id": f"candidate-{index:02d}",
                    },
                )
                score_paths.append(path)
            with patch.object(
                compat,
                "PINNED_SCORER_SOURCE_SHA256",
                compat.file_sha256(scorer_path),
            ), patch.object(
                compat,
                "PINNED_MACE_SOURCE_SHA256",
                compat.file_sha256(mace_path),
            ):
                rows, binding = compat.validate_formal_score_receipts_isolated(
                    formal_v3_method_root=method_root,
                    formal_v3_source_revision=compat.PINNED_SOURCE_REVISION,
                    formal_v3_source_archive_sha256=_digest("archive"),
                    score_paths=score_paths,
                    python_executable=Path(sys.executable).resolve(),
                )
                self.assertEqual(len(rows), 40)
                self.assertTrue(
                    binding["receipt_validator_executed_in_isolated_python"]
                )
                self.assertFalse(
                    binding["active_repository_scorer_imported_by_isolated_validator"]
                )

                bad = json.loads(score_paths[0].read_bytes())
                bad["schema_version"] = (
                    "bernini-pair-v5-frozen-t2v-global-energy-score-v4"
                )
                _write(score_paths[0], bad)
                with self.assertRaisesRegex(
                    compat.PairV5T2VScoreV3CompatibilityError,
                    "isolated d541801 formal-v3 validation failed",
                ):
                    compat.validate_formal_score_receipts_isolated(
                        formal_v3_method_root=method_root,
                        formal_v3_source_revision=compat.PINNED_SOURCE_REVISION,
                        formal_v3_source_archive_sha256=_digest("archive"),
                        score_paths=score_paths,
                        python_executable=Path(sys.executable).resolve(),
                    )

    def test_bank_manifest_validation_never_opens_media(self) -> None:
        spec = json.loads(T2V_SPEC.read_bytes())
        spec_sha = mainline.file_sha256(T2V_SPEC)
        flattened = [
            (group["group_id"], candidate)
            for group in spec["groups"]
            for candidate in group["candidates"]
        ]
        candidate_rows = []
        cells: dict[str, list[tuple[dict[str, object], str]]] = {}
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
                    "receipt_path": f"/does/not/exist/{candidate_id}.json",
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

    def _run_bundle(
        self, *, optimizer_authorized: bool = True, mutate_score: bool = False
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text).resolve()
            score_root = root / "scores"
            scalar_root = root / "calibration"
            score_root.mkdir()
            scalar_root.mkdir()
            spec_path = root / "bank-spec.json"
            bank_path = root / "bank-receipt.json"
            spec_sha = _write(spec_path, {"fixture": "formal-core4"})
            bank_sha = _write(bank_path, {"fixture": "rendered-bank"})
            bank_digest = _digest("rendered-bank-object")
            checkpoint_tree = _digest("checkpoint-tree")
            archive_sha = _digest("d541801-archive")

            branches = list(mainline.calibration.BRANCH_ORDER)
            cells = (
                ("sp4-a", "fit", "dog", "cell-fit-dog"),
                ("sp4-a", "fit", "human", "cell-fit-human"),
                ("sp4-b", "confirmation", "dog", "cell-confirm-dog"),
                ("sp4-b", "confirmation", "human", "cell-confirm-human"),
            )
            manifest_rows = []
            scores = []
            audits = []
            scalar_rows = []
            registry_by_cell = {}
            candidates = []
            for group_id, split, family, cell_id in cells:
                cell_candidates = []
                registry = {}
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
                    cell_candidates.append((group_id, candidate))
                    registry[branch] = {
                        "candidate_id": candidate_id,
                        "semantic_branch": branch,
                        "checkpoint_tree_sha256": checkpoint_tree,
                    }
                registry_by_cell[cell_id] = registry
                candidates.extend(cell_candidates)

            for ordinal, (group_id, candidate) in enumerate(candidates):
                candidate_id = candidate["candidate_id"]
                raw = float(ordinal)
                generation_digest = _digest(f"generation-{candidate_id}")
                generation_sha = _digest(f"generation-file-{candidate_id}")
                mp4_sha = _digest(f"mp4-{candidate_id}")
                clean_sha = _digest(f"clean-{candidate_id}")
                gaussian_sha = _digest(f"noise-{candidate_id}")
                bank_row = {
                    "receipt_digest": generation_digest,
                    "receipt_sha256": generation_sha,
                    "mp4_sha256": mp4_sha,
                    "predecode_clean_latent_sha256": clean_sha,
                    "official_initial_gaussian_sha256": gaussian_sha,
                }
                manifest_rows.append((group_id, candidate, bank_row))
                identity = {
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
                }
                frozen_digest = _digest(f"frozen-{candidate_id}")
                score = {
                    **identity,
                    "schema_version": compat.FORMAL_SCORE_SCHEMA,
                    "root_spec_raw_sha256": spec_sha,
                    "bank_receipt_digest": bank_digest,
                    "generation_receipt_digest": generation_digest,
                    "generation_receipt_file_sha256": generation_sha,
                    "generated_mp4_sha256": mp4_sha,
                    "clean_latent_artifact_sha256": clean_sha,
                    "official_gaussian_artifact_sha256": gaussian_sha,
                    "geometry_source_video_sha256": candidate[
                        "geometry_source_video_sha256"
                    ],
                    "full_t2v_caption_utf8_sha256": hashlib.sha256(
                        candidate["full_t2v_caption"].encode("utf-8")
                    ).hexdigest(),
                    "generation_runtime_binding_by_branch": registry_by_cell[
                        candidate["calibration_group_id"]
                    ],
                    "frozen_scorer_receipt_digest": frozen_digest,
                    "raw_global_action_energy_score": raw,
                    "receipt_digest": _digest(f"score-{candidate_id}"),
                }
                if mutate_score and ordinal == 0:
                    score["raw_global_action_energy_score"] += 0.25
                scores.append(score)
                _write(
                    score_root
                    / group_id
                    / candidate_id
                    / compat.FORMAL_SCORE_FILENAME,
                    score,
                )
                audit = {
                    **identity,
                    "generation_receipt_digest": generation_digest,
                    "receipt_digest": _digest(f"audit-{candidate_id}"),
                }
                scalar = {
                    **identity,
                    "generation_receipt_digest": generation_digest,
                    "event_audit_receipt_digest": audit["receipt_digest"],
                    "frozen_scorer_receipt_digest": frozen_digest,
                    "raw_global_action_energy_score": raw,
                    "row_digest": _digest(f"row-{candidate_id}"),
                }
                audits.append(audit)
                scalar_rows.append(scalar)
                _write(
                    scalar_root / "event-audits" / f"{candidate_id}.json", audit
                )
                _write(
                    scalar_root / "score-rows" / f"{candidate_id}.json", scalar
                )

            prereg = {"preregistration_digest": _digest("preregistration")}
            prereg_sha = _write(
                scalar_root / "preregistration-v3.json", prereg
            )
            stored = {
                "action_family_order": ["dog", "human"],
                "mapping_by_family": {
                    family: {"mapping_digest": _digest(f"map-{family}")}
                    for family in ("dog", "human")
                },
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
            source_binding_unsigned = {
                "source_revision": compat.PINNED_SOURCE_REVISION,
                "source_archive_sha256": archive_sha,
                "critical_source_manifest": {
                    "score": compat.PINNED_SCORER_SOURCE_SHA256,
                    "mace": compat.PINNED_MACE_SOURCE_SHA256,
                },
                "formal_score_schema": compat.FORMAL_SCORE_SCHEMA,
                "formal_score_filename": compat.FORMAL_SCORE_FILENAME,
                "receipt_validator_executed_in_isolated_python": True,
                "active_repository_scorer_imported_by_isolated_validator": False,
            }
            source_binding = {
                **source_binding_unsigned,
                "binding_digest": mainline.object_sha256(source_binding_unsigned),
            }

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        mainline,
                        "_load_formal_bank_manifest",
                        return_value=(
                            {"fixture": "formal-core4"},
                            {
                                "receipt_digest": bank_digest,
                                "file_sha256": bank_sha,
                            },
                            manifest_rows,
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        mainline.formal_v3_compat,
                        "validate_formal_score_receipts_isolated",
                        return_value=(scores, source_binding),
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
                    formal_v3_method_root=root,
                    formal_v3_source_revision=compat.PINNED_SOURCE_REVISION,
                    formal_v3_source_archive_sha256=archive_sha,
                    python_executable=Path(sys.executable).resolve(),
                )

    def test_recomputes_exact40_v3_scalar_provenance(self) -> None:
        bundle = self._run_bundle()
        authorization = bundle["authorization"]
        self.assertEqual(authorization["score_count"], 40)
        self.assertEqual(
            authorization["formal_score_schema"], compat.FORMAL_SCORE_SCHEMA
        )
        self.assertEqual(
            authorization["formal_score_scalar_definition"],
            compat.V3_SCALAR_DEFINITION,
        )
        self.assertTrue(
            authorization["formal_receipts_validated_by_isolated_d541801_code"]
        )
        self.assertFalse(authorization["active_repository_score_schema_consumed"])
        self.assertFalse(authorization["active_repository_action_scalar_consumed"])
        self.assertFalse(authorization["decimal_or_log1p_action_scalar_consumed"])
        self.assertTrue(authorization["calibration_maps_authorized"])
        self.assertFalse(authorization["native_rv2v_optimizer_authorized"])
        self.assertEqual(len(bundle["formal_score_bindings"]), 40)

    def test_rejects_scalar_mutation_and_non_go(self) -> None:
        with self.assertRaisesRegex(
            mainline.PairV5MainlineCalibrationError,
            "scalar-provenance join differs",
        ):
            self._run_bundle(mutate_score=True)
        with self.assertRaisesRegex(
            mainline.PairV5MainlineCalibrationError,
            "scalar calibration is not GO",
        ):
            self._run_bundle(optimizer_authorized=False)

    def test_unique_validator_has_no_active_scorer_or_cagd_dependency(self) -> None:
        source = Path(mainline.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("score_pair_v5_t2v_energy_bank_v3", source)
        self.assertNotIn("guidance_distill", source)
        self.assertNotIn("cagd", source)
        self.assertIn("pair_v5_t2v_score_d541801_v3_compat", source)
        option_strings = {
            option
            for action in mainline.build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--formal-v3-method-root", option_strings)
        self.assertIn("--formal-v3-source-revision", option_strings)
        self.assertIn("--formal-v3-source-archive-sha256", option_strings)


if __name__ == "__main__":
    unittest.main()
