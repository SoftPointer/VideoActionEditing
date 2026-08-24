from __future__ import annotations

from copy import deepcopy
import ast
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import caper_stage1_preference_admission_v1 as caper  # noqa: E402


def _sha(character: str) -> str:
    return character * 64


try:
    import av as _av  # noqa: F401
except ImportError:
    _AV_AVAILABLE = False
else:
    _AV_AVAILABLE = True


@lru_cache(maxsize=1)
def _exact81_mp4_bytes() -> bytes:
    """Create one tiny, real 81-frame/25-fps MP4 shared by all fixtures."""

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "exact81.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x204080:s=16x16:r=25",
                "-frames:v",
                "81",
                "-c:v",
                "mpeg4",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
        return path.read_bytes()


class CAPERManifestFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temporary.name)
        self._ledger_entries: list[dict[str, object]] = []
        self._population_rows: list[dict[str, object]] = []
        self._rewrite_counter = 0
        self.population_k = 2
        self.source_revision_sha256 = _sha("d")
        self.action_taxonomy_sha256 = _sha("e")
        self.audit_protocol_sha256 = _sha("f")
        self.policy = {
            "policy_id": "bernini-r13-deployable-v2v",
            "model_family": "Bernini-R-1.3B",
            "checkpoint_tree_sha256": _sha("a"),
            "inference_contract_sha256": _sha("b"),
            "deployable": True,
            "source_conditioned": True,
            "weights_frozen_during_rollout": True,
        }
        self.reward_evaluator = {
            "evaluator_id": "caper-action-evaluator-v1",
            "evaluator_type": caper.REWARD_EVALUATOR_TYPE,
            "evaluator_version": "1.0.0",
            "weights_artifact": self._artifact("evaluator-weights.safetensors"),
            "runtime_artifact": self._artifact("evaluator-runtime.json"),
            "input_contract_sha256": _sha("9"),
            "required_inputs": list(caper.REWARD_EVALUATOR_INPUTS),
            "optional_inputs": list(caper.REWARD_EVALUATOR_OPTIONAL_INPUTS),
            "forbidden_inputs": list(caper.REWARD_EVALUATOR_FORBIDDEN_INPUTS),
        }
        fit = self._pair(
            split="fit",
            prefix="fit-dog",
            identity_id="identity-fit-dog",
            scene_id="scene-fit-studio",
            action_family="sit-down",
        )
        heldout = self._pair(
            split="heldout",
            prefix="heldout-human",
            identity_id="identity-heldout-human",
            scene_id="scene-heldout-street",
            action_family="stand-up",
        )
        owner = self._owner()
        population_ledger = self._write_population_ledger()
        ledger = self._write_ledger()
        self.manifest = {
            "schema_version": caper.SCHEMA_VERSION,
            "manifest_id": "caper-stage1-unit-fixture",
            "created_utc": "2026-08-09T09:30:00Z",
            "purpose": "Unit-test same-source on-policy preference admission",
            "admission_contract": {
                "stage": "CAPER-stage1",
                "preference_target_mode": caper.PREFERENCE_TARGET_MODE,
                "pure_t2v_role": caper.PURE_T2V_ROLE,
                "pure_t2v_preference_target_allowed": False,
                "target_video_dependency_allowed": False,
                "scalar_reward_admission_allowed": False,
                "scalar_compensation_allowed": False,
                "preservation_gate_semantics": caper.PRESERVATION_SEMANTICS,
                "preservation_gate_names": list(caper.PRESERVATION_GATES),
                "split_isolation_axes": list(caper.SPLIT_ISOLATION_AXES),
                "no_valid_fit_pair_behavior": caper.NO_PAIR_BEHAVIOR,
            },
            "deployable_policy": deepcopy(self.policy),
            "reward_evaluator": deepcopy(self.reward_evaluator),
            "rollout_population_ledger": population_ledger,
            "exposure_ledger": ledger,
            "pure_t2v_calibration_owners": [owner],
            "splits": {"fit": [fit], "heldout": [heldout]},
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _artifact(self, name: str) -> dict[str, object]:
        relative = Path("artifacts") / name
        path = self.base_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".mp4"):
            # Appending a distinct, ignored trailer keeps each artifact
            # commitment unique while preserving one valid exact81 stream.
            payload = _exact81_mp4_bytes() + f"\nfixture:{name}\n".encode("utf-8")
        else:
            payload = (f"sealed artifact: {name}\n").encode("utf-8")
        path.write_bytes(payload)
        return {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    def _fake_decode(
        self, artifact: dict[str, object], *, base_dir: Path | None, label: str
    ) -> dict[str, object]:
        caper._verify_artifact(artifact, base_dir=base_dir)
        return {
            "decoded_media_contract": caper.DECODED_MEDIA_CONTRACT,
            "decoded_frame_count": 81,
            "decoded_fps_numerator": 25,
            "decoded_fps_denominator": 1,
            "decoded_height": 16,
            "decoded_width": 16,
            "decoded_rgb24_sha256": hashlib.sha256(
                f"fixture-rgb:{artifact['sha256']}".encode("utf-8")
            ).hexdigest(),
        }

    def _decoded(self, artifact: dict[str, object]) -> dict[str, object]:
        if _AV_AVAILABLE:
            return caper._decode_exact81_media_artifact(
                artifact, base_dir=self.base_dir, label="fixture media"
            )
        return self._fake_decode(
            artifact, base_dir=self.base_dir, label="fixture media"
        )

    def _evaluator_binding(self) -> dict[str, object]:
        return {
            "evaluator_id": self.reward_evaluator["evaluator_id"],
            "evaluator_type": self.reward_evaluator["evaluator_type"],
            "evaluator_version": self.reward_evaluator["evaluator_version"],
            "weights_artifact_sha256": self.reward_evaluator["weights_artifact"][
                "sha256"
            ],
            "runtime_artifact_sha256": self.reward_evaluator["runtime_artifact"][
                "sha256"
            ],
            "input_contract_sha256": self.reward_evaluator[
                "input_contract_sha256"
            ],
        }

    def _milestones(self, action_grade: str) -> list[dict[str, object]]:
        scores = {
            "terminal_reached_and_held": (0.80, 0.90, 0.90, 0.90),
            "terminal_reached": (0.80, 0.90, 0.90, 0.30),
            "transition_incomplete": (0.80, 0.90, 0.30, 0.20),
            "started_only": (0.80, 0.30, 0.20, 0.20),
            "absent_or_wrong": (0.20, 0.20, 0.20, 0.20),
        }[action_grade]
        return [
            {
                "milestone": milestone,
                "frame_index": frame_index,
                "score": score,
                "threshold": 0.5,
                "margin": score - 0.5,
                "uncertainty": 0.02,
            }
            for milestone, frame_index, score in zip(
                caper.MILESTONES, (0, 20, 50, 75), scores
            )
        ]

    def _json_artifact(
        self, name: str, value: dict[str, object], *, seal_field: str
    ) -> dict[str, object]:
        payload = deepcopy(value)
        payload[seal_field] = caper.object_sha256(payload)
        relative = Path("artifacts") / name
        path = self.base_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = caper.canonical_json_bytes(payload) + b"\n"
        path.write_bytes(raw)
        return {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }

    def _write_ledger(self, *, suffix: str = "base") -> dict[str, object]:
        return self._json_artifact(
            f"exposure-ledger-{suffix}.json",
            {
                "schema_version": caper.EXPOSURE_LEDGER_SCHEMA_VERSION,
                "ledger_id": "caper-unit-exposure-ledger",
                "checkpoint_tree_sha256": self.policy["checkpoint_tree_sha256"],
                "inference_contract_sha256": self.policy[
                    "inference_contract_sha256"
                ],
                "source_revision_sha256": self.source_revision_sha256,
                "action_taxonomy_sha256": self.action_taxonomy_sha256,
                "reward_audit_protocol_sha256": self.audit_protocol_sha256,
                "entries": sorted(
                    deepcopy(self._ledger_entries), key=lambda row: row["rollout_id"]
                ),
            },
            seal_field="ledger_sha256",
        )

    def _write_population_ledger(
        self, *, suffix: str = "base"
    ) -> dict[str, object]:
        return self._json_artifact(
            f"population-ledger-{suffix}.json",
            {
                "schema_version": caper.POPULATION_LEDGER_SCHEMA_VERSION,
                "ledger_id": "caper-unit-population-ledger",
                "policy_id": self.policy["policy_id"],
                "checkpoint_tree_sha256": self.policy["checkpoint_tree_sha256"],
                "inference_contract_sha256": self.policy[
                    "inference_contract_sha256"
                ],
                "source_revision_sha256": self.source_revision_sha256,
                "action_taxonomy_sha256": self.action_taxonomy_sha256,
                "population_size_k": self.population_k,
                "preregistered_seed_ids": [
                    f"seed-{index}" for index in range(self.population_k)
                ],
                "pair_selection_rule": caper.PAIR_SELECTION_RULE,
                "seed_balance_rule": caper.SEED_BALANCE_RULE,
                "populations": sorted(
                    deepcopy(self._population_rows),
                    key=lambda row: row["population_id"],
                ),
            },
            seal_field="ledger_sha256",
        )

    def _binding(self) -> dict[str, str]:
        return {
            "policy_id": self.policy["policy_id"],
            "checkpoint_tree_sha256": self.policy["checkpoint_tree_sha256"],
            "inference_contract_sha256": self.policy[
                "inference_contract_sha256"
            ],
        }

    def _candidate(
        self,
        *,
        role: str,
        prefix: str,
        source_id: str,
        identity_id: str,
        scene_id: str,
        action_family: str,
        edit_instruction: str,
        source_sha256: str,
        population_id: str,
        seed_id: str,
        action_score_override: float | None = None,
    ) -> dict[str, object]:
        candidate_id = f"{prefix}-{role}"
        output = self._artifact(f"{prefix}-{role}.mp4")
        lineage = hashlib.sha256(f"lineage:{candidate_id}".encode()).hexdigest()
        gaussian = hashlib.sha256(f"gaussian:{candidate_id}".encode()).hexdigest()
        instruction_digest = caper.text_sha256(edit_instruction)
        decoded = self._decoded(output)
        action_grade = (
            "terminal_reached_and_held"
            if role == "winner"
            else "transition_incomplete"
        )
        action_score = (
            action_score_override
            if action_score_override is not None
            else (0.90 if role == "winner" else 0.45)
        )
        action_uncertainty = 0.02
        receipt_body: dict[str, object] = {
            "schema_version": caper.ROLLOUT_RECEIPT_SCHEMA_VERSION,
            "rollout_id": candidate_id,
            "lineage_root_sha256": lineage,
            "rollout_role": caper.SOURCE_CANDIDATE_ROLE,
            "split": "fit" if prefix.startswith("fit-") else "heldout",
            **self._binding(),
            "source_id": source_id,
            "source_media_sha256": source_sha256,
            "action_family": action_family,
            "instruction_text_sha256": instruction_digest,
            "accepted_inputs": list(caper.SOURCE_CONDITIONING_INPUTS),
            "sealed_gaussian_sha256": gaussian,
            "population_id": population_id,
            "seed_id": seed_id,
            "attempt_status": "success",
            "failure_code": None,
            "target_video_read": False,
            "target_video_latent_read": False,
            "paired_target_read": False,
            "pure_t2v_media_read": False,
            "pure_t2v_latent_read": False,
            "output_media_sha256": output["sha256"],
            "output_media_size_bytes": output["size_bytes"],
            "frame_count": 81,
            "fps": 25,
            **decoded,
        }
        receipt_body["source_role_digest"] = caper.source_role_digest(
            {
                key: receipt_body[key]
                for key in (
                    "lineage_root_sha256",
                    "rollout_role",
                    "split",
                    "policy_id",
                    "checkpoint_tree_sha256",
                    "inference_contract_sha256",
                    "source_id",
                    "source_media_sha256",
                    "action_family",
                    "instruction_text_sha256",
                    "accepted_inputs",
                    "sealed_gaussian_sha256",
                    "population_id",
                    "seed_id",
                    "attempt_status",
                )
            }
        )
        receipt = self._json_artifact(
            f"{candidate_id}-rollout.json",
            receipt_body,
            seal_field="receipt_sha256",
        )
        gates = {gate: True for gate in caper.PRESERVATION_GATES}
        audit = self._json_artifact(
            f"{candidate_id}-reward.json",
            {
                "schema_version": caper.REWARD_AUDIT_SCHEMA_VERSION,
                "audit_id": f"audit-{candidate_id}",
                "rollout_id": candidate_id,
                "lineage_root_sha256": lineage,
                "source_role_digest": receipt_body["source_role_digest"],
                "output_media_sha256": output["sha256"],
                "rollout_receipt_artifact_sha256": receipt["sha256"],
                "audit_role": caper.SOURCE_AUDIT_ROLE,
                "checkpoint_tree_sha256": self.policy["checkpoint_tree_sha256"],
                "action_family": action_family,
                "rubric_id": "event-terminal-hold-v1",
                "audit_protocol_sha256": self.audit_protocol_sha256,
                "evaluator_binding": self._evaluator_binding(),
                "evaluator_inputs": [
                    *caper.REWARD_EVALUATOR_INPUTS,
                    *caper.REWARD_EVALUATOR_OPTIONAL_INPUTS,
                ],
                "mask_read": False,
                "track_read": False,
                "pose_read": False,
                "flow_read": False,
                "pure_t2v_origin_metadata_read": False,
                "action_grade": action_grade,
                "action_score": action_score,
                "action_uncertainty": action_uncertainty,
                "action_margin_to_grade_threshold": action_score - 0.5,
                "milestone_frame_evidence": self._milestones(action_grade),
                "preservation_hard_gates": gates,
                "scalar_compensation_used": False,
                "target_video_dependency": False,
            },
            seal_field="audit_sha256",
        )
        self._ledger_entries.append(
            {
                "rollout_id": candidate_id,
                "lineage_root_sha256": lineage,
                "rollout_role": caper.SOURCE_CANDIDATE_ROLE,
                "split": receipt_body["split"],
                "population_id": population_id,
                "seed_id": seed_id,
                "policy_id": self.policy["policy_id"],
                "checkpoint_tree_sha256": self.policy["checkpoint_tree_sha256"],
                "inference_contract_sha256": self.policy[
                    "inference_contract_sha256"
                ],
                "source_id": source_id,
                "source_media_sha256": source_sha256,
                "action_family": action_family,
                "instruction_text_sha256": instruction_digest,
                "output_media_sha256": output["sha256"],
                "rollout_receipt_artifact_sha256": receipt["sha256"],
                "reward_audit_artifact_sha256": audit["sha256"],
                "source_role_digest": receipt_body["source_role_digest"],
            }
        )
        return {
            "candidate_id": candidate_id,
            "population_id": population_id,
            "seed_id": seed_id,
            "declared_role": role,
            "generation_mode": "source_conditioned_video_editing",
            "eligible_as_preference_target": True,
            "source_id": source_id,
            "identity_id": identity_id,
            "scene_id": scene_id,
            "action_family": action_family,
            "edit_instruction": edit_instruction,
            "policy_binding": self._binding(),
            "conditioning_attestation": {
                "input_kinds": list(caper.SOURCE_CONDITIONING_INPUTS),
                "source_media_sha256": source_sha256,
                "target_video_read": False,
                "target_video_latent_read": False,
                "paired_target_read": False,
                "pure_t2v_media_read": False,
                "pure_t2v_latent_read": False,
            },
            "output_media": output,
            "rollout_receipt": receipt,
            "reward_audit": audit,
            "preservation_hard_gates": gates,
        }

    def _pair(
        self,
        *,
        split: str,
        prefix: str,
        identity_id: str,
        scene_id: str,
        action_family: str,
    ) -> dict[str, object]:
        source_id = f"source-{prefix}"
        population_id = f"population-{prefix}"
        instruction = f"Make the subject perform {action_family}."
        source = self._artifact(f"{prefix}-source.mp4")
        source_intake_receipt = self._artifact(f"{prefix}-source-intake.json")
        context = {
            "prefix": prefix,
            "source_id": source_id,
            "identity_id": identity_id,
            "scene_id": scene_id,
            "action_family": action_family,
            "edit_instruction": instruction,
            "source_sha256": source["sha256"],
            "population_id": population_id,
        }
        winner = self._candidate(role="winner", seed_id="seed-0", **context)
        loser = self._candidate(role="loser", seed_id="seed-1", **context)
        self._population_rows.append(
            {
                "population_id": population_id,
                "split": split,
                "source_id": source_id,
                "identity_id": identity_id,
                "scene_id": scene_id,
                "action_family": action_family,
                "edit_instruction": instruction,
                "source_media": source,
                "source_intake_receipt": source_intake_receipt,
                "attempts": [
                    {
                        "seed_id": endpoint["seed_id"],
                        "candidate_id": endpoint["candidate_id"],
                        "attempt_status": "success",
                        "failure_code": None,
                        "output_media": endpoint["output_media"],
                        "rollout_receipt": endpoint["rollout_receipt"],
                        "reward_audit": endpoint["reward_audit"],
                        "source_role_digest": next(
                            entry["source_role_digest"]
                            for entry in self._ledger_entries
                            if entry["rollout_id"] == endpoint["candidate_id"]
                        ),
                    }
                    for endpoint in (winner, loser)
                ],
            }
        )
        return {
            "pair_id": f"pair-{prefix}",
            "population_id": population_id,
            "split": split,
            "source_id": source_id,
            "identity_id": identity_id,
            "scene_id": scene_id,
            "action_family": action_family,
            "edit_instruction": instruction,
            "source_media": source,
            "source_intake_receipt": source_intake_receipt,
            "target_video_dependency": False,
            "scalar_reward_used_for_admission": False,
            "scalar_compensation_used": False,
            "winner": winner,
            "loser": loser,
            "action_ordering": {
                "rubric_id": "event-terminal-hold-v1",
                "winner_action_grade": "terminal_reached_and_held",
                "loser_action_grade": "transition_incomplete",
                "strict_preference_claimed": True,
                "selection_basis": caper.ACTION_SELECTION_BASIS,
                "pair_selection_rule": caper.PAIR_SELECTION_RULE,
                "winner_action_score": 0.90,
                "winner_action_uncertainty": 0.02,
                "loser_action_score": 0.45,
                "loser_action_uncertainty": 0.02,
                "pairwise_confidence_margin": 0.41,
                "minimum_required_pairwise_margin": 0.05,
            },
        }

    def _owner(self) -> dict[str, object]:
        owner_id = "t2v-calibration-owner-fit-sit"
        prompt = "A dog completes a sit-down action and holds the pose."
        output = self._artifact("owner-t2v.mp4")
        generator = {
            "generator_id": "bernini-r13-pure-t2v",
            "checkpoint_tree_sha256": _sha("a"),
            "inference_contract_sha256": _sha("c"),
        }
        lineage = hashlib.sha256(f"lineage:{owner_id}".encode()).hexdigest()
        gaussian = hashlib.sha256(f"gaussian:{owner_id}".encode()).hexdigest()
        instruction_digest = caper.text_sha256(prompt)
        decoded = self._decoded(output)
        action_grade = "terminal_reached_and_held"
        action_score = 0.92
        action_uncertainty = 0.02
        receipt_body: dict[str, object] = {
            "schema_version": caper.ROLLOUT_RECEIPT_SCHEMA_VERSION,
            "rollout_id": owner_id,
            "lineage_root_sha256": lineage,
            "rollout_role": caper.PURE_T2V_LEDGER_ROLE,
            "split": "fit",
            "policy_id": generator["generator_id"],
            "checkpoint_tree_sha256": generator["checkpoint_tree_sha256"],
            "inference_contract_sha256": generator["inference_contract_sha256"],
            "source_id": None,
            "source_media_sha256": None,
            "action_family": "sit-down",
            "instruction_text_sha256": instruction_digest,
            "accepted_inputs": list(caper.PURE_T2V_INPUTS),
            "sealed_gaussian_sha256": gaussian,
            "population_id": None,
            "seed_id": None,
            "attempt_status": "success",
            "failure_code": None,
            "target_video_read": False,
            "target_video_latent_read": False,
            "paired_target_read": False,
            "pure_t2v_media_read": False,
            "pure_t2v_latent_read": False,
            "output_media_sha256": output["sha256"],
            "output_media_size_bytes": output["size_bytes"],
            "frame_count": 81,
            "fps": 25,
            **decoded,
        }
        receipt_body["source_role_digest"] = caper.source_role_digest(
            {
                key: receipt_body[key]
                for key in (
                    "lineage_root_sha256",
                    "rollout_role",
                    "split",
                    "policy_id",
                    "checkpoint_tree_sha256",
                    "inference_contract_sha256",
                    "source_id",
                    "source_media_sha256",
                    "action_family",
                    "instruction_text_sha256",
                    "accepted_inputs",
                    "sealed_gaussian_sha256",
                    "population_id",
                    "seed_id",
                    "attempt_status",
                )
            }
        )
        receipt = self._json_artifact(
            "owner-t2v-rollout.json", receipt_body, seal_field="receipt_sha256"
        )
        audit = self._json_artifact(
            "owner-t2v-reward.json",
            {
                "schema_version": caper.REWARD_AUDIT_SCHEMA_VERSION,
                "audit_id": "audit-owner-t2v",
                "rollout_id": owner_id,
                "lineage_root_sha256": lineage,
                "source_role_digest": receipt_body["source_role_digest"],
                "output_media_sha256": output["sha256"],
                "rollout_receipt_artifact_sha256": receipt["sha256"],
                "audit_role": caper.PURE_T2V_AUDIT_ROLE,
                "checkpoint_tree_sha256": self.policy["checkpoint_tree_sha256"],
                "action_family": "sit-down",
                "rubric_id": "pure-t2v-action-calibration-v1",
                "audit_protocol_sha256": self.audit_protocol_sha256,
                "evaluator_binding": self._evaluator_binding(),
                "evaluator_inputs": list(caper.REWARD_EVALUATOR_INPUTS),
                "mask_read": False,
                "track_read": False,
                "pose_read": False,
                "flow_read": False,
                "pure_t2v_origin_metadata_read": False,
                "action_grade": action_grade,
                "action_score": action_score,
                "action_uncertainty": action_uncertainty,
                "action_margin_to_grade_threshold": action_score - 0.5,
                "milestone_frame_evidence": self._milestones(action_grade),
                "preservation_hard_gates": None,
                "scalar_compensation_used": False,
                "target_video_dependency": False,
            },
            seal_field="audit_sha256",
        )
        self._ledger_entries.append(
            {
                "rollout_id": owner_id,
                "lineage_root_sha256": lineage,
                "rollout_role": caper.PURE_T2V_LEDGER_ROLE,
                "split": "fit",
                "population_id": None,
                "seed_id": None,
                "policy_id": generator["generator_id"],
                "checkpoint_tree_sha256": generator["checkpoint_tree_sha256"],
                "inference_contract_sha256": generator["inference_contract_sha256"],
                "source_id": None,
                "source_media_sha256": None,
                "action_family": "sit-down",
                "instruction_text_sha256": instruction_digest,
                "output_media_sha256": output["sha256"],
                "rollout_receipt_artifact_sha256": receipt["sha256"],
                "reward_audit_artifact_sha256": audit["sha256"],
                "source_role_digest": receipt_body["source_role_digest"],
            }
        )
        return {
            "owner_id": owner_id,
            "split": "fit",
            "role": caper.PURE_T2V_ROLE,
            "generation_mode": "pure_t2v",
            "action_family": "sit-down",
            "prompt": prompt,
            "generator_binding": generator,
            "output_media": output,
            "rollout_receipt": receipt,
            "reward_audit": audit,
            "eligible_as_preference_target": False,
            "eligible_as_training_target": False,
            "target_video_dependency": False,
        }

    def _rewrite_candidate_audit(
        self,
        manifest: dict,
        *,
        candidate_id: str,
        gate: str | None = None,
        gate_value: bool | None = None,
        action_grade: str | None = None,
        updates: dict[str, object] | None = None,
    ) -> None:
        self._rewrite_counter += 1
        endpoint = next(
            pair[role]
            for split in caper.SPLITS
            for pair in manifest["splits"][split]
            for role in ("winner", "loser")
            if pair[role]["candidate_id"] == candidate_id
        )
        audit_path = self.base_dir / endpoint["reward_audit"]["path"]
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        payload.pop("audit_sha256")
        if gate is not None:
            payload["preservation_hard_gates"][gate] = gate_value
            endpoint["preservation_hard_gates"][gate] = gate_value
        if action_grade is not None:
            payload["action_grade"] = action_grade
            payload["milestone_frame_evidence"] = self._milestones(action_grade)
        if updates:
            payload.update(deepcopy(updates))
        endpoint["reward_audit"] = self._json_artifact(
            f"rewrite-{self._rewrite_counter}-{candidate_id}-audit.json",
            payload,
            seal_field="audit_sha256",
        )
        ledger_path = self.base_dir / manifest["exposure_ledger"]["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger.pop("ledger_sha256")
        for entry in ledger["entries"]:
            if entry["rollout_id"] == candidate_id:
                entry["reward_audit_artifact_sha256"] = endpoint["reward_audit"][
                    "sha256"
                ]
        manifest["exposure_ledger"] = self._json_artifact(
            f"rewrite-{self._rewrite_counter}-ledger.json",
            ledger,
            seal_field="ledger_sha256",
        )
        population_path = self.base_dir / manifest["rollout_population_ledger"][
            "path"
        ]
        population_ledger = json.loads(population_path.read_text(encoding="utf-8"))
        population_ledger.pop("ledger_sha256")
        for population in population_ledger["populations"]:
            for attempt in population["attempts"]:
                if attempt["candidate_id"] == candidate_id:
                    attempt["reward_audit"] = endpoint["reward_audit"]
        manifest["rollout_population_ledger"] = self._json_artifact(
            f"rewrite-{self._rewrite_counter}-population-ledger.json",
            population_ledger,
            seal_field="ledger_sha256",
        )

    def _sync_ledger_to_manifest(self, manifest: dict) -> None:
        self._rewrite_counter += 1
        allowed = {
            pair[role]["candidate_id"]
            for split in caper.SPLITS
            for pair in manifest["splits"][split]
            for role in ("winner", "loser")
        } | {
            owner["owner_id"]
            for owner in manifest["pure_t2v_calibration_owners"]
        }
        ledger_path = self.base_dir / manifest["exposure_ledger"]["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger.pop("ledger_sha256")
        ledger["entries"] = [
            entry for entry in ledger["entries"] if entry["rollout_id"] in allowed
        ]
        manifest["exposure_ledger"] = self._json_artifact(
            f"sync-{self._rewrite_counter}-ledger.json",
            ledger,
            seal_field="ledger_sha256",
        )
        population_path = self.base_dir / manifest["rollout_population_ledger"][
            "path"
        ]
        population_ledger = json.loads(population_path.read_text(encoding="utf-8"))
        population_ledger.pop("ledger_sha256")
        population_ids = {
            pair["population_id"]
            for split in caper.SPLITS
            for pair in manifest["splits"][split]
        }
        population_ledger["populations"] = [
            population
            for population in population_ledger["populations"]
            if population["population_id"] in population_ids
        ]
        manifest["rollout_population_ledger"] = self._json_artifact(
            f"sync-{self._rewrite_counter}-population-ledger.json",
            population_ledger,
            seal_field="ledger_sha256",
        )

    def _unsigned_json_artifact(self, artifact: dict[str, object]) -> dict:
        payload = json.loads(
            (self.base_dir / str(artifact["path"])).read_text(encoding="utf-8")
        )
        for field in ("ledger_sha256", "receipt_sha256", "audit_sha256"):
            payload.pop(field, None)
        return payload

    def _promote_fit_population_to_k4(
        self, manifest: dict, *, seed2_score: float = 0.80
    ) -> tuple[dict[str, object], dict[str, object]]:
        pair = manifest["splits"]["fit"][0]
        manifest["splits"]["heldout"] = []
        context = {
            "source_id": pair["source_id"],
            "identity_id": pair["identity_id"],
            "scene_id": pair["scene_id"],
            "action_family": pair["action_family"],
            "edit_instruction": pair["edit_instruction"],
            "source_sha256": pair["source_media"]["sha256"],
            "population_id": pair["population_id"],
        }
        seed2 = self._candidate(
            role="winner",
            prefix="fit-k4-seed2",
            seed_id="seed-2",
            action_score_override=seed2_score,
            **context,
        )
        seed3 = self._candidate(
            role="loser",
            prefix="fit-k4-seed3",
            seed_id="seed-3",
            action_score_override=0.60,
            **context,
        )
        population_ledger = self._unsigned_json_artifact(
            manifest["rollout_population_ledger"]
        )
        population_ledger["population_size_k"] = 4
        population_ledger["preregistered_seed_ids"] = [
            "seed-0",
            "seed-1",
            "seed-2",
            "seed-3",
        ]
        population_ledger["populations"] = [
            row
            for row in population_ledger["populations"]
            if row["population_id"] == pair["population_id"]
        ]
        population = next(
            row
            for row in population_ledger["populations"]
            if row["population_id"] == pair["population_id"]
        )
        for endpoint in (seed2, seed3):
            ledger_entry = next(
                row
                for row in self._ledger_entries
                if row["rollout_id"] == endpoint["candidate_id"]
            )
            population["attempts"].append(
                {
                    "seed_id": endpoint["seed_id"],
                    "candidate_id": endpoint["candidate_id"],
                    "attempt_status": "success",
                    "failure_code": None,
                    "output_media": endpoint["output_media"],
                    "rollout_receipt": endpoint["rollout_receipt"],
                    "reward_audit": endpoint["reward_audit"],
                    "source_role_digest": ledger_entry["source_role_digest"],
                }
            )
        self._rewrite_counter += 1
        manifest["rollout_population_ledger"] = self._json_artifact(
            f"k4-{self._rewrite_counter}-population-ledger.json",
            population_ledger,
            seal_field="ledger_sha256",
        )
        exposure = self._unsigned_json_artifact(manifest["exposure_ledger"])
        exposure["entries"] = [
            row
            for row in exposure["entries"]
            if row["split"] != "heldout"
        ]
        extra_ids = {seed2["candidate_id"], seed3["candidate_id"]}
        exposure["entries"].extend(
            deepcopy(row)
            for row in self._ledger_entries
            if row["rollout_id"] in extra_ids
        )
        exposure["entries"] = sorted(
            exposure["entries"], key=lambda row: row["rollout_id"]
        )
        manifest["exposure_ledger"] = self._json_artifact(
            f"k4-{self._rewrite_counter}-exposure-ledger.json",
            exposure,
            seal_field="ledger_sha256",
        )
        return seed2, seed3

    def validate(self, manifest=None, *, verify_files=False):
        return caper.validate_manifest(
            self.manifest if manifest is None else manifest,
            verify_files=verify_files,
            base_dir=self.base_dir,
        )

    def materialize(self, manifest=None, *, verify_files=True):
        kwargs = {
            "verify_files": verify_files,
            "base_dir": self.base_dir,
        }
        payload = self.manifest if manifest is None else manifest
        if _AV_AVAILABLE or not verify_files:
            return caper.materialize_admission(payload, **kwargs)
        with mock.patch.object(
            caper,
            "_decode_exact81_media_artifact",
            side_effect=self._fake_decode,
        ):
            return caper.materialize_admission(payload, **kwargs)


class ValidAdmissionTests(CAPERManifestFixture):
    def test_valid_fit_and_heldout_pairs_materialize_with_hash_bindings(self) -> None:
        result = self.materialize()
        self.assertTrue(result["artifact_bytes_verified"])
        self.assertEqual(len(result["optimizer_pairs"]), 1)
        self.assertEqual(len(result["heldout_audit_records"]), 1)
        self.assertEqual(result["rejected_pairs"], [])
        self.assertEqual(
            result["authorized_optimizer_pair_ids"], ["pair-fit-dog"]
        )
        self.assertTrue(result["optimizer_step_allowed"])
        self.assertIsNone(result["zero_update_certificate"])

        materialized = result["optimizer_pairs"][0]
        self.assertTrue(materialized["optimizer_target_allowed"])
        for role in ("winner", "loser"):
            self.assertEqual(
                materialized[role]["generation_mode"],
                "source_conditioned_video_editing",
            )
            self.assertTrue(
                materialized[role]["eligible_as_preference_target"]
            )
            self.assertEqual(
                materialized[role]["conditioning_attestation"][
                    "input_kinds"
                ],
                list(caper.SOURCE_CONDITIONING_INPUTS),
            )
        original = self.manifest["splits"]["fit"][0]
        for role in ("winner", "loser"):
            for artifact in ("output_media", "rollout_receipt", "reward_audit"):
                self.assertEqual(
                    materialized[role][artifact]["sha256"],
                    original[role][artifact]["sha256"],
                )
        self.assertNotIn("path", json.dumps(result["heldout_audit_records"]))
        self.assertTrue(result["semantic_evidence_verified"])
        self.assertEqual(
            result["bindings"]["exposure_ledger_artifact_sha256"],
            self.manifest["exposure_ledger"]["sha256"],
        )
        caper.verify_materialization_seal(result)

    def test_pure_t2v_is_retained_only_as_calibration_owner(self) -> None:
        result = self.materialize()
        owner = result["pure_t2v_calibration_owners"][0]
        self.assertEqual(owner["role"], caper.PURE_T2V_ROLE)
        self.assertEqual(owner["generation_mode"], "pure_t2v")
        self.assertFalse(owner["eligible_as_preference_target"])
        self.assertFalse(owner["eligible_as_training_target"])
        endpoint_ids = {
            row[role]["candidate_id"]
            for row in result["optimizer_pairs"]
            for role in ("winner", "loser")
        }
        self.assertNotIn(owner["owner_id"], endpoint_ids)
        self.assertNotIn("path", json.dumps(owner))

    def test_dependency_light_module_has_no_training_side_effect(self) -> None:
        source = Path(caper.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("torch.distributed", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("backward(", source)

    def test_json_schema_is_closed_at_every_security_boundary(self) -> None:
        schema_path = (
            METHOD_ROOT
            / "schemas"
            / "bernini_caper_stage1_preference_candidates_v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        for definition in (
            "artifact",
            "admissionContract",
            "deployablePolicy",
            "rewardEvaluator",
            "policyBinding",
            "conditioningAttestation",
            "preservationHardGates",
            "preferenceEndpoint",
            "actionOrdering",
            "preferencePairCandidate",
            "pureT2VCalibrationOwner",
        ):
            self.assertFalse(schema["$defs"][definition]["additionalProperties"])
        endpoint = schema["$defs"]["preferenceEndpoint"]["properties"]
        self.assertEqual(
            endpoint["generation_mode"]["const"],
            "source_conditioned_video_editing",
        )

    def test_cli_validates_and_materializes_without_overwrite(self) -> None:
        manifest_path = self.base_dir / "manifest.json"
        output_path = self.base_dir / "admitted.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        def run_cli(arguments):
            if _AV_AVAILABLE:
                return caper.main(arguments)
            with mock.patch.object(
                caper,
                "_decode_exact81_media_artifact",
                side_effect=self._fake_decode,
            ):
                return caper.main(arguments)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(
                run_cli(
                    [
                        "validate",
                        "--manifest",
                        str(manifest_path),
                        "--verify-files",
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(stdout.getvalue())["status"], "valid")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(
                run_cli(
                    [
                        "materialize",
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
        written = caper.load_manifest(output_path)
        caper.verify_materialization_seal(written)
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "refusing to overwrite"):
            run_cli(
                [
                    "materialize",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                ]
            )


class PureT2VBoundaryTests(CAPERManifestFixture):
    def test_empty_owner_registry_is_rejected_even_when_ledger_is_present(self) -> None:
        bad = deepcopy(self.manifest)
        bad["pure_t2v_calibration_owners"] = []
        self._sync_ledger_to_manifest(bad)
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "must be non-empty"):
            self.materialize(bad)

    def test_pure_t2v_endpoint_is_contract_error(self) -> None:
        bad = deepcopy(self.manifest)
        bad["splits"]["fit"][0]["winner"]["generation_mode"] = "pure_t2v"
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "pure T2V"):
            self.validate(bad)

    def test_owner_cannot_claim_preference_or_training_target_eligibility(self) -> None:
        for key in (
            "eligible_as_preference_target",
            "eligible_as_training_target",
        ):
            with self.subTest(key=key):
                bad = deepcopy(self.manifest)
                bad["pure_t2v_calibration_owners"][0][key] = True
                with self.assertRaises(caper.CAPERAdmissionError):
                    self.validate(bad)

    def test_owner_media_receipt_or_audit_cannot_alias_endpoint(self) -> None:
        for artifact in ("output_media", "rollout_receipt", "reward_audit"):
            with self.subTest(artifact=artifact):
                bad = deepcopy(self.manifest)
                bad["pure_t2v_calibration_owners"][0][artifact] = deepcopy(
                    bad["splits"]["fit"][0]["winner"][artifact]
                )
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, "aliases a preference endpoint"
                ):
                    self.validate(bad)

    def test_owner_id_cannot_alias_candidate_id(self) -> None:
        bad = deepcopy(self.manifest)
        bad["pure_t2v_calibration_owners"][0]["owner_id"] = bad["splits"][
            "fit"
        ][0]["winner"]["candidate_id"]
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "owner IDs alias"):
            self.validate(bad)

    def test_transcoded_renamed_t2v_lineage_cannot_launder_as_endpoint(self) -> None:
        bad = deepcopy(self.manifest)
        owner = bad["pure_t2v_calibration_owners"][0]
        winner = bad["splits"]["fit"][0]["winner"]
        # Simulate a transcode/rename: both path and bytes now differ.
        owner_bytes = (self.base_dir / owner["output_media"]["path"]).read_bytes()
        relative = Path("artifacts") / "renamed-transcoded-owner-as-winner.mp4"
        raw = owner_bytes + b"transcoded-container"
        (self.base_dir / relative).write_bytes(raw)
        winner["output_media"] = {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        owner_receipt = json.loads(
            (self.base_dir / owner["rollout_receipt"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        owner_receipt.pop("receipt_sha256")
        owner_receipt["rollout_id"] = winner["candidate_id"]
        owner_receipt["output_media_sha256"] = winner["output_media"]["sha256"]
        owner_receipt["output_media_size_bytes"] = winner["output_media"][
            "size_bytes"
        ]
        winner["rollout_receipt"] = self._json_artifact(
            "laundered-owner-rollout.json",
            owner_receipt,
            seal_field="receipt_sha256",
        )
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "population attempt|rollout_role differs"
        ):
            self.materialize(bad)

    def test_relabelled_t2v_receipt_still_conflicts_with_immutable_lineage_ledger(self) -> None:
        bad = deepcopy(self.manifest)
        owner = bad["pure_t2v_calibration_owners"][0]
        pair = bad["splits"]["fit"][0]
        winner = pair["winner"]
        owner_receipt = json.loads(
            (self.base_dir / owner["rollout_receipt"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        owner_receipt.pop("receipt_sha256")
        owner_receipt.update(
            {
                "rollout_id": winner["candidate_id"],
                "rollout_role": caper.SOURCE_CANDIDATE_ROLE,
                "policy_id": self.policy["policy_id"],
                "inference_contract_sha256": self.policy[
                    "inference_contract_sha256"
                ],
                "source_id": pair["source_id"],
                "source_media_sha256": pair["source_media"]["sha256"],
                "action_family": pair["action_family"],
                "instruction_text_sha256": caper.text_sha256(
                    pair["edit_instruction"]
                ),
                "accepted_inputs": list(caper.SOURCE_CONDITIONING_INPUTS),
                "population_id": pair["population_id"],
                "seed_id": winner["seed_id"],
                "attempt_status": "success",
                "failure_code": None,
                "output_media_sha256": winner["output_media"]["sha256"],
                "output_media_size_bytes": winner["output_media"]["size_bytes"],
            }
        )
        owner_receipt["source_role_digest"] = caper.source_role_digest(
            {
                key: owner_receipt[key]
                for key in (
                    "lineage_root_sha256",
                    "rollout_role",
                    "split",
                    "policy_id",
                    "checkpoint_tree_sha256",
                    "inference_contract_sha256",
                    "source_id",
                    "source_media_sha256",
                    "action_family",
                    "instruction_text_sha256",
                    "accepted_inputs",
                    "sealed_gaussian_sha256",
                    "population_id",
                    "seed_id",
                    "attempt_status",
                )
            }
        )
        winner["rollout_receipt"] = self._json_artifact(
            "relabelled-owner-lineage.json",
            owner_receipt,
            seal_field="receipt_sha256",
        )
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "population attempt|ledger entry.*differs"
        ):
            self.materialize(bad)


class SameSourceOnPolicyTests(CAPERManifestFixture):
    def test_every_pair_coordinate_must_match_both_endpoints(self) -> None:
        mutations = {
            "source_id": "other-source",
            "identity_id": "other-identity",
            "scene_id": "other-scene",
            "action_family": "other-action",
            "edit_instruction": "Perform a different instruction.",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                bad = deepcopy(self.manifest)
                bad["splits"]["fit"][0]["loser"][key] = value
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, f"{key} differs"
                ):
                    self.validate(bad)

    def test_each_endpoint_must_bind_same_deployable_policy_checkpoint(self) -> None:
        mutations = {
            "policy_id": "off-policy-editor",
            "checkpoint_tree_sha256": _sha("d"),
            "inference_contract_sha256": _sha("e"),
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                bad = deepcopy(self.manifest)
                bad["splits"]["fit"][0]["winner"]["policy_binding"][key] = value
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, "not the sealed deployable policy"
                ):
                    self.validate(bad)

    def test_candidate_conditioning_must_bind_pair_source_media(self) -> None:
        bad = deepcopy(self.manifest)
        bad["splits"]["fit"][0]["winner"]["conditioning_attestation"][
            "source_media_sha256"
        ] = _sha("f")
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "different source media"):
            self.validate(bad)

    def test_endpoint_ids_and_artifacts_cannot_alias(self) -> None:
        bad_id = deepcopy(self.manifest)
        pair = bad_id["splits"]["fit"][0]
        pair["loser"]["candidate_id"] = pair["winner"]["candidate_id"]
        with self.assertRaises(caper.CAPERAdmissionError):
            self.validate(bad_id)

        for artifact in ("output_media", "rollout_receipt", "reward_audit"):
            with self.subTest(artifact=artifact):
                bad = deepcopy(self.manifest)
                pair = bad["splits"]["fit"][0]
                pair["loser"][artifact] = deepcopy(pair["winner"][artifact])
                with self.assertRaisesRegex(caper.CAPERAdmissionError, "alias"):
                    self.validate(bad)

    def test_media_receipt_and_audit_roles_cannot_alias_within_candidate(self) -> None:
        bad = deepcopy(self.manifest)
        winner = bad["splits"]["fit"][0]["winner"]
        winner["reward_audit"] = deepcopy(winner["rollout_receipt"])
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "aliases two security roles"
        ):
            self.validate(bad)

    def test_artifact_cannot_alias_across_candidates_or_source_role(self) -> None:
        cross_candidate = deepcopy(self.manifest)
        fit_winner = cross_candidate["splits"]["fit"][0]["winner"]
        heldout_winner = cross_candidate["splits"]["heldout"][0]["winner"]
        heldout_winner["reward_audit"] = deepcopy(fit_winner["reward_audit"])
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "aliases two security roles"
        ):
            self.validate(cross_candidate)

        source_alias = deepcopy(self.manifest)
        pair = source_alias["splits"]["fit"][0]
        pair["winner"]["reward_audit"] = deepcopy(pair["source_intake_receipt"])
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "source media/receipt"):
            self.validate(source_alias)


class SplitIsolationTests(CAPERManifestFixture):
    def _set_heldout_coordinate(self, key: str, value: str) -> dict:
        bad = deepcopy(self.manifest)
        pair = bad["splits"]["heldout"][0]
        pair[key] = value
        pair["winner"][key] = value
        pair["loser"][key] = value
        return bad

    def test_fit_heldout_identity_scene_and_action_family_are_independently_disjoint(self) -> None:
        fit = self.manifest["splits"]["fit"][0]
        for axis in ("identity_id", "scene_id", "action_family"):
            with self.subTest(axis=axis):
                bad = self._set_heldout_coordinate(axis, fit[axis])
                with self.assertRaisesRegex(caper.CAPERAdmissionError, f"{axis} leakage"):
                    self.validate(bad)

    def test_same_source_media_digest_cannot_cross_splits_under_a_new_id(self) -> None:
        bad = deepcopy(self.manifest)
        fit = bad["splits"]["fit"][0]
        heldout = bad["splits"]["heldout"][0]
        heldout["source_media"] = deepcopy(fit["source_media"])
        for role in ("winner", "loser"):
            heldout[role]["conditioning_attestation"][
                "source_media_sha256"
            ] = fit["source_media"]["sha256"]
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "source-media digest|source_media_sha256 leakage"
        ):
            self.validate(bad)

    def test_pair_split_must_match_container(self) -> None:
        bad = deepcopy(self.manifest)
        bad["splits"]["fit"][0]["split"] = "heldout"
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "split container"):
            self.validate(bad)


class NoCompensationAndNoTargetVideoTests(CAPERManifestFixture):
    def test_target_video_access_is_forbidden_at_pair_and_candidate_boundary(self) -> None:
        bad_pair = deepcopy(self.manifest)
        bad_pair["splits"]["fit"][0]["target_video_dependency"] = True
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "must be exactly false"):
            self.validate(bad_pair)

        for key in (
            "target_video_read",
            "target_video_latent_read",
            "paired_target_read",
        ):
            with self.subTest(key=key):
                bad = deepcopy(self.manifest)
                bad["splits"]["fit"][0]["winner"][
                    "conditioning_attestation"
                ][key] = True
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, "must be exactly false"
                ):
                    self.validate(bad)

    def test_unknown_target_video_path_is_rejected_by_closed_schema(self) -> None:
        bad = deepcopy(self.manifest)
        bad["splits"]["fit"][0]["target_video_path"] = "/secret/gt.mp4"
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "extra=.*target_video_path"):
            self.validate(bad)

    def test_scalar_reward_or_compensation_claim_is_contract_error(self) -> None:
        for key in (
            "scalar_reward_used_for_admission",
            "scalar_compensation_used",
        ):
            with self.subTest(key=key):
                bad = deepcopy(self.manifest)
                bad["splits"]["fit"][0][key] = True
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, "must be exactly false"
                ):
                    self.validate(bad)

    def test_weighted_score_cannot_be_smuggled_into_pair_or_endpoint(self) -> None:
        for container in ("pair", "endpoint"):
            with self.subTest(container=container):
                bad = deepcopy(self.manifest)
                pair = bad["splits"]["fit"][0]
                target = pair if container == "pair" else pair["winner"]
                target["weighted_scalar_score"] = 0.99
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, "weighted_scalar_score"
                ):
                    self.validate(bad)


class AdmissionAndZeroUpdateTests(CAPERManifestFixture):
    def test_unverified_materialization_is_forbidden(self) -> None:
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "requires verified"):
            self.materialize(verify_files=False)

    def test_failed_preservation_gate_rejects_pair_without_compensation(self) -> None:
        bad = deepcopy(self.manifest)
        bad["splits"]["fit"][0]["winner"]["preservation_hard_gates"][
            "identity"
        ] = False
        self._rewrite_candidate_audit(
            bad,
            candidate_id="fit-dog-winner",
            gate="identity",
            gate_value=False,
        )
        result = self.materialize(bad)
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertEqual(result["authorized_optimizer_pair_ids"], [])
        self.assertEqual(
            result["rejected_pairs"][0]["reasons"],
            ["winner_preservation_gate_failed:identity"],
        )
        # A valid held-out pair remains evaluation evidence but cannot enable a step.
        heldout = result["heldout_audit_records"]
        self.assertEqual(len(heldout), 1)
        self.assertFalse(heldout[0]["optimizer_target_allowed"])
        certificate = result["zero_update_certificate"]
        self.assertIsNotNone(certificate)
        self.assertFalse(certificate["optimizer_step_allowed"])
        self.assertEqual(certificate["authorized_optimizer_steps"], 0)
        self.assertFalse(certificate["gradient_application_allowed"])
        self.assertFalse(certificate["parameter_update_required"])
        caper.verify_materialization_seal(result)

    def test_action_tie_or_reversal_is_not_a_preference_pair(self) -> None:
        for winner_grade in (
            "transition_incomplete",
            "started_only",
        ):
            with self.subTest(winner_grade=winner_grade):
                bad = deepcopy(self.manifest)
                bad["splits"]["fit"][0]["action_ordering"][
                    "winner_action_grade"
                ] = winner_grade
                self._rewrite_candidate_audit(
                    bad,
                    candidate_id="fit-dog-winner",
                    action_grade=winner_grade,
                )
                result = self.materialize(bad)
                self.assertFalse(result["optimizer_step_allowed"])
                fit_rejected = [
                    row for row in result["rejected_pairs"] if row["split"] == "fit"
                ]
                self.assertEqual(
                    fit_rejected[0]["reasons"],
                    ["winner_action_not_strictly_better"],
                )

    def test_empty_fit_split_emits_explicit_zero_update_certificate(self) -> None:
        empty = deepcopy(self.manifest)
        empty["splits"]["fit"] = []
        self._sync_ledger_to_manifest(empty)
        # The fit T2V owner remains a calibration artifact and never enables training.
        result = self.materialize(empty)
        self.assertFalse(result["optimizer_step_allowed"])
        self.assertEqual(result["authorized_optimizer_pair_ids"], [])
        certificate = result["zero_update_certificate"]
        self.assertEqual(
            certificate["reason"],
            "no_valid_fit_same_source_on_policy_preference_pair",
        )
        self.assertEqual(certificate["rejected_fit_pair_ids"], [])
        self.assertEqual(certificate["authorized_optimizer_steps"], 0)

    def test_materialization_or_zero_certificate_tampering_breaks_seal(self) -> None:
        result = self.materialize()
        tampered = deepcopy(result)
        tampered["authorized_optimizer_pair_ids"] = []
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "SHA-256 differs"):
            caper.verify_materialization_seal(tampered)

        no_pair = deepcopy(self.manifest)
        no_pair["splits"]["fit"] = []
        self._sync_ledger_to_manifest(no_pair)
        zero = self.materialize(no_pair)
        zero["zero_update_certificate"]["authorized_optimizer_steps"] = 1
        # Re-sealing only the outer object cannot forge the inner certificate.
        unsigned = dict(zero)
        del unsigned["materialization_sha256"]
        zero["materialization_sha256"] = caper.object_sha256(unsigned)
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "certificate SHA-256 differs"
        ):
            caper.verify_materialization_seal(zero)

    def test_outer_resealed_heldout_path_injection_is_still_rejected(self) -> None:
        result = self.materialize()
        result["heldout_audit_records"][0]["path"] = "secret.mp4"
        unsigned = dict(result)
        del unsigned["materialization_sha256"]
        result["materialization_sha256"] = caper.object_sha256(unsigned)
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "extra=.*path"):
            caper.verify_materialization_seal(result)

    def test_outer_and_inner_resealed_t2v_optimizer_injection_is_rejected(self) -> None:
        result = self.materialize()
        pair = result["optimizer_pairs"][0]
        pair["winner"]["generation_mode"] = "pure_t2v"
        pair["pair_sha256"] = caper.object_sha256(
            {key: pair[key] for key in caper._PAIR_FIELDS}
        )
        unsigned = dict(result)
        del unsigned["materialization_sha256"]
        result["materialization_sha256"] = caper.object_sha256(unsigned)
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "decoded-evidence|pure T2V"
        ):
            caper.verify_materialization_seal(result)


class ArtifactAndParserAdversarialTests(CAPERManifestFixture):
    def test_opaque_text_file_cannot_pose_as_rollout_receipt(self) -> None:
        bad = deepcopy(self.manifest)
        bad["splits"]["fit"][0]["winner"]["rollout_receipt"] = self._artifact(
            "opaque-rollout-receipt.json"
        )
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "population attempt|canonical JSON"
        ):
            self.materialize(bad)

    def test_artifact_path_rejects_traversal_dot_and_ambiguous_separators(self) -> None:
        for path in (
            "../escape.mp4",
            "artifacts/../escape.mp4",
            "artifacts/./candidate.mp4",
            "artifacts//candidate.mp4",
        ):
            with self.subTest(path=path):
                bad = deepcopy(self.manifest)
                bad["splits"]["fit"][0]["winner"]["output_media"]["path"] = path
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, "dot/traversal components"
                ):
                    self.validate(bad)

    def test_media_receipt_and_reward_audit_hashes_are_verified(self) -> None:
        fields = (
            ("source_media", None),
            ("output_media", "winner"),
            ("rollout_receipt", "winner"),
            ("reward_audit", "loser"),
        )
        for artifact, role in fields:
            with self.subTest(artifact=artifact, role=role):
                bad = deepcopy(self.manifest)
                pair = bad["splits"]["fit"][0]
                target = pair[artifact] if role is None else pair[role][artifact]
                target["sha256"] = _sha("f")
                if artifact == "source_media":
                    for endpoint in ("winner", "loser"):
                        pair[endpoint]["conditioning_attestation"][
                            "source_media_sha256"
                        ] = _sha("f")
                with self.assertRaisesRegex(
                    caper.CAPERAdmissionError, "SHA-256 differs"
                ):
                    self.materialize(bad)

    def test_file_size_tampering_is_detected_before_hash(self) -> None:
        bad = deepcopy(self.manifest)
        bad["splits"]["fit"][0]["winner"]["rollout_receipt"][
            "size_bytes"
        ] += 1
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "size differs"):
            self.materialize(bad)

    def test_symlinked_evidence_is_rejected(self) -> None:
        link = self.base_dir / "artifacts" / "symlinked-reward.json"
        target_record = self._artifact("unregistered-symlink-target.json")
        target = self.base_dir / str(target_record["path"])
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable")
        bad = deepcopy(self.manifest)
        artifact = bad["splits"]["fit"][0]["winner"]["reward_audit"]
        artifact["path"] = link.relative_to(self.base_dir).as_posix()
        artifact["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        artifact["size_bytes"] = target.stat().st_size
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "symlinked"):
            self.materialize(bad)

    def test_duplicate_json_key_is_rejected(self) -> None:
        path = self.base_dir / "duplicate.json"
        path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "duplicate JSON key"):
            caper.load_manifest(path)

    def test_nonfinite_json_constant_is_rejected(self) -> None:
        path = self.base_dir / "nan.json"
        path.write_text('{"value":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "non-finite"):
            caper.load_manifest(path)


class PopulationMediaAndEvaluatorAttackTests(CAPERManifestFixture):
    def test_dynamic_k4_population_is_accepted_when_complete_and_fixed_rule_wins(self) -> None:
        manifest = deepcopy(self.manifest)
        self._promote_fit_population_to_k4(manifest)
        result = self.materialize(manifest)
        self.assertEqual(result["authorized_optimizer_pair_ids"], ["pair-fit-dog"])

    def test_k4_best_seed_cherry_pick_is_rejected(self) -> None:
        manifest = deepcopy(self.manifest)
        self._promote_fit_population_to_k4(manifest, seed2_score=0.99)
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "complete-population selection rule"
        ):
            self.materialize(manifest)

    def test_missing_preregistered_rollout_is_rejected(self) -> None:
        manifest = deepcopy(self.manifest)
        ledger = self._unsigned_json_artifact(
            manifest["rollout_population_ledger"]
        )
        ledger["populations"][0]["attempts"].pop()
        manifest["rollout_population_ledger"] = self._json_artifact(
            "attack-missing-rollout.json", ledger, seal_field="ledger_sha256"
        )
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "every preregistered attempt"
        ):
            self.materialize(manifest)

    def test_seed_imbalance_or_duplicate_seed_is_rejected(self) -> None:
        manifest = deepcopy(self.manifest)
        ledger = self._unsigned_json_artifact(
            manifest["rollout_population_ledger"]
        )
        ledger["populations"][0]["attempts"][1]["seed_id"] = "seed-0"
        manifest["rollout_population_ledger"] = self._json_artifact(
            "attack-seed-imbalance.json", ledger, seal_field="ledger_sha256"
        )
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "seed balance"):
            self.materialize(manifest)

    def test_fake_decoded_frame_metadata_is_rejected_against_media_bytes(self) -> None:
        manifest = deepcopy(self.manifest)
        endpoint = manifest["splits"]["fit"][0]["winner"]
        receipt = self._unsigned_json_artifact(endpoint["rollout_receipt"])
        receipt["decoded_rgb24_sha256"] = _sha("8")
        endpoint["rollout_receipt"] = self._json_artifact(
            "attack-fake-decoded-metadata.json",
            receipt,
            seal_field="receipt_sha256",
        )
        population = self._unsigned_json_artifact(
            manifest["rollout_population_ledger"]
        )
        for row in population["populations"]:
            for attempt in row["attempts"]:
                if attempt["candidate_id"] == endpoint["candidate_id"]:
                    attempt["rollout_receipt"] = endpoint["rollout_receipt"]
        manifest["rollout_population_ledger"] = self._json_artifact(
            "attack-fake-decoded-population.json",
            population,
            seal_field="ledger_sha256",
        )
        with self.assertRaisesRegex(
            caper.CAPERAdmissionError, "actual decoded bytes"
        ):
            self.materialize(manifest)

    def test_media_byte_tamper_is_rejected_before_receipt_trust(self) -> None:
        manifest = deepcopy(self.manifest)
        artifact = manifest["splits"]["fit"][0]["winner"]["output_media"]
        path = self.base_dir / artifact["path"]
        path.write_bytes(path.read_bytes() + b"post-receipt-tamper")
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "size differs"):
            self.materialize(manifest)

    def test_changed_evaluator_weights_cannot_reuse_old_audits(self) -> None:
        manifest = deepcopy(self.manifest)
        artifact = manifest["reward_evaluator"]["weights_artifact"]
        path = self.base_dir / artifact["path"]
        raw = path.read_bytes() + b"changed evaluator weights"
        path.write_bytes(raw)
        artifact["sha256"] = hashlib.sha256(raw).hexdigest()
        artifact["size_bytes"] = len(raw)
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "evaluator_binding differs"):
            self.materialize(manifest)

    def test_evaluator_forbidden_mask_read_is_rejected(self) -> None:
        manifest = deepcopy(self.manifest)
        self._rewrite_candidate_audit(
            manifest,
            candidate_id="fit-dog-winner",
            updates={"mask_read": True},
        )
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "mask_read"):
            self.materialize(manifest)

    def test_milestone_margin_tamper_is_rejected(self) -> None:
        manifest = deepcopy(self.manifest)
        endpoint = manifest["splits"]["fit"][0]["winner"]
        audit = self._unsigned_json_artifact(endpoint["reward_audit"])
        milestones = audit["milestone_frame_evidence"]
        milestones[1]["margin"] += 0.1
        self._rewrite_candidate_audit(
            manifest,
            candidate_id=endpoint["candidate_id"],
            updates={"milestone_frame_evidence": milestones},
        )
        with self.assertRaisesRegex(caper.CAPERAdmissionError, "margin differs"):
            self.materialize(manifest)

    @unittest.skipUnless(_AV_AVAILABLE, "real PyAV check runs in AUH vace")
    def test_real_pyav_decodes_exact81_fixed_rgb_geometry(self) -> None:
        artifact = self.manifest["splits"]["fit"][0]["winner"]["output_media"]
        decoded = caper._decode_exact81_media_artifact(
            artifact, base_dir=self.base_dir, label="real-PyAV attack fixture"
        )
        self.assertEqual(decoded["decoded_frame_count"], 81)
        self.assertEqual(decoded["decoded_fps_numerator"], 25)
        self.assertEqual(decoded["decoded_fps_denominator"], 1)
        self.assertRegex(decoded["decoded_rgb24_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
