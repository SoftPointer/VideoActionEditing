from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_pair_v5_native_rv2v4_rollout as rollout
import pair_v5_native_rollout_spec as contract


LAUNCHER = METHOD_ROOT / "scripts/auh_pair_v5_native_rv2v4_rollout_dual4.sbatch"
ACTION_CANDIDATE_SPEC = (
    METHOD_ROOT / "assets/pair_v5_native_rv2v4_action_candidates_v1.json"
)
CORE4_ACTION_POPULATION_SPEC = (
    METHOD_ROOT / "assets/pair_v5_native_rv2v4_core4_action_population_v1.json"
)
T2V_CORE4_BANK_SPEC = (
    METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"
)


def _candidate(candidate_id: str, seed: int, *, txt: float = 4.0) -> dict[str, object]:
    caption = f"A small brown dog in a kitchen picks up a bone and holds it, variant {candidate_id}."
    return {
        "candidate_id": candidate_id,
        "source_video": f"/dataset/{candidate_id}.mp4",
        "source_video_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(),
        "complete_caption": caption,
        "complete_caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
        "caption_contract": contract.CAPTION_CONTRACT,
        "seed": seed,
        "guidance": {"omega_txt": txt, "omega_vid": 1.25, "omega_img": 4.5},
    }


def _spec() -> dict[str, object]:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "sampling_contract": contract.SAMPLING_CONTRACT,
        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
        "groups": [
            {
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "candidates": [_candidate("a-seed11", 11), _candidate("a-seed29", 29, txt=5.0)],
            },
            {
                "group_id": "sp4-b",
                "visible_gpus": [4, 5, 6, 7],
                "candidates": [_candidate("b-seed47", 47)],
            },
        ],
    }


class PairV5NativeRolloutSpecTests(unittest.TestCase):
    def _write_spec(self, root: Path, value: object | None = None) -> tuple[Path, str]:
        path = root / "sealed.json"
        path.write_bytes(contract.canonical_json_bytes(_spec() if value is None else value) + b"\n")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_exact81_two_sp4_and_multiple_non_hardcoded_seeds(self) -> None:
        normalized = contract.validate_root_spec(_spec())
        self.assertEqual(normalized["sampling_contract"]["num_frames"], 81)
        self.assertEqual(normalized["sampling_contract"]["latent_frames"], 21)
        self.assertEqual(normalized["sampling_contract"]["num_inference_steps"], 40)
        self.assertEqual(normalized["sampling_contract"]["source_reference_indices"], [0, 27, 53, 80])
        self.assertEqual([g["visible_gpus"] for g in normalized["groups"]], [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(
            [c["seed"] for g in normalized["groups"] for c in g["candidates"]],
            [11, 29, 47],
        )
        self.assertEqual(normalized["groups"][0]["candidates"][1]["guidance"]["omega_txt"], 5.0)

    def test_preregistered_action_bank_is_two_sources_by_four_seeds(self) -> None:
        payload = json.loads(ACTION_CANDIDATE_SPEC.read_text(encoding="utf-8"))
        normalized = contract.validate_root_spec(payload)
        groups = normalized["groups"]
        self.assertEqual([len(group["candidates"]) for group in groups], [4, 4])
        self.assertEqual(
            [[candidate["seed"] for candidate in group["candidates"]] for group in groups],
            [[2026080811, 2026080812, 2026080813, 2026080814]] * 2,
        )
        self.assertEqual(
            len({candidate["source_video_sha256"] for group in groups for candidate in group["candidates"]}),
            2,
        )
        self.assertTrue(
            all(
                candidate["guidance"] == contract.DEFAULT_GUIDANCE
                for group in groups
                for candidate in group["candidates"]
            )
        )

    def test_core4_action_population_is_source_bound_and_action_only(self) -> None:
        t2v = json.loads(T2V_CORE4_BANK_SPEC.read_text(encoding="utf-8"))
        t2v_groups = {
            group["group_id"]: group["candidates"] for group in t2v["groups"]
        }
        action_rows = [
            row
            for rows in t2v_groups.values()
            for row in rows
            if row["semantic_branch"] == "action"
        ]
        self.assertEqual(len(action_rows), 4)
        self.assertEqual(len({row["geometry_source_video_sha256"] for row in action_rows}), 4)

        action_by_binding: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in action_rows:
            action_by_binding[
                (row["geometry_source_video_sha256"], row["full_t2v_caption"])
            ].append(row)
        self.assertTrue(all(len(rows) == 1 for rows in action_by_binding.values()))

        payload = json.loads(CORE4_ACTION_POPULATION_SPEC.read_text(encoding="utf-8"))
        normalized = contract.validate_root_spec(payload)
        self.assertEqual([len(group["candidates"]) for group in normalized["groups"]], [4, 4])
        population = [
            candidate
            for group in normalized["groups"]
            for candidate in group["candidates"]
        ]
        self.assertEqual(len(population), 8)

        candidates_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
        matched_cells: dict[str, dict[str, object]] = {}
        negative_bindings = {
            (row["geometry_source_video_sha256"], row["full_t2v_caption"])
            for rows in t2v_groups.values()
            for row in rows
            if row["semantic_branch"] != "action"
        }
        forbidden_branch_tokens = (
            "noop", "incomplete", "reverse", "shuffle", "wrong_actor",
            "wrong_object", "camera_only", "appearance_only",
            "generic_wrong_motion", "negative",
        )
        for candidate in population:
            binding = (
                candidate["source_video_sha256"], candidate["complete_caption"]
            )
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertIn(binding, action_by_binding)
                self.assertNotIn(binding, negative_bindings)
                self.assertFalse(
                    any(token in candidate["candidate_id"].lower() for token in forbidden_branch_tokens)
                )
                row = action_by_binding[binding][0]
                self.assertEqual(candidate["source_video"], row["geometry_source_video"])
                self.assertEqual(
                    candidate["complete_caption_sha256"],
                    row["full_t2v_caption_utf8_sha256"],
                )
                matched_cells[candidate["source_video_sha256"]] = row
                candidates_by_source[candidate["source_video_sha256"]].append(candidate)

        expected_sources = {
            row["geometry_source_video_sha256"] for row in action_rows
        }
        self.assertEqual(set(candidates_by_source), expected_sources)
        for candidates in candidates_by_source.values():
            self.assertEqual(len(candidates), 2)
            self.assertEqual(
                {candidate["seed"] for candidate in candidates},
                {2026080901, 2026080902},
            )

        expected_family_splits: dict[str, set[str]] = defaultdict(set)
        for row in action_rows:
            expected_family_splits[row["action_family_id"]].add(row["analysis_split"])
        observed_family_splits: dict[str, set[str]] = defaultdict(set)
        for row in matched_cells.values():
            observed_family_splits[row["action_family_id"]].add(row["analysis_split"])
        self.assertEqual(observed_family_splits, expected_family_splits)
        self.assertEqual(set(observed_family_splits), {"dog-sit-facing-camera", "human-rise-to-stand"})
        self.assertTrue(
            all(splits == {"fit", "confirmation"} for splits in observed_family_splits.values())
        )

        for group in normalized["groups"]:
            expected_group_sources = {
                row["geometry_source_video_sha256"]
                for row in t2v_groups[group["group_id"]]
                if row["semantic_branch"] == "action"
            }
            self.assertEqual(
                {candidate["source_video_sha256"] for candidate in group["candidates"]},
                expected_group_sources,
            )

    def test_schema_is_closed_against_privileged_inputs(self) -> None:
        for forbidden in (
            "target_video", "t2v_proposal_media", "donor_video", "mask", "flow",
            "pose", "track", "trajectory", "initial_noise",
        ):
            value = _spec()
            value["groups"][0]["candidates"][0][forbidden] = "/forbidden"
            with self.subTest(forbidden=forbidden), self.assertRaises(contract.PairRolloutSpecError):
                contract.validate_root_spec(value)
        self.assertEqual(contract.SEMANTIC_INPUT_CLOSURE["accepted"], ["source_video", "complete_caption"])
        self.assertTrue(all(contract.SEMANTIC_INPUT_CLOSURE[key] is False for key in contract.SEMANTIC_INPUT_CLOSURE if key != "accepted"))

    def test_raw_spec_hash_seals_plan_and_candidate_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = self._write_spec(root)
            manifest = contract.materialize_plan(
                spec_path=path, expected_sha256=digest, output_dir=root / "plan"
            )
            self.assertEqual(manifest["root_spec_raw_sha256"], digest)
            self.assertEqual(len(manifest["candidate_records"]), 3)
            envelope = contract.load_candidate_envelope(
                manifest["candidate_records"][1]["path"], digest
            )
            self.assertEqual(envelope["candidate"]["seed"], 29)
            self.assertRegex(envelope["candidate_envelope_sha256"], r"^[0-9a-f]{64}$")
            with self.assertRaises(contract.PairRolloutSpecError):
                contract.load_sealed_spec(path, "f" * 64)

    def test_candidate_wrapper_forwards_sealed_seed_and_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = self._write_spec(root)
            manifest = contract.materialize_plan(
                spec_path=path, expected_sha256=digest, output_dir=root / "plan"
            )
            candidate_path = manifest["candidate_records"][1]["path"]
            captured: list[str] = []
            old_main, old_bind = rollout.native.main, rollout._bind_receipt
            old_values = (rollout.native.OMEGA_TEXT, rollout.native.OMEGA_VIDEO, rollout.native.OMEGA_IMAGE)
            old_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
            try:
                os.environ["ROCR_VISIBLE_DEVICES"] = "0,1,2,3"
                rollout.native.main = lambda argv: captured.extend(argv) or 0
                rollout._bind_receipt = lambda args, envelope: None
                status = rollout.main([
                    "--candidate-spec", candidate_path,
                    "--expected-root-spec-sha256", digest,
                    "--output-dir", str(root / "candidate-output"),
                    "--bernini-root", "/bernini", "--veomni-root", "/veomni",
                    "--checkpoint", "/checkpoint",
                    "--checkpoint-content-manifest", "/checkpoint.sha256",
                    "--method-source-revision", "a" * 40,
                    "--method-source-archive-sha256", "b" * 64,
                ])
                self.assertEqual(status, 0)
                self.assertEqual(captured[captured.index("--seed") + 1], "29")
                self.assertEqual(captured[captured.index("--arms") + 1], "rv2v")
                self.assertEqual(captured[captured.index("--num-inference-steps") + 1], "40")
                self.assertEqual(rollout.native.OMEGA_TEXT, 5.0)
                self.assertEqual(rollout.native.OMEGA_VIDEO, 1.25)
                self.assertEqual(rollout.native.OMEGA_IMAGE, 4.5)
            finally:
                rollout.native.main, rollout._bind_receipt = old_main, old_bind
                rollout.native.OMEGA_TEXT, rollout.native.OMEGA_VIDEO, rollout.native.OMEGA_IMAGE = old_values
                if old_visible is None:
                    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                else:
                    os.environ["ROCR_VISIBLE_DEVICES"] = old_visible

    def test_pair_receipt_binds_spec_candidate_native_receipt_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = self._write_spec(root)
            manifest = contract.materialize_plan(
                spec_path=path, expected_sha256=digest, output_dir=root / "plan"
            )
            envelope = contract.load_candidate_envelope(manifest["candidate_records"][0]["path"], digest)
            output = root / "output"; output.mkdir()
            candidate = envelope["candidate"]
            artifact = {"path": str(output / "artifact"), "sha256": "1" * 64}
            native_receipt = {
                "receipt_digest": "2" * 64,
                "sampling": {"rv2v": {
                    "num_frames": 81, "num_inference_steps": 40, "seed": candidate["seed"],
                    **candidate["guidance"], "target_initialization": contract.TARGET_INITIALIZATION,
                }},
                "outputs": {"rv2v": {"path": str(output / "rv2v.mp4"), "sha256": "3" * 64,
                    "normalized_clean_latent": {**artifact, "native_sampler_before_vae_decode": True}}},
                "initial_noise_artifacts": {"rv2v": artifact},
            }
            (output / "receipt.json").write_bytes(contract.canonical_json_bytes(native_receipt) + b"\n")
            args = argparse.Namespace(output_dir=str(output))
            old_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
            os.environ["ROCR_VISIBLE_DEVICES"] = "0,1,2,3"
            try:
                rollout._bind_receipt(args, envelope)
            finally:
                if old_visible is None:
                    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                else:
                    os.environ["ROCR_VISIBLE_DEVICES"] = old_visible
            receipt = json.loads((output / "pair-v5-rollout-receipt.json").read_bytes())
            declared = receipt.pop("receipt_digest")
            self.assertEqual(hashlib.sha256(contract.canonical_json_bytes(receipt)).hexdigest(), declared)
            self.assertEqual(receipt["root_spec_raw_sha256"], digest)
            self.assertEqual(receipt["candidate"]["seed"], 11)
            self.assertEqual(receipt["runtime_topology"]["rocr_visible_devices"], "0,1,2,3")
            self.assertEqual(receipt["artifacts"]["predecode_clean_latent"]["native_sampler_before_vae_decode"], True)


class PairV5NativeRolloutLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_and_dual_sp4_topology(self) -> None:
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.text)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.text)
        self.assertEqual(self.text.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.text)

    def test_launcher_is_create_only_and_uses_sealed_spec_not_fixed_seed(self) -> None:
        self.assertIn("PAIR_V5_ROLLOUT_SPEC_SHA256", self.text)
        self.assertIn('[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]]', self.text)
        self.assertIn('[[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]]', self.text)
        self.assertIn("seed_source=sealed_spec", self.text)
        self.assertNotIn("expected_seed=2027", self.text)
        self.assertNotIn('--seed "2027"', self.text)

    def test_launcher_runtime_has_only_deploy_available_semantic_inputs(self) -> None:
        invocation = self.text.split('"${method_root}/infer_pair_v5_native_rv2v4_rollout.py"', 1)[1].split("\n  done", 1)[0]
        for forbidden in (
            "--target", "--proposal", "--donor", "--mask", "--flow", "--pose",
            "--track", "--trajectory", "--initial-noise",
        ):
            self.assertNotIn(forbidden, invocation)
        self.assertIn("target=false t2v_proposal=false donor=false", self.text)
        self.assertIn("PAIR_V5_NATIVE_RV2V4_DUAL4_STRONG_AUDIT_OK", self.text)


if __name__ == "__main__":
    unittest.main()
