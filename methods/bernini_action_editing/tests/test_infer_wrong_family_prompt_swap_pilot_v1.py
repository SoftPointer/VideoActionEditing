from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for entry in (str(METHOD_ROOT), str(TEST_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import infer_wrong_family_prompt_swap_pilot_v1 as runtime  # noqa: E402
import test_pair_v5_t2v_calibration_bank as bank_fixtures  # noqa: E402
import wrong_family_prompt_swap_pilot_v1 as pilot  # noqa: E402


REGISTRY = METHOD_ROOT / "assets" / pilot.REGISTRY_ASSET_BASENAME
SOURCE_BANK = METHOD_ROOT / "assets" / pilot.SOURCE_BANK_BASENAME
ASSET_ROOT = METHOD_ROOT / "assets"
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_generate_wrong_family_prompt_swap_pilot_v1_dual4.sbatch"
)


def _reseal(value: dict, digest_field: str) -> dict:
    result = copy.deepcopy(value)
    result.pop(digest_field, None)
    result[digest_field] = pilot.sha256_bytes(pilot.canonical_json_bytes(result))
    return result


def _add_gaussian_tensor_key(output: Path) -> None:
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["initial_noise_artifacts"]["t2v"][
        "tensor_key"
    ] = "official_initial_gaussian"
    receipt = _reseal(receipt, "receipt_digest")
    receipt_path.write_bytes(pilot.canonical_json_bytes(receipt) + b"\n")


class WrongFamilyRuntimeFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.plan = pilot.build_generation_plan(
            registry_path=REGISTRY.resolve(),
            expected_registry_sha256=pilot.REGISTRY_RAW_SHA256,
            source_bank_path=SOURCE_BANK.resolve(),
            seed_scan_roots=[ASSET_ROOT.resolve()],
        )
        self.plan_path = self.root / "generation-plan.json"
        self.plan_path.write_bytes(pilot.canonical_json_bytes(self.plan) + b"\n")
        self.plan_sha = pilot.file_sha256(self.plan_path)
        self.runtime_root = self.root / "runtime"
        self.manifest = runtime.materialize_runtime_plan(
            generation_plan_path=self.plan_path,
            expected_generation_plan_sha256=self.plan_sha,
            output_dir=self.runtime_root,
        )
        self.manifest_path = self.runtime_root / runtime.RUNTIME_MANIFEST_BASENAME
        self.manifest_sha = pilot.file_sha256(self.manifest_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()


class WrongFamilyRuntimePlanTests(WrongFamilyRuntimeFixture):
    def test_materialization_is_exact20_fixed_dual_sp4_and_replayable(self) -> None:
        checked, raw_sha, _ = runtime.load_runtime_plan(
            self.manifest_path,
            self.manifest_sha,
            plan=self.plan,
            generation_plan_raw_sha256=self.plan_sha,
        )
        self.assertEqual(raw_sha, self.manifest_sha)
        self.assertEqual(checked["candidate_count"], 20)
        self.assertEqual(
            [(row["group_id"], row["visible_gpus"], row["iid"]) for row in checked["group_layout"]],
            [
                ("sp4-a", [0, 1, 2, 3], pilot.PROSPECTIVE_IIDS[0]),
                ("sp4-b", [4, 5, 6, 7], pilot.PROSPECTIVE_IIDS[1]),
            ],
        )
        self.assertEqual(
            [row["ordinal"] for row in checked["candidate_records"][:10]],
            list(range(10)),
        )
        self.assertEqual(
            [row["ordinal"] for row in checked["candidate_records"][10:]],
            list(range(10)),
        )
        for record in checked["candidate_records"]:
            envelope, raw, _ = runtime.load_planned_candidate_envelope(
                envelope_path=record["candidate_envelope_path"],
                plan=self.plan,
                generation_plan_raw_sha256=self.plan_sha,
                runtime_plan=checked,
            )
            self.assertEqual(raw, record["candidate_envelope_sha256"])
            self.assertEqual(
                envelope["candidate_envelope_digest"],
                record["candidate_envelope_digest"],
            )
            self.assertEqual(
                envelope["candidate"]["seed"],
                pilot.FRESH_SEEDS[record["iid"]],
            )

    def test_runtime_manifest_or_envelope_authority_tamper_fails(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["candidate_records"][0]["visible_gpus"] = [4, 5, 6, 7]
        changed = _reseal(changed, "runtime_plan_digest")
        with self.assertRaises(runtime.WrongFamilyRuntimeError):
            runtime.validate_runtime_plan(
                changed,
                plan=self.plan,
                generation_plan_raw_sha256=self.plan_sha,
            )

        record = self.manifest["candidate_records"][0]
        envelope = json.loads(
            Path(record["candidate_envelope_path"]).read_text(encoding="utf-8")
        )
        envelope["candidate"]["seed"] += 1
        envelope = _reseal(envelope, "candidate_envelope_digest")
        with self.assertRaises(runtime.WrongFamilyRuntimeError):
            runtime.validate_candidate_envelope(
                envelope,
                plan=self.plan,
                generation_plan_raw_sha256=self.plan_sha,
            )

    def test_render_forwards_only_sealed_generation_caption_and_native_t2v(self) -> None:
        record = self.manifest["candidate_records"][0]
        candidate = self.plan["prospective_cells"][0]["generation_candidates"][0]
        query_texts = {
            prompt["utf8_text"]
            for prompt in self.plan["prospective_cells"][0]["query_prompts"].values()
        }
        captured = []
        prior_main = runtime.native.main
        prior_env = {
            key: os.environ.get(key)
            for key in ("ROCR_VISIBLE_DEVICES", "RANK", "WORLD_SIZE")
        }
        try:
            os.environ["ROCR_VISIBLE_DEVICES"] = "0,1,2,3"
            os.environ["RANK"] = "1"
            os.environ["WORLD_SIZE"] = "4"
            runtime.native.main = lambda argv: captured.append(list(argv)) or 0
            status = runtime.render_candidate(
                argparse.Namespace(
                    generation_plan=str(self.plan_path),
                    expected_generation_plan_sha256=self.plan_sha,
                    runtime_plan=str(self.manifest_path),
                    expected_runtime_plan_sha256=self.manifest_sha,
                    candidate_envelope=record["candidate_envelope_path"],
                    output_dir=str(self.root / "uncreated-output"),
                    bernini_root="/bernini",
                    veomni_root="/veomni",
                    checkpoint="/checkpoint",
                    checkpoint_content_manifest="/checkpoint.sha256",
                    method_source_revision="a" * 40,
                    method_source_archive_sha256="b" * 64,
                )
            )
        finally:
            runtime.native.main = prior_main
            for key, value in prior_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(status, 0)
        self.assertEqual(len(captured), 1)
        argv = captured[0]
        self.assertEqual(argv[argv.index("--arms") + 1], "t2v")
        self.assertEqual(argv[argv.index("--num-inference-steps") + 1], "40")
        self.assertEqual(argv[argv.index("--seed") + 1], str(candidate["seed"]))
        prompt = argv[argv.index("--action-prompt") + 1]
        self.assertEqual(prompt, candidate["full_t2v_caption"])
        self.assertNotIn(prompt, query_texts)
        for forbidden in (
            "--target",
            "--reference",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--initial-noise",
        ):
            self.assertNotIn(forbidden, argv)


class WrongFamilyRenderedTupleTests(WrongFamilyRuntimeFixture):
    def _render_fake_tuple(self) -> Path:
        output_root = self.root / "rendered"
        output_root.mkdir()
        prior_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
        try:
            for record in self.manifest["candidate_records"]:
                envelope, envelope_sha, envelope_path = (
                    runtime.load_planned_candidate_envelope(
                        envelope_path=record["candidate_envelope_path"],
                        plan=self.plan,
                        generation_plan_raw_sha256=self.plan_sha,
                        runtime_plan=self.manifest,
                    )
                )
                candidate = envelope["candidate"]
                candidate_root = output_root / candidate["candidate_id"]
                candidate_root.mkdir()
                gaussian_value = ("official-gaussian:" + record["iid"]).encode()
                bank_fixtures._native_receipt(
                    candidate_root,
                    candidate,
                    gaussian_payload=gaussian_value,
                    gaussian_container_payload=(
                        gaussian_value
                        + b":container:"
                        + candidate["semantic_branch"].encode()
                    ),
                )
                _add_gaussian_tensor_key(candidate_root)
                os.environ["ROCR_VISIBLE_DEVICES"] = ",".join(
                    str(item) for item in record["visible_gpus"]
                )
                runtime.bind_candidate_receipt(
                    output_dir=candidate_root,
                    envelope=envelope,
                    envelope_path=envelope_path,
                    envelope_raw_sha256=envelope_sha,
                    generation_plan=self.plan,
                    generation_plan_raw_sha256=self.plan_sha,
                    runtime_plan=self.manifest,
                    runtime_plan_raw_sha256=self.manifest_sha,
                )
        finally:
            if prior_visible is None:
                os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            else:
                os.environ["ROCR_VISIBLE_DEVICES"] = prior_visible
        return output_root

    def test_exact20_outputs_bind_latents_mp4_gaussians_and_master(self) -> None:
        output_root = self._render_fake_tuple()
        master = runtime.audit_rendered_pilot(
            generation_plan_path=self.plan_path,
            expected_generation_plan_sha256=self.plan_sha,
            runtime_plan_path=self.manifest_path,
            expected_runtime_plan_sha256=self.manifest_sha,
            output_dir=output_root,
        )
        self.assertEqual(master["candidate_count"], 20)
        self.assertEqual(master["cell_count"], 2)
        self.assertEqual(len(master["candidate_receipts"]), 20)
        self.assertEqual(len(master["same_cell_gaussian_proofs"]), 2)
        self.assertTrue(
            all(
                proof["all_ten_tensor_values_equal"]
                for proof in master["same_cell_gaussian_proofs"]
            )
        )
        self.assertNotEqual(
            master["same_cell_gaussian_proofs"][0]["raw_value_sha256"],
            master["same_cell_gaussian_proofs"][1]["raw_value_sha256"],
        )
        self.assertTrue((output_root / runtime.GAUSSIAN_BINDING_BASENAME).is_file())
        self.assertTrue((output_root / runtime.AUDIT_PLAN_BASENAME).is_file())
        self.assertTrue(
            (output_root / runtime.BLINDED_REVIEW_PACKET_BASENAME).is_file()
        )
        master_path = output_root / runtime.MASTER_RECEIPT_BASENAME
        self.assertTrue(master_path.is_file())
        checked = runtime.validate_master_receipt(
            json.loads(master_path.read_text(encoding="utf-8"))
        )
        self.assertFalse(checked["interpretation"]["editor"])
        self.assertFalse(checked["interpretation"]["scientific_critic"])
        self.assertFalse(checked["interpretation"]["optimizer_authorized"])
        self.assertFalse(
            checked["interpretation"]["query_prompts_consumed_by_generator"]
        )
        binding = json.loads(
            (output_root / runtime.GAUSSIAN_BINDING_BASENAME).read_text(
                encoding="utf-8"
            )
        )
        audit = json.loads(
            (output_root / runtime.AUDIT_PLAN_BASENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(len(binding["bindings"]), 20)
        self.assertEqual(audit["judgment_count"], 24)
        blinded = json.loads(
            (output_root / runtime.BLINDED_REVIEW_PACKET_BASENAME).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(blinded["review_item_count"], 24)
        self.assertEqual(blinded["unique_media_count"], 20)
        self.assertEqual(
            len({row["opaque_media_id"] for row in blinded["review_items"]}), 20
        )
        blinded_text = json.dumps(blinded, sort_keys=True)
        for forbidden_key in (
            '"candidate_id"',
            '"semantic_branch"',
            '"required_outcome"',
            '"full_t2v_caption"',
        ):
            self.assertNotIn(forbidden_key, blinded_text)
        for candidate in (
            row
            for cell in self.plan["prospective_cells"]
            for row in cell["generation_candidates"]
        ):
            self.assertNotIn(candidate["candidate_id"], blinded_text)
            self.assertNotIn(candidate["full_t2v_caption"], blinded_text)
        for row in blinded["review_items"]:
            self.assertEqual(
                Path(row["opaque_media_path"]).parent.name,
                runtime.BLINDED_MEDIA_DIR_BASENAME,
            )
            self.assertTrue(row["semantic_branch_hidden"])
            self.assertTrue(row["required_outcome_hidden"])
        for row in master["candidate_receipts"]:
            receipt = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
            self.assertIn("predecode_clean_latent", receipt["artifacts"])
            self.assertEqual(receipt["artifacts"]["mp4"]["frame_count"], 81)
            self.assertEqual(
                receipt["artifacts"]["official_initial_gaussian"]["dtype"],
                "torch.float32",
            )

    def test_candidate_receipt_cannot_gain_query_or_optimizer_authority(self) -> None:
        output_root = self._render_fake_tuple()
        record = self.manifest["candidate_records"][0]
        receipt_path = (
            output_root / record["candidate_id"] / runtime.CANDIDATE_RECEIPT_BASENAME
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["interpretation"]["query_prompts_consumed_by_generator"] = True
        receipt["interpretation"]["optimizer_authorized"] = True
        receipt = _reseal(receipt, "receipt_digest")
        with self.assertRaises(runtime.WrongFamilyRuntimeError):
            runtime.validate_candidate_receipt(
                receipt,
                receipt_path=receipt_path,
                candidate_root=receipt_path.parent,
                plan=self.plan,
                generation_plan_raw_sha256=self.plan_sha,
                runtime_plan=self.manifest,
                runtime_plan_raw_sha256=self.manifest_sha,
            )


class WrongFamilyAuhLauncherTests(unittest.TestCase):
    def test_launcher_is_all8_dual_sp4_exact81_native_and_strong_audited(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn('run_group sp4-a "0,1,2,3"', text)
        self.assertIn('run_group sp4-b "4,5,6,7"', text)
        self.assertEqual(text.count("--nproc_per_node=4"), 1)
        self.assertIn("wrong_family_prompt_swap_pilot_v1.py\" build-plan", text)
        self.assertIn('--seed-scan-root "${registry}"', text)
        self.assertIn('--seed-scan-root "${seed_inventory_root}"', text)
        self.assertIn(
            '"methods/bernini_action_editing/scripts/'
            'auh_generate_wrong_family_prompt_swap_pilot_v1_dual4.sbatch"',
            text,
        )
        self.assertIn('${BASH_SOURCE[0]}', text)
        self.assertIn('running launcher differs from source archive', text)
        self.assertIn("infer_wrong_family_prompt_swap_pilot_v1.py\" render-candidate", text)
        self.assertIn("infer_wrong_family_prompt_swap_pilot_v1.py\" audit-bank", text)
        self.assertIn("wrong-family-generation-master-receipt.json", text)
        self.assertIn("wrong-family-gaussian-binding.json", text)
        self.assertIn("wrong-family-private-adjudication-plan.json", text)
        self.assertIn("wrong-family-blinded-review-packet.json", text)
        self.assertNotIn("rv2v", text.lower())
        for forbidden in ("--target", "--mask", "--flow", "--pose", "--track"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
