from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_t2v_factorial_bank as bank  # noqa: E402
import infer_native_multivideo_motion_donor_oracle as runtime  # noqa: E402


MICRO_SPEC = METHOD_ROOT / "assets/dmiq_cdf_dog_t2v_micro_spec_v2.json"
REVISION = "2" * 40
ARCHIVE_SHA256 = "3" * 64


def _manifest() -> dict:
    spec = json.loads(MICRO_SPEC.read_text(encoding="utf-8"))
    return bank.build_manifest(
        spec,
        method_source_revision=REVISION,
        method_source_archive_sha256=ARCHIVE_SHA256,
        attempt_rung=0,
    )


def _seal(value: dict, field_name: str = "receipt_digest") -> dict:
    value.pop(field_name, None)
    value[field_name] = runtime.object_sha256(value)
    return value


def _bank_receipt(manifest: dict) -> dict:
    entries = []
    for index, entry in enumerate(manifest["entries"]):
        raw_noise = "a" * 64 if entry["proposal_cell_index"] == 0 else "b" * 64
        entries.append(
            {
                "entry_id": entry["entry_id"],
                "semantic_branch": entry["semantic_branch"],
                "proposal_cell_id": entry["proposal_cell_id"],
                "design_slot_id": entry["design_slot_id"],
                "analysis_split": entry["analysis_split"],
                "execution_group": entry["execution_group"],
                "seed_replicate_id": entry["seed_replicate_id"],
                "seed": entry["seed"],
                "attempt_rung": entry["attempt_rung"],
                "native_receipt_path": f"/bank/{entry['entry_id']}/receipt.json",
                "native_receipt_file_sha256": f"{index + 1:064x}",
                "native_receipt_digest": f"{index + 21:064x}",
                "video_path": f"/bank/{entry['entry_id']}/t2v.mp4",
                "video_sha256": f"{index + 41:064x}",
                "clean_latent_path": f"/bank/{entry['entry_id']}/t2v.normalized-clean-latent.safetensors",
                "clean_latent_sha256": f"{index + 61:064x}",
                "initial_noise_path": f"/bank/{entry['entry_id']}/t2v.initial-noise.safetensors",
                "initial_noise_file_sha256": f"{index + 81:064x}",
                "initial_noise_tensor_value_sha256": raw_noise,
                "initial_noise_value_digest_independently_recomputed": True,
                "method_source_revision": REVISION,
                "method_source_archive_sha256": ARCHIVE_SHA256,
                "pure_t2v_condition_audit_pass": True,
            }
        )
    return _seal(
        {
            "schema_version": bank.BANK_RECEIPT_SCHEMA,
            "bank_id": manifest["bank_id"],
            "profile": "engineering_micro",
            "attempt_rung": 0,
            "manifest_digest": manifest["manifest_digest"],
            "entry_count": 20,
            "proposal_cell_count": 2,
            "entries": entries,
            "native_method_provenance": {
                "method_source_revision": REVISION,
                "method_source_archive_sha256": ARCHIVE_SHA256,
                "preregistered_in_manifest_before_render": True,
                "all_entries_exact": True,
            },
            "condition_closure": {
                "renderer_arm": "t2v",
                "source_video_role": "exact81_bucket_selection_and_hash_verification_only",
                "source_latent_or_reference_consumed": False,
                "target_video_consumed": False,
                "mask_flow_pose_track_trajectory_consumed": False,
                "all_native_entry_audits_pass": True,
                "all_cells_share_exact_initial_noise_across_ten_branches": True,
                "all_initial_noise_value_digests_independently_recomputed": True,
            },
            "interpretation": {
                "factorial_render_complete": True,
                "optimizer_update": "null",
                "training_performed": False,
            },
        }
    )


class _FakeTransformer:
    dtype = "bf16"

    def patch_vae_latent(self, hidden_states, source_id=None):
        return (hidden_states, source_id), ("rotary", source_id)


class _FakeScheduler:
    def __init__(self) -> None:
        self.bad_early = False

    def step(self, model_output, timestep, sample, return_dict=False):
        return (model_output, timestep, sample, return_dict)


class _FakeDiffusion:
    def __init__(self, *, bad_source_id: bool = False) -> None:
        self.transformer = _FakeTransformer()
        self.transformer_2 = None
        self.scheduler = _FakeScheduler()
        self.bad_source_id = bad_source_id
        self.returned = object()

    def shared_step(
        self, model_id, noisy_latents, timesteps, cond_embeds,
        rotary_embs, batch_vae_seqlen, batch_text_seqlen,
    ):
        return noisy_latents

    def sample(
        self,
        prompt_embeds=None,
        uncond_prompt_embeds=None,
        prompt_embeds_t2=None,
        uncond_embeds_t2=None,
        num_frames=1,
        width=832,
        height=480,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_inference_steps=50,
        guidance_mode="rv2v",
        omega_vid=3.0,
        omega_img=3.0,
        omega_txt=4.0,
        omega_scale=0.75,
        flow_shift=5.0,
        seed=42,
        device="cuda",
        eta=1.0,
        norm_threshold=(50.0, 50.0),
        momentum=0.0,
    ):
        del prompt_embeds_t2, uncond_embeds_t2, num_frames, width, height
        del image_vae_latents, multi_image_vae_latents, omega_vid, omega_img
        del omega_txt, omega_scale, flow_shift, seed, device, eta
        del norm_threshold, momentum
        count = len(multi_video_vae_latents)
        for step in range(num_inference_steps):
            for index, condition in enumerate(multi_video_vae_latents, start=1):
                source_id = float(index)
                if self.bad_source_id and step == 0 and index == count:
                    source_id = 5.0
                self.transformer.patch_vae_latent(condition, source_id=source_id)
            target = object()
            self.transformer.patch_vae_latent(target, source_id=0)
            if guidance_mode == "rv2v":
                visuals = [object(), object(), object()]
                rotaries = [object(), object(), object()]
                lengths = [
                    runtime.PATCH_TOKENS,
                    2 * runtime.PATCH_TOKENS,
                    (count + 1) * runtime.PATCH_TOKENS,
                    (count + 1) * runtime.PATCH_TOKENS,
                ]
                prompts = [
                    uncond_prompt_embeds,
                    uncond_prompt_embeds,
                    uncond_prompt_embeds,
                    prompt_embeds,
                ]
                visual_objects = [visuals[0], visuals[1], visuals[2], visuals[2]]
                rotary_objects = [rotaries[0], rotaries[1], rotaries[2], rotaries[2]]
            else:
                shared_visual = object()
                shared_rotary = object()
                lengths = [(count + 1) * runtime.PATCH_TOKENS] * 2
                prompts = [uncond_prompt_embeds, prompt_embeds]
                visual_objects = [shared_visual, shared_visual]
                rotary_objects = [shared_rotary, shared_rotary]
            for length, prompt, visual, rotary in zip(
                lengths, prompts, visual_objects, rotary_objects
            ):
                self.shared_step(
                    model_id="transformer_1",
                    noisy_latents=visual,
                    timesteps=step,
                    cond_embeds=prompt,
                    rotary_embs=rotary,
                    batch_vae_seqlen=[length],
                    batch_text_seqlen=[1],
                )
            self.scheduler.step(object(), step, object(), return_dict=False)
        return self.returned


def _sample_kwargs(
    *, conditions: list, prompt: object, negative: object,
    guidance_mode: str, steps: int,
) -> dict:
    spec = next(
        item for item in runtime.ARM_SPECS
        if item.guidance_mode == guidance_mode
    )
    return {
        "prompt_embeds": prompt,
        "uncond_prompt_embeds": negative,
        "image_vae_latents": None,
        "multi_video_vae_latents": conditions,
        "multi_image_vae_latents": None,
        "width": runtime.WIDTH,
        "height": runtime.HEIGHT,
        "device": "cuda",
        **runtime._sampling_values(spec, steps=steps),
    }


class NativeMotionDonorOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = Path(runtime.__file__).resolve()
        cls.source = cls.source_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.manifest = _manifest()
        cls.bank_receipt = _bank_receipt(cls.manifest)

    def test_fixed_arm_order_and_source_first_closure(self) -> None:
        self.assertEqual(
            runtime.ARM_ORDER,
            (
                "O0", "Z0", "D-action", "D-noop", "D-reverse",
                "D-wrong-actor", "D-duplicate-source",
                "D-action-order-swap", "A-source-v2v-apg",
            ),
        )
        self.assertEqual(runtime.DONOR_BRANCHES, (
            "full_action", "noop", "reverse_action", "wrong_actor"
        ))
        for spec in runtime.arm_plan():
            if spec.arm_id == "D-action-order-swap":
                self.assertEqual(spec.condition_roles[0], "registered_full_action_donor")
                self.assertIn("not_pure_role_swap", spec.diagnostic)
            else:
                self.assertEqual(spec.condition_roles[0], "source_video")
        self.assertFalse(runtime.ARM_SPECS[0].observed)
        self.assertTrue(runtime.ARM_SPECS[1].observed)
        self.assertEqual(runtime.ARM_SPECS[-1].guidance_mode, "v2v_apg")

    def test_each_world4_group_binds_one_whole_cell(self) -> None:
        prompt_hashes = set()
        for group in bank.GROUPS:
            bound = runtime.bind_registered_donors(
                manifest=self.manifest,
                bank_receipt=self.bank_receipt,
                execution_group=group,
            )
            self.assertEqual(set(bound["donor_rows"]), set(runtime.DONOR_BRANCHES))
            self.assertTrue(all(
                row["manifest"]["execution_group"] == group
                for row in bound["donor_rows"].values()
            ))
            prompt_hashes.add(bound["target_prompt_sha256"])
        self.assertEqual(len(prompt_hashes), 1)
        broken = deepcopy(self.bank_receipt)
        broken["condition_closure"]["all_native_entry_audits_pass"] = False
        _seal(broken)
        with self.assertRaises(runtime.MotionDonorOracleError):
            runtime.bind_registered_donors(
                manifest=self.manifest,
                bank_receipt=broken,
                execution_group="sp4-a",
            )

    def _run_fake_audit(self, *, count: int, mode: str, steps: int = 1):
        diffusion = _FakeDiffusion()
        conditions = [object() for _ in range(count)]
        prompt, negative = object(), object()
        audit = runtime.NativeMultiVideoConditionAudit(
            diffusion,
            condition_list=conditions,
            condition_roles=tuple(f"condition_{index}" for index in range(count)),
            guidance_mode=mode,
            expected_steps=steps,
            expected_seed=runtime.TARGET_SEED,
            prompt_embeds=prompt,
            uncond_prompt_embeds=negative,
        )
        original_sample = diffusion.sample
        original_shared = diffusion.shared_step
        original_patch = diffusion.transformer.patch_vae_latent
        original_scheduler = diffusion.scheduler.step
        audit.install()
        result = diffusion.sample(**_sample_kwargs(
            conditions=conditions,
            prompt=prompt,
            negative=negative,
            guidance_mode=mode,
            steps=steps,
        ))
        audit.restore()
        self.assertIs(result, diffusion.returned)
        self.assertEqual(diffusion.sample, original_sample)
        self.assertEqual(diffusion.shared_step, original_shared)
        self.assertEqual(diffusion.transformer.patch_vae_latent, original_patch)
        self.assertEqual(diffusion.scheduler.step, original_scheduler)
        self.assertTrue(audit.restored)
        return audit.trace

    def test_source_only_rv2v_observer_exact_call_sequence(self) -> None:
        trace = self._run_fake_audit(count=1, mode="rv2v", steps=1)
        self.assertEqual(trace["source_id_order_per_step"], [1.0, 0.0])
        self.assertEqual(
            trace["shared_visual_token_lengths_per_step"],
            [19_530, 39_060, 39_060, 39_060],
        )
        self.assertEqual(trace["scheduler_calls"], 1)
        self.assertFalse(trace["observer_modified_numerics"])

    def test_two_video_rv2v_observer_exact40_sequence(self) -> None:
        trace = self._run_fake_audit(count=2, mode="rv2v", steps=40)
        self.assertEqual(trace["source_id_order_per_step"], [1.0, 2.0, 0.0])
        self.assertEqual(
            trace["shared_visual_token_lengths_per_step"],
            [19_530, 39_060, 58_590, 58_590],
        )
        self.assertEqual(trace["step_count"], 40)
        self.assertEqual(len(trace["step_records"]), 40)

    def test_v2v_apg_anchor_uses_two_official_forwards(self) -> None:
        trace = self._run_fake_audit(count=1, mode="v2v_apg", steps=1)
        self.assertEqual(trace["source_id_order_per_step"], [1.0, 0.0])
        self.assertEqual(trace["shared_visual_token_lengths_per_step"], [39_060, 39_060])

    def test_observer_fails_closed_and_restores_on_wrong_source_id(self) -> None:
        diffusion = _FakeDiffusion(bad_source_id=True)
        conditions = [object(), object()]
        prompt, negative = object(), object()
        audit = runtime.NativeMultiVideoConditionAudit(
            diffusion,
            condition_list=conditions,
            condition_roles=("source", "donor"),
            guidance_mode="rv2v",
            expected_steps=1,
            expected_seed=runtime.TARGET_SEED,
            prompt_embeds=prompt,
            uncond_prompt_embeds=negative,
        )
        original_sample = diffusion.sample
        audit.install()
        try:
            with self.assertRaises(runtime.MotionDonorOracleError):
                diffusion.sample(**_sample_kwargs(
                    conditions=conditions,
                    prompt=prompt,
                    negative=negative,
                    guidance_mode="rv2v",
                    steps=1,
                ))
        finally:
            audit.restore()
        self.assertEqual(diffusion.sample, original_sample)
        self.assertTrue(audit.restored)

    def test_missing_clean_latent_fails_before_tensor_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            entry = root / "entries/example"
            entry.mkdir(parents=True)
            clean = entry / "t2v.normalized-clean-latent.safetensors"
            receipt = _seal(
                {
                    "schema_version": bank.NATIVE_RECEIPT_SCHEMA,
                    "arms": ["t2v"],
                    "interpretation": {"training_performed": False},
                    "outputs": {
                        "t2v": {
                            "normalized_clean_latent": {
                                "path": str(clean),
                                "sha256": "4" * 64,
                            }
                        }
                    },
                }
            )
            receipt_path = entry / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            row = {
                "manifest": {
                    "entry_id": "example",
                    "semantic_branch": "full_action",
                    "prompt_utf8_sha256": "5" * 64,
                    "proposal_cell_id": "cell",
                    "execution_group": "sp4-a",
                    "seed": 1,
                    "output_subdir": "entries/example",
                },
                "bank": {
                    "native_receipt_path": str(receipt_path),
                    "native_receipt_file_sha256": runtime.file_sha256(receipt_path),
                    "native_receipt_digest": receipt["receipt_digest"],
                    "clean_latent_path": str(clean),
                    "clean_latent_sha256": "4" * 64,
                },
            }
            called = False

            def forbidden_loader(path):
                nonlocal called
                called = True
                raise AssertionError(path)

            with self.assertRaises((runtime.MotionDonorOracleError, FileNotFoundError)):
                runtime.load_registered_clean_donor(
                    row=row, bank_root=root, tensor_loader=forbidden_loader
                )
            self.assertFalse(called)

    def test_runtime_is_frozen_no_training_or_donor_mp4_read_path(self) -> None:
        calls = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
        ]
        forbidden_call_names = {"backward", "step", "train"}
        # scheduler.step is called only through the untouched captured callable;
        # no optimizer-shaped owner/name occurs in this runtime.
        for call in calls:
            if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden_call_names:
                owner = ast.unparse(call.func.value)
                self.assertNotIn("optimizer", owner.lower())
                self.assertNotEqual(call.func.attr, "backward")
                self.assertNotEqual(call.func.attr, "train")
        self.assertNotIn("torch.optim", self.source)
        self.assertNotIn("loss.backward", self.source)
        self.assertNotIn("bank_row[\"video_path\"]", self.source)
        self.assertNotIn("_vae_encode(vae, donor", self.source)
        self.assertIn(
            "Z0 condition observer is not byte-exact to the condition-audit-free O0",
            self.source,
        )
        self.assertIn("donor_mp4_consumed\": False", self.source)
        self.assertIn("latent_raw_storage_sha256_in_order", self.source)
        self.assertIn("condition_latent_artifact_file_sha256_in_order", self.source)
        self.assertIn("result.dtype != torch.float32", self.source)
        self.assertIn("result.is_contiguous()", self.source)
        self.assertIn("_output_staging_directory(output_dir)", self.source)
        self.assertIn("_commit_output_transaction(staging=artifact_dir, final=output_dir)", self.source)
        self.assertIn("native_return_dtype_asserted_before_cpu_copy", self.source)
        self.assertNotIn("o0_official_unwrapped", self.source)
        self.assertIn("--num-inference-steps", self.source)
        self.assertEqual(runtime.ALLOWED_STEPS, (1, 40))

    def test_group_output_transaction_rebases_then_atomically_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            final = parent / "sp4-a"
            with mock.patch.dict(
                "os.environ",
                {"BERNINI_OUTPUT_TRANSACTION_ID": "job1-sp4-a-step1"},
                clear=False,
            ):
                staging = runtime._output_staging_directory(final)
            artifact = staging / "example.safetensors"
            artifact.write_bytes(b"sealed")
            rebased = runtime._rebase_artifact_paths(
                {"path": str(artifact), "external": "/checkpoint/model"},
                old_root=staging,
                new_root=final,
            )
            self.assertEqual(rebased["path"], str(final / artifact.name))
            self.assertEqual(rebased["external"], "/checkpoint/model")
            runtime._commit_output_transaction(staging=staging, final=final)
            self.assertFalse(staging.exists())
            self.assertEqual((final / artifact.name).read_bytes(), b"sealed")


if __name__ == "__main__":
    unittest.main()
