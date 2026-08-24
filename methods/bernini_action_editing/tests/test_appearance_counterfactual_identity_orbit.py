from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import appearance_counterfactual_identity_orbit as orbit
from tools import materialize_appearance_counterfactual_identity_orbit as materializer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(orbit.canonical_json_bytes(value) + b"\n")


class AppearanceOrbitFixture:
    def __init__(
        self,
        root: Path,
        *,
        variant_a_arm: str = "rv2v",
        variant_b_arm: str = "rv2v",
        qualified: bool = False,
        same_prompt: bool = False,
        duplicate_variant_content: bool = False,
        mutate_a_receipt=None,
        mutate_b_receipt=None,
        mutate_qualification=None,
    ) -> None:
        self.root = root.resolve()
        self.iid = "orbit-example"
        self.variant_a_arm = variant_a_arm
        self.variant_b_arm = variant_b_arm
        self.source = self.root / "source.mp4"
        self.variant_a = self.root / "variant_a.mp4"
        self.variant_b = self.root / "variant_b.mp4"
        self.source.write_bytes(b"source-exact81-placeholder")
        self.variant_a.write_bytes(b"variant-a-exact81-placeholder")
        self.variant_b.write_bytes(
            b"variant-a-exact81-placeholder"
            if duplicate_variant_content
            else b"variant-b-exact81-placeholder"
        )
        self.member_sha = {
            "source": _sha(self.source),
            "variant_a": _sha(self.variant_a),
            "variant_b": _sha(self.variant_b),
        }
        prompt_a = "6" * 64
        prompt_b = prompt_a if same_prompt else "7" * 64
        self.variant_a_receipt = self._native_receipt(
            arm=variant_a_arm,
            output_path=self.variant_a,
            output_sha=self.member_sha["variant_a"],
            prompt_sha=prompt_a,
            seed=2027,
        )
        self.variant_b_receipt = self._native_receipt(
            arm=variant_b_arm,
            output_path=self.variant_b,
            output_sha=self.member_sha["variant_b"],
            prompt_sha=prompt_b,
            seed=2028,
        )
        if mutate_a_receipt is not None:
            mutate_a_receipt(self.variant_a_receipt)
        if mutate_b_receipt is not None:
            mutate_b_receipt(self.variant_b_receipt)
        self._reseal(self.variant_a_receipt)
        self._reseal(self.variant_b_receipt)
        self.variant_a_receipt_path = self.root / "variant_a.receipt.json"
        self.variant_b_receipt_path = self.root / "variant_b.receipt.json"
        _write_json(self.variant_a_receipt_path, self.variant_a_receipt)
        _write_json(self.variant_b_receipt_path, self.variant_b_receipt)

        qualification_binding = None
        self.qualification_path = self.root / "qualification.json"
        if qualified:
            qualification = orbit.qualification_seal_body(
                iid=self.iid,
                members={
                    "source": {
                        "video_path": str(self.source),
                        "video_sha256": self.member_sha["source"],
                    },
                    "variant_a": {
                        "video_path": str(self.variant_a),
                        "video_sha256": self.member_sha["variant_a"],
                        "native_arm": variant_a_arm,
                    },
                    "variant_b": {
                        "video_path": str(self.variant_b),
                        "video_sha256": self.member_sha["variant_b"],
                        "native_arm": variant_b_arm,
                    },
                },
                qualifier_id="external-full-video-reviewer-A",
                protocol_sha256="9" * 64,
            )
            if mutate_qualification is not None:
                mutate_qualification(qualification)
            self._reseal(qualification)
            _write_json(self.qualification_path, qualification)
            qualification_binding = {
                "path": str(self.qualification_path),
                "file_sha256": _sha(self.qualification_path),
                "digest": qualification["receipt_digest"],
            }

        spec = orbit.build_materialization_spec(
            [
                {
                    "iid": self.iid,
                    "source": {
                        "video_path": str(self.source),
                        "video_sha256": self.member_sha["source"],
                    },
                    "variant_a": self._spec_variant(
                        path=self.variant_a,
                        sha=self.member_sha["variant_a"],
                        arm=variant_a_arm,
                        receipt=self.variant_a_receipt,
                        receipt_path=self.variant_a_receipt_path,
                    ),
                    "variant_b": self._spec_variant(
                        path=self.variant_b,
                        sha=self.member_sha["variant_b"],
                        arm=variant_b_arm,
                        receipt=self.variant_b_receipt,
                        receipt_path=self.variant_b_receipt_path,
                    ),
                    "qualification": qualification_binding,
                }
            ]
        )
        self.spec_path = self.root / "orbit-spec.json"
        _write_json(self.spec_path, spec)
        self.spec_sha256 = _sha(self.spec_path)

    @staticmethod
    def _reseal(value: dict) -> None:
        value.pop("receipt_digest", None)
        value["receipt_digest"] = orbit.object_sha256(value)

    @staticmethod
    def _spec_variant(
        *, path: Path, sha: str, arm: str, receipt: dict, receipt_path: Path
    ) -> dict:
        return {
            "video_path": str(path),
            "video_sha256": sha,
            "native_arm": arm,
            "native_receipt_path": str(receipt_path),
            "native_receipt_file_sha256": _sha(receipt_path),
            "native_receipt_digest": receipt["receipt_digest"],
        }

    def _native_receipt(
        self,
        *,
        arm: str,
        output_path: Path,
        output_sha: str,
        prompt_sha: str,
        seed: int,
    ) -> dict:
        arms = [arm]
        if arm == "r2v":
            videos, refs, indices = 0, 5, [0, 20, 40, 60, 80]
            video_source_ids, reference_source_ids = [], [1, 2, 3, 4, 5]
            guidance = "r2v_apg"
        elif arm == "rv2v":
            videos, refs, indices = 1, 4, [0, 27, 53, 80]
            video_source_ids, reference_source_ids = [1], [2, 3, 4, 5]
            guidance = "rv2v"
        else:
            # Invalid-arm tests must still be able to build the spec; the
            # contract rejects it before reading this placeholder receipt.
            videos, refs, indices = 0, 0, []
            video_source_ids, reference_source_ids = [], []
            guidance = arm
        sampling = {
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_mode": guidance,
            "omega_vid": 1.25,
            "omega_img": 4.5,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "flow_shift": 5.0,
            "seed": seed,
            "eta": 0.5,
            "norm_threshold": [50.0, 50.0],
            "momentum": 0.0,
            "target_initialization": "official_gen_wanx22_fresh_gaussian",
            "target_mixed_with_source_latent": False,
            "custom_sampler_or_scheduler": False,
            "same_seed_and_target_shape_across_arms": True,
            "single_expert": "transformer_1",
            "ulysses_size": 4,
        }
        conditioning = {
            "full_source_video_count": videos,
            "source_derived_reference_count": refs,
            "source_frame_indices": indices,
            "reference_encoding": "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]",
            "reference_from_temporal_video_latent_slice": False,
            "source_ids": {
                "target_source_id": 0,
                "video_source_ids": video_source_ids,
                "reference_source_ids": reference_source_ids,
                "conditioning_source_count": videos + refs,
                "max_conditioning_source_id": videos + refs,
                "within_pretrained_source_ids_1_through_5": videos + refs <= 5,
                "source_id_interpolation_required": False,
            },
        }
        receipt = {
            "schema_version": orbit.NATIVE_RECEIPT_SCHEMA,
            "method": orbit.NATIVE_METHOD,
            "method_source_revision": "1" * 40,
            "method_source_archive_sha256": "2" * 64,
            "bernini_commit": "3" * 40,
            "veomni_commit": "4" * 40,
            "bernini_inference_files": {},
            "checkpoint": {
                "path": "/checkpoint",
                "tree_sha256": "5" * 64,
                "content": {
                    "manifest_path": "/audit/checkpoint.sha256",
                    "manifest_sha256_computed": "6" * 64,
                    "manifest_sha256_expected": "6" * 64,
                    "verified_entries_digest": "7" * 64,
                    "verified_file_count": 23,
                    "every_file_sha256_verified": True,
                },
            },
            "arms": arms,
            "input": {
                "source_video_path": str(self.source),
                "source_video_sha256": self.member_sha["source"],
                "action_prompt_utf8_sha256": prompt_sha,
                "action_prompt_utf8_bytes": 42,
                "accepted_external_conditions": ["source_video", "action_prompt"],
                "target_video": False,
                "external_reference_image_or_video": False,
                "external_mask_flow_pose_track_trajectory": False,
                "external_first_frame_anchor": False,
            },
            "preprocessing": {
                "frame_count": 81,
                "fps": 25.0,
                "reported_fps": 25.0,
                "source_input_hw": [496, 480],
                "source_derived_bucket_hw": [496, 480],
                "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
                "external_shared_i0": False,
            },
            "prompt_contract": {arm: {}},
            "conditioning": {arm: conditioning},
            "sampling": {arm: sampling},
            "latent_geometry": {},
            "condition_identities": {},
            "source_condition_artifact": {},
            "initial_noise_artifacts": {arm: {}},
            "generated_identities": {arm: {}},
            "outputs": {
                arm: {
                    "path": str(output_path),
                    "sha256": output_sha,
                    "frame_count": 81,
                    "fps": 25.0,
                    "height": 496,
                    "width": 480,
                    "normalized_clean_latent": {},
                }
            },
            "freeze_certificate": {
                "base_frozen": True,
                "trainable_parameter_tensors": 0,
                "trainable_parameter_elements": 0,
                "lora_module_count": 0,
            },
            "runtime_versions": {},
            "interpretation": {
                "purpose": "test_native_identity_conditioned_generation_before_training",
                "quality_claim": False,
                "training_performed": False,
                "best_arm_selected": False,
            },
            "experimental_canary": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        self._reseal(receipt)
        return receipt

    def rewrite_spec(self, mutation) -> None:
        value = json.loads(self.spec_path.read_text(encoding="utf-8"))
        mutation(value)
        value.pop("spec_digest", None)
        value["spec_digest"] = orbit.object_sha256(value)
        _write_json(self.spec_path, value)
        self.spec_sha256 = _sha(self.spec_path)

    def load(self):
        audit = orbit.FileMutationAudit()
        loaded = orbit.load_materialization_spec(
            self.spec_path,
            expected_sha256=self.spec_sha256,
            audit=audit,
        )
        return loaded, audit


class AppearanceCounterfactualOrbitContractTests(unittest.TestCase):
    def test_v3_rv2v4_plus_rv2v4_is_first_class_and_unqualified_defaults_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory))
            loaded, audit = fixture.load()
            row = loaded.rows[0]
            self.assertEqual(orbit.SPEC_SCHEMA.rsplit("-", 1)[-1], "v3")
            self.assertEqual(
                loaded.reference_encoding_contract,
                orbit.reference_encoding_contract(),
            )
            self.assertEqual(orbit.REFERENCE_INDICES, (0, 27, 53, 80))
            self.assertEqual(row.variant_a.native_arm, "rv2v")
            self.assertEqual(row.variant_b.native_arm, "rv2v")
            self.assertNotEqual(
                row.variant_a.receipt["input"]["action_prompt_utf8_sha256"],
                row.variant_b.receipt["input"]["action_prompt_utf8_sha256"],
            )
            self.assertFalse(row.scientific_use_authorized)
            self.assertEqual(len(audit.finalize()), 6)

    def test_mixed_r2v_rv2v_is_also_explicitly_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(
                Path(directory), variant_a_arm="r2v", variant_b_arm="rv2v"
            )
            loaded, audit = fixture.load()
            self.assertEqual(
                (loaded.rows[0].variant_a.native_arm, loaded.rows[0].variant_b.native_arm),
                ("r2v", "rv2v"),
            )
            audit.finalize()

    def test_checkpoint_identity_uses_verified_content_not_manifest_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(
                Path(directory),
                mutate_b_receipt=lambda value: value["checkpoint"]["content"].__setitem__(
                    "manifest_path", "/another/immutable-copy/checkpoint.sha256"
                ),
            )
            loaded, audit = fixture.load()
            self.assertEqual(len(loaded.rows), 1)
            audit.finalize()

    def test_checkpoint_content_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(
                Path(directory),
                mutate_b_receipt=lambda value: value["checkpoint"].__setitem__(
                    "tree_sha256", "8" * 64
                ),
            )
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "checkpoint"):
                fixture.load()

    def test_old_role_schema_and_implicit_arm_are_rejected_without_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory))
            fixture.rewrite_spec(
                lambda value: value.__setitem__(
                    "schema_version",
                    "bernini-appearance-counterfactual-identity-orbit-spec-v1",
                )
            )
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "schema"):
                fixture.load()
        row = {
            "iid": "x",
            "source": {"video_path": "/s.mp4", "video_sha256": "1" * 64},
            "variant_a": {
                "video_path": "/a.mp4",
                "video_sha256": "2" * 64,
                "native_receipt_path": "/a.json",
                "native_receipt_file_sha256": "3" * 64,
                "native_receipt_digest": "4" * 64,
            },
            "variant_b": {
                "video_path": "/b.mp4",
                "video_sha256": "5" * 64,
                "native_arm": "rv2v",
                "native_receipt_path": "/b.json",
                "native_receipt_file_sha256": "6" * 64,
                "native_receipt_digest": "7" * 64,
            },
            "qualification": None,
        }
        with self.assertRaisesRegex(orbit.AppearanceOrbitError, "closure"):
            orbit.build_materialization_spec([row])

    def test_variant_native_arm_must_be_allowed_and_match_receipt_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory))
            fixture.rewrite_spec(
                lambda value: value["rows"][0]["variant_a"].__setitem__(
                    "native_arm", "r2v"
                )
            )
            with self.assertRaises(orbit.AppearanceOrbitError):
                fixture.load()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "native_arm"):
                AppearanceOrbitFixture(Path(directory), variant_a_arm="t2v")

    def test_distinct_prompt_hashes_and_variant_content_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory), same_prompt=True)
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "prompt hashes"):
                fixture.load()
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(
                Path(directory), duplicate_variant_content=True
            )
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "content"):
                fixture.load()

    def test_external_qualification_binds_arms_and_semantic_orbit_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory), qualified=True)
            loaded, audit = fixture.load()
            row = loaded.rows[0]
            self.assertTrue(row.scientific_use_authorized)
            seal = row.qualification.receipt
            self.assertEqual(seal["members"]["variant_a"]["native_arm"], "rv2v")
            self.assertTrue(
                seal["qualification_gates"]["cross_member"][
                    "variant_a_and_variant_b_semantic_identities_distinct"
                ]
            )
            self.assertTrue(
                seal["qualification_gates"]["cross_member"][
                    "same_motion_across_all_members"
                ]
            )
            self.assertEqual(len(audit.finalize()), 7)

    def test_negative_or_nonblind_qualification_cannot_authorize(self) -> None:
        mutations = {
            "arm mismatch": lambda value: value["members"]["variant_a"].__setitem__(
                "native_arm", "r2v"
            ),
            "identity unchanged": lambda value: value["qualification_gates"][
                "variant_a"
            ].__setitem__("appearance_identity_changed_from_source", False),
            "identities not distinct": lambda value: value["qualification_gates"][
                "cross_member"
            ].__setitem__(
                "variant_a_and_variant_b_semantic_identities_distinct", False
            ),
            "motion differs": lambda value: value["qualification_gates"][
                "cross_member"
            ].__setitem__("same_motion_across_all_members", False),
            "camera differs": lambda value: value["qualification_gates"][
                "cross_member"
            ].__setitem__("same_camera_across_all_members", False),
            "training seen": lambda value: value.__setitem__(
                "downstream_training_results_seen", True
            ),
            "not external": lambda value: value["evaluation_protocol"].__setitem__(
                "external_to_materializer", False
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = AppearanceOrbitFixture(
                    Path(directory),
                    qualified=True,
                    mutate_qualification=mutation,
                )
                with self.assertRaises(orbit.AppearanceOrbitError):
                    fixture.load()

    def test_native_receipt_validation_fails_closed_per_required_field(self) -> None:
        arm = "rv2v"
        mutations = {
            "source hash": lambda value: value["input"].__setitem__(
                "source_video_sha256", "0" * 64
            ),
            "output path": lambda value: value["outputs"][arm].__setitem__(
                "path", "/tmp/not-the-bound-output.mp4"
            ),
            "81 frames": lambda value: value["outputs"][arm].__setitem__(
                "frame_count", 80
            ),
            "25 fps": lambda value: value["outputs"][arm].__setitem__("fps", 24.0),
            "40 steps": lambda value: value["sampling"][arm].__setitem__(
                "num_inference_steps", 39
            ),
            "frozen": lambda value: value["freeze_certificate"].__setitem__(
                "base_frozen", False
            ),
            "no training": lambda value: value["interpretation"].__setitem__(
                "training_performed", True
            ),
            "no target": lambda value: value["input"].__setitem__(
                "target_video", True
            ),
            "no mask-flow-pose-track": lambda value: value["input"].__setitem__(
                "external_mask_flow_pose_track_trajectory", True
            ),
            "native source IDs": lambda value: value["conditioning"][arm][
                "source_ids"
            ].__setitem__("reference_source_ids", [1, 2, 3, 4]),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = AppearanceOrbitFixture(
                    Path(directory), mutate_a_receipt=mutation
                )
                with self.assertRaises(orbit.AppearanceOrbitError):
                    fixture.load()

    def test_native_receipt_path_file_hash_and_digest_are_all_bound(self) -> None:
        mutations = {
            "path": lambda row, root: row["variant_a"].__setitem__(
                "native_receipt_path", str(root / "missing.json")
            ),
            "file hash": lambda row, root: row["variant_a"].__setitem__(
                "native_receipt_file_sha256", "0" * 64
            ),
            "digest": lambda row, root: row["variant_a"].__setitem__(
                "native_receipt_digest", "0" * 64
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = AppearanceOrbitFixture(Path(directory))
                fixture.rewrite_spec(
                    lambda value, mutation=mutation: mutation(
                        value["rows"][0], fixture.root
                    )
                )
                with self.assertRaises(orbit.AppearanceOrbitError):
                    fixture.load()

    def test_per_file_post_mutation_audit_detects_late_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory))
            _, audit = fixture.load()
            fixture.variant_b.write_bytes(b"late mutation")
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "mutated"):
                audit.finalize()

    def test_spec_surface_and_iids_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory))
            value = json.loads(fixture.spec_path.read_text(encoding="utf-8"))
            row = value["rows"][0]
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "unique"):
                orbit.build_materialization_spec([row, dict(row)])
            invalid = dict(row)
            invalid["target_video_path"] = "/forbidden.mp4"
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "closure"):
                orbit.build_materialization_spec([invalid])

    def test_spec_fails_closed_on_three_reference_or_twelve_call_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = AppearanceOrbitFixture(Path(directory))
            fixture.rewrite_spec(
                lambda value: value["reference_encoding_contract"].update(
                    {
                        "reference_rgb_indices": [0, 40, 80],
                        "reference_count": 3,
                        "independent_rgb_reference_encode_calls_per_row": 9,
                        "independent_vae_encode_calls_per_row": 12,
                    }
                )
            )
            with self.assertRaisesRegex(orbit.AppearanceOrbitError, "RV2V-4"):
                fixture.load()


class AppearanceCounterfactualOrbitMaterializerTests(unittest.TestCase):
    class FakeVideo:
        def __init__(self, name: str):
            self.name = name

        def __getitem__(self, key):
            return self.FakeReference(self.name, key)

        class FakeReference:
            def __init__(self, name, key):
                self.name = name
                self.key = key

            def contiguous(self):
                return self

    def test_exactly_three_full_and_twelve_independent_rgb_reference_encodes(self) -> None:
        videos = {name: self.FakeVideo(name) for name in orbit.MEMBER_NAMES}
        calls = []

        def fake_encode(encoder, rgb, *, phases, role, call_index):
            calls.append((rgb, phases, role, call_index))
            return f"blob-{call_index}".encode(), {
                "encode_call_index": call_index,
                "encoded_independently": True,
                "reference_from_full_video_posterior_slice": False,
                "posterior_parameters_shape": [1, 32, phases, 2, 2],
            }

        blobs, metadata = materializer._encode_all_posteriors(
            videos, encoder=object(), encode_one=fake_encode
        )
        self.assertEqual(tuple(blobs), materializer.POSTERIOR_FIELDS)
        self.assertEqual(tuple(metadata), materializer.POSTERIOR_FIELDS)
        self.assertEqual(len(calls), 15)
        self.assertEqual([call[1] for call in calls[:3]], [21, 21, 21])
        self.assertEqual([call[1] for call in calls[3:]], [1] * 12)
        self.assertEqual([call[3] for call in calls], list(range(15)))
        self.assertEqual(
            [call[2] for call in calls],
            [
                materializer.POSTERIOR_ARTIFACT_ROLES[field]
                for field in materializer.POSTERIOR_FIELDS
            ],
        )
        self.assertEqual(
            materializer.POSTERIOR_ARTIFACT_ROLES["source_full_posterior_blob"],
            "source_full_exact81_rgb",
        )
        self.assertIn("variant_a_full_posterior_blob", blobs)
        self.assertIn("variant_a_ref27_posterior_blob", blobs)
        self.assertIn("variant_b_ref53_posterior_blob", blobs)
        self.assertIn("variant_b_ref80_posterior_blob", blobs)

    def test_output_is_create_only_and_cli_has_no_privileged_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory).resolve() / "already_exists"
            existing.mkdir()
            with self.assertRaisesRegex(
                materializer.OrbitMaterializationError, "create-only"
            ):
                materializer._resolve_output(existing)
        destinations = {action.dest for action in materializer.build_parser()._actions}
        for forbidden in (
            "target_video",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "train_manifest",
            "locality",
        ):
            self.assertNotIn(forbidden, destinations)

    def test_publication_is_exclusive_and_receipt_is_last_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            output = parent / "sealed_orbit"
            stage = parent / ".sealed_orbit.staging"
            stage.mkdir()
            (stage / "dataset.parquet").write_bytes(b"parquet")
            (stage / "receipt.json").write_bytes(b"receipt\n")
            materializer._publish_create_only_bundle(stage, output)
            self.assertFalse(stage.exists())
            self.assertEqual((output / "dataset.parquet").read_bytes(), b"parquet")
            self.assertEqual((output / "receipt.json").read_bytes(), b"receipt\n")

    def test_v3_rv2v4_materializer_has_no_training_or_locality_dependency(self) -> None:
        source = (
            METHOD_ROOT
            / "tools/materialize_appearance_counterfactual_identity_orbit.py"
        ).read_text(encoding="utf-8")
        self.assertTrue(materializer.ROW_SCHEMA.endswith("-v3"))
        self.assertTrue(materializer.RECEIPT_SCHEMA.endswith("-v3"))
        self.assertIn("import source_self_runtime as runtime", source)
        self.assertIn("PinnedBerniniWanPosteriorEncoder", source)
        self.assertNotIn("import train_ramp", source)
        self.assertNotIn("import train_source_self", source)
        self.assertNotIn("build_latent_locality", source)
        self.assertNotIn("source_self_identity_orbit_v4", source)


if __name__ == "__main__":
    unittest.main()
