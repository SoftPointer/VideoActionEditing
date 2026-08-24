from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "pair_v7_fit_only_geometry_authority.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v7_fit_only_geometry_authority as authority


def _sha(character: str) -> str:
    return character * 64


class FitOnlyAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkpoint = _sha("1")
        self.adapter = _sha("2")

    def _write(self, name: str, payload: bytes) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path.resolve()

    def _prompts(self, prefix: str) -> dict[str, str]:
        return {
            branch: f"{prefix} {ordinal} {branch}"
            for ordinal, branch in enumerate(authority.BRANCH_ORDER)
        }

    def _drafts(self) -> list[dict[str, object]]:
        drafts = []
        for index, (candidate, family) in enumerate(
            (("fit-dog-sit", "sit"), ("fit-human-turn", "turn"))
        ):
            drafts.append(
                {
                    "event_id": candidate,
                    "fit_candidate_id": candidate,
                    "action_family": family,
                    "prompt_by_branch": self._prompts(f"prompt-{index}"),
                    "source_sample_id": f"source-{index}",
                    "source_video_path": str(
                        self._write(f"source-{index}.mp4", f"video-{index}".encode())
                    ),
                    "raw_caption_by_branch": self._prompts(f"caption-{index}"),
                    "clean_latent_path": str(
                        self._write(f"clean-{index}.safetensors", f"clean-{index}".encode())
                    ),
                    "clean_latent_tensor_key": "latent",
                    "official_gaussian_path": str(
                        self._write(f"noise-{index}.safetensors", f"noise-{index}".encode())
                    ),
                    "official_gaussian_tensor_key": "noise",
                }
            )
        return drafts

    @staticmethod
    def _tensor_inspection(path: Path, key: str, *, label: str):
        del label
        digest = hashlib.sha256(f"{path}:{key}".encode()).hexdigest()
        shape = (1, 16, 21, 60, 62) if "-0." in path.name else (1, 16, 21, 64, 58)
        return authority.TensorInspection(digest, shape)

    def _author_manifest(self):
        output = self.root / "fit-only.json"
        with patch.object(
            authority, "_inspect_tensor_artifact", side_effect=self._tensor_inspection
        ), patch.object(
            authority, "_load_tensor_artifact", return_value=None
        ), patch.object(
            authority,
            "_inspect_source_media",
            return_value={"frame_count": 81, "fps": 25.0},
        ):
            authored = authority.author_fit_only_manifest(
                output_path=output,
                checkpoint_tree_sha256=self.checkpoint,
                action_adapter_schema_sha256=self.adapter,
                event_drafts=self._drafts(),
            )
            loaded, runtimes = authority.load_fit_only_manifest(
                output,
                expected_file_sha256=authority._file_sha256(output),
                expected_checkpoint_tree_sha256=self.checkpoint,
                expected_action_adapter_schema_sha256=self.adapter,
            )
        return output, authored, loaded, runtimes

    def _checkpoint_identity(self, filename: str = "checkpoint.sha256"):
        manifest = self._write(filename, b"same checkpoint manifest bytes\n")
        manifest_sha = authority._file_sha256(manifest)
        return {
            "manifest_path": str(manifest),
            "manifest_sha256_computed": manifest_sha,
            "manifest_sha256_expected": manifest_sha,
            "verified_file_count": 23,
            "every_file_sha256_verified": True,
            "verified_entries_digest": _sha("3"),
        }

    def test_manifest_is_create_only_exact81_correct_source_and_no_update(self) -> None:
        output, authored, loaded, runtimes = self._author_manifest()
        self.assertEqual(authored["schedule_indices"], [33])
        self.assertEqual(authored["first_schedule_index"], 33)
        self.assertEqual(len(loaded.events), 2)
        self.assertTrue(all(item.event_latent_cpu is None for item in runtimes))
        for event in authored["events"]:
            self.assertEqual(event["source_frame_count"], 81)
            self.assertEqual(event["source_fps"], 25.0)
            self.assertEqual(event["source_reference_indices"], [0, 27, 53, 80])
            self.assertNotIn("wrong_source", json.dumps(event, sort_keys=True))
            for field in authority._NO_UPDATE_CLAIMS:
                self.assertFalse(event[field])
        for field in authority._NO_UPDATE_CLAIMS:
            self.assertFalse(authored[field])
        with patch.object(
            authority, "_inspect_tensor_artifact", side_effect=self._tensor_inspection
        ), patch.object(
            authority, "_load_tensor_artifact", return_value=None
        ), patch.object(
            authority,
            "_inspect_source_media",
            return_value={"frame_count": 81, "fps": 25.0},
        ):
            with self.assertRaisesRegex(authority.PairV7FitOnlyAuthorityError, "create-only"):
                authority.author_fit_only_manifest(
                    output_path=output,
                    checkpoint_tree_sha256=self.checkpoint,
                    action_adapter_schema_sha256=self.adapter,
                    event_drafts=self._drafts(),
                )

    def test_source_media_inspector_rejects_wrong_frames_and_fps(self) -> None:
        source = self._write("source.mp4", b"video")
        runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("container.decode(stream)", runtime_source)
        self.assertNotIn('shutil.which("ffprobe")', runtime_source)

        def fake_av(frame_count: int, fps: str):
            stream = SimpleNamespace(average_rate=fps)

            class Container:
                streams = SimpleNamespace(video=[stream])

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def decode(self, selected):
                    if selected is not stream:
                        raise AssertionError("unexpected video stream")
                    return iter(range(frame_count))

            return SimpleNamespace(open=lambda *_args, **_kwargs: Container())

        with patch.dict(sys.modules, {"av": fake_av(81, "25/1")}):
            self.assertEqual(
                authority._inspect_source_media(source),
                {"frame_count": 81, "fps": 25.0},
            )
        for frame_count, fps in ((80, "25/1"), (81, "24/1")):
            with patch.dict(sys.modules, {"av": fake_av(frame_count, fps)}):
                with self.assertRaisesRegex(
                    authority.PairV7FitOnlyAuthorityError, "exact81"
                ):
                    authority._inspect_source_media(source)

    def test_checkpoint_identity_is_path_independent_for_cast_binding(self) -> None:
        left = self._checkpoint_identity("left.sha256")
        right_path = self._write("nested/right.sha256", Path(left["manifest_path"]).read_bytes())
        right = {**left, "manifest_path": str(right_path)}
        self.assertNotEqual(authority.object_sha256(left), authority.object_sha256(right))
        self.assertEqual(
            authority._checkpoint_content_identity_binding(left),
            authority._checkpoint_content_identity_binding(right),
        )
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("PHASE_A_CHECKPOINT_CONTENT_RECEIPT_DIGEST", source)

    def test_sail_master_strictly_binds_both_measured_vjp_children(self) -> None:
        descriptors = {}
        expected_children = {}
        for role in ("dog", "human"):
            unsigned = {
                "candidate_id": f"candidate-{role}",
                "mechanism_probe_only": True,
                "editor_parameter_or_update_authorized": False,
                "scientific_claim_authorized": False,
                "action_editing_success_claim_authorized": False,
                "training_performed": False,
                "source_condition_in_live_query": False,
                "live_vjp_proof": {
                    "real_sp4_autograd_collective_observed": True,
                    "replica_consensus_observed": True,
                    "same_x_sigma_object_for_action_noop": True,
                },
            }
            child = {
                **unsigned,
                "receipt_digest": authority.object_sha256(unsigned),
            }
            child_path = self.root / role / "receipt.json"
            child_path.parent.mkdir()
            child_path.write_bytes(authority.canonical_json_bytes(child) + b"\n")
            file_sha = authority._file_sha256(child_path)
            descriptors[role] = {
                "candidate_id": unsigned["candidate_id"],
                "receipt_digest": child["receipt_digest"],
                "receipt_file_sha256": file_sha,
                "receipt_path": str(child_path),
            }
            expected_children[role] = {
                "file_sha256": file_sha,
                "receipt_digest": child["receipt_digest"],
            }
        master_unsigned = {
            "schema_version": "bernini-sail-relational-motion-dual4-master-v1",
            "children": descriptors,
            "postflight_complete": True,
            "all_six_mp4_exact81": True,
            "scientific_claim_authorized": False,
            "action_editing_success_claim_authorized": False,
            "training_performed": False,
            "source_condition_in_live_query": False,
        }
        master = {
            **master_unsigned,
            "receipt_digest": authority.object_sha256(master_unsigned),
        }
        master_path = self.root / "master.json"
        master_path.write_bytes(authority.canonical_json_bytes(master) + b"\n")
        with patch.object(
            authority,
            "SAIL_PRIOR_NO_SUCCESS_FILE_SHA256",
            authority._file_sha256(master_path),
        ), patch.object(
            authority,
            "SAIL_PRIOR_NO_SUCCESS_RECEIPT_DIGEST",
            master["receipt_digest"],
        ), patch.object(authority, "SAIL_CHILD_BINDINGS", expected_children):
            checked = authority._validate_sail_prior_no_success(master_path)
        self.assertEqual(len(checked["child_receipts"]), 2)
        self.assertEqual(
            {row["role"] for row in checked["child_receipts"]}, {"dog", "human"}
        )

    def test_external_evidence_postflight_closure_catches_child_score_mutation(self) -> None:
        checkpoint_identity = self._checkpoint_identity()

        def descriptor(name: str):
            path = self._write(f"closure/{name}.json", name.encode())
            return {"path": str(path), "file_sha256": authority._file_sha256(path)}

        candidates = [descriptor(f"candidate-{index:02d}") for index in range(40)]
        groups = []
        for index in range(2):
            groups.append(
                {
                    **descriptor(f"group-{index}"),
                    "candidate_receipts": candidates[index * 20 : (index + 1) * 20],
                }
            )
        children = [descriptor("sail-dog"), descriptor("sail-human")]
        cast_bindings = {
            "cast_v4_method_archive": descriptor("method-archive"),
            "cast_v4_root_spec": descriptor("root-spec"),
            "cast_v4_groups": groups,
            "negative_boundaries": [
                descriptor("legacy-boundary"),
                {**descriptor("sail-master"), "child_receipts": children},
            ],
        }
        bindings = authority._external_evidence_file_bindings(
            cast_bindings, checkpoint_identity
        )
        self.assertEqual(len(bindings), 49)
        mutated_path = Path(candidates[7]["path"])
        mutated_path.write_bytes(b"mutated")
        selected = next(item for item in bindings if item.path == mutated_path)
        with self.assertRaisesRegex(authority.PairV7FitOnlyAuthorityError, "changed"):
            selected.assert_unchanged()

    def _event(self, candidate: str, family: str, source_sha: str, prefix: str):
        return SimpleNamespace(
            event_id=candidate,
            action_family=family,
            clean_latent_tensor_sha256=_sha(prefix),
            official_gaussian_tensor_sha256=_sha(chr(ord(prefix) + 1)),
            prompt_by_branch=self._prompts(f"prompt-{prefix}"),
            raw_caption_by_branch=self._prompts(f"caption-{prefix}"),
            source_video=SimpleNamespace(sha256=source_sha),
            latent_shape=(1, 16, 21, 60, 62),
        )

    def test_collect_binds_all_40_and_selected_correct_source_score(self) -> None:
        events = (
            self._event("fit-dog-sit", "sit", _sha("a"), "4"),
            self._event("fit-human-turn", "turn", _sha("b"), "6"),
        )
        manifest = SimpleNamespace(events=events)
        checkpoint_identity = self._checkpoint_identity()
        checkpoint_digest = authority.object_sha256(checkpoint_identity)
        candidates = []
        for index in range(40):
            candidate_id = f"filler-{index:02d}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "receipt_digest": hashlib.sha256(candidate_id.encode()).hexdigest(),
                }
            )
        for index, event in enumerate(events):
            candidates[index * 20] = {
                "candidate_id": event.event_id,
                "path": f"/score/{event.event_id}.json",
                "file_sha256": _sha(("8", "9")[index]),
                "receipt_digest": _sha(("a", "b")[index]),
                "analysis_split": "fit",
                "action_family_id": event.action_family,
                "semantic_branch": "action",
                "clean_latent_tensor_sha256": event.clean_latent_tensor_sha256,
                "official_gaussian_tensor_sha256": event.official_gaussian_tensor_sha256,
                "prompt_by_branch": dict(event.prompt_by_branch),
                "full_t2v_caption_by_branch": dict(event.raw_caption_by_branch),
                "geometry_source_video_sha256": event.source_video.sha256,
                "candidate_shape": list(event.latent_shape),
                "raw_global_action_energy_score": float(index),
            }
        groups = [
            {
                "group_id": group_id,
                "frozen_checkpoint_receipt_digest": _sha("c"),
                "candidate_receipts": candidates[offset : offset + 20],
            }
            for group_id, offset in (("A", 0), ("B", 20))
        ]
        method = {
            "path": "/archive.tar",
            "file_sha256": _sha("d"),
            "git_archive_revision": authority.CAST_V4_METHOD_REVISION,
        }
        root_spec = {"path": "/root.json", "file_sha256": _sha("e")}

        def boundary(**kwargs):
            return {
                "boundary_id": kwargs["boundary_id"],
                "path": f"/{kwargs['boundary_id']}.json",
                "file_sha256": kwargs["expected_file_sha256"],
                "embedded_digest": kwargs["expected_embedded_digest"],
                "required_boolean_observations": dict(kwargs["required_booleans"]),
                "inherited_as_population_or_update_authority": False,
            }

        sail_boundary = {
            "boundary_id": "sail_prior_frozen_intervention_no_success",
            "path": "/sail.json",
            "file_sha256": authority.SAIL_PRIOR_NO_SUCCESS_FILE_SHA256,
            "embedded_digest": authority.SAIL_PRIOR_NO_SUCCESS_RECEIPT_DIGEST,
            "required_boolean_observations": {
                "postflight_complete": True,
                "all_six_mp4_exact81": True,
                "scientific_claim_authorized": False,
                "action_editing_success_claim_authorized": False,
                "training_performed": False,
                "source_condition_in_live_query": False,
            },
            "child_receipts": [],
            "inherited_as_population_or_update_authority": False,
        }

        with patch.object(authority, "_validate_cast_method_archive", return_value=method), patch.object(
            authority, "_validate_cast_root_spec", return_value=root_spec
        ), patch.object(
            authority, "_validate_cast_group", side_effect=groups
        ), patch.object(
            authority, "_validate_negative_boundary", side_effect=boundary
        ) as legacy_mock, patch.object(
            authority,
            "_validate_sail_prior_no_success",
            return_value=sail_boundary,
        ) as sail_mock:
            result = authority._collect_cast_v4_bindings(
                manifest=manifest,
                checkpoint_content_identity=checkpoint_identity,
                checkpoint_content_receipt_digest=checkpoint_digest,
                cast_method_archive_path="archive",
                expected_cast_method_archive_sha256=_sha("d"),
                expected_cast_method_revision=authority.CAST_V4_METHOD_REVISION,
                cast_root_spec_path="root",
                expected_cast_root_spec_sha256=_sha("e"),
                cast_group_receipt_paths=("A", "B"),
                expected_cast_group_receipt_sha256=(_sha("1"), _sha("2")),
                legacy_v3_no_go_path="legacy",
                sail_prior_no_success_path="sail",
            )
        self.assertEqual(
            set(result["selected_action_score_by_event"]),
            {event.event_id for event in events},
        )
        self.assertEqual(
            legacy_mock.call_args_list[0].kwargs["required_booleans"],
            {
                "optimizer_authorized": False,
                "gates.confirmation_overall": False,
            },
        )
        self.assertEqual(
            result["negative_boundaries"][1]["required_boolean_observations"],
            sail_boundary["required_boolean_observations"],
        )
        sail_mock.assert_called_once_with("sail")

    def test_author_and_validate_evidence_remain_fit_only(self) -> None:
        manifest_path, _, manifest, _ = self._author_manifest()
        checkpoint_identity = self._checkpoint_identity()
        checkpoint_digest = authority.object_sha256(checkpoint_identity)
        root_file = self._write("root.json", b"root")
        legacy_file = self._write("legacy.json", b"legacy")
        sail_file = self._write("sail.json", b"sail")
        cast_bindings = {
            "cast_v4_method_archive": {"path": "/archive", "file_sha256": _sha("4")},
            "cast_v4_root_spec": {
                "path": str(root_file),
                "file_sha256": authority._file_sha256(root_file),
            },
            "cast_v4_groups": [{"receipt_digest": _sha("5")}, {"receipt_digest": _sha("6")}],
            "selected_action_score_by_event": {
                event.event_id: {"receipt_digest": _sha(("7", "8")[index])}
                for index, event in enumerate(manifest.events)
            },
            "negative_boundaries": [
                {
                    "boundary_id": "d541801_v3_confirmation_no_optimizer_go",
                    "path": str(legacy_file),
                    "file_sha256": authority._file_sha256(legacy_file),
                    "embedded_digest": authority.LEGACY_V3_NO_GO_RECEIPT_DIGEST,
                    "required_boolean_observations": {
                        "optimizer_authorized": False,
                        "gates.confirmation_overall": False,
                    },
                    "inherited_as_population_or_update_authority": False,
                },
                {
                    "boundary_id": "sail_prior_frozen_intervention_no_success",
                    "path": str(sail_file),
                    "file_sha256": authority._file_sha256(sail_file),
                    "embedded_digest": authority.SAIL_PRIOR_NO_SUCCESS_RECEIPT_DIGEST,
                    "required_boolean_observations": {
                        "postflight_complete": True,
                        "all_six_mp4_exact81": True,
                        "scientific_claim_authorized": False,
                        "action_editing_success_claim_authorized": False,
                        "training_performed": False,
                        "source_condition_in_live_query": False,
                    },
                    "inherited_as_population_or_update_authority": False,
                },
            ],
        }
        evidence_path = self.root / "fit-evidence.json"
        tensor_patch = patch.object(
            authority, "_inspect_tensor_artifact", side_effect=self._tensor_inspection
        )
        load_patch = patch.object(
            authority, "_load_tensor_artifact", return_value=None
        )
        media_patch = patch.object(
            authority,
            "_inspect_source_media",
            return_value={"frame_count": 81, "fps": 25.0},
        )
        with tensor_patch, load_patch, media_patch, patch.object(
            authority, "_collect_cast_v4_bindings", return_value=cast_bindings
        ):
            evidence = authority.author_fit_only_evidence(
                output_path=evidence_path,
                manifest_path=manifest_path,
                expected_manifest_file_sha256=authority._file_sha256(manifest_path),
                checkpoint_tree_sha256=self.checkpoint,
                checkpoint_content_identity=checkpoint_identity,
                checkpoint_content_receipt_digest=checkpoint_digest,
                action_adapter_schema_sha256=self.adapter,
                cast_method_archive_path="archive",
                expected_cast_method_archive_sha256=_sha("4"),
                expected_cast_method_revision=authority.CAST_V4_METHOD_REVISION,
                cast_root_spec_path="root",
                expected_cast_root_spec_sha256=_sha("5"),
                cast_group_receipt_paths=("A", "B"),
                expected_cast_group_receipt_sha256=(_sha("6"), _sha("7")),
                legacy_v3_no_go_path=legacy_file,
                sail_prior_no_success_path=sail_file,
            )
        with patch.object(
            authority, "_inspect_tensor_artifact", side_effect=self._tensor_inspection
        ), patch.object(
            authority, "_load_tensor_artifact", return_value=None
        ), patch.object(
            authority,
            "_inspect_source_media",
            return_value={"frame_count": 81, "fps": 25.0},
        ), patch.object(
            authority, "_collect_cast_v4_bindings", return_value=cast_bindings
        ), patch.object(
            authority, "_external_evidence_file_bindings", return_value=tuple()
        ):
            _, _, checked = authority.validate_fit_only_geometry_authority(
                manifest_path=manifest_path,
                expected_manifest_file_sha256=authority._file_sha256(manifest_path),
                evidence_path=evidence_path,
                expected_evidence_file_sha256=authority._file_sha256(evidence_path),
                expected_checkpoint_tree_sha256=self.checkpoint,
                checkpoint_content_identity=checkpoint_identity,
                expected_checkpoint_content_receipt_digest=checkpoint_digest,
                expected_action_adapter_schema_sha256=self.adapter,
                cast_method_archive_path="archive",
                expected_cast_method_archive_sha256=_sha("4"),
                expected_cast_method_revision=authority.CAST_V4_METHOD_REVISION,
                cast_group_receipt_paths=("A", "B"),
                expected_cast_group_receipt_sha256=(_sha("6"), _sha("7")),
            )
        self.assertTrue(evidence["geometry_measurement_authorized"])
        self.assertTrue(checked.validation_receipt["geometry_measurement_authorized"])
        for receipt in (evidence, checked.validation_receipt):
            for field in authority._NO_UPDATE_CLAIMS:
                self.assertFalse(receipt[field])

    def test_strict_json_rejects_duplicate_keys_and_cli_has_no_free_jsonpath(self) -> None:
        duplicate = self._write("duplicate.json", b'{"x":1,"x":2}\n')
        with self.assertRaisesRegex(authority.PairV7FitOnlyAuthorityError, "invalid"):
            authority._strict_json(duplicate, label="duplicate")
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("--legacy-v3-optimizer-false-json-path", source)
        self.assertNotIn("--legacy-v3-confirmation-false-json-path", source)
        self.assertIn('LEGACY_V3_OPTIMIZER_FALSE_JSON_PATH = "optimizer_authorized"', source)
        self.assertIn(
            'LEGACY_V3_CONFIRMATION_FALSE_JSON_PATH = "gates.confirmation_overall"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
