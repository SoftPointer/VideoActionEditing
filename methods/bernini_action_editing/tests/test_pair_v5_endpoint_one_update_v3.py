from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_endpoint_one_update_v3 as endpoint  # noqa: E402


def _write_bytes(path: Path, value: bytes) -> str:
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> str:
    raw = endpoint.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _seal(value: dict[str, object], field: str = "receipt_digest") -> dict[str, object]:
    return {**value, field: endpoint.object_sha256(value)}


def _write_exact81_video(path: Path, *, offset: int) -> str:
    width, height = 8, 6
    frame_bytes = width * height * 3
    raw = b"".join(
        bytes(
            ((offset + frame_index * 17 + byte_index * 3) % 256)
            for byte_index in range(frame_bytes)
        )
        for frame_index in range(81)
    )
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}", "-r", "25", "-i", "pipe:0",
            "-frames:v", "81", "-an", "-c:v", "ffv1", "-level", "3",
            "-g", "1", str(path),
        ],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr.decode("utf-8", errors="replace"))
    return endpoint.file_sha256(path)


class EndpointPopulationFixture:
    CHECKPOINT = "a" * 64

    def __init__(
        self,
        root: Path,
        *,
        confirmation_candidate: str | None = None,
        low_axis_by_source: dict[int, str] | None = None,
        low_temporal_sources: set[int] | None = None,
        low_event_sources: set[int] | None = None,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.policy = endpoint.make_parent_policy(
            generation_round=0,
            checkpoint_tree_sha256=self.CHECKPOINT,
        )
        self.temporal_evaluator = endpoint.decoded_temporal_event_evaluator_binding()
        self.event_evaluator = endpoint.decoded_temporal_event_evaluator_binding()
        self.gate = endpoint.make_gate_policy(
            temporal_counterfactual_evaluator=self.temporal_evaluator,
            event81_evaluator=self.event_evaluator,
            identity_min=0.75,
            background_min=0.75,
            camera_min=0.75,
            temporal_min=0.75,
            quality_min=0.75,
            action_min=0.80,
            action_margin=0.20,
            counterfactual_margin_min=0.70,
            temporal_order_min=0.70,
            event_start_min=0.75,
            event_transition_min=0.75,
            event_terminal_min=0.75,
            event_terminal_hold_min=0.75,
        )
        self.inputs: list[dict[str, dict[str, str]]] = []
        self.records: dict[str, dict[str, Path]] = {}
        low_axis_by_source = low_axis_by_source or {}
        low_temporal_sources = low_temporal_sources or set()
        low_event_sources = low_event_sources or set()
        self.observer_registration_by_ordinal: dict[int, dict[str, object]] = {}
        for ordinal in (1, 2):
            authority = self.root / f"observer-{ordinal}-authority.txt"
            runtime = self.root / f"observer-{ordinal}-runtime.txt"
            protocol = self.root / f"observer-{ordinal}-protocol.txt"
            _write_bytes(authority, f"external authority {ordinal}\n".encode())
            _write_bytes(runtime, f"external runtime {ordinal}\n".encode())
            _write_bytes(protocol, f"external protocol {ordinal}\n".encode())
            self.observer_registration_by_ordinal[ordinal] = (
                endpoint.decoded_event.make_observer_registration(
                    observer_id=f"endpoint-observer-{ordinal}",
                    observer_kind="frozen_external_event_model",
                    observer_authority_artifact=endpoint.file_binding(authority),
                    observer_runtime_artifact=endpoint.file_binding(runtime),
                    model_or_protocol_artifact=endpoint.file_binding(protocol),
                )
            )
        for source_index in range(2):
            source = self.root / f"source-{source_index}.mp4"
            source_sha = _write_bytes(
                source, f"exact81-source-{source_index}".encode("ascii")
            )
            caption = (
                f"A static source scene {source_index}; the main subject performs "
                "the requested body action while the view remains fixed."
            )
            caption_sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
            event_spec = endpoint.decoded_event.make_event_spec(
                action_family_id="body-action",
                source_video_sha256=source_sha,
                complete_caption_sha256=caption_sha,
                actor_binding=f"the same main subject in source scene {source_index}",
                start_state_question="Is the same subject in the registered start state?",
                transition_question="Is the same subject executing the requested transition?",
                terminal_state_question="Is the same subject in the requested terminal state?",
                terminal_hold_question="Does the same subject hold the terminal state at the end?",
                registered_observers=list(
                    self.observer_registration_by_ordinal.values()
                ),
            )
            event_spec_path = self.root / f"source-{source_index}-event-spec.json"
            _write_json(event_spec_path, event_spec)
            for endpoint_index, score in enumerate((0.95, 0.55)):
                candidate_id = f"source{source_index}-candidate{endpoint_index}"
                split = (
                    "confirmation"
                    if confirmation_candidate == candidate_id
                    else "fit"
                )
                self._candidate(
                    candidate_id=candidate_id,
                    source=source,
                    source_sha=source_sha,
                    caption=caption,
                    caption_sha=caption_sha,
                    seed=1000 + source_index * 10 + endpoint_index,
                    score=score,
                    split=split,
                    low_axis=low_axis_by_source.get(source_index),
                    event_spec_path=event_spec_path,
                    event_spec=event_spec,
                    low_temporal=source_index in low_temporal_sources,
                    low_event=source_index in low_event_sources,
                )

    def _candidate(
        self,
        *,
        candidate_id: str,
        source: Path,
        source_sha: str,
        caption: str,
        caption_sha: str,
        seed: int,
        score: float,
        split: str,
        low_axis: str | None,
        event_spec_path: Path,
        event_spec: dict[str, object],
        low_temporal: bool,
        low_event: bool,
    ) -> None:
        directory = self.root / candidate_id
        directory.mkdir()
        mp4 = directory / "rv2v.mkv"
        clean = directory / "rv2v.normalized-clean-latent.safetensors"
        mp4_sha = _write_exact81_video(
            mp4, offset=sum(candidate_id.encode("ascii")) % 251
        )
        clean_sha = _write_bytes(clean, f"latent-{candidate_id}".encode("ascii"))

        native_unsigned: dict[str, object] = {
            "schema_version": endpoint.NATIVE_RECEIPT_SCHEMA,
            "checkpoint": {"tree_sha256": self.CHECKPOINT},
            "freeze_certificate": {
                "base_frozen": True,
                "lora_module_count": 0,
                "trainable_parameter_tensors": 0,
            },
        }
        native = _seal(native_unsigned)
        native_path = directory / "receipt.json"
        native_sha = _write_json(native_path, native)

        envelope_sha = hashlib.sha256(
            f"envelope-{candidate_id}".encode("ascii")
        ).hexdigest()
        legacy_unsigned: dict[str, object] = {
            "schema_version": endpoint.LEGACY_ROLLOUT_SCHEMA,
            "candidate": {
                "candidate_id": candidate_id,
                "source_video": str(source),
                "source_video_sha256": source_sha,
                "complete_caption": caption,
                "complete_caption_sha256": caption_sha,
                "caption_contract": (
                    "complete_source_content_caption_with_requested_new_action"
                ),
                "seed": seed,
                "guidance": {
                    "omega_txt": 4.0,
                    "omega_vid": 1.25,
                    "omega_img": 4.5,
                },
            },
            "candidate_envelope_sha256": envelope_sha,
            "sampling_contract": {
                "condition_mode": "rv2v4",
                "num_frames": 81,
                "latent_frames": 21,
                "fps": 25,
                "num_inference_steps": 40,
                "source_reference_indices": [0, 27, 53, 80],
                "target_initialization": "official_gen_wanx22_fresh_gaussian",
            },
            "semantic_input_closure": {
                "accepted": ["source_video", "complete_caption"],
                "target_video": False,
                "t2v_proposal_media": False,
                "donor_video": False,
                "external_reference": False,
                "mask": False,
                "flow": False,
                "pose": False,
                "track": False,
                "trajectory": False,
            },
            "native_receipt_path": str(native_path),
            "native_receipt_sha256": native_sha,
            "native_receipt_digest": native["receipt_digest"],
            "artifacts": {
                "mp4": {
                    "path": str(mp4),
                    "sha256": mp4_sha,
                    "frame_count": 81,
                    "fps": 25,
                    "width": 8,
                    "height": 6,
                },
                "predecode_clean_latent": {
                    "path": str(clean),
                    "sha256": clean_sha,
                    "tensor_key": "normalized_clean_latent",
                    "shape": [1, 16, 21, 8, 8],
                },
            },
        }
        legacy = _seal(legacy_unsigned)
        legacy_path = directory / "pair-v5-rollout-receipt.json"
        legacy_sha = _write_json(legacy_path, legacy)

        action_unsigned: dict[str, object] = {
            "schema_version": endpoint.ACTIVE_ACTION_SCHEMA,
            "candidate": {
                "candidate_id": candidate_id,
                "analysis_split": split,
                "action_family_id": "body-action",
                "source_video_sha256": source_sha,
                "complete_caption_utf8_sha256": caption_sha,
                "seed": seed,
            },
            "source": {
                "source_video_path": str(source),
                "source_video_sha256_declared": source_sha,
                "source_video_sha256_recomputed": source_sha,
                "calibration_geometry_source_sha256": source_sha,
            },
            "rollout": {
                "pair_receipt_path": str(legacy_path),
                "pair_receipt_file_sha256": legacy_sha,
                "pair_receipt_digest": legacy["receipt_digest"],
                "native_receipt_digest": native["receipt_digest"],
                "candidate_envelope_sha256": envelope_sha,
                "native_condition_mode": "rv2v4",
                "checkpoint_tree_sha256": self.CHECKPOINT,
                "generated_mp4_sha256": mp4_sha,
                "generated_mp4_consumed_by_scorer": False,
            },
            "calibration": {
                "action_family_id": "body-action",
                "family_mapping": {
                    "kind": "clipped_affine_fit_only",
                    "anchor_source_split": "fit",
                    "clip_min": 0.0,
                    "clip_max": 1.0,
                    "lower_raw_anchor": 0.0,
                    "upper_raw_anchor": 1.0,
                },
            },
            "prompts": {
                "full_t2v_caption_by_branch": {"action": caption},
                "full_t2v_caption_utf8_sha256_by_branch": {
                    "action": caption_sha
                },
            },
            "artifacts": {
                "clean_latent_artifact_sha256": clean_sha,
                "clean_and_gaussian_are_same_candidate_artifacts": True,
            },
            "mace": {
                "raw_global_action_energy_score": float(score),
                "calibrated_family_action_score": float(score),
                "decision_threshold": 0.8,
                "passes_calibrated_action_metric": score >= 0.8,
            },
            "optimizer_authorized": False,
            "scientific_action_editing_claim": False,
        }
        action = _seal(action_unsigned)
        action_path = directory / "pair-v5-native-rv2v-action-score-v4.json"
        _write_json(action_path, action)

        scores = {
            "source_identity_appearance_proxy": 0.90,
            "background_appearance_fixed_grid_proxy": 0.90,
            "source_bound_spatial_layout_viewpoint_proxy": 0.90,
            "non_target_temporal_consistency_proxy": 0.90,
            "decode_video_quality_diagnostic": 0.90,
            # Deliberately adversarial; these fields must remain diagnostic-only.
            "source_identity_appearance_wrong_source_proxy": 0.99,
            "source_identity_appearance_correct_minus_wrong_margin": -0.09,
            "background_appearance_wrong_source_fixed_grid_proxy": 0.99,
            "background_appearance_correct_minus_wrong_margin": -0.09,
            "source_bound_spatial_layout_wrong_source_proxy": 0.99,
            "source_bound_spatial_layout_correct_minus_wrong_margin": -0.09,
        }
        if low_axis is not None:
            scores[endpoint.PRIMARY_PRESERVATION_METRICS[low_axis]] = 0.10
        preservation_unsigned: dict[str, object] = {
            "schema_version": endpoint.DECODED_PRESERVATION_SCHEMA,
            "candidate_id": candidate_id,
            "correct_source_video_sha256": source_sha,
            "wrong_source_video_sha256": "f" * 64,
            "candidate_mp4_sha256": mp4_sha,
            "predecode_clean_latent_sha256": clean_sha,
            "candidate_envelope_sha256": envelope_sha,
            "rollout_receipt_digest": legacy["receipt_digest"],
            "native_rollout_receipt_digest": native["receipt_digest"],
            "metrics": scores,
            "evidence_valid": True,
            "eligible_for_downstream_calibration": False,
            "absolute_source_preservation_pass_claim": False,
        }
        preservation = _seal(preservation_unsigned)
        preservation_path = directory / "preservation.json"
        _write_json(preservation_path, preservation)

        salt_path = directory / "blind-salt.bin"
        salt_path.write_bytes(hashlib.sha256(candidate_id.encode("ascii")).digest())
        challenge_dir = directory / "decoded-event-challenge"
        public, private = endpoint.decoded_event.prepare_challenge(
            rollout_receipt_path=legacy_path,
            event_spec_path=event_spec_path,
            blind_salt_path=salt_path,
            preparer_id="endpoint-fixture-preparer",
            output_dir=challenge_dir,
        )
        base_probability = {
            "start": [0.95 if index < 16 else 0.05 for index in range(81)],
            "transition": [
                0.95 if 30 <= index < 35 else 0.05 for index in range(81)
            ],
            "terminal": [
                (0.50 if low_event else 0.95) if index >= 61 else 0.05
                for index in range(81)
            ],
            "terminal_hold": [
                (0.50 if low_event else 0.95) if index >= 73 else 0.05
                for index in range(81)
            ],
        }
        observer_paths = []
        for ordinal in (1, 2):
            observations: dict[str, object] = {}
            offset = 0.0 if ordinal == 1 else -0.01
            for blind_id in public["blind_arm_order"]:
                mapping = private["transform_by_blind_id"][blind_id][
                    "frame_index_map"
                ]
                if low_temporal:
                    mapping = list(range(81))
                probabilities = {
                    name: [
                        max(
                            0.0,
                            min(1.0, base_probability[name][source_frame] + offset),
                        )
                        for source_frame in mapping
                    ]
                    for name in endpoint.decoded_event.EVIDENCE_ORDER
                }
                observations[blind_id] = {
                    "blind_arm_id": blind_id,
                    "review_media_sha256": public["blind_arms"][blind_id][
                        "review_media"
                    ]["sha256"],
                    "frame_indices": list(range(81)),
                    "start_probability_by_frame": probabilities["start"],
                    "transition_probability_by_frame": probabilities["transition"],
                    "terminal_probability_by_frame": probabilities["terminal"],
                    "terminal_hold_probability_by_frame": probabilities[
                        "terminal_hold"
                    ],
                    "ambiguous_or_unreviewable": False,
                }
            evidence_artifact = directory / f"observer-{ordinal}-evidence.txt"
            _write_bytes(
                evidence_artifact,
                f"detached blind evidence {candidate_id} observer {ordinal}\n".encode(),
            )
            registration = self.observer_registration_by_ordinal[ordinal]
            observer_unsigned = {
                "schema_version": endpoint.decoded_event.OBSERVER_RECEIPT_SCHEMA,
                "observer_id": f"endpoint-observer-{ordinal}",
                "observer_kind": "frozen_external_event_model",
                "observer_authority_digest": registration[
                    "observer_authority_digest"
                ],
                "challenge_digest": public["challenge_digest"],
                "event_spec_digest": event_spec["event_spec_digest"],
                "blind_arm_order": public["blind_arm_order"],
                "arm_observations_by_blind_id": observations,
                "detached_evidence_artifact": endpoint.file_binding(
                    evidence_artifact
                ),
                "observer_runtime_artifact": registration[
                    "observer_runtime_artifact"
                ],
                "model_or_protocol_digest": registration[
                    "model_or_protocol_digest"
                ],
                "independent_from_candidate_generator": True,
                "independent_from_challenge_preparer": True,
                "transform_identity_was_hidden": True,
                "candidate_identity_was_hidden": True,
                "labels_not_inferred_from_filename_branch_or_seed": True,
                "annotation_complete": True,
                "receipt_self_signature_authorizes_optimizer": False,
            }
            observer = _seal(observer_unsigned)
            observer_path = directory / f"observer-{ordinal}.json"
            _write_json(observer_path, observer)
            observer_paths.append(observer_path)
        master = endpoint.decoded_event.build_master_receipt(
            public_challenge_file=endpoint.file_binding(
                challenge_dir / "public-challenge.json"
            ),
            private_transform_key_file=endpoint.file_binding(
                challenge_dir / "private-transform-key.json"
            ),
            observer_receipt_files=[
                endpoint.file_binding(path) for path in observer_paths
            ],
        )
        master_path = directory / "decoded-temporal-event-master-v1.json"
        _write_json(master_path, master)
        temporal, event = endpoint.decoded_event.make_endpoint_projections(
            master, master_file=endpoint.file_binding(master_path)
        )
        temporal_path = directory / "temporal-counterfactual.json"
        _write_json(temporal_path, temporal)
        event_path = directory / "event81.json"
        _write_json(event_path, event)

        rollout_evidence = endpoint.make_rollout_evidence(
            parent_policy=self.policy,
            action_receipt_file=endpoint.file_binding(action_path),
        )
        rollout_path = directory / "round-rollout-evidence-v3.json"
        _write_json(rollout_path, rollout_evidence)
        self.inputs.append(
            {
                "rollout_evidence": endpoint.file_binding(rollout_path),
                "action_receipt": endpoint.file_binding(action_path),
                "preservation_receipt": endpoint.file_binding(preservation_path),
                "temporal_counterfactual_receipt": endpoint.file_binding(
                    temporal_path
                ),
                "event81_receipt": endpoint.file_binding(event_path),
            }
        )
        self.records[candidate_id] = {
            "action": action_path,
            "preservation": preservation_path,
            "rollout": rollout_path,
            "temporal": temporal_path,
            "event": event_path,
            "master": master_path,
        }

    def request_and_manifest(self) -> tuple[dict[str, object], dict[str, object], Path]:
        request = endpoint.make_build_request(
            gate_policy=self.gate, candidates=self.inputs
        )
        request_path = self.root / "build-request.json"
        _write_json(request_path, request)
        manifest = endpoint.assemble_one_update_manifest(
            request, build_request_file=endpoint.file_binding(request_path)
        )
        return request, manifest, request_path


class EndpointOneUpdateV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Cache immutable exact81 decodes within this integration suite.

        The production validator still performs every file-binding and replay
        check.  This test-only cache avoids repeatedly invoking ffmpeg for the
        same immutable lossless review media while assembling and replaying a
        manifest.
        """

        cls._original_decode_exact81 = (
            endpoint.decoded_event.decode_exact81_rgb24
        )
        cls._decode_cache: dict[tuple[object, ...], object] = {}

        def cached_decode(
            path: str | Path,
            *,
            ffmpeg: str = "ffmpeg",
            ffprobe: str = "ffprobe",
        ) -> object:
            resolved = Path(path).resolve(strict=True)
            stat = resolved.stat()
            key = (
                str(resolved),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                ffmpeg,
                ffprobe,
            )
            if key not in cls._decode_cache:
                cls._decode_cache[key] = cls._original_decode_exact81(
                    resolved, ffmpeg=ffmpeg, ffprobe=ffprobe
                )
            return cls._decode_cache[key]

        endpoint.decoded_event.decode_exact81_rgb24 = cached_decode

    @classmethod
    def tearDownClass(cls) -> None:
        endpoint.decoded_event.decode_exact81_rgb24 = (
            cls._original_decode_exact81
        )
        cls._decode_cache.clear()

    @staticmethod
    def _rewrite_sealed(path: Path, mutate: object) -> None:
        value = json.loads(path.read_text())
        value.pop("receipt_digest")
        mutate(value)
        _write_json(path, _seal(value))

    def test_two_source_round0_manifest_replays_and_is_one_step_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            _, manifest, _ = fixture.request_and_manifest()
            self.assertEqual(manifest["generation_round"], 0)
            self.assertEqual(manifest["source_count"], 2)
            self.assertEqual(manifest["pair_count"], 2)
            self.assertEqual(manifest["optimizer_update_count"], 1)
            self.assertTrue(manifest["round0_static_pair_one_update_canary_allowed"])
            self.assertFalse(manifest["wrong_source_consumed_as_optimizer_gate"])
            self.assertEqual(
                endpoint.validate_one_update_manifest(manifest), manifest
            )
            authorized = endpoint.authorize_manifest_for_single_step(
                manifest,
                expected_generation_round=0,
                expected_parent_policy_digest=fixture.policy["policy_digest"],
                optimizer_step_index=0,
            )
            self.assertEqual(authorized["manifest_digest"], manifest["manifest_digest"])

    def test_confirmation_candidate_is_rejected_not_silently_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(
                Path(temporary), confirmation_candidate="source0-candidate0"
            )
            request = endpoint.make_build_request(
                gate_policy=fixture.gate, candidates=fixture.inputs
            )
            request_path = Path(temporary) / "request.json"
            _write_json(request_path, request)
            with self.assertRaisesRegex(
                endpoint.PairV5EndpointV3Error, "confirmation"
            ):
                endpoint.assemble_one_update_manifest(
                    request, build_request_file=endpoint.file_binding(request_path)
                )

    def test_each_preservation_axis_is_a_noncompensatory_gate(self) -> None:
        for axis in endpoint.PRIMARY_PRESERVATION_METRICS:
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as temporary:
                fixture = EndpointPopulationFixture(
                    Path(temporary), low_axis_by_source={0: axis}
                )
                request = endpoint.make_build_request(
                    gate_policy=fixture.gate, candidates=fixture.inputs
                )
                request_path = Path(temporary) / "request.json"
                _write_json(request_path, request)
                with self.assertRaises(endpoint.NoAuthorizedPairsError):
                    endpoint.assemble_one_update_manifest(
                        request, build_request_file=endpoint.file_binding(request_path)
                    )

    def test_wrong_source_diagnostics_cannot_veto_or_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            _, manifest, _ = fixture.request_and_manifest()
            self.assertEqual(manifest["pair_count"], 2)
            for pair in manifest["pairs"]:
                for endpoint_name in ("winner", "loser"):
                    self.assertGreaterEqual(
                        pair[endpoint_name]["preservation_scores"]["identity"],
                        0.75,
                    )

    def test_receipt_swap_across_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            swapped = deepcopy(fixture.inputs)
            swapped[0]["preservation_receipt"] = fixture.inputs[2][
                "preservation_receipt"
            ]
            request = endpoint.make_build_request(
                gate_policy=fixture.gate, candidates=swapped
            )
            request_path = Path(temporary) / "request.json"
            _write_json(request_path, request)
            with self.assertRaisesRegex(
                endpoint.PairV5EndpointV3Error, "join differs"
            ):
                endpoint.assemble_one_update_manifest(
                    request, build_request_file=endpoint.file_binding(request_path)
                )

    def test_caller_score_or_flag_fields_are_not_in_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            forged = dict(fixture.inputs[0])
            forged["action_score"] = 1.0
            with self.assertRaisesRegex(
                endpoint.PairV5EndpointV3Error, "closure differs"
            ):
                endpoint.make_build_request(
                    gate_policy=fixture.gate,
                    candidates=[forged, *fixture.inputs[1:]],
                )

    def test_missing_temporal_or_event_evidence_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            for missing in ("temporal_counterfactual_receipt", "event81_receipt"):
                with self.subTest(missing=missing):
                    candidate = dict(fixture.inputs[0])
                    candidate.pop(missing)
                    with self.assertRaisesRegex(
                        endpoint.PairV5EndpointV3Error, "closure differs"
                    ):
                        endpoint.make_build_request(
                            gate_policy=fixture.gate,
                            candidates=[candidate, *fixture.inputs[1:]],
                        )

    def test_legacy_thin_temporal_and_event_packets_are_fail_closed(self) -> None:
        evaluator = endpoint.decoded_temporal_event_evaluator_binding()
        legacy_packets = (
            (
                endpoint.adapt_temporal_counterfactual_receipt,
                {"schema_version": endpoint.LEGACY_TEMPORAL_COUNTERFACTUAL_SCHEMA},
            ),
            (
                endpoint.adapt_event81_receipt,
                {"schema_version": endpoint.LEGACY_EVENT81_SCHEMA},
            ),
        )
        for adapter, packet in legacy_packets:
            with self.subTest(schema=packet["schema_version"]):
                with self.assertRaisesRegex(
                    endpoint.PairV5EndpointV3Error, "master replay"
                ):
                    adapter(packet, expected_evaluator=evaluator)

    def test_resigned_temporal_packet_cannot_change_pinned_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            malicious = Path(temporary) / "self_signed_temporal.py"
            _write_bytes(malicious, b"caller-authored replacement evaluator\n")
            path = fixture.records["source0-candidate0"]["temporal"]

            def mutate(value: dict[str, object]) -> None:
                value["evaluator_implementation"] = endpoint.file_binding(malicious)
                value["branch_energy_by_name"] = {
                    "target": 0.0,
                    "reverse": 1.0,
                    "shuffle": 1.0,
                    "freeze": 1.0,
                }

            self._rewrite_sealed(path, mutate)
            fixture.inputs[0]["temporal_counterfactual_receipt"] = endpoint.file_binding(
                path
            )
            request = endpoint.make_build_request(
                gate_policy=fixture.gate, candidates=fixture.inputs
            )
            request_path = Path(temporary) / "request.json"
            _write_json(request_path, request)
            with self.assertRaisesRegex(
                endpoint.PairV5EndpointV3Error, "master replay"
            ):
                endpoint.assemble_one_update_manifest(
                    request, build_request_file=endpoint.file_binding(request_path)
                )

    def test_temporal_and_full81_evidence_are_independent_hard_gates(self) -> None:
        fixture_kwargs = {
            "temporal": {"low_temporal_sources": {0}},
            "event": {"low_event_sources": {0}},
        }
        for evidence_name, kwargs in fixture_kwargs.items():
            with self.subTest(evidence=evidence_name), tempfile.TemporaryDirectory() as temporary:
                fixture = EndpointPopulationFixture(Path(temporary), **kwargs)
                request = endpoint.make_build_request(
                    gate_policy=fixture.gate, candidates=fixture.inputs
                )
                request_path = Path(temporary) / "request.json"
                _write_json(request_path, request)
                with self.assertRaises(endpoint.NoAuthorizedPairsError):
                    endpoint.assemble_one_update_manifest(
                        request, build_request_file=endpoint.file_binding(request_path)
                    )

    def test_temporal_and_event_projections_must_share_one_master_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            record = fixture.records["source0-candidate0"]
            master = json.loads(record["master"].read_text())
            copied_master_path = Path(temporary) / "copied-master.json"
            _write_json(copied_master_path, master)
            _, copied_event = endpoint.decoded_event.make_endpoint_projections(
                master,
                master_file=endpoint.file_binding(copied_master_path),
            )
            copied_event_path = Path(temporary) / "copied-master-event81.json"
            _write_json(copied_event_path, copied_event)
            fixture.inputs[0]["event81_receipt"] = endpoint.file_binding(
                copied_event_path
            )
            request = endpoint.make_build_request(
                gate_policy=fixture.gate, candidates=fixture.inputs
            )
            request_path = Path(temporary) / "request.json"
            _write_json(request_path, request)
            with self.assertRaisesRegex(
                endpoint.PairV5EndpointV3Error, "join differs"
            ):
                endpoint.assemble_one_update_manifest(
                    request, build_request_file=endpoint.file_binding(request_path)
                )

    def test_gate_policy_rejects_caller_supplied_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "caller-evaluator.py"
            _write_bytes(fake, b"caller supplied evaluator\n")
            registered = endpoint.decoded_temporal_event_evaluator_binding()
            with self.assertRaisesRegex(
                endpoint.PairV5EndpointV3Error, "registered decoded master evaluator"
            ):
                endpoint.make_gate_policy(
                    temporal_counterfactual_evaluator=registered,
                    event81_evaluator=endpoint.file_binding(fake),
                    identity_min=0.75,
                    background_min=0.75,
                    camera_min=0.75,
                    temporal_min=0.75,
                    quality_min=0.75,
                    action_min=0.80,
                    action_margin=0.20,
                    counterfactual_margin_min=0.70,
                    temporal_order_min=0.70,
                    event_start_min=0.75,
                    event_transition_min=0.75,
                    event_terminal_min=0.75,
                    event_terminal_hold_min=0.75,
                )

    def test_single_source_population_never_authorizes_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            request = endpoint.make_build_request(
                gate_policy=fixture.gate, candidates=fixture.inputs[:2]
            )
            request_path = Path(temporary) / "request.json"
            _write_json(request_path, request)
            with self.assertRaises(endpoint.NoAuthorizedPairsError):
                endpoint.assemble_one_update_manifest(
                    request, build_request_file=endpoint.file_binding(request_path)
                )

    def test_second_step_or_next_round_reuse_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            _, manifest, _ = fixture.request_and_manifest()
            with self.assertRaisesRegex(endpoint.PairV5EndpointV3Error, "stale"):
                endpoint.authorize_manifest_for_single_step(
                    manifest,
                    expected_generation_round=0,
                    expected_parent_policy_digest=fixture.policy["policy_digest"],
                    optimizer_step_index=1,
                )
            with self.assertRaisesRegex(endpoint.PairV5EndpointV3Error, "stale"):
                endpoint.authorize_manifest_for_single_step(
                    manifest,
                    expected_generation_round=1,
                    expected_parent_policy_digest=fixture.policy["policy_digest"],
                    optimizer_step_index=0,
                )

    def test_resigned_manifest_cannot_replace_trusted_action_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EndpointPopulationFixture(Path(temporary))
            _, manifest, _ = fixture.request_and_manifest()
            forged = deepcopy(manifest)
            forged["pairs"][0]["winner"]["action_score"] = 1.0
            pair = forged["pairs"][0]
            pair["action_margin"] = (
                pair["winner"]["action_score"] - pair["loser"]["action_score"]
            )
            pair_unsigned = dict(pair)
            pair_unsigned.pop("pair_digest")
            pair["pair_digest"] = endpoint.object_sha256(pair_unsigned)
            manifest_unsigned = dict(forged)
            manifest_unsigned.pop("manifest_digest")
            forged["manifest_digest"] = endpoint.object_sha256(manifest_unsigned)
            with self.assertRaisesRegex(endpoint.PairV5EndpointV3Error, "replay"):
                endpoint.validate_one_update_manifest(forged)


if __name__ == "__main__":
    unittest.main()
