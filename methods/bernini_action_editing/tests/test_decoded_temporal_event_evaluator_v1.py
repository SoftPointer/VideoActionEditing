from __future__ import annotations

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

import decoded_temporal_event_evaluator_v1 as evaluator  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(evaluator.canonical_json_bytes(value) + b"\n")


def _seal(unsigned: dict[str, object]) -> dict[str, object]:
    return {**unsigned, "receipt_digest": evaluator.object_sha256(unsigned)}


def _run(command: list[str], *, input_bytes: bytes | None = None) -> None:
    completed = subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr.decode("utf-8", errors="replace"))


class RealPreparedFixture:
    width = 8
    height = 6

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.candidate_video = self.root / "rv2v.mkv"
        frame_size = self.width * self.height * 3
        raw_frames = []
        for frame_index in range(evaluator.FRAME_COUNT):
            # Every decoded frame is byte-distinct; transform equality cannot
            # pass merely because the synthetic fixture happened to be static.
            frame = bytes(
                ((frame_index * 17 + byte_index * 3) % 256)
                for byte_index in range(frame_size)
            )
            raw_frames.append(frame)
        _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{self.width}x{self.height}",
                "-r",
                str(evaluator.FPS),
                "-i",
                "pipe:0",
                "-frames:v",
                str(evaluator.FRAME_COUNT),
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-g",
                "1",
                str(self.candidate_video),
            ],
            input_bytes=b"".join(raw_frames),
        )
        self.source_video = self.root / "source.mkv"
        self.source_video.write_bytes(self.candidate_video.read_bytes())
        source_sha = evaluator.file_sha256(self.source_video)
        candidate_sha = evaluator.file_sha256(self.candidate_video)
        self.caption = (
            "A fixed camera shows one dog standing; the same dog lowers its hips, "
            "sits facing the camera, and holds the seated pose through the end."
        )
        self.caption_sha = hashlib.sha256(self.caption.encode("utf-8")).hexdigest()

        native_unsigned = {
            "schema_version": "fixture-native-receipt-v1",
            "checkpoint": {"tree_sha256": "a" * 64},
        }
        self.native = _seal(native_unsigned)
        self.native_path = self.root / "receipt.json"
        _write_json(self.native_path, self.native)

        candidate_id = "fixture-source-action-s1"
        rollout_unsigned = {
            "schema_version": evaluator.ROLLOUT_SCHEMA,
            "candidate": {
                "candidate_id": candidate_id,
                "source_video": str(self.source_video),
                "source_video_sha256": source_sha,
                "complete_caption": self.caption,
                "complete_caption_sha256": self.caption_sha,
                "caption_contract": (
                    "complete_source_content_caption_with_requested_new_action"
                ),
                "seed": 12345,
                "guidance": {"omega_txt": 4.0, "omega_vid": 1.25, "omega_img": 4.5},
            },
            "sampling_contract": {
                "condition_mode": "rv2v4",
                "num_frames": evaluator.FRAME_COUNT,
                "latent_frames": evaluator.LATENT_PHASES,
                "fps": evaluator.FPS,
                "num_inference_steps": 40,
                "source_reference_indices": [0, 27, 53, 80],
                "target_initialization": "official_gen_wanx22_fresh_gaussian",
            },
            "semantic_input_closure": {
                "accepted": ["source_video", "complete_caption"],
                "mask": False,
            },
            "native_receipt_path": str(self.native_path),
            "native_receipt_sha256": evaluator.file_sha256(self.native_path),
            "native_receipt_digest": self.native["receipt_digest"],
            "artifacts": {
                "mp4": {
                    "path": str(self.candidate_video),
                    "sha256": candidate_sha,
                    "frame_count": evaluator.FRAME_COUNT,
                    "fps": evaluator.FPS,
                    "width": self.width,
                    "height": self.height,
                    "normalized_clean_latent": {"not_consumed": True},
                }
            },
        }
        self.rollout = _seal(rollout_unsigned)
        self.rollout_path = self.root / "pair-v5-rollout-receipt.json"
        _write_json(self.rollout_path, self.rollout)

        self.registration_by_ordinal: dict[int, dict[str, object]] = {}
        for ordinal in (1, 2):
            authority = self.root / f"observer-{ordinal}-authority.txt"
            runtime = self.root / f"observer-{ordinal}-runtime.txt"
            protocol = self.root / f"observer-{ordinal}-protocol.txt"
            authority.write_text(f"external authority or public key {ordinal}\n")
            runtime.write_text(f"external observer runtime {ordinal}\n")
            protocol.write_text(f"frozen event model manifest or rubric {ordinal}\n")
            self.registration_by_ordinal[ordinal] = evaluator.make_observer_registration(
                observer_id=f"external-observer-{ordinal}",
                observer_kind="frozen_external_event_model",
                observer_authority_artifact=evaluator.file_binding(authority),
                observer_runtime_artifact=evaluator.file_binding(runtime),
                model_or_protocol_artifact=evaluator.file_binding(protocol),
            )
        self.event_spec = evaluator.make_event_spec(
            action_family_id="dog-sit",
            source_video_sha256=source_sha,
            complete_caption_sha256=self.caption_sha,
            actor_binding="the same single dog visible at the beginning",
            start_state_question="Is the same dog still in the initial standing state?",
            transition_question="Is the same dog actively lowering from stand to sit?",
            terminal_state_question="Is the same dog in the requested seated terminal state?",
            terminal_hold_question="Does the same dog hold the seated state without reversing?",
            registered_observers=list(self.registration_by_ordinal.values()),
        )
        self.event_spec_path = self.root / "event-spec.json"
        _write_json(self.event_spec_path, self.event_spec)
        self.salt_path = self.root / "blind-salt.bin"
        self.salt_path.write_bytes(bytes(range(32)))
        self.challenge_dir = self.root / "challenge"
        self.public, self.private = evaluator.prepare_challenge(
            rollout_receipt_path=self.rollout_path,
            event_spec_path=self.event_spec_path,
            blind_salt_path=self.salt_path,
            preparer_id="preparer-fixture",
            output_dir=self.challenge_dir,
        )
        self.public_path = self.challenge_dir / "public-challenge.json"
        self.private_path = self.challenge_dir / "private-transform-key.json"

    @staticmethod
    def _original_probabilities() -> dict[str, list[float]]:
        return {
            "start": [0.95 if frame < 16 else 0.05 for frame in range(81)],
            "transition": [0.95 if 28 <= frame <= 36 else 0.05 for frame in range(81)],
            "terminal": [0.95 if frame >= 61 else 0.05 for frame in range(81)],
            "terminal_hold": [0.95 if frame >= 73 else 0.05 for frame in range(81)],
        }

    def write_observer(
        self,
        ordinal: int,
        *,
        probability_offset: float = 0.0,
        shared_evidence_bytes: bytes | None = None,
    ) -> Path:
        evidence = self.root / f"observer-{ordinal}-evidence.txt"
        evidence.write_bytes(
            shared_evidence_bytes
            if shared_evidence_bytes is not None
            else f"detached blind evidence {ordinal}\n".encode()
        )
        registration = self.registration_by_ordinal[ordinal]
        base = self._original_probabilities()
        observations: dict[str, object] = {}
        for blind_id in self.public["blind_arm_order"]:
            mapping = self.private["transform_by_blind_id"][blind_id]["frame_index_map"]
            values: dict[str, list[float]] = {}
            for state in evaluator.EVIDENCE_ORDER:
                values[state] = [
                    max(0.0, min(1.0, base[state][source_frame] + probability_offset))
                    for source_frame in mapping
                ]
            observations[blind_id] = {
                "blind_arm_id": blind_id,
                "review_media_sha256": self.public["blind_arms"][blind_id][
                    "review_media"
                ]["sha256"],
                "frame_indices": list(range(81)),
                "start_probability_by_frame": values["start"],
                "transition_probability_by_frame": values["transition"],
                "terminal_probability_by_frame": values["terminal"],
                "terminal_hold_probability_by_frame": values["terminal_hold"],
                "ambiguous_or_unreviewable": False,
            }
        unsigned = {
            "schema_version": evaluator.OBSERVER_RECEIPT_SCHEMA,
            "observer_id": f"external-observer-{ordinal}",
            "observer_kind": "frozen_external_event_model",
            "observer_authority_digest": registration[
                "observer_authority_digest"
            ],
            "challenge_digest": self.public["challenge_digest"],
            "event_spec_digest": self.event_spec["event_spec_digest"],
            "blind_arm_order": self.public["blind_arm_order"],
            "arm_observations_by_blind_id": observations,
            "detached_evidence_artifact": evaluator.file_binding(evidence),
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
        receipt = _seal(unsigned)
        path = self.root / f"observer-{ordinal}.json"
        _write_json(path, receipt)
        return path


class DecodedTemporalEventEvaluatorTest(unittest.TestCase):
    def test_exact81_transform_maps_are_fixed_and_same_multiset_where_required(self) -> None:
        self.assertEqual(evaluator.temporal_index_map("target"), tuple(range(81)))
        self.assertEqual(evaluator.temporal_index_map("reverse"), tuple(range(80, -1, -1)))
        self.assertEqual(evaluator.temporal_index_map("freeze"), (0,) * 81)
        shuffle = evaluator.temporal_index_map("shuffle")
        self.assertEqual(len(shuffle), 81)
        self.assertEqual(sorted(shuffle), list(range(81)))
        self.assertNotEqual(shuffle, tuple(range(81)))
        self.assertNotEqual(shuffle, tuple(range(80, -1, -1)))
        with self.assertRaisesRegex(evaluator.DecodedTemporalEventError, "81"):
            evaluator.apply_frame_map([b"frame"] * 80, "target")

    def test_event_spec_is_source_caption_bound_and_seed_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registrations = []
            for ordinal in (1, 2):
                authority = root / f"authority-{ordinal}"
                runtime = root / f"runtime-{ordinal}"
                protocol = root / f"protocol-{ordinal}"
                authority.write_bytes(f"authority {ordinal}".encode())
                runtime.write_bytes(f"runtime {ordinal}".encode())
                protocol.write_bytes(f"protocol {ordinal}".encode())
                registrations.append(
                    evaluator.make_observer_registration(
                        observer_id=f"observer-{ordinal}",
                        observer_kind="human_blind_annotation",
                        observer_authority_artifact=evaluator.file_binding(authority),
                        observer_runtime_artifact=evaluator.file_binding(runtime),
                        model_or_protocol_artifact=evaluator.file_binding(protocol),
                    )
                )
            spec = evaluator.make_event_spec(
                action_family_id="human-rise",
                source_video_sha256="a" * 64,
                complete_caption_sha256="b" * 64,
                actor_binding="the same single person present in the source",
                start_state_question="Is the person in the initial crouching state?",
                transition_question="Is the person actively rising toward standing?",
                terminal_state_question="Is the person fully upright and standing?",
                terminal_hold_question="Is upright standing held through the end?",
                registered_observers=registrations,
            )
            self.assertEqual(evaluator.validate_event_spec(spec), spec)
            self.assertNotIn("candidate_id", spec)
            self.assertNotIn("seed", spec)
            tampered = json.loads(json.dumps(spec))
            tampered["frame_windows"]["terminal"] = [60, 80]
            with self.assertRaisesRegex(evaluator.DecodedTemporalEventError, "differs"):
                evaluator.validate_event_spec(tampered)

    def test_real_ffmpeg_prepare_master_and_endpoint_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RealPreparedFixture(Path(temporary))
            public, private = evaluator.validate_prepared_challenge(
                fixture.public, fixture.private, replay_media=True
            )
            self.assertEqual(public, fixture.public)
            self.assertEqual(private, fixture.private)
            for transform in evaluator.TRANSFORM_ORDER:
                blind_id = next(
                    blind
                    for blind, item in private["transform_by_blind_id"].items()
                    if item["transform_name"] == transform
                )
                self.assertNotIn(transform, Path(public["blind_arms"][blind_id]["review_media"]["path"]).name)

            observer_a = fixture.write_observer(1)
            observer_b = fixture.write_observer(2, probability_offset=-0.01)
            master = evaluator.build_master_receipt(
                public_challenge_file=evaluator.file_binding(fixture.public_path),
                private_transform_key_file=evaluator.file_binding(fixture.private_path),
                observer_receipt_files=[
                    evaluator.file_binding(observer_a),
                    evaluator.file_binding(observer_b),
                ],
            )
            self.assertTrue(master["evidence_valid"])
            self.assertEqual(master["observer_count"], 2)
            self.assertTrue(master["independent_observer_gate_passed"])
            self.assertTrue(master["observer_agreement_gate_passed"])
            target = master["branch_energy_by_name"]["target"]
            for negative in ("reverse", "shuffle", "freeze"):
                self.assertLess(target, master["branch_energy_by_name"][negative])
            master_path = fixture.root / "master.json"
            _write_json(master_path, master)
            self.assertEqual(evaluator.validate_master_receipt(master), master)
            temporal, event81 = evaluator.make_endpoint_projections(
                master, master_file=evaluator.file_binding(master_path)
            )
            self.assertEqual(
                evaluator.validate_temporal_projection(temporal), temporal
            )
            self.assertEqual(evaluator.validate_event81_projection(event81), event81)
            self.assertEqual(
                event81["terminal_probability_by_frame"],
                master["chronological_event_probability_by_frame"]["terminal"],
            )

            # Even a correctly resealed thin packet cannot replace master
            # evidence with caller-chosen branch energies.
            forged_unsigned = dict(temporal)
            forged_unsigned.pop("receipt_digest")
            forged_unsigned["branch_energy_by_name"]["target"] = 0.0
            forged = _seal(forged_unsigned)
            with self.assertRaisesRegex(
                evaluator.DecodedTemporalEventError, "projection differs"
            ):
                evaluator.validate_temporal_projection(forged)

    def test_duplicate_detached_evidence_produces_non_authorizing_master(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RealPreparedFixture(Path(temporary))
            shared_evidence = b"the same detached evidence cannot count twice\n"
            observer_a = fixture.write_observer(
                1, shared_evidence_bytes=shared_evidence
            )
            observer_b = fixture.write_observer(
                2, shared_evidence_bytes=shared_evidence
            )
            master = evaluator.build_master_receipt(
                public_challenge_file=evaluator.file_binding(fixture.public_path),
                private_transform_key_file=evaluator.file_binding(fixture.private_path),
                observer_receipt_files=[
                    evaluator.file_binding(observer_a),
                    evaluator.file_binding(observer_b),
                ],
            )
            self.assertFalse(master["independent_observer_gate_passed"])
            self.assertFalse(master["evidence_valid"])
            self.assertIn("independent_observer_authority", master["failure_reasons"])

    def test_one_observer_and_observer_file_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RealPreparedFixture(Path(temporary))
            observer_a = fixture.write_observer(1)
            with self.assertRaisesRegex(
                evaluator.DecodedTemporalEventError, "at least two"
            ):
                evaluator.build_master_receipt(
                    public_challenge_file=evaluator.file_binding(fixture.public_path),
                    private_transform_key_file=evaluator.file_binding(fixture.private_path),
                    observer_receipt_files=[evaluator.file_binding(observer_a)],
                )
            observer_b = fixture.write_observer(2)
            unregistered = json.loads(observer_b.read_text())
            unregistered.pop("receipt_digest")
            unregistered["observer_authority_digest"] = "e" * 64
            unregistered = _seal(unregistered)
            unregistered_path = fixture.root / "unregistered-observer.json"
            _write_json(unregistered_path, unregistered)
            with self.assertRaisesRegex(
                evaluator.DecodedTemporalEventError, "preregistered"
            ):
                evaluator.build_master_receipt(
                    public_challenge_file=evaluator.file_binding(fixture.public_path),
                    private_transform_key_file=evaluator.file_binding(fixture.private_path),
                    observer_receipt_files=[
                        evaluator.file_binding(observer_a),
                        evaluator.file_binding(unregistered_path),
                    ],
                )
            bindings = [
                evaluator.file_binding(observer_a),
                evaluator.file_binding(observer_b),
            ]
            master = evaluator.build_master_receipt(
                public_challenge_file=evaluator.file_binding(fixture.public_path),
                private_transform_key_file=evaluator.file_binding(fixture.private_path),
                observer_receipt_files=bindings,
            )
            observer_b.write_bytes(observer_b.read_bytes() + b" ")
            with self.assertRaisesRegex(
                evaluator.DecodedTemporalEventError, "SHA-256 differs"
            ):
                evaluator.validate_master_receipt(master)


if __name__ == "__main__":
    unittest.main()
