#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for location in (METHOD_ROOT, TOOLS_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import pair_v5_t2v_calibration_bank_spec as contract  # noqa: E402
import build_caper_t2v_reward_review as review  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(review.canonical_json_bytes(value) + b"\n")


def _seal(unsigned: dict[str, object]) -> dict[str, object]:
    return {**unsigned, "receipt_digest": review.object_sha256(unsigned)}


def _caption(split: str, family: str, branch: str) -> str:
    return (
        f"A fixed camera observes the registered {split} {family} scene under stable lighting. "
        f"The only performer executes the complete {branch.replace('_', ' ')} semantic event "
        "with a clear start transition terminal state and continuous temporal hold."
    )


def _ffprobe_result(command: tuple[str, ...], *, bad_name: str | None = None):
    frames = "80" if bad_name and Path(command[-1]).parent.name == bad_name else "81"
    payload = {
        "streams": [
            {
                "width": 480,
                "height": 496,
                "avg_frame_rate": "25/1",
                "nb_read_frames": frames,
            }
        ]
    }
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=b"",
    )


class RewardBankFixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.bank_root = self.root / "bank"
        self.bank_root.mkdir()
        self.spec_path = self.root / "sealed-core4-spec.json"
        self.bank_receipt_path = self.bank_root / "pair-v5-t2v-calibration-bank-receipt.json"
        self.output = self.root / "review" / "index.html"
        self.spec = self._make_spec()
        _write_json(self.spec_path, self.spec)
        self.spec_sha = review.file_sha256(self.spec_path)
        self.bank = self._materialize_bank()
        _write_json(self.bank_receipt_path, self.bank)

    @staticmethod
    def _make_spec() -> dict[str, object]:
        cells = (
            ("fit", "dog-sit", "dog-fit", 1601, "sp4-a"),
            ("confirmation", "dog-sit", "dog-confirm", 3501, "sp4-a"),
            ("fit", "human-stand", "human-fit", 1602, "sp4-b"),
            ("confirmation", "human-stand", "human-confirm", 3502, "sp4-b"),
        )
        groups = {
            "sp4-a": {"group_id": "sp4-a", "visible_gpus": [0, 1, 2, 3], "candidates": []},
            "sp4-b": {"group_id": "sp4-b", "visible_gpus": [4, 5, 6, 7], "candidates": []},
        }
        for split, family, cell_id, seed, group_id in cells:
            actor = f"actor-{cell_id}"
            scene = f"scene-{cell_id}"
            action = f"action-{cell_id}"
            for branch in contract.MACE_BRANCH_ORDER:
                caption = _caption(split, family, branch)
                groups[group_id]["candidates"].append(
                    {
                        "candidate_id": f"{cell_id}-{branch}",
                        "analysis_split": split,
                        "action_family_id": family,
                        "calibration_group_id": cell_id,
                        "prompt_group_id": f"{actor}--{scene}",
                        "action_family_group_id": action,
                        "actor_group_id": actor,
                        "scene_group_id": scene,
                        "action_group_id": action,
                        "geometry_source_video": f"/sealed/geometry/{cell_id}.mp4",
                        "geometry_source_video_sha256": hashlib.sha256(
                            f"geometry:{cell_id}".encode()
                        ).hexdigest(),
                        "geometry_contract": contract.GEOMETRY_CONTRACT,
                        "semantic_branch": branch,
                        "full_t2v_caption": caption,
                        "full_t2v_caption_utf8_sha256": hashlib.sha256(
                            caption.encode("utf-8")
                        ).hexdigest(),
                        "caption_contract": contract.CAPTION_CONTRACT,
                        "seed": seed,
                    }
                )
        value = {
            "schema_version": contract.SCHEMA_VERSION_V2,
            "sampling_contract": contract.SAMPLING_CONTRACT,
            "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
            "artifact_use_contract": contract.ARTIFACT_USE_CONTRACT,
            "split_contract": contract.SPLIT_CONTRACT,
            "groups": [groups["sp4-a"], groups["sp4-b"]],
        }
        return contract.validate_root_spec(value)

    @staticmethod
    def _artifact(path: Path, payload: bytes) -> dict[str, object]:
        path.write_bytes(payload)
        return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}

    def _native_receipt(
        self, candidate: dict[str, object], *, cell_gaussian: bytes
    ) -> tuple[dict[str, object], dict[str, object]]:
        candidate_dir = self.bank_root / str(candidate["candidate_id"])
        candidate_dir.mkdir()
        mp4 = self._artifact(
            candidate_dir / "t2v.mp4",
            f"fake-mp4:{candidate['candidate_id']}".encode(),
        )
        clean = self._artifact(
            candidate_dir / "t2v.normalized-clean-latent.safetensors",
            f"clean:{candidate['candidate_id']}".encode(),
        )
        clean.update(
            {
                "shape": [1, 16, 21, 62, 60],
                "native_sampler_before_vae_decode": True,
                "mp4_decode_reencode_used": False,
            }
        )
        mp4.update(
            {
                "frame_count": 81,
                "fps": 25,
                "height": 496,
                "width": 480,
                "normalized_clean_latent": clean,
            }
        )
        gaussian = self._artifact(
            candidate_dir / "t2v.official-initial-gaussian.safetensors",
            cell_gaussian + b":" + str(candidate["semantic_branch"]).encode(),
        )
        gaussian.update(
            {
                "shape": [1, 16, 21, 62, 60],
                "dtype": "torch.float32",
                "stored_dtype": "torch.float32",
                "generator_initial_seed": candidate["seed"],
                "captured_from_native_sampler": True,
                "external_initial_noise_injection": False,
                "source_or_target_derived": False,
                "observer_changed_return_value": False,
                "official_randn_tensor_call_count": 1,
                "raw_value_sha256": hashlib.sha256(cell_gaussian).hexdigest(),
                "content_sha256": hashlib.sha256(b"content:" + cell_gaussian).hexdigest(),
            }
        )
        guidance = contract.SAMPLING_CONTRACT["guidance"]
        native = _seal(
            {
                "schema_version": "bernini-native-identity-generation-canary-v1",
                "method": "frozen-bernini-native-identity-generation-canary",
                "arms": ["t2v"],
                "input": {
                    "source_video_sha256": candidate["geometry_source_video_sha256"],
                    "action_prompt_utf8_sha256": candidate[
                        "full_t2v_caption_utf8_sha256"
                    ],
                    "target_video": False,
                    "external_reference_image_or_video": False,
                    "external_mask_flow_pose_track_trajectory": False,
                    "external_first_frame_anchor": False,
                },
                "preprocessing": {
                    "frame_count": 81,
                    "fps": 25,
                    "source_derived_bucket_hw": [496, 480],
                },
                "conditioning": {
                    "t2v": {
                        "full_source_video_count": 0,
                        "source_derived_reference_count": 0,
                        "source_frame_indices": [],
                        "reference_encoding": "none",
                        "source_ids": {
                            "target_source_id": 0,
                            "video_source_ids": [],
                            "reference_source_ids": [],
                            "conditioning_source_count": 0,
                            "max_conditioning_source_id": 0,
                            "within_pretrained_source_ids_1_through_5": True,
                            "source_id_interpolation_required": False,
                        },
                    }
                },
                "condition_identities": {
                    "rank_zero_broadcasts": {
                        "references": {},
                        "full_source_video": None,
                    },
                    "references": {},
                    "full_source_video": None,
                },
                "source_condition_artifact": None,
                "sampling": {
                    "t2v": {
                        "num_frames": 81,
                        "num_inference_steps": 40,
                        "guidance_mode": "t2v_apg",
                        "seed": candidate["seed"],
                        "omega_txt": guidance["omega_txt"],
                        "omega_vid": guidance["omega_vid"],
                        "omega_img": guidance["omega_img"],
                        "target_initialization": contract.TARGET_INITIALIZATION,
                        "target_mixed_with_source_latent": False,
                        "custom_sampler_or_scheduler": False,
                        "ulysses_size": 4,
                    }
                },
                "latent_geometry": {"video_latent_shape": [1, 16, 21, 62, 60]},
                "outputs": {"t2v": mp4},
                "initial_noise_artifacts": {"t2v": gaussian},
                "interpretation": {"training_performed": False},
            }
        )
        native_path = candidate_dir / "receipt.json"
        _write_json(native_path, native)
        artifacts = {
            "mp4": mp4,
            "predecode_clean_latent": clean,
            "official_initial_gaussian": gaussian,
        }
        return native, artifacts

    def _materialize_bank(self) -> dict[str, object]:
        receipt_rows = []
        cell_proofs = []
        split_membership = {
            split: {axis: set() for axis in contract.SPLIT_GROUP_AXES}
            for split in contract.ANALYSIS_SPLITS
        }
        for group in self.spec["groups"]:
            cell_rows: dict[tuple[str, str, str], list[tuple[dict[str, object], dict[str, object]]]] = {}
            visible = group["visible_gpus"]
            for ordinal, candidate in enumerate(group["candidates"]):
                cell_gaussian = f"gaussian:{candidate['calibration_group_id']}".encode()
                native, artifacts = self._native_receipt(candidate, cell_gaussian=cell_gaussian)
                candidate_dir = self.bank_root / candidate["candidate_id"]
                native_path = candidate_dir / "receipt.json"
                pair = _seal(
                    {
                        "schema_version": contract.RECEIPT_SCHEMA_VERSION,
                        "root_spec_raw_sha256": self.spec_sha,
                        "candidate_envelope_sha256": "e" * 64,
                        "group_id": group["group_id"],
                        "visible_gpus": visible,
                        "runtime_topology": {
                            "world_size": 4,
                            "ulysses_size": 4,
                            "rocr_visible_devices": ",".join(str(item) for item in visible),
                        },
                        "ordinal": ordinal,
                        "candidate": candidate,
                        "sampling_contract": contract.SAMPLING_CONTRACT,
                        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
                        "artifact_use_contract": contract.ARTIFACT_USE_CONTRACT,
                        "split_contract": contract.SPLIT_CONTRACT,
                        "geometry_use_certificate": {
                            "video_sha256": candidate["geometry_source_video_sha256"],
                            "bucket_hw": [496, 480],
                            "latent_shape": [1, 16, 21, 62, 60],
                            "used_to_derive_bucket_shape": True,
                            "vae_latent_created": False,
                            "pixels_entered_transformer": False,
                            "content_conditioning_count": 0,
                        },
                        "native_receipt_path": str(native_path),
                        "native_receipt_sha256": review.file_sha256(native_path),
                        "native_receipt_digest": native["receipt_digest"],
                        "artifacts": artifacts,
                        "interpretation": dict(review._PAIR_INTERPRETATION),
                    }
                )
                pair_path = candidate_dir / review.PAIR_RECEIPT_FILENAME
                _write_json(pair_path, pair)
                receipt_rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "analysis_split": candidate["analysis_split"],
                        "action_family_id": candidate["action_family_id"],
                        "calibration_group_id": candidate["calibration_group_id"],
                        "semantic_branch": candidate["semantic_branch"],
                        "receipt_path": str(pair_path),
                        "receipt_sha256": review.file_sha256(pair_path),
                        "receipt_digest": pair["receipt_digest"],
                        "mp4_sha256": artifacts["mp4"]["sha256"],
                        "predecode_clean_latent_sha256": artifacts[
                            "predecode_clean_latent"
                        ]["sha256"],
                        "official_initial_gaussian_sha256": artifacts[
                            "official_initial_gaussian"
                        ]["sha256"],
                    }
                )
                key = (
                    candidate["analysis_split"],
                    candidate["action_family_id"],
                    candidate["calibration_group_id"],
                )
                cell_rows.setdefault(key, []).append((candidate, artifacts))
                for axis in contract.SPLIT_GROUP_AXES:
                    split_membership[candidate["analysis_split"]][axis].add(
                        candidate[axis]
                    )
            for key, rows in cell_rows.items():
                first = rows[0][1]["official_initial_gaussian"]
                cell_proofs.append(
                    {
                        "analysis_split": key[0],
                        "action_family_id": key[1],
                        "calibration_group_id": key[2],
                        "semantic_branch_count": 10,
                        "semantic_branch_order": list(contract.MACE_BRANCH_ORDER),
                        "all_ten_official_gaussian_tensor_values_byte_equal": True,
                        "all_container_files_individually_sha256_verified": True,
                        "official_gaussian_file_sha256_by_branch": {
                            candidate["semantic_branch"]: artifacts[
                                "official_initial_gaussian"
                            ]["sha256"]
                            for candidate, artifacts in rows
                        },
                        "official_gaussian_raw_value_sha256": first[
                            "raw_value_sha256"
                        ],
                        "official_gaussian_content_sha256": first[
                            "content_sha256"
                        ],
                        "seed": first["generator_initial_seed"],
                    }
                )
        return _seal(
            {
                "schema_version": contract.BANK_RECEIPT_SCHEMA_VERSION,
                "root_spec_raw_sha256": self.spec_sha,
                "candidate_count": 40,
                "cell_count": 4,
                "mace_branch_order": list(contract.MACE_BRANCH_ORDER),
                "sampling_contract": contract.SAMPLING_CONTRACT,
                "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
                "artifact_use_contract": contract.ARTIFACT_USE_CONTRACT,
                "split_contract": contract.SPLIT_CONTRACT,
                "split_group_membership": {
                    split: {
                        axis: sorted(values) for axis, values in axes.items()
                    }
                    for split, axes in split_membership.items()
                },
                "fit_confirmation_all_registered_axes_disjoint": True,
                "same_cell_gaussian_proofs": cell_proofs,
                "candidate_receipts": receipt_rows,
                "interpretation": dict(review._BANK_INTERPRETATION),
            }
        )

    def reseal_bank(self) -> None:
        unsigned = dict(self.bank)
        unsigned.pop("receipt_digest", None)
        self.bank = _seal(unsigned)
        _write_json(self.bank_receipt_path, self.bank)

    def build(self, *, bad_name: str | None = None) -> dict[str, object]:
        with mock.patch.object(
            review.subprocess,
            "run",
            side_effect=lambda command, **_kwargs: _ffprobe_result(
                command, bad_name=bad_name
            ),
        ):
            return review.build_review(
                bank_root=self.bank_root,
                bank_receipt_path=self.bank_receipt_path,
                root_spec_path=self.spec_path,
                output_html=self.output,
                ffprobe="fixture-ffprobe",
            )


class CaperT2VRewardReviewTests(unittest.TestCase):
    def test_complete_core4_writes_fixed_four_column_review_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            audit = fixture.build()
            self.assertTrue(audit["review_complete"])
            self.assertEqual(audit["candidate_count"], 40)
            self.assertEqual(len(audit["cells"]), 4)
            self.assertEqual(audit["status_counts"], {"valid": 40})
            self.assertFalse(audit["authority"]["donor_selection_performed"])
            self.assertFalse(audit["authority"]["target_selection_performed"])
            self.assertEqual(
                audit["selection_policy"]["displayed_video_branches"],
                ["action", "noop", "incomplete", "reverse"],
            )
            self.assertEqual(len(audit["selection_policy"]["audit_only_branches"]), 6)
            page = fixture.output.read_text(encoding="utf-8")
            self.assertEqual(page.count('<article class="card valid">'), 16)
            self.assertIn("generic_wrong_motion", page)
            self.assertIn('data-command="pause"', page)
            self.assertIn('data-seek="dog-fit"', page)
            self.assertIn('data-rate="all"', page)
            self.assertIn("not editor targets, donors, conditions", page)
            stored = json.loads(fixture.output.with_name("index.audit.json").read_bytes())
            self.assertEqual(stored["audit_digest"], audit["audit_digest"])

    def test_missing_candidate_is_visible_and_main_defaults_to_return_three(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            missing_id = fixture.spec["groups"][0]["candidates"][0]["candidate_id"]
            (fixture.bank_root / missing_id / review.PAIR_RECEIPT_FILENAME).unlink()
            with mock.patch.object(
                review.subprocess,
                "run",
                side_effect=lambda command, **_kwargs: _ffprobe_result(command),
            ):
                status = review.main(
                    [
                        "--bank-root",
                        str(fixture.bank_root),
                        "--bank-receipt",
                        str(fixture.bank_receipt_path),
                        "--root-spec",
                        str(fixture.spec_path),
                        "--output",
                        str(fixture.output),
                        "--ffprobe",
                        "fixture-ffprobe",
                    ]
                )
            self.assertEqual(status, 3)
            audit = json.loads(fixture.output.with_name("index.audit.json").read_bytes())
            media = audit["cells"][0]["branches"]["action"]["media"]
            self.assertEqual(media["status"], "missing")
            self.assertIn("absent", media["message"])
            page = fixture.output.read_text(encoding="utf-8")
            self.assertIn("MISSING", page)
            self.assertIn(str(missing_id), page)

    def test_allow_incomplete_is_explicit_cli_override_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            missing_id = fixture.spec["groups"][0]["candidates"][0]["candidate_id"]
            (fixture.bank_root / missing_id / review.PAIR_RECEIPT_FILENAME).unlink()
            with mock.patch.object(
                review.subprocess,
                "run",
                side_effect=lambda command, **_kwargs: _ffprobe_result(command),
            ):
                status = review.main(
                    [
                        "--bank-root", str(fixture.bank_root),
                        "--bank-receipt", str(fixture.bank_receipt_path),
                        "--root-spec", str(fixture.spec_path),
                        "--output", str(fixture.output),
                        "--ffprobe", "fixture-ffprobe",
                        "--allow-incomplete",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertFalse(
                json.loads(fixture.output.with_name("index.audit.json").read_bytes())[
                    "review_complete"
                ]
            )

    def test_mp4_sha_tamper_invalidates_candidate_and_whole_cell_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            candidate_id = fixture.spec["groups"][0]["candidates"][0]["candidate_id"]
            (fixture.bank_root / candidate_id / "t2v.mp4").write_bytes(b"tampered")
            audit = fixture.build()
            self.assertFalse(audit["review_complete"])
            cell = audit["cells"][0]
            self.assertEqual(cell["branches"]["action"]["media"]["status"], "invalid")
            self.assertIn("SHA-256 differs", cell["branches"]["action"]["media"]["message"])
            self.assertEqual(cell["same_cell_gaussian"]["status"], "invalid")

    def test_ffprobe_80_frames_is_fail_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            candidate_id = fixture.spec["groups"][0]["candidates"][0]["candidate_id"]
            audit = fixture.build(bad_name=candidate_id)
            media = audit["cells"][0]["branches"]["action"]["media"]
            self.assertEqual(media["status"], "invalid")
            self.assertIn("not exact81/25fps", media["message"])

    def test_resealed_false_same_gaussian_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            fixture.bank["same_cell_gaussian_proofs"][0][
                "official_gaussian_raw_value_sha256"
            ] = "f" * 64
            fixture.reseal_bank()
            audit = fixture.build()
            self.assertFalse(audit["review_complete"])
            self.assertEqual(audit["cells"][0]["same_cell_gaussian"]["status"], "invalid")
            self.assertIn(
                "observed Gaussian identity differs",
                audit["cells"][0]["same_cell_gaussian"]["message"],
            )

    def test_bank_receipt_seal_and_exact40_closure_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            fixture.bank["candidate_count"] = 39
            fixture.reseal_bank()
            with self.assertRaisesRegex(review.CaperT2VReviewError, "closure"):
                fixture.build()
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            fixture.bank["candidate_count"] = 39
            _write_json(fixture.bank_receipt_path, fixture.bank)  # stale embedded seal
            with self.assertRaisesRegex(review.CaperT2VReviewError, "seal"):
                fixture.build()

    def test_candidate_receipt_cannot_authorize_target_or_donor_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RewardBankFixture(Path(directory))
            candidate = fixture.spec["groups"][0]["candidates"][0]
            candidate_dir = fixture.bank_root / candidate["candidate_id"]
            pair_path = candidate_dir / review.PAIR_RECEIPT_FILENAME
            pair = json.loads(pair_path.read_bytes())
            pair["interpretation"]["donor_or_pseudo_target_use_forbidden"] = False
            unsigned = dict(pair)
            unsigned.pop("receipt_digest")
            pair = _seal(unsigned)
            _write_json(pair_path, pair)
            bank_row = fixture.bank["candidate_receipts"][0]
            bank_row["receipt_sha256"] = review.file_sha256(pair_path)
            bank_row["receipt_digest"] = pair["receipt_digest"]
            fixture.reseal_bank()
            audit = fixture.build()
            media = audit["cells"][0]["branches"]["action"]["media"]
            self.assertEqual(media["status"], "invalid")
            self.assertIn("calibration-only binding differs", media["message"])

    def test_real_ffprobe_accepts_exact81_and_rejects_80(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("ffmpeg/ffprobe are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for count in (81, 80):
                path = root / f"frames-{count}.mp4"
                completed = subprocess.run(
                    [
                        ffmpeg,
                        "-v", "error",
                        "-f", "lavfi",
                        "-i", "color=c=black:s=16x16:r=25",
                        "-frames:v", str(count),
                        "-an",
                        "-c:v", "mpeg4",
                        "-q:v", "2",
                        str(path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                videos.append(path)
            self.assertEqual(
                review.probe_exact81_video(videos[0], ffprobe=ffprobe)["frame_count"],
                81,
            )
            with self.assertRaisesRegex(review.CaperT2VReviewError, "not exact81"):
                review.probe_exact81_video(videos[1], ffprobe=ffprobe)


if __name__ == "__main__":
    unittest.main()
