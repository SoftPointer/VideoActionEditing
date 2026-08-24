from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for path in (METHOD_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_pair_v6_scaid_manifest as builder
import pair_v6_scaid_source_coordinate as scaid
import train_pair_v6_scaid as trainer


def _prompts(prefix: str) -> dict[str, str]:
    return {
        branch: f"{prefix} complete authoritative caption for {branch}."
        for branch in scaid.BRANCH_ORDER
    }


class PairV6SCAIDManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # macOS exposes /var through the /private/var canonical path.  Keep the
        # fixture aligned with the production provenance gate, which rejects
        # non-canonical absolute evidence/media paths.
        self.root = Path(self.temp.name).resolve(strict=True)
        self.assertEqual(self.root, self.root.resolve(strict=True))
        self.evidence = self.root / "evidence.json"
        self.manifest_sha = hashlib.sha256(b"authoritative guidance manifest").hexdigest()
        self.spec_path = self.root / "source-spec.json"
        self.spec_path.write_text("{}\n", encoding="ascii")
        self.spec_sha = hashlib.sha256(self.spec_path.read_bytes()).hexdigest()
        self.evidence.write_text(
            json.dumps(
                {
                    "guidance_manifest": {
                        "path": str(self.root / "guidance.json"),
                        "file_sha256": self.manifest_sha,
                    },
                    "source_bank_spec": {
                        "path": str(self.spec_path),
                        "file_sha256": self.spec_sha,
                    },
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )
        self.evidence_sha = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        self.checkpoint_sha = trainer.legacy.CHECKPOINT_TREE_SHA256
        self.raw_prompts = (_prompts("dog"), _prompts("human"))
        self.prompts = tuple(
            {
                branch: builder.native_infer.build_task_prompt(
                    "t2v", raw[branch], prompt_cleaner=lambda value: value
                )
                for branch in scaid.BRANCH_ORDER
            }
            for raw in self.raw_prompts
        )
        self.authoritative_events = tuple(
            SimpleNamespace(
                event_id=candidate,
                action_family=family,
                analysis_split="fit",
                prompt_by_branch=prompts,
                prompt_bank_sha256=scaid.object_sha256(prompts),
            )
            for candidate, family, prompts in (
                ("validated-dog-fit", "dog-sit", self.prompts[0]),
                ("validated-human-fit", "human-stand", self.prompts[1]),
            )
        )
        self.guidance_manifest = SimpleNamespace(
            raw_sha256=self.manifest_sha, events=self.authoritative_events
        )
        self.source_spec = {
            "groups": [
                {
                    "candidates": [
                        {
                            "candidate_id": (
                                candidate if branch == "action" else f"{candidate}-{branch}"
                            ),
                            "analysis_split": "fit",
                            "semantic_branch": branch,
                            "action_family_id": family,
                            "calibration_group_id": f"{candidate}-cell",
                            "full_t2v_caption": raw[branch],
                        }
                        for branch in scaid.BRANCH_ORDER
                    ]
                }
                for candidate, family, raw in (
                    ("validated-dog-fit", "dog-sit", self.raw_prompts[0]),
                    ("validated-human-fit", "human-stand", self.raw_prompts[1]),
                )
            ]
        }
        media: list[Path] = []
        for index, content in enumerate(
            (b"dog correct exact81", b"dog wrong exact81", b"human correct exact81", b"human wrong exact81")
        ):
            path = self.root / f"media-{index}.mp4"
            path.write_bytes(content)
            media.append(path)
        for group, source_path in zip(
            self.source_spec["groups"], (media[0], media[2])
        ):
            for candidate in group["candidates"]:
                candidate["geometry_source_video"] = str(source_path)
                candidate["geometry_source_video_sha256"] = self._digest(
                    source_path
                )
        self.events = (
            self._event(
                "validated-dog-fit", "validated-dog-fit", media[0], media[1], "dog-donor"
            ),
            self._event(
                "validated-human-fit", "validated-human-fit", media[2], media[3], "human-donor"
            ),
        )

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _event(
        self, sample: str, candidate: str, source: Path, wrong: Path, wrong_iid: str
    ) -> builder.EventInput:
        review_artifact = self.root / f"{candidate}-wrong-source-review.json"
        review_artifact.write_text(
            json.dumps({"candidate": candidate, "wrong_source_iid": wrong_iid}),
            encoding="ascii",
        )
        audit_source_archive = self.root / f"{candidate}-audit-method-source.tar"
        audit_source_archive.write_bytes(b"external audit method source archive")
        audit_unsigned = {
            "schema_version": trainer.WRONG_SOURCE_AUDIT_SCHEMA,
            "candidate_sample_id": sample,
            "candidate_source_video_sha256": self._digest(source),
            "wrong_source_iid": wrong_iid,
            "wrong_source_video_sha256": self._digest(wrong),
            "criteria": {
                name: True for name in trainer.WRONG_SOURCE_AUDIT_CRITERIA
            },
            "reviewer": "external-test-reviewer",
            "review_artifact_path": str(review_artifact),
            "review_artifact_sha256": self._digest(review_artifact),
            "preprocessing_contract": dict(
                trainer.WRONG_SOURCE_PREPROCESSING_CONTRACT
            ),
            "audit_method_source_revision": "a" * 40,
            "audit_method_source_archive_path": str(audit_source_archive),
            "audit_method_source_archive_sha256": self._digest(
                audit_source_archive
            ),
        }
        audit = {
            **audit_unsigned,
            "audit_digest": trainer.object_sha256(audit_unsigned),
        }
        audit_path = self.root / f"{candidate}-wrong-source-audit.json"
        audit_path.write_bytes(trainer.canonical_json_bytes(audit) + b"\n")
        return builder.EventInput(
            sample,
            candidate,
            source,
            self._digest(source),
            wrong,
            self._digest(wrong),
            wrong_iid,
            audit_path,
            self._digest(audit_path),
            audit["audit_digest"],
        )

    def _gate(self, evidence: Path, **kwargs: object) -> SimpleNamespace:
        del evidence
        candidate = str(kwargs["fit_candidate_id"])
        event = next(row for row in self.authoritative_events if row.event_id == candidate)
        return SimpleNamespace(
            fit_candidate_id=candidate,
            action_family=event.action_family,
            prompt_bank_sha256=event.prompt_bank_sha256,
        )

    def _build(self, output: Path) -> dict[str, object]:
        geometry = {
            "width": 736,
            "height": 704,
            "avg_frame_rate": "25/1",
            "nb_read_frames": "81",
        }
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(self.source_spec, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ), mock.patch.object(
            builder.scaid,
            "load_authoritative_v3_authorization",
            side_effect=self._gate,
        ) as validator, mock.patch.object(
            builder.native_dpo, "_ffprobe_exact81", return_value=geometry
        ) as ffprobe:
            receipt = builder.build_manifest(
                evidence_path=self.evidence,
                expected_evidence_sha256=self.evidence_sha,
                checkpoint_tree_sha256=self.checkpoint_sha,
                events=self.events,
                output_path=output,
            )
        self.assertEqual(validator.call_count, 2)
        self.assertEqual(ffprobe.call_count, 8)
        return dict(receipt)

    def test_deterministic_manifest_binds_authoritative_prompts_and_families(self) -> None:
        output_a = self.root / "manifest-a.json"
        output_b = self.root / "manifest-b.json"
        receipt_a = self._build(output_a)
        receipt_b = self._build(output_b)
        self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
        self.assertEqual(receipt_a["file_sha256"], receipt_b["file_sha256"])
        decoded = json.loads(output_a.read_text(encoding="ascii"))
        self.assertEqual(
            [row["action_family"] for row in decoded["events"]],
            ["dog-sit", "human-stand"],
        )
        self.assertEqual(
            decoded["events"][0]["raw_caption_by_branch"], self.raw_prompts[0]
        )
        self.assertEqual(
            decoded["events"][1]["raw_caption_by_branch"], self.raw_prompts[1]
        )
        self.assertNotIn("prompt_by_branch", decoded["events"][0])
        self.assertEqual(
            decoded["events"][0]["raw_caption_bank_sha256"],
            scaid.object_sha256(self.raw_prompts[0]),
        )
        self.assertEqual(decoded["checkpoint_tree_sha256"], self.checkpoint_sha)
        self.assertTrue(receipt_a["authoritative_v3_recomputed"])

    def test_wrong_source_requires_external_identity_class_and_pose_audit(self) -> None:
        audit = json.loads(
            self.events[0].wrong_source_audit_path.read_text(encoding="ascii")
        )
        audit["criteria"]["same_actor_class"] = False
        unsigned = dict(audit)
        unsigned.pop("audit_digest")
        audit["audit_digest"] = trainer.object_sha256(unsigned)
        bad_audit = self.root / "rejected-wrong-source-audit.json"
        bad_audit.write_bytes(trainer.canonical_json_bytes(audit) + b"\n")
        bad = builder.EventInput(
            **{
                **self.events[0].__dict__,
                "wrong_source_audit_path": bad_audit,
                "wrong_source_audit_file_sha256": self._digest(bad_audit),
                "wrong_source_audit_digest": audit["audit_digest"],
            }
        )
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(self.source_spec, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ):
            with self.assertRaisesRegex(
                builder.PairV6SCAIDManifestBuildError,
                "identity/class/initial-pose audit",
            ):
                builder.build_manifest(
                    evidence_path=self.evidence,
                    expected_evidence_sha256=self.evidence_sha,
                    checkpoint_tree_sha256=self.checkpoint_sha,
                    events=(bad, self.events[1]),
                    output_path=self.root / "bad.json",
                )

    def test_correct_wrong_and_cross_dp_media_must_be_hash_distinct(self) -> None:
        reused_audit = json.loads(
            self.events[1].wrong_source_audit_path.read_text(encoding="ascii")
        )
        reused_audit["wrong_source_video_sha256"] = self.events[0].source_video_sha256
        unsigned = dict(reused_audit)
        unsigned.pop("audit_digest")
        reused_audit["audit_digest"] = trainer.object_sha256(unsigned)
        reused_audit_path = self.root / "reused-wrong-source-audit.json"
        reused_audit_path.write_bytes(
            trainer.canonical_json_bytes(reused_audit) + b"\n"
        )
        reused = builder.EventInput(
            **{
                **self.events[1].__dict__,
                "wrong_source_video_path": self.events[0].source_video_path,
                "wrong_source_video_sha256": self.events[0].source_video_sha256,
                "wrong_source_audit_path": reused_audit_path,
                "wrong_source_audit_file_sha256": self._digest(reused_audit_path),
                "wrong_source_audit_digest": reused_audit["audit_digest"],
            }
        )
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(self.source_spec, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ), mock.patch.object(
            builder.scaid,
            "load_authoritative_v3_authorization",
            side_effect=self._gate,
        ), mock.patch.object(
            builder.native_dpo,
            "_ffprobe_exact81",
            return_value={"width": 736, "height": 704, "avg_frame_rate": "25/1"},
        ):
            with self.assertRaisesRegex(builder.PairV6SCAIDManifestBuildError, "reused"):
                builder.build_manifest(
                    evidence_path=self.evidence,
                    expected_evidence_sha256=self.evidence_sha,
                    checkpoint_tree_sha256=self.checkpoint_sha,
                    events=(self.events[0], reused),
                    output_path=self.root / "reused.json",
                )

    def test_output_is_fresh_create_only(self) -> None:
        output = self.root / "manifest.json"
        self._build(output)
        original = output.read_bytes()
        with self.assertRaisesRegex(builder.PairV6SCAIDManifestBuildError, "fresh"):
            self._build(output)
        self.assertEqual(output.read_bytes(), original)

    def test_source_bank_candidate_flatten_is_one_to_one(self) -> None:
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(self.source_spec, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ):
            events, raw_by_event, anchor_by_event = (
                builder._load_authoritative_events_and_raw_captions(self.evidence)
            )
        expected = {event.event_id for event in self.authoritative_events}
        self.assertEqual(set(events), expected)
        self.assertEqual(set(raw_by_event), expected)
        self.assertEqual(set(anchor_by_event), expected)
        self.assertEqual(
            sum(len(group["candidates"]) for group in self.source_spec["groups"]),
            len(scaid.BRANCH_ORDER) * len(expected),
        )

    def test_tampered_hash_bound_source_spec_is_rejected(self) -> None:
        self.spec_path.write_text('{"tampered":true}\n', encoding="ascii")
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ):
            with self.assertRaisesRegex(
                builder.PairV6SCAIDManifestBuildError, "source bank spec"
            ):
                builder._load_authoritative_events_and_raw_captions(self.evidence)

    def test_arbitrary_native_correct_source_forgery_is_rejected(self) -> None:
        forged_path = self.root / "forged-correct.mp4"
        forged_path.write_bytes(b"arbitrary caller-selected exact81 source")
        forged = builder.EventInput(
            **{
                **self.events[0].__dict__,
                "source_video_path": forged_path,
                "source_video_sha256": self._digest(forged_path),
            }
        )
        geometry = {
            "width": 736,
            "height": 704,
            "avg_frame_rate": "25/1",
            "nb_read_frames": "81",
        }
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(self.source_spec, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ), mock.patch.object(
            builder.native_dpo, "_ffprobe_exact81", return_value=geometry
        ):
            with self.assertRaisesRegex(
                builder.PairV6SCAIDManifestBuildError,
                "evidence-bound geometry anchor",
            ):
                builder.build_manifest(
                    evidence_path=self.evidence,
                    expected_evidence_sha256=self.evidence_sha,
                    checkpoint_tree_sha256=self.checkpoint_sha,
                    events=(forged, self.events[1]),
                    output_path=self.root / "forged-source.json",
                )

    def test_caller_cannot_relabel_evidence_bound_fit_sample_id(self) -> None:
        relabeled = builder.EventInput(
            **{**self.events[0].__dict__, "sample_id": "caller-selected-sample"}
        )
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(self.source_spec, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ):
            with self.assertRaisesRegex(
                builder.PairV6SCAIDManifestBuildError,
                "sample ID is not the evidence-bound",
            ):
                builder.build_manifest(
                    evidence_path=self.evidence,
                    expected_evidence_sha256=self.evidence_sha,
                    checkpoint_tree_sha256=self.checkpoint_sha,
                    events=(relabeled, self.events[1]),
                    output_path=self.root / "relabeled-source.json",
                )

    def test_source_cell_to_guidance_prompt_mapping_mismatch_is_rejected(self) -> None:
        mismatched = copy.deepcopy(self.source_spec)
        mismatched["groups"][0]["candidates"][0][
            "full_t2v_caption"
        ] = "Tampered raw action caption."
        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(mismatched, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ):
            with self.assertRaisesRegex(
                builder.PairV6SCAIDManifestBuildError,
                "raw-to-T2V prompt mapping differs",
            ):
                builder._load_authoritative_events_and_raw_captions(self.evidence)

    def test_correct_wrong_width_height_fps_must_match(self) -> None:
        def geometry(path: Path) -> dict[str, object]:
            return {
                "width": 640 if path == self.events[0].wrong_source_video_path else 736,
                "height": 704,
                "avg_frame_rate": "25/1",
                "nb_read_frames": "81",
            }

        with mock.patch.object(
            builder.cagd_trainer, "load_manifest", return_value=self.guidance_manifest
        ), mock.patch.object(
            builder.bank_spec,
            "load_sealed_spec",
            return_value=(self.source_spec, self.spec_sha),
        ), mock.patch.object(
            builder,
            "_build_t2v_task_prompt",
            side_effect=lambda raw: builder.native_infer.build_task_prompt(
                "t2v", raw, prompt_cleaner=lambda value: value
            ),
        ), mock.patch.object(
            builder.native_dpo, "_ffprobe_exact81", side_effect=geometry
        ):
            with self.assertRaisesRegex(
                builder.PairV6SCAIDManifestBuildError, "width-height-fps"
            ):
                builder.build_manifest(
                    evidence_path=self.evidence,
                    expected_evidence_sha256=self.evidence_sha,
                    checkpoint_tree_sha256=self.checkpoint_sha,
                    events=self.events,
                    output_path=self.root / "geometry-mismatch.json",
                )


if __name__ == "__main__":
    unittest.main()
