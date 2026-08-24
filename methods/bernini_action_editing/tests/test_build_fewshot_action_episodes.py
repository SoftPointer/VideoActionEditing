from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_fewshot_action_episodes as builder  # noqa: E402


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EpisodeFixture:
    SIGNATURES = {
        "clap": "clap_both_hands_overhead",
        "wave": "open_palm_wave_right",
    }

    def __init__(
        self,
        root: Path,
        *,
        primitive_counts: dict[str, int] | None = None,
    ) -> None:
        self.root = root
        self.preview = root / "preview.jsonl"
        self.index = root / "vae-index.jsonl"
        self.output = root / "out" / "episodes.jsonl"
        self.preview_rows: list[dict[str, object]] = []
        self.index_rows: list[dict[str, object]] = []
        counts = primitive_counts or {"clap": 4, "wave": 4}
        for primitive, count in sorted(counts.items()):
            for ordinal in range(count):
                self.add_row(
                    iid=f"{primitive}-{ordinal:02d}",
                    signature=self.SIGNATURES[primitive],
                )
        self.write()

    def add_row(
        self,
        *,
        iid: str,
        signature: str,
        group_id: str | None = None,
        source_sha256: str | None = None,
    ) -> dict[str, object]:
        group = group_id or f"group-{iid}"
        source_sha = source_sha256 or _sha_text(f"source:{iid}")
        target_sha = _sha_text(f"target:{iid}")
        instruction = f"Make the only actor perform {signature}."
        generation = f"Starting at frame zero, perform {signature}."
        subject_id = "subject_01"
        census = {
            "iid": iid,
            "dynamic_subjects": [
                {
                    "subject_id": subject_id,
                    "dynamic": True,
                    "stable_reference": f"the actor in {iid}",
                    "source_action_signature": "walk_forward",
                    "source_motion": "walks forward",
                }
            ],
            "camera": {"motion_class": "locked_off"},
            "all_dynamic_subjects_enumerated": True,
            "crowd_or_unresolved_motion": False,
            "confidence": "high",
        }
        plan = {
            "iid": iid,
            "dynamic_subject_targets": [
                {
                    "subject_id": subject_id,
                    "target_action_signature": signature,
                    "target_motion": f"performs {signature.replace('_', ' ')}",
                    "substantive_change": True,
                }
            ],
            "camera_target": {
                "motion_class": "locked_off",
                "relation": "preserve_static",
            },
            "confidence": "high",
        }
        row: dict[str, object] = {
            "schema_version": builder.PREVIEW_ROW_SCHEMA,
            "iid": iid,
            "group_id": group,
            # This intentionally disagrees with the target action.  The builder
            # must never use it for ontology assignment.
            "family": "source-running-family",
            "source_video_path": f"/fixture/source/{iid}.mp4",
            "source_video_sha256": source_sha,
            "target_video_path": f"/fixture/target/{iid}.mp4",
            "target_video_sha256": target_sha,
            "edit_instruction": instruction,
            "edit_instruction_sha256": _sha_text(instruction),
            "instruction_source": "natural",
            "generation_instruction": generation,
            "generation_instruction_sha256": _sha_text(generation),
            "source_census": census,
            "target_plan": plan,
            "selection_gates": {
                "single_dynamic_actor": True,
                "source_camera_locked_off": True,
                "target_camera_locked_off": True,
                "target_camera_preserve_static": True,
                "source_census_high_confidence": True,
                "target_plan_high_confidence": True,
            },
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_eligible": False,
            "post_video_acceptance": "pending",
            "provenance": {"fixture": True},
        }
        row["row_digest"] = builder.object_sha256(row)
        self.preview_rows.append(row)
        self.index_rows.append(
            {
                "schema_version": builder.VAE_INDEX_ROW_SCHEMA,
                "iid": iid,
                "parquet_path": f"/fixture/shards/{iid}.parquet",
                "parquet_sha256": _sha_text(f"parquet:{iid}"),
                "materialized_row_digest": _sha_text(f"materialized:{iid}"),
                "bucket_hw": [480, 496],
                "posterior_parameters_shape": [2, 32, 21, 60, 62],
                "sample_receipt_path": f"/fixture/receipts/{iid}.json",
                "sample_receipt_sha256": _sha_text(f"receipt:{iid}"),
                "preview_only": True,
                "production_claim_forbidden": True,
            }
        )
        return row

    @staticmethod
    def rebind_preview_row(row: dict[str, object]) -> None:
        row.pop("row_digest", None)
        row["row_digest"] = builder.object_sha256(row)

    def row(self, iid: str) -> dict[str, object]:
        return next(value for value in self.preview_rows if value["iid"] == iid)

    def index_row(self, iid: str) -> dict[str, object]:
        return next(value for value in self.index_rows if value["iid"] == iid)

    def write(self) -> None:
        self.preview.write_bytes(
            b"".join(
                builder.canonical_json_bytes(row) + b"\n"
                for row in sorted(self.preview_rows, key=lambda value: str(value["iid"]))
            )
        )
        self.index.write_bytes(
            b"".join(
                builder.canonical_json_bytes(row) + b"\n"
                for row in sorted(self.index_rows, key=lambda value: str(value["iid"]))
            )
        )

    @property
    def preview_sha256(self) -> str:
        return hashlib.sha256(self.preview.read_bytes()).hexdigest()

    @property
    def index_sha256(self) -> str:
        return hashlib.sha256(self.index.read_bytes()).hexdigest()

    def build(self) -> builder.EpisodeBuildPayload:
        return builder.build_fewshot_action_episodes(
            preview_manifest=self.preview,
            expected_preview_manifest_sha256=self.preview_sha256,
            vae_index=self.index,
            expected_vae_index_sha256=self.index_sha256,
            output_jsonl=self.output,
        )


class AtomicOntologyTests(unittest.TestCase):
    def test_accepts_one_explicit_primitive_and_rejects_composites(self) -> None:
        match, reason = builder.classify_atomic_action_signature(
            "clap_both_hands_overhead"
        )
        self.assertEqual(reason, "eligible")
        self.assertIsNotNone(match)
        self.assertEqual(match.primitive_id, "clap")

        match, reason = builder.classify_atomic_action_signature(
            "crouch_and_touch_floor"
        )
        self.assertIsNone(match)
        self.assertEqual(reason, "composite_target_action_signature")

        match, reason = builder.classify_atomic_action_signature(
            "lower_grip_raise"
        )
        self.assertIsNone(match)
        self.assertEqual(reason, "ambiguous_target_action_signature")

        match, reason = builder.classify_atomic_action_signature(
            "event_specific_subject_motion"
        )
        self.assertIsNone(match)
        self.assertEqual(reason, "unsupported_target_action_signature")


class FewShotEpisodeBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dry_run_is_canonical_deterministic_disjoint_and_preview_only(self) -> None:
        fixture = EpisodeFixture(self.root)
        first = fixture.build()
        second = fixture.build()
        self.assertEqual(first.jsonl_bytes, second.jsonl_bytes)
        self.assertEqual(first.receipt_bytes, second.receipt_bytes)
        self.assertFalse(fixture.output.exists())
        self.assertFalse(Path(f"{fixture.output}.receipt.json").exists())

        lines = first.jsonl_bytes.splitlines()
        episodes = [json.loads(line) for line in lines]
        self.assertEqual(len(episodes), 2)
        self.assertEqual([row["primitive_id"] for row in episodes], ["clap", "wave"])
        for line, episode in zip(lines, episodes):
            self.assertEqual(line, builder.canonical_json_bytes(episode))
            unsigned = dict(episode)
            digest = unsigned.pop("episode_digest")
            self.assertEqual(builder.object_sha256(unsigned), digest)
            self.assertEqual(len(episode["positive_supports"]), 2)
            self.assertEqual(len(episode["negative_action"]["supports"]), 2)
            self.assertNotEqual(
                episode["primitive_id"], episode["negative_action"]["primitive_id"]
            )
            self.assertTrue(episode["experimental_only"])
            self.assertFalse(episode["training_authorized"])
            self.assertFalse(episode["production_claim"])
            self.assertFalse(episode["manual_review_performed"])
            self.assertEqual(episode["target_post_video_acceptance"], "pending")
            self.assertFalse(episode["ontology"]["source_family_used"])
            labels = [control["label"] for control in episode["phase_controls"]]
            self.assertEqual(
                labels,
                [
                    "correct",
                    "reverse_nonboundary",
                    "shuffle_nonboundary",
                    "negate",
                    "wrong_action_support",
                ],
            )
            for control in episode["phase_controls"]:
                self.assertEqual(control["phase_permutation"][0], 0)
                self.assertEqual(sorted(control["phase_permutation"]), list(range(21)))
            members = [
                *episode["positive_supports"],
                episode["train_query"],
                episode["heldout_query"],
                *episode["negative_action"]["supports"],
            ]
            self.assertNotIn("family", episode["train_query"])
            self.assertTrue(
                all(member["primitive_id"] != "source-running-family" for member in members)
            )

        support = {
            member[field]
            for episode in episodes
            for member in episode["positive_supports"]
            for field in ["iid"]
        }
        train = {episode["train_query"]["iid"] for episode in episodes}
        heldout = {episode["heldout_query"]["iid"] for episode in episodes}
        self.assertFalse(support & train)
        self.assertFalse(support & heldout)
        self.assertFalse(train & heldout)

        receipt = json.loads(first.receipt_bytes)
        self.assertEqual(
            first.receipt_bytes, builder.canonical_json_bytes(receipt) + b"\n"
        )
        unsigned_receipt = dict(receipt)
        receipt_digest = unsigned_receipt.pop("receipt_digest")
        self.assertEqual(builder.object_sha256(unsigned_receipt), receipt_digest)
        self.assertTrue(receipt["experimental_only"])
        self.assertFalse(receipt["training_authorized"])
        self.assertTrue(receipt["training_use_forbidden"])
        self.assertFalse(receipt["production_claim"])
        self.assertFalse(receipt["manual_review_performed"])
        self.assertEqual(receipt["human_review_status"], "not_performed")
        self.assertEqual(receipt["target_post_video_acceptance"], "pending")
        self.assertFalse(receipt["target_video_quality_verified"])
        self.assertEqual(receipt["ontology"]["sha256"], builder.ONTOLOGY_SHA256)
        self.assertFalse(receipt["source_family_field_used"])

    def test_requires_caller_pinned_hashes_and_exact_iid_join(self) -> None:
        fixture = EpisodeFixture(self.root)
        with self.assertRaisesRegex(builder.FewShotEpisodeError, "caller-pinned"):
            builder.build_fewshot_episode_payloads(
                preview_manifest=fixture.preview,
                expected_preview_manifest_sha256="0" * 64,
                vae_index=fixture.index,
                expected_vae_index_sha256=fixture.index_sha256,
                output_jsonl=fixture.output,
            )

        fixture.index_rows.pop()
        fixture.write()
        with self.assertRaisesRegex(builder.FewShotEpisodeError, "IID membership"):
            fixture.build()
        self.assertFalse(fixture.output.exists())

    def test_filters_failed_gate_and_composite_without_claiming_review(self) -> None:
        fixture = EpisodeFixture(
            self.root, primitive_counts={"clap": 5, "wave": 5}
        )
        moving = fixture.row("wave-04")
        moving["source_census"]["camera"]["motion_class"] = "pan_left"
        moving["selection_gates"]["source_camera_locked_off"] = False
        fixture.rebind_preview_row(moving)
        fixture.add_row(iid="unclear-00", signature="turn_and_point")
        fixture.write()

        payload = fixture.build()
        receipt = payload.receipt
        self.assertEqual(receipt["episode_count"], 2)
        self.assertEqual(
            receipt["exclusion_counts"]["source_camera_not_locked_off"], 1
        )
        self.assertEqual(
            receipt["exclusion_counts"]["composite_target_action_signature"], 1
        )
        self.assertFalse(receipt["manual_review_performed"])
        iids = {
            member["iid"]
            for episode in map(json.loads, payload.jsonl_bytes.splitlines())
            for member in [
                *episode["positive_supports"],
                episode["train_query"],
                episode["heldout_query"],
            ]
        }
        self.assertNotIn("wave-04", iids)
        self.assertNotIn("unclear-00", iids)

    def test_requires_at_least_four_independent_groups_per_primitive(self) -> None:
        fixture = EpisodeFixture(
            self.root, primitive_counts={"clap": 3, "wave": 3}
        )
        with self.assertRaisesRegex(
            builder.FewShotEpisodeError, "need 4 independent"
        ):
            fixture.build()

    def test_group_and_source_components_are_deduplicated_before_split(self) -> None:
        fixture = EpisodeFixture(
            self.root, primitive_counts={"clap": 6, "wave": 6}
        )
        duplicate_group = fixture.row("wave-05")
        duplicate_group["group_id"] = fixture.row("wave-04")["group_id"]
        fixture.rebind_preview_row(duplicate_group)
        duplicate_source = fixture.row("clap-05")
        duplicate_source["source_video_sha256"] = fixture.row("clap-04")[
            "source_video_sha256"
        ]
        fixture.rebind_preview_row(duplicate_source)
        fixture.write()

        payload = fixture.build()
        self.assertEqual(
            payload.receipt["exclusion_counts"][
                "group_or_source_leak_component_duplicate"
            ],
            2,
        )
        episodes = [json.loads(line) for line in payload.jsonl_bytes.splitlines()]
        support_members = [
            member for episode in episodes for member in episode["positive_supports"]
        ]
        train_members = [episode["train_query"] for episode in episodes]
        heldout_members = [episode["heldout_query"] for episode in episodes]
        for field in ("iid", "group_id", "source_video_sha256"):
            support = {member[field] for member in support_members}
            train = {member[field] for member in train_members}
            heldout = {member[field] for member in heldout_members}
            self.assertFalse(support & train)
            self.assertFalse(support & heldout)
            self.assertFalse(train & heldout)

    def test_forged_true_gate_fails_closed(self) -> None:
        fixture = EpisodeFixture(self.root)
        row = fixture.row("wave-00")
        row["source_census"]["camera"]["motion_class"] = "pan_left"
        # Deliberately retain source_camera_locked_off=true.
        fixture.rebind_preview_row(row)
        fixture.write()
        with self.assertRaisesRegex(
            builder.FewShotEpisodeError, "selection gates disagree"
        ):
            fixture.build()

    def test_cli_is_dry_run_by_default_and_publish_is_create_only(self) -> None:
        fixture = EpisodeFixture(self.root)
        args = [
            "--preview-manifest",
            str(fixture.preview),
            "--expected-preview-manifest-sha256",
            fixture.preview_sha256,
            "--vae-index",
            str(fixture.index),
            "--expected-vae-index-sha256",
            fixture.index_sha256,
            "--output-jsonl",
            str(fixture.output),
        ]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(builder.main(args), 0)
        status = json.loads(stdout.getvalue())
        self.assertTrue(status["dry_run"])
        self.assertFalse(status["published"])
        self.assertFalse(fixture.output.exists())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(builder.main([*args, "--publish"]), 0)
        status = json.loads(stdout.getvalue())
        self.assertFalse(status["dry_run"])
        self.assertTrue(status["published"])
        receipt_path = Path(f"{fixture.output}.receipt.json")
        self.assertTrue(fixture.output.is_file())
        self.assertTrue(receipt_path.is_file())
        self.assertEqual(
            hashlib.sha256(fixture.output.read_bytes()).hexdigest(),
            json.loads(receipt_path.read_bytes())["output_jsonl_sha256"],
        )

        old_jsonl = fixture.output.read_bytes()
        old_receipt = receipt_path.read_bytes()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(builder.main([*args, "--publish"]), 2)
        self.assertIn("create-only output exists", stderr.getvalue())
        self.assertEqual(fixture.output.read_bytes(), old_jsonl)
        self.assertEqual(receipt_path.read_bytes(), old_receipt)


if __name__ == "__main__":
    unittest.main()
