from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "goku_action_review_packet.py"
SPEC = importlib.util.spec_from_file_location(
    "goku_action_review_packet",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
REVIEW_PACKET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW_PACKET)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _row(
    index: int,
    *,
    remote_root: str,
    source_payload: bytes,
    anchor_payload: bytes,
    prompt: str | None = None,
) -> dict[str, object]:
    iid = f"proposal-{index:03d}"
    return {
        "iid": iid,
        "group_id": f"group-{index:03d}",
        "source_caption": f"A subject walks in source clip {index}.",
        "edited_caption": f"The subject jumps in target clip {index}.",
        "prompt": prompt or f"Make subject {index} jump.",
        "resolved_src_video": (
            f"{remote_root}/videos/{iid}/source.mp4"
        ),
        "resolved_anchor_image": f"{remote_root}/anchors/{iid}.png",
        "source_video_sha256": _sha(source_payload),
        "anchor_sha256": _sha(anchor_payload),
        "qwen_action_anchor": {
            "anchor_observation": {
                "initial_state": "The subject stands with both feet visible.",
                "source_action": "The subject walks steadily to the left.",
            },
            "compatibility": {
                "source_action_normalized": "walk left",
                "target_action_normalized": "jump vertically",
                "target_action_verb": "jump",
                "rewritten_edit_instruction": (
                    "Have the subject jump vertically from the visible pose."
                ),
                "causal_bridge": "direct",
                "causal_bridge_description": (
                    "The visible standing pose directly supports takeoff."
                ),
                "causal_stages": [
                    "Bend the knees from the visible standing pose.",
                    "Push off and jump vertically.",
                ],
                "absolute_target_prompt": (
                    "The same subject starts in the shown pose, bends both "
                    "knees, and jumps vertically while appearance, background, "
                    "and camera remain unchanged."
                ),
            },
        },
        "action_anchor_finalization": {
            "schema_version": REVIEW_PACKET.PROPOSAL_ROW_SCHEMA,
            "policy_version": "test-strict-policy",
            "hard_gate_passed": True,
            "hard_gate_failures": [],
            "selection_bucket": "proposed",
            "human_review_status": "pending",
            "human_label": False,
            "generation_authorized": False,
            "target_support_evidence": {
                "requires_proposal_bound_human_review": True,
                "lexically_verified_fields": [
                    "target_action_normalized",
                    "rewritten_edit_instruction",
                ],
                "lexically_unverified_fields": ["causal_stages"],
            },
        },
    }


def _write_media(
    local_root: Path,
    *,
    row: dict[str, object],
    source_payload: bytes,
    anchor_payload: bytes,
) -> None:
    iid = str(row["iid"])
    source = local_root / "videos" / iid / "source.mp4"
    anchor = local_root / "anchors" / f"{iid}.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    anchor.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(source_payload)
    anchor.write_bytes(anchor_payload)


def _fixture(
    root: Path,
    *,
    count: int = 2,
    prompt: str | None = None,
) -> tuple[Path, Path, list[dict[str, object]], list[tuple[bytes, bytes]]]:
    remote_root = "/vast/auh/goku"
    local_root = root / "local-mirror"
    local_root.mkdir()
    rows: list[dict[str, object]] = []
    payloads: list[tuple[bytes, bytes]] = []
    for index in range(count):
        source = f"source-video-{index}\x00full".encode("utf-8")
        anchor = b"\x89PNG\r\n\x1a\n" + f"anchor-{index}".encode("utf-8")
        row = _row(
            index,
            remote_root=remote_root,
            source_payload=source,
            anchor_payload=anchor,
            prompt=prompt,
        )
        _write_media(
            local_root,
            row=row,
            source_payload=source,
            anchor_payload=anchor,
        )
        rows.append(row)
        payloads.append((source, anchor))
    proposal = root / REVIEW_PACKET.PROPOSAL_COPY_NAME
    proposal.write_bytes(_jsonl_bytes(rows))
    return proposal, local_root, rows, payloads


def _thaw_packet_tree(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    path.chmod(0o755)
    for current_text, directory_names, file_names in os.walk(path):
        current = Path(current_text)
        current.chmod(0o755)
        for name in directory_names:
            (current / name).chmod(0o755)
        for name in file_names:
            (current / name).chmod(0o644)


class GokuActionReviewPacketTests(unittest.TestCase):
    def test_builds_exact_draft_only_static_packet_with_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            hostile = '</script><img src=x onerror="alert(1)">'
            proposal, local_root, rows, payloads = _fixture(
                root,
                prompt=hostile,
            )
            proposal_raw = proposal.read_bytes()
            output = root / "packet"

            ready = REVIEW_PACKET.build_review_packet(
                proposal_path=proposal,
                output_dir=output,
                path_map_specs=[f"/vast/auh/goku={local_root}"],
            )
            try:
                for current_text, directory_names, file_names in os.walk(
                    output
                ):
                    current = Path(current_text)
                    self.assertEqual(
                        stat.S_IMODE(os.lstat(current).st_mode),
                        0o555,
                    )
                    for name in directory_names:
                        self.assertEqual(
                            stat.S_IMODE(
                                os.lstat(current / name).st_mode
                            ),
                            0o555,
                        )
                    for name in file_names:
                        self.assertEqual(
                            stat.S_IMODE(
                                os.lstat(current / name).st_mode
                            ),
                            0o444,
                        )
            finally:
                # The production artifact remains frozen; only this temporary
                # test fixture is made writable so TemporaryDirectory can
                # remove it without relying on platform-specific behavior.
                _thaw_packet_tree(output)

            proposal_sha = _sha(proposal_raw)
            self.assertEqual(
                (output / REVIEW_PACKET.PROPOSAL_COPY_NAME).read_bytes(),
                proposal_raw,
            )
            manifest_raw = (output / REVIEW_PACKET.MANIFEST_NAME).read_bytes()
            manifest = json.loads(manifest_raw)
            self.assertEqual(
                set(manifest),
                {
                    "schema_version",
                    "review_scope",
                    "production_authorization",
                    "draft_schema_version",
                    "incompatible_official_approval_schema",
                    "builder",
                    "proposal",
                    "path_maps",
                    "rows",
                },
            )
            self.assertEqual(
                set(manifest["builder"]),
                {"implementation_path", "implementation_sha256"},
            )
            self.assertEqual(
                set(manifest["proposal"]),
                {
                    "original_path",
                    "packet_path",
                    "sha256",
                    "bytes",
                    "rows",
                },
            )
            self.assertTrue(
                all(
                    set(record) == {"remote_prefix", "local_prefix"}
                    for record in manifest["path_maps"]
                )
            )
            self.assertEqual(manifest["schema_version"], REVIEW_PACKET.PACKET_SCHEMA)
            self.assertEqual(manifest["review_scope"], "draft_only")
            self.assertFalse(manifest["production_authorization"])
            self.assertEqual(manifest["proposal"]["sha256"], proposal_sha)
            self.assertEqual(manifest["proposal"]["rows"], 2)
            self.assertEqual(ready["proposal_sha256"], proposal_sha)
            builder_sha = _sha(SCRIPT.read_bytes())
            self.assertEqual(
                manifest["builder"]["implementation_path"],
                str(SCRIPT),
            )
            self.assertEqual(
                manifest["builder"]["implementation_sha256"],
                builder_sha,
            )
            self.assertEqual(
                ready["builder_implementation_sha256"],
                builder_sha,
            )
            self.assertEqual(
                set(ready),
                {
                    "schema_version",
                    "review_scope",
                    "production_authorization",
                    "proposal_sha256",
                    "packet_manifest_sha256",
                    "sha256sums_sha256",
                    "builder_implementation_sha256",
                    "row_count",
                    "draft_schema_version",
                    "incompatible_official_approval_schema",
                },
            )
            self.assertEqual(
                ready["packet_manifest_sha256"],
                _sha(manifest_raw),
            )
            self.assertEqual(
                ready["sha256sums_sha256"],
                _sha((output / REVIEW_PACKET.SUMS_NAME).read_bytes()),
            )
            self.assertFalse(ready["production_authorization"])
            self.assertNotEqual(
                ready["draft_schema_version"],
                ready["incompatible_official_approval_schema"],
            )

            for index, manifest_row in enumerate(manifest["rows"]):
                self.assertEqual(
                    set(manifest_row),
                    {
                        "iid",
                        "proposal_line",
                        "proposal_row_sha256",
                        "assets",
                        "review",
                        "draft_decision_default",
                    },
                )
                self.assertEqual(
                    set(manifest_row["assets"]),
                    {"source_video", "anchor_image"},
                )
                for asset in manifest_row["assets"].values():
                    self.assertEqual(
                        set(asset),
                        {
                            "original_path",
                            "mapped_local_input_path",
                            "packet_path",
                            "sha256",
                            "bytes",
                        },
                    )
                self.assertEqual(
                    manifest_row["proposal_row_sha256"],
                    _sha(proposal_raw.splitlines(keepends=True)[index]),
                )
                self.assertEqual(
                    manifest_row["draft_decision_default"],
                    "undecided",
                )
                source_record = manifest_row["assets"]["source_video"]
                anchor_record = manifest_row["assets"]["anchor_image"]
                self.assertEqual(
                    (output / source_record["packet_path"]).read_bytes(),
                    payloads[index][0],
                )
                self.assertEqual(
                    (output / anchor_record["packet_path"]).read_bytes(),
                    payloads[index][1],
                )
                self.assertEqual(
                    source_record["original_path"],
                    rows[index]["resolved_src_video"],
                )
                self.assertEqual(
                    anchor_record["original_path"],
                    rows[index]["resolved_anchor_image"],
                )
                self.assertEqual(
                    Path(source_record["mapped_local_input_path"]),
                    (
                        local_root
                        / "videos"
                        / str(rows[index]["iid"])
                        / "source.mp4"
                    ),
                )
                self.assertIn(
                    "target_support_evidence",
                    manifest_row["review"],
                )

            sums = (output / REVIEW_PACKET.SUMS_NAME).read_text(
                encoding="utf-8"
            )
            summed_names: set[str] = set()
            for line in sums.splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(_sha((output / name).read_bytes()), digest)
                summed_names.add(name)
            self.assertIn(REVIEW_PACKET.PROPOSAL_COPY_NAME, summed_names)
            self.assertIn(REVIEW_PACKET.MANIFEST_NAME, summed_names)
            self.assertIn(REVIEW_PACKET.APP_NAME, summed_names)
            self.assertNotIn(REVIEW_PACKET.READY_NAME, summed_names)

            index_html = (output / REVIEW_PACKET.INDEX_NAME).read_text()
            app = (output / REVIEW_PACKET.APP_NAME).read_text()
            self.assertNotIn(hostile, index_html)
            self.assertIn("Content-Security-Policy", index_html)
            self.assertIn("Exact full source video", app)
            self.assertIn("Target-support evidence", app)
            self.assertIn('"undecided"', app)
            self.assertIn(REVIEW_PACKET.DRAFT_SCHEMA, app)
            self.assertIn("production_authorization: false", app)
            self.assertIn(
                'document.getElementById("export-draft").disabled = !complete',
                app,
            )
            self.assertIn('entry.decision !== "undecided"', app)
            self.assertIn("entry.notes.trim().length > 0", app)
            self.assertIn("Refusing incomplete draft export", app)
            self.assertIn("notes: entry.notes.trim()", app)
            self.assertIn("textContent", app)
            self.assertNotIn("innerHTML", app)
            self.assertNotIn("insertAdjacentHTML", app)
            self.assertNotIn("document.write", app)
            self.assertNotIn("approve all", app.casefold())
            self.assertIn(proposal_sha, app)
            self.assertIn(_sha(manifest_raw), app)
            node = shutil.which("node")
            if node is not None:
                subprocess.run(
                    [node, "--check", str(output / REVIEW_PACKET.APP_NAME)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_longest_path_component_prefix_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            remote_root = "/vast/datasets/goku"
            source = b"correct-special-source"
            anchor = b"\x89PNG\r\n\x1a\ncorrect-special-anchor"
            row = _row(
                0,
                remote_root=remote_root,
                source_payload=source,
                anchor_payload=anchor,
            )
            proposal = root / REVIEW_PACKET.PROPOSAL_COPY_NAME
            proposal.write_bytes(_jsonl_bytes([row]))

            broad = root / "broad"
            specific = root / "specific"
            broad.mkdir()
            specific.mkdir()
            # If the broad prefix were selected, the hash gate would fail.
            broad_source = (
                broad / "goku" / "videos" / str(row["iid"]) / "source.mp4"
            )
            broad_anchor = broad / "goku" / "anchors" / f"{row['iid']}.png"
            broad_source.parent.mkdir(parents=True)
            broad_anchor.parent.mkdir(parents=True)
            broad_source.write_bytes(b"wrong")
            broad_anchor.write_bytes(b"wrong")
            _write_media(
                specific,
                row=row,
                source_payload=source,
                anchor_payload=anchor,
            )

            output = root / "packet"
            REVIEW_PACKET.build_review_packet(
                proposal_path=proposal,
                output_dir=output,
                path_map_specs=[
                    f"/vast/datasets/goku={specific}",
                    f"/vast/datasets={broad}",
                ],
            )
            _thaw_packet_tree(output)
            manifest = json.loads(
                (output / REVIEW_PACKET.MANIFEST_NAME).read_text()
            )
            mapped = manifest["rows"][0]["assets"]["source_video"][
                "mapped_local_input_path"
            ]
            self.assertTrue(mapped.startswith(str(specific)))
            self.assertEqual(
                [
                    record["remote_prefix"]
                    for record in manifest["path_maps"]
                ],
                ["/vast/datasets/goku", "/vast/datasets"],
            )

    def test_bad_media_hash_fails_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proposal, local_root, rows, _ = _fixture(root, count=1)
            rows[0]["source_video_sha256"] = "0" * 64
            proposal.write_bytes(_jsonl_bytes(rows))
            output = root / "packet"

            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "source video SHA-256 differs",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal,
                    output_dir=output,
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )
            self.assertFalse(os.path.lexists(output))

    def test_rejects_symlink_inputs_and_path_anomalies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proposal, local_root, rows, _ = _fixture(root, count=1)

            proposal_link = root / "proposal-link"
            proposal_link.symlink_to(proposal)
            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "symlink",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal_link,
                    output_dir=root / "packet-link-proposal",
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )

            anchor = local_root / "anchors" / f"{rows[0]['iid']}.png"
            real_anchor = local_root / "anchors" / "real.png"
            real_anchor.write_bytes(anchor.read_bytes())
            anchor.unlink()
            anchor.symlink_to(real_anchor)
            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "symlink",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal,
                    output_dir=root / "packet-link-media",
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )

            anchor.unlink()
            anchor.write_bytes(real_anchor.read_bytes())
            anomalous = json.loads(json.dumps(rows[0]))
            anomalous["resolved_src_video"] = (
                "/vast/auh/goku/videos/../escape/source.mp4"
            )
            proposal.write_bytes(_jsonl_bytes([anomalous]))
            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "canonical absolute POSIX path",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal,
                    output_dir=root / "packet-anomalous",
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )

    def test_existing_output_and_duplicate_iid_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proposal, local_root, rows, _ = _fixture(root, count=1)
            output = root / "packet"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("do not replace", encoding="utf-8")
            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "refusing overwrite",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal,
                    output_dir=output,
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )
            self.assertEqual(sentinel.read_text(), "do not replace")

            proposal.write_bytes(_jsonl_bytes([rows[0], rows[0]]))
            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "duplicate proposal IID",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal,
                    output_dir=root / "duplicate-packet",
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )

    def test_duplicate_source_content_is_rejected_for_distinct_iids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proposal, local_root, rows, _ = _fixture(root, count=2)
            rows[1]["resolved_src_video"] = rows[0]["resolved_src_video"]
            rows[1]["source_video_sha256"] = rows[0][
                "source_video_sha256"
            ]
            proposal.write_bytes(_jsonl_bytes(rows))

            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "duplicate proposal source video",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal,
                    output_dir=root / "packet",
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )

    def test_false_lexical_review_evidence_flag_is_still_reviewable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proposal, local_root, rows, _ = _fixture(root, count=1)
            finalization = rows[0]["action_anchor_finalization"]
            assert isinstance(finalization, dict)
            evidence = finalization["target_support_evidence"]
            assert isinstance(evidence, dict)
            evidence["requires_proposal_bound_human_review"] = False
            proposal.write_bytes(_jsonl_bytes(rows))

            ready = REVIEW_PACKET.build_review_packet(
                proposal_path=proposal,
                output_dir=root / "packet",
                path_map_specs=[f"/vast/auh/goku={local_root}"],
            )
            _thaw_packet_tree(root / "packet")
            self.assertEqual(ready["row_count"], 1)
            self.assertFalse(ready["production_authorization"])

    def test_rejects_any_row_that_is_not_pending_proposed_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            proposal, local_root, rows, _ = _fixture(root, count=1)
            finalization = rows[0]["action_anchor_finalization"]
            assert isinstance(finalization, dict)
            finalization["generation_authorized"] = True
            proposal.write_bytes(_jsonl_bytes(rows))

            with self.assertRaisesRegex(
                REVIEW_PACKET.ReviewPacketError,
                "generation_authorized must be False",
            ):
                REVIEW_PACKET.build_review_packet(
                    proposal_path=proposal,
                    output_dir=root / "packet",
                    path_map_specs=[f"/vast/auh/goku={local_root}"],
                )


if __name__ == "__main__":
    unittest.main()
