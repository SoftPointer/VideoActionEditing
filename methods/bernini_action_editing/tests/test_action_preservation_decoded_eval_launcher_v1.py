from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_launcher_v1 as launcher
import action_preservation_decoded_eval_plan_v1 as plan
import action_preservation_decoded_eval_deployment_controller_v1 as controller
import action_preservation_decoded_eval_aggregate_v2 as aggregate


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def action_contract(iid: str) -> dict:
    description = (
        f"Complete the fitted action for source {iid}, then hold the terminal pose."
    )
    value = {
        "schema_version": plan.ACTION_REVIEW_CONTRACT_SCHEMA,
        "action_order_description": description,
        "action_order_description_sha256": plan.text_sha256(description),
        "expected_onset_frame_min": 4,
        "expected_onset_frame_max": 20,
        "terminal_hold_start_frame_min": 65,
        "terminal_hold_end_frame": 80,
        "full_video_frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
    }
    value["contract_digest"] = plan.object_sha256(value)
    return value


def bundle(evaluation_root: pathlib.Path) -> dict:
    sources = []
    for index, iid in enumerate(plan.FITTED_IIDS):
        instruction = f"Perform the fitted action for source {iid}."
        sources.append(
            {
                "iid": iid,
                "source_video_sha256": digest(f"source:{iid}"),
                "source_receipt_sha256": digest(f"source-receipt:{iid}"),
                "instruction": instruction,
                "instruction_sha256": plan.text_sha256(instruction),
                "action_review_contract": action_contract(iid),
                "seed": 2026081801 + index,
            }
        )
    checkpoints = [
        {
            "arm": arm,
            "checkpoint_step": step,
            "checkpoint_receipt_sha256": digest(f"checkpoint:{arm}:{step}"),
            "adapter_sha256": digest(f"adapter:{arm}:{step}"),
        }
        for arm in plan.ARMS
        for step in plan.CHECKPOINT_STEPS
    ]
    input_spec = plan.build_input_spec(
        evaluation_id="preservation-v2-launch-root-authority-test",
        evaluation_root=evaluation_root,
        pins={key: digest(key) for key in plan.PIN_FIELDS},
        sources=sources,
        checkpoints=checkpoints,
    )
    value = plan.build_bundle(input_spec)
    value["publication_receipt"] = plan.build_publication_receipt(value)
    return value


class LauncherPublicationAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = pathlib.Path(self.temporary.name).resolve()
        self.bundle = bundle(self.parent / "evaluation")

    def manifest(self, root: pathlib.Path) -> dict:
        return launcher.build_launch_manifest(
            bundle=self.bundle,
            launch_root=root,
            python_identity={"path": "/stub/python", "sha256": digest("python")},
            executor_identity={
                "path": "/stub/executor.py",
                "sha256": digest("executor"),
            },
            decoder_identity={
                "path": "/stub/decoder.py",
                "sha256": digest("decoder"),
            },
            ffprobe_identity={
                "path": "/stub/ffprobe",
                "sha256": digest("ffprobe"),
            },
            physical_bindings_identity={
                "path": "/stub/physical-bindings.json",
                "sha256": digest("physical-bindings"),
            },
            blinding_key_identity={
                "path": "/stub/blinding-key",
                "sha256": digest("blinding-key"),
                "size": 32,
                "mode": 0o400,
            },
            aggregate_root=root.parent / "aggregate",
            verify_tools=False,
        )

    def deployment_authority(self) -> dict:
        work_root = self.parent / "signed-work-root"
        identity = {
            "device": 1, "inode": 2, "uid": 3, "gid": 4,
            "mode": stat.S_IFDIR | 0o700, "nlink": 2, "rdev": 0,
            "size": 64, "blocks": 0, "mtime_ns": 5, "ctime_ns": 6,
        }
        immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
        work = {
            "schema_version": launcher.WORK_ROOT_AUTHORITY_SCHEMA,
            "path": str(work_root),
            "parent_path": str(work_root.parent),
            "creation_identity": identity,
            "immutable_identity": {
                key: identity[key] for key in immutable_fields
            },
            "parent_immutable_identity": {
                "device": 1, "inode": 7, "uid": 3, "gid": 4,
                "mode": stat.S_IFDIR | 0o700, "rdev": 0,
            },
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        work["authority_digest"] = launcher.object_sha256(work)
        value = {
            "schema_version": launcher.DEPLOYMENT_AUTHORITY_SCHEMA,
            "work_root_authority": work,
            "deployment_receipt": {
                "path": str(work_root / "deployment-receipt.json"),
                "sha256": digest("deployment-receipt-file"),
            },
            "source_spec_authority": {
                "path": str(work_root / "source-spec-authority.json"),
                "sha256": digest("source-spec-authority-file"),
            },
            "deployment_receipt_digest": digest("deployment-receipt-object"),
            "source_spec_authority_digest": digest("source-authority-object"),
        }
        value["authority_digest"] = launcher.object_sha256(value)
        return value

    def materialized_bundle(self, name: str = "materialized-evaluation") -> dict:
        value = bundle(self.parent / name)
        source = {
            key: value[key]
            for key in ("input_spec", "review_contract", "manifest", "shards")
        }
        published = plan.publish_bundle_authorized(source)
        return {
            **source,
            "publication_receipt": published["publication_receipt"],
            "directory_authority": published["directory_authority"],
        }

    def completion_anchors(self, value: dict) -> list[dict]:
        reservations = value["publication_receipt"][
            "holder_completion_reservations"
        ]
        result = []
        for index, (holder, reservation) in enumerate(
            zip(plan.HOLDER_ROWS, reservations)
        ):
            initial = reservation["identity"]
            anchor = {
                "schema_version": launcher.HOLDER_COMPLETION_ANCHOR_SCHEMA,
                "holder_job_id": holder["job_id"],
                "completion_path": reservation["path"],
                "initial_inode_identity": {
                    field: initial[field]
                    for field in ("device", "inode", "uid", "gid", "rdev")
                },
                "completion_sha256": digest(f"completion-file:{index}"),
                "completion_size": 1000 + index,
                "completion_mode": (
                    plan.HOLDER_DIRECTORY_COMPLETION_SEALED_MODE
                ),
                "completion_digest": digest(f"completion-object:{index}"),
                "holder_summary_digest": digest(f"holder-summary:{index}"),
            }
            anchor["anchor_digest"] = launcher.object_sha256(anchor)
            result.append(anchor)
        return result

    def test_positive_publication_is_exact_relative_and_sealed(self) -> None:
        root = self.parent / "positive-launch"
        value = self.manifest(root)
        output = launcher.publish_launch_manifest(value, bundle=self.bundle)

        self.assertEqual(output, root / launcher.FILENAME)
        self.assertEqual(
            value["launch_manifest_anchor_channel"],
            launcher._launch_manifest_anchor_channel(required=False),
        )
        self.assertEqual(
            output.read_bytes(), launcher.canonical_json_bytes(value) + b"\n"
        )
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o400)
        self.assertEqual(
            {item.name for item in root.iterdir()},
            {launcher.FILENAME},
        )
        with self.assertRaisesRegex(
            launcher.DecodedEvaluationLauncherError, "not fresh"
        ):
            launcher.publish_launch_manifest(value, bundle=self.bundle)

    def test_controller_run_target_argv_uses_direct_signed_work_root_captures(self) -> None:
        authority = launcher._deployment_authority(
            self.deployment_authority()
        )
        captures = launcher._executor_capture_rows(authority)
        self.assertEqual(len(captures), 4)
        self.assertEqual(
            len({row["basename"] for row in captures}), 4
        )
        work_root = pathlib.Path(authority["work_root_authority"]["path"])
        self.assertTrue(
            all(pathlib.Path(row["path"]).parent == work_root for row in captures)
        )
        capture = captures[0]
        argv = launcher._controller_target_argv(
            root_python={"path": "/usr/bin/python3.10", "sha256": digest("root")},
            controller={"path": "/bundle/controller.py", "sha256": digest("controller")},
            deployment_authority=authority,
            target="action_preservation_decoded_eval_executor_v2.py",
            target_arguments=["--holder-job-id", capture["holder_job_id"]],
            capture_receipt_path=capture["path"],
        )
        self.assertEqual(
            argv[:8],
            [
                "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                launcher.ROOT_CONTROLLER_BOOTSTRAP_SOURCE,
                "/bundle/controller.py", digest("controller"),
            ],
        )
        self.assertEqual(argv[8], "run-target")
        self.assertNotIn("verified-run", argv)
        self.assertEqual(
            argv[argv.index("--capture-receipt") + 1], capture["path"]
        )
        self.assertEqual(
            launcher.ROOT_CONTROLLER_BOOTSTRAP_SOURCE,
            controller.ROOT_CONTROLLER_BOOTSTRAP_SOURCE,
        )
        with self.assertRaisesRegex(
            launcher.DecodedEvaluationLauncherError,
            "direct work-root member",
        ):
            launcher._controller_target_argv(
                root_python={
                    "path": "/usr/bin/python3.10", "sha256": digest("root")
                },
                controller={"path": "/bundle/controller.py", "sha256": digest("controller")},
                deployment_authority=authority,
                target="action_preservation_decoded_eval_executor_v2.py",
                target_arguments=[],
                capture_receipt_path=str(work_root / "nested" / "capture.json"),
            )

    def test_online_anchor_exact_four_and_stdout_fail_closed(self) -> None:
        value = self.materialized_bundle()
        anchors = self.completion_anchors(value)
        literals = [
            launcher.canonical_json_bytes(anchor).decode("utf-8")
            for anchor in reversed(anchors)
        ]
        authority = launcher.build_holder_completion_dynamic_authority(
            literals, bundle=value
        )
        self.assertEqual(
            [item["holder_job_id"] for item in authority["anchors"]],
            [item["job_id"] for item in plan.HOLDER_ROWS],
        )
        arguments = launcher.holder_completion_anchor_arguments(
            authority, bundle=value
        )
        self.assertEqual(
            arguments.count("--holder-completion-anchor"), 4
        )
        self.assertEqual(
            arguments[1::2],
            [
                launcher.canonical_json_bytes(anchor).decode("utf-8")
                for anchor in anchors
            ],
        )

        first_literal = launcher.canonical_json_bytes(anchors[0]).decode(
            "utf-8"
        )
        observed = launcher.parse_holder_completion_anchor_stdout(
            first_literal.encode("utf-8") + b"\n",
            return_code=0,
            bundle=value,
            expected_holder_job_id=anchors[0]["holder_job_id"],
        )
        self.assertEqual(observed, anchors[0])
        hostile_stdout = (
            b"",
            first_literal.encode("utf-8"),
            first_literal.encode("utf-8") + b"\nextra",
            first_literal.encode("utf-8") + b"\n" + first_literal.encode("utf-8") + b"\n",
            b"partial-" + first_literal.encode("utf-8") + b"\n",
        )
        for raw in hostile_stdout:
            with self.subTest(stdout=raw[:24]), self.assertRaises(
                launcher.DecodedEvaluationLauncherError
            ):
                launcher.parse_holder_completion_anchor_stdout(
                    raw,
                    return_code=0,
                    bundle=value,
                    expected_holder_job_id=anchors[0]["holder_job_id"],
                )
        with self.assertRaises(launcher.DecodedEvaluationLauncherError):
            launcher.parse_holder_completion_anchor_stdout(
                first_literal.encode("utf-8") + b"\n",
                return_code=70,
                bundle=value,
                expected_holder_job_id=anchors[0]["holder_job_id"],
            )

        with self.assertRaises(launcher.DecodedEvaluationLauncherError):
            launcher.build_holder_completion_dynamic_authority(
                [literals[0], literals[0], *literals[2:]], bundle=value
            )
        with self.assertRaises(launcher.DecodedEvaluationLauncherError):
            launcher.build_holder_completion_dynamic_authority(
                literals[:3], bundle=value
            )
        with self.assertRaises(launcher.DecodedEvaluationLauncherError):
            launcher.parse_holder_completion_anchor_literal(
                first_literal + " ", bundle=value
            )

        for field, replacement in (
            ("completion_path", anchors[1]["completion_path"]),
            (
                "initial_inode_identity",
                {**anchors[0]["initial_inode_identity"], "inode": 999999},
            ),
            ("completion_mode", 0o600),
        ):
            hostile = copy.deepcopy(anchors[0])
            hostile[field] = replacement
            hostile.pop("anchor_digest")
            hostile["anchor_digest"] = launcher.object_sha256(hostile)
            with self.subTest(field=field), self.assertRaises(
                launcher.DecodedEvaluationLauncherError
            ):
                launcher.parse_holder_completion_anchor_literal(
                    launcher.canonical_json_bytes(hostile).decode("utf-8"),
                    bundle=value,
                )
        extra = copy.deepcopy(anchors[0])
        extra["hostile_extra"] = True
        extra.pop("anchor_digest")
        extra["anchor_digest"] = launcher.object_sha256(extra)
        with self.assertRaises(launcher.DecodedEvaluationLauncherError):
            launcher.parse_holder_completion_anchor_literal(
                launcher.canonical_json_bytes(extra).decode("utf-8"),
                bundle=value,
            )

    def test_aggregate_command_plan_binds_manifest_and_literal_anchors(self) -> None:
        value = self.materialized_bundle("aggregate-plan-evaluation")
        anchors = self.completion_anchors(value)
        literals = [
            launcher.canonical_json_bytes(anchor).decode("utf-8")
            for anchor in anchors
        ]
        authority = launcher._deployment_authority(
            self.deployment_authority()
        )
        work_root = pathlib.Path(authority["work_root_authority"]["path"])
        launch = self.manifest(work_root / "launch")
        launch["tool_files_verified"] = True
        launch["verified_runtime"] = {
            "root_python": {
                "path": "/usr/bin/python3.10", "sha256": digest("root")
            },
            "controller": {
                "path": "/bundle/controller.py",
                "sha256": digest("controller"),
            },
        }
        launch["deployment_authority"] = authority
        launch["aggregate_root"] = str(work_root / "aggregate")
        launch["aggregate_runtime_capture_receipt_path"] = str(
            work_root / launcher.AGGREGATE_CAPTURE_BASENAME
        )
        launch_path = pathlib.Path(launch["launch_root"]) / launcher.FILENAME
        launch_sha = hashlib.sha256(
            launcher.canonical_json_bytes(launch) + b"\n"
        ).hexdigest()
        launch_payload_size = len(launcher.canonical_json_bytes(launch) + b"\n")
        launch_anchor = launcher.build_launch_manifest_anchor(
            launch_manifest=launch,
            path=launch_path,
            identity={
                "device": 1, "inode": 9, "uid": 3, "gid": 4,
                "mode": stat.S_IFREG | 0o400, "nlink": 1, "rdev": 0,
                "size": launch_payload_size, "blocks": 0,
                "mtime_ns": 5, "ctime_ns": 6,
            },
        )
        with mock.patch.object(
            launcher, "validate_launch_manifest", return_value=launch
        ):
            command = launcher.build_aggregate_command_plan(
                launch_manifest=launch,
                launch_manifest_path=launch_path,
                launch_manifest_sha256=launch_sha,
                launch_manifest_anchor=launch_anchor,
                holder_completion_anchor_literals=literals,
                bundle=value,
            )
            self.assertEqual(
                command["launch_manifest_anchor"]["sha256"], launch_sha
            )
            self.assertEqual(
                command["launch_manifest_anchor_literal"],
                launcher.canonical_json_bytes(launch_anchor).decode("utf-8"),
            )
            self.assertEqual(
                command["holder_completion_anchor_literals"], literals
            )
            self.assertEqual(
                command["argv"].count("--holder-completion-anchor"), 4
            )
            self.assertIn("--blinding-key-sha256", command["argv"])
            self.assertEqual(
                command["argv"][
                    command["argv"].index(
                        "--aggregate-runtime-capture-receipt"
                    ) + 1
                ],
                launch["aggregate_runtime_capture_receipt_path"],
            )
            self.assertFalse(command["command_execution_performed"])
            self.assertFalse(command["subprocess_spawned"])
            self.assertEqual(
                command["aggregate_completion_anchor_channel"],
                launcher._aggregate_completion_anchor_channel(),
            )
            aggregate_arguments = command["argv"][
                command["argv"].index("--") + 1:
            ]
            with mock.patch.object(
                aggregate.bridge.verified_release,
                "load_inherited_work_root_environment",
                side_effect=(
                    aggregate.bridge.verified_release
                    .DecodedEvalVerifiedReleaseError("parser-cross-sentinel")
                ),
            ), self.assertRaisesRegex(
                aggregate.DecodedEvaluationAggregateError,
                "parser-cross-sentinel",
            ):
                aggregate.main(aggregate_arguments)
            command_stdout = launcher.canonical_json_bytes(command) + b"\n"
            self.assertEqual(
                launcher.parse_aggregate_command_plan_stdout(
                    command_stdout,
                    return_code=0,
                    bundle=value,
                    launch_manifest=launch,
                ),
                command,
            )
            for hostile_stdout in (
                command_stdout + command_stdout,
                command_stdout[:-1],
                b"partial-" + command_stdout,
            ):
                with self.assertRaises(
                    launcher.DecodedEvaluationLauncherError
                ):
                    launcher.parse_aggregate_command_plan_stdout(
                        hostile_stdout,
                        return_code=0,
                        bundle=value,
                        launch_manifest=launch,
                    )

            hostile = copy.deepcopy(command)
            hostile["argv"][0] = "/hostile/python"
            hostile.pop("plan_digest")
            hostile["plan_digest"] = launcher.object_sha256(hostile)
            with self.assertRaises(launcher.DecodedEvaluationLauncherError):
                launcher.validate_aggregate_command_plan(
                    hostile, bundle=value, launch_manifest=launch
                )
            extra = copy.deepcopy(command)
            extra["hostile_extra"] = True
            extra.pop("plan_digest")
            extra["plan_digest"] = launcher.object_sha256(extra)
            with self.assertRaises(launcher.DecodedEvaluationLauncherError):
                launcher.validate_aggregate_command_plan(
                    extra, bundle=value, launch_manifest=launch
                )

            directory_identity = {
                "device": 1, "inode": 2, "uid": 3, "gid": 4,
                "mode": stat.S_IFDIR | 0o555, "nlink": 2, "rdev": 0,
                "size": 64, "blocks": 0, "mtime_ns": 5, "ctime_ns": 6,
            }

            def aggregate_file(
                relative_path: str, mode: int, label: str
            ) -> dict:
                size = 100 + len(label)
                return {
                    "relative_path": relative_path,
                    "sha256": digest(f"aggregate-file:{label}"),
                    "size": size,
                    "mode": mode,
                    "identity": {
                        **directory_identity,
                        "inode": 20 + len(label),
                        "mode": stat.S_IFREG | mode,
                        "nlink": 1,
                        "size": size,
                    },
                    "object_digest": digest(f"aggregate-object:{label}"),
                }

            aggregate_anchor = {
                "schema_version": (
                    launcher.AGGREGATE_COMPLETION_ANCHOR_SCHEMA
                ),
                "evaluation_id": launch["evaluation_id"],
                "aggregate_root": launch["aggregate_root"],
                "aggregate_root_identity": directory_identity,
                "aggregate_file": aggregate_file(
                    "evaluation_complete.json", 0o444, "aggregate"
                ),
                "private_file": aggregate_file(
                    "private_blind_mapping.json", 0o400, "private"
                ),
                "public_file": aggregate_file(
                    "blind_review_packet.json", 0o444, "public"
                ),
                "media_directory_identity": {
                    **directory_identity, "inode": 3
                },
                "media_file_count": 264,
                "media_rows_digest": digest("aggregate-media-rows"),
            }
            aggregate_anchor["media_tree_digest"] = launcher.object_sha256(
                {
                    "media_directory_identity": aggregate_anchor[
                        "media_directory_identity"
                    ],
                    "media_file_count": aggregate_anchor["media_file_count"],
                    "media_rows_digest": aggregate_anchor["media_rows_digest"],
                }
            )
            aggregate_anchor["anchor_digest"] = launcher.object_sha256(
                aggregate_anchor
            )
            aggregate_stdout = (
                launcher.canonical_json_bytes(aggregate_anchor) + b"\n"
            )
            self.assertEqual(
                launcher.parse_aggregate_completion_anchor_stdout(
                    aggregate_stdout,
                    return_code=0,
                    launch_manifest=launch,
                ),
                aggregate_anchor,
            )
            maximum_media_count = (
                plan.TOTAL_DECODE_COUNT
                + len(getattr(plan, "SOURCE_IDS", plan.FITTED_IIDS))
            )
            maximum_anchor = copy.deepcopy(aggregate_anchor)
            maximum_anchor["media_file_count"] = maximum_media_count
            maximum_anchor["media_tree_digest"] = launcher.object_sha256(
                {
                    "media_directory_identity": maximum_anchor[
                        "media_directory_identity"
                    ],
                    "media_file_count": maximum_anchor["media_file_count"],
                    "media_rows_digest": maximum_anchor["media_rows_digest"],
                }
            )
            maximum_anchor.pop("anchor_digest")
            maximum_anchor["anchor_digest"] = launcher.object_sha256(
                maximum_anchor
            )
            self.assertEqual(
                launcher.validate_aggregate_completion_anchor(
                    maximum_anchor, launch_manifest=launch
                ),
                maximum_anchor,
            )
            over = copy.deepcopy(maximum_anchor)
            over["media_file_count"] = maximum_media_count + 1
            over["media_tree_digest"] = launcher.object_sha256(
                {
                    "media_directory_identity": over[
                        "media_directory_identity"
                    ],
                    "media_file_count": over["media_file_count"],
                    "media_rows_digest": over["media_rows_digest"],
                }
            )
            over.pop("anchor_digest")
            over["anchor_digest"] = launcher.object_sha256(over)
            with self.assertRaises(launcher.DecodedEvaluationLauncherError):
                launcher.validate_aggregate_completion_anchor(
                    over, launch_manifest=launch
                )
            for hostile_stdout in (
                aggregate_stdout + aggregate_stdout,
                aggregate_stdout[:-1],
                b"partial-" + aggregate_stdout,
            ):
                with self.assertRaises(
                    launcher.DecodedEvaluationLauncherError
                ):
                    launcher.parse_aggregate_completion_anchor_stdout(
                        hostile_stdout,
                        return_code=0,
                        launch_manifest=launch,
                    )

    def test_pinned_manifest_held_read_rejects_replacement_and_extra(self) -> None:
        for hostile_kind in (
            None,
            "manifest-replacement",
            "work-root-extra",
            "work-root-replacement",
        ):
            with self.subTest(hostile=hostile_kind):
                work_root = self.parent / f"pinned-work-{hostile_kind}"
                work_root.mkdir(mode=0o700)
                value = bundle(work_root / "evaluation")
                source = {
                    key: value[key]
                    for key in (
                        "input_spec", "review_contract", "manifest", "shards"
                    )
                }
                published = plan.publish_bundle_authorized(source)
                materialized = {
                    **source,
                    "publication_receipt": published["publication_receipt"],
                    "directory_authority": published["directory_authority"],
                }
                launch_root = work_root / "launch"
                manifest = launcher.build_launch_manifest(
                    bundle=materialized,
                    launch_root=launch_root,
                    python_identity={
                        "path": "/stub/python", "sha256": digest("python")
                    },
                    executor_identity={
                        "path": "/stub/executor.py",
                        "sha256": digest("executor"),
                    },
                    decoder_identity={
                        "path": "/stub/decoder.py",
                        "sha256": digest("decoder"),
                    },
                    ffprobe_identity={
                        "path": "/stub/ffprobe", "sha256": digest("ffprobe")
                    },
                    physical_bindings_identity={
                        "path": "/stub/physical-bindings.json",
                        "sha256": digest("physical-bindings"),
                    },
                    blinding_key_identity={
                        "path": "/stub/blinding-key",
                        "sha256": digest("blinding-key"),
                        "size": 32,
                        "mode": 0o400,
                    },
                    aggregate_root=work_root / "aggregate",
                    verify_tools=False,
                )
                launch_publication = launcher.publish_launch_manifest_authorized(
                    manifest, bundle=materialized
                )
                output = pathlib.Path(launch_publication["output"])
                launch_anchor = launch_publication["launch_manifest_anchor"]
                manifest_sha = hashlib.sha256(output.read_bytes()).hexdigest()
                self.assertEqual(launch_anchor["sha256"], manifest_sha)
                self.assertEqual(
                    launcher.parse_launch_manifest_anchor_stdout(
                        launcher.canonical_json_bytes(launch_anchor) + b"\n",
                        return_code=0,
                        launch_manifest=manifest,
                    ),
                    launch_anchor,
                )
                launch_anchor_stdout = (
                    launcher.canonical_json_bytes(launch_anchor) + b"\n"
                )
                for hostile_stdout in (
                    launch_anchor_stdout + launch_anchor_stdout,
                    launch_anchor_stdout[:-1],
                    b"partial-" + launch_anchor_stdout,
                ):
                    with self.assertRaises(
                        launcher.DecodedEvaluationLauncherError
                    ):
                        launcher.parse_launch_manifest_anchor_stdout(
                            hostile_stdout,
                            return_code=0,
                            launch_manifest=manifest,
                        )
                parent_fd = os.open(
                    work_root.parent,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                root_fd = os.open(
                    work_root,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                binding = {
                    "path": str(work_root),
                    "parent_path": str(work_root.parent),
                    "parent_fd": parent_fd,
                    "root_fd": root_fd,
                    "root_immutable_identity": (
                        launcher._immutable_directory_row(os.fstat(root_fd))
                    ),
                    "root_identity": launcher.executor._stat_identity_row(
                        os.fstat(root_fd)
                    ),
                    "parent_immutable_identity": (
                        launcher._immutable_directory_row(os.fstat(parent_fd))
                    ),
                    "parent_identity": launcher.executor._stat_identity_row(
                        os.fstat(parent_fd)
                    ),
                    "entries": sorted(os.listdir(root_fd)),
                    "work_root_authority_digest": digest("work-root-authority"),
                    "deployment_receipt_digest": digest("deployment-receipt"),
                    "source_spec_authority_digest": digest("source-authority"),
                    "target": "action_preservation_decoded_eval_launcher_v1.py",
                    "capture_receipt_path": str(
                        work_root / "aggregate-launcher-capture.json"
                    ),
                }
                fired = False

                def barrier(stage: str) -> None:
                    nonlocal fired
                    if (
                        hostile_kind is None
                        or fired
                        or stage != "aggregate_launch_before_final_manifest_replay"
                    ):
                        return
                    fired = True
                    if hostile_kind == "manifest-replacement":
                        launch_root.chmod(0o755)
                        raw = output.read_bytes()
                        output.rename(launch_root / (launcher.FILENAME + ".held"))
                        output.write_bytes(raw)
                        output.chmod(0o400)
                        launch_root.chmod(0o555)
                    elif hostile_kind == "work-root-extra":
                        (work_root / "hostile-extra").write_bytes(b"x")
                    elif hostile_kind == "work-root-replacement":
                        work_root.rename(
                            self.parent / f"{work_root.name}-displaced"
                        )
                        work_root.mkdir(mode=0o700)

                try:
                    with mock.patch.object(
                        launcher.executor.bridge.verified_release,
                        "validate_inherited_work_root_binding",
                        return_value=binding,
                    ), mock.patch.object(
                        launcher, "_publication_barrier", side_effect=barrier
                    ):
                        if hostile_kind is None:
                            observed, loaded = launcher.load_pinned_launch_manifest(
                                output,
                                expected_sha256=manifest_sha,
                                work_root_binding=binding,
                                expected_anchor=launch_anchor,
                            )
                            self.assertEqual(observed, manifest)
                            self.assertEqual(
                                loaded["publication_receipt"]["publication_digest"],
                                materialized["publication_receipt"]["publication_digest"],
                            )
                            with self.assertRaises(
                                launcher.DecodedEvaluationLauncherError
                            ):
                                launcher.load_pinned_launch_manifest(
                                    output,
                                    expected_sha256=digest("wrong-manifest"),
                                    work_root_binding=binding,
                                    expected_anchor=launch_anchor,
                                )
                        else:
                            with self.assertRaises(
                                launcher.DecodedEvaluationLauncherError
                            ):
                                launcher.load_pinned_launch_manifest(
                                    output,
                                    expected_sha256=manifest_sha,
                                    work_root_binding=binding,
                                    expected_anchor=launch_anchor,
                                )
                            self.assertTrue(fired)
                finally:
                    os.close(root_fd)
                    os.close(parent_fd)

    def test_production_publication_uses_held_work_root_and_no_fd_projection(self) -> None:
        for hostile in (False, True):
            with self.subTest(hostile=hostile):
                authority_parent = self.parent / f"authority-parent-{hostile}"
                authority_parent.mkdir()
                work_root = authority_parent / "work-root"
                work_root.mkdir(mode=0o700)
                launch_root = work_root / "launch"
                parent_fd = os.open(
                    authority_parent,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                root_fd = os.open(
                    work_root,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                binding = {
                    "path": str(work_root),
                    "parent_path": str(authority_parent),
                    "parent_fd": parent_fd,
                    "root_fd": root_fd,
                    "root_immutable_identity": (
                        launcher._immutable_directory_row(os.fstat(root_fd))
                    ),
                    "parent_immutable_identity": (
                        launcher._immutable_directory_row(os.fstat(parent_fd))
                    ),
                    "entries": [],
                    "work_root_authority_digest": digest("work-root-authority"),
                    "deployment_receipt_digest": digest("deployment-receipt"),
                    "source_spec_authority_digest": digest("source-authority"),
                    "target": "action_preservation_decoded_eval_launcher_v1.py",
                    "capture_receipt_path": str(
                        work_root / "launcher-verified-runtime-capture.json"
                    ),
                }
                value = self.manifest(launch_root)
                value["tool_files_verified"] = True
                value["launcher_work_root"] = (
                    launcher._launcher_work_root_projection(binding)
                )
                displaced = authority_parent / "work-root-displaced"

                def barrier(stage: str) -> None:
                    if hostile and stage == "before_root_mkdir":
                        os.rename(work_root, displaced)
                        work_root.mkdir(mode=0o700)

                try:
                    with mock.patch.object(
                        launcher, "validate_launch_manifest", return_value=value
                    ), mock.patch.object(
                        launcher.executor.bridge.verified_release,
                        "validate_inherited_work_root_binding",
                        return_value=binding,
                    ), mock.patch.object(
                        launcher, "_publication_barrier", side_effect=barrier
                    ):
                        if hostile:
                            with self.assertRaisesRegex(
                                launcher.DecodedEvaluationLauncherError,
                                "work-root|authority|identity",
                            ):
                                launcher.publish_launch_manifest(
                                    value,
                                    bundle=self.bundle,
                                    work_root_binding=binding,
                                )
                        else:
                            output = launcher.publish_launch_manifest(
                                value,
                                bundle=self.bundle,
                                work_root_binding=binding,
                            )
                            self.assertEqual(output.parent, launch_root)
                            self.assertEqual(
                                {item.name for item in launch_root.iterdir()},
                                {launcher.FILENAME},
                            )
                            serialized = output.read_text()
                            self.assertNotIn('"root_fd"', serialized)
                            self.assertNotIn('"parent_fd"', serialized)
                finally:
                    os.close(root_fd)
                    os.close(parent_fd)
                if hostile:
                    self.assertTrue(displaced.is_dir())
                    self.assertEqual(list(work_root.iterdir()), [])

    def test_descriptor_pair_failure_does_not_leak(self) -> None:
        first = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY)
        second = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            baseline = len(os.listdir("/dev/fd"))
            with mock.patch.object(
                launcher.os,
                "set_inheritable",
                side_effect=OSError("injected inheritable failure"),
            ), self.assertRaises(launcher.DecodedEvaluationLauncherError):
                launcher._duplicate_noninheritable_pair(
                    first, second, label="test pair"
                )
            self.assertEqual(len(os.listdir("/dev/fd")), baseline)
        finally:
            os.close(second)
            os.close(first)

    def test_canonical_parent_descriptor_rejects_symlink_and_exchange(self) -> None:
        real_parent = self.parent / "real-parent"
        real_parent.mkdir()
        alias_parent = self.parent / "alias-parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        symlink_root = alias_parent / "launch"
        with self.assertRaises(launcher.DecodedEvaluationLauncherError):
            launcher.publish_launch_manifest(
                self.manifest(symlink_root), bundle=self.bundle
            )

        transaction_parent = self.parent / "transaction-parent"
        transaction_parent.mkdir()
        root = transaction_parent / "launch"
        value = self.manifest(root)
        fired = False

        def exchange_parent(stage: str) -> None:
            nonlocal fired
            if stage == "before_root_mkdir" and not fired:
                fired = True
                transaction_parent.rename(self.parent / "held-transaction-parent")
                transaction_parent.mkdir()

        with mock.patch.object(
            launcher, "_publication_barrier", side_effect=exchange_parent
        ), self.assertRaises(launcher.DecodedEvaluationLauncherError):
            launcher.publish_launch_manifest(value, bundle=self.bundle)
        self.assertTrue(fired)

    def test_deterministic_barriers_fail_closed_for_hostile_name_changes(self) -> None:
        def root_collision(root: pathlib.Path) -> None:
            root.mkdir()

        def root_symlink(root: pathlib.Path) -> None:
            target = root.with_name(root.name + "-symlink-target")
            target.mkdir()
            root.symlink_to(target, target_is_directory=True)

        def root_exchange(root: pathlib.Path) -> None:
            root.rename(root.with_name(root.name + "-held"))
            root.mkdir()

        def manifest_collision(root: pathlib.Path) -> None:
            (root / launcher.FILENAME).write_bytes(b"hostile collision\n")

        def manifest_symlink(root: pathlib.Path) -> None:
            target = root.parent / (root.name + "-manifest-symlink-target")
            target.write_bytes(b"hostile symlink target\n")
            (root / launcher.FILENAME).symlink_to(target)

        def manifest_hardlink(root: pathlib.Path) -> None:
            os.link(
                root / launcher.FILENAME,
                root.parent / (root.name + "-external-hardlink"),
            )

        def manifest_exchange(root: pathlib.Path) -> None:
            manifest = root / launcher.FILENAME
            manifest.rename(root / (launcher.FILENAME + "-held"))
            manifest.write_bytes(b"hostile replacement\n")
            manifest.chmod(0o400)

        def sealed_manifest_exchange(root: pathlib.Path) -> None:
            root.chmod(0o700)
            try:
                manifest_exchange(root)
            finally:
                root.chmod(0o555)

        cases = (
            ("root-collision", "before_root_mkdir", root_collision),
            ("root-symlink", "before_root_mkdir", root_symlink),
            ("root-exchange", "after_root_open", root_exchange),
            ("manifest-collision", "before_manifest_create", manifest_collision),
            ("manifest-symlink", "before_manifest_create", manifest_symlink),
            ("manifest-hardlink", "after_manifest_sync", manifest_hardlink),
            ("manifest-exchange", "after_manifest_sync", manifest_exchange),
            (
                "sealed-manifest-exchange",
                "after_seal_before_final_replay",
                sealed_manifest_exchange,
            ),
        )
        for case_name, hostile_stage, attack in cases:
            with self.subTest(case=case_name):
                root = self.parent / ("hostile-" + case_name)
                value = self.manifest(root)
                fired = False

                def barrier(stage: str) -> None:
                    nonlocal fired
                    if not fired and stage == hostile_stage:
                        fired = True
                        attack(root)

                with mock.patch.object(
                    launcher, "_publication_barrier", side_effect=barrier
                ), self.assertRaises(launcher.DecodedEvaluationLauncherError):
                    launcher.publish_launch_manifest(value, bundle=self.bundle)
                self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
