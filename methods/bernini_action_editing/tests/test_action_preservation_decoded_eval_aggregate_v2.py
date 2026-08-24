from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_aggregate_v2 as aggregate
import action_preservation_decoded_eval_decoder_adapter_v1 as decoder
import action_preservation_decoded_eval_executor_v2 as executor
import action_preservation_decoded_eval_model_authority_v2 as authority
import action_preservation_decoded_eval_plan_v1 as plan


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def seal(row: dict, field: str) -> dict:
    row.pop(field, None)
    row[field] = aggregate.object_sha256(row)
    return row


def identity(*, directory: bool, mode: int, inode: int, size: int = 7) -> dict:
    return {
        "device": 9,
        "inode": inode,
        "uid": 1001,
        "gid": 1002,
        "mode": (stat.S_IFDIR if directory else stat.S_IFREG) | mode,
        "nlink": 2 if directory else 1,
        "rdev": 0,
        "size": 0 if directory else size,
        "blocks": 8,
        "mtime_ns": 100000 + inode,
        "ctime_ns": 200000 + inode,
    }


def capture(*, adapter: bool, task_id: str = "candidate-A") -> dict:
    root = Path("/dead/checkpoint" if adapter else "/dead/model")
    view = Path("/dead/adapter-view" if adapter else "/dead/model-view")
    relatives = (
        authority.ADAPTER_RELATIVE_FILES
        if adapter
        else authority.MODEL_RELATIVE_FILES
    )
    directories = (
        authority.ADAPTER_RELATIVE_DIRECTORIES
        if adapter
        else authority.MODEL_RELATIVE_DIRECTORIES
    )
    first_fd = 100 if adapter else 10
    file_mode = 0o444 if adapter else 0o644
    files = []
    links = {}
    for index, relative in enumerate(relatives):
        fd = first_fd + index
        target = f"/dev/fd/{fd}"
        links[relative] = target
        files.append(
            {
                "relative_path": relative,
                "path": str(root / relative),
                "sha256": sha(f"file:{adapter}:{relative}"),
                "identity": identity(
                    directory=False,
                    mode=file_mode,
                    inode=1000 + fd,
                    size=20 + index,
                ),
                "authority_fd": fd,
                "proc_fd_path": target,
            }
        )
    source_directories = []
    view_directories = []
    directory_fd = 200 if adapter else 300
    for index, relative in enumerate(directories):
        source_directories.append(
            {
                "relative_path": relative,
                "path": str(root if relative == "." else root / relative),
                "authority_fd": directory_fd + index,
                "identity": identity(
                    directory=True, mode=0o755, inode=2000 + directory_fd + index
                ),
            }
        )
        view_directories.append(
            {
                "relative_path": relative,
                "path": str(view if relative == "." else view / relative),
                "authority_fd": directory_fd + 50 + index,
                "identity": identity(
                    directory=True,
                    mode=0o700,
                    inode=3000 + directory_fd + index,
                ),
            }
        )
    common = {
        "schema_version": (
            authority.ADAPTER_CAPTURE_SCHEMA
            if adapter
            else authority.MODEL_CAPTURE_SCHEMA
        ),
        "executor_pid": 4242,
        "file_count": len(files),
        "source_directory_count": len(source_directories),
        "view_directory_count": len(view_directories),
        "private_parent": {
            "relative_path": ".",
            "path": str(view.parent),
            "authority_fd": 900 if adapter else 901,
            "identity": identity(
                directory=True, mode=0o700, inode=9900
            ),
        },
        "private_root_name": view.name,
        "view_created_only_via_held_parent_fd": True,
        "files": files,
        "source_directories": source_directories,
        "view_directories": view_directories,
        "view_links": links,
        "files_digest": aggregate.object_sha256(files),
        "source_directories_digest": aggregate.object_sha256(source_directories),
        "view_directories_digest": aggregate.object_sha256(view_directories),
        "view_links_digest": aggregate.object_sha256(links),
        "same_fd_double_hash_complete": True,
        "full_identity_captured": True,
        "file_and_directory_fds_retained": True,
        "fd_view_leaf_target_kind": "injected_test_fd_prefix",
    }
    if adapter:
        row = {
            **common,
            "task_id": task_id,
            "checkpoint_root": str(root),
            "adapter_view_root": str(view),
            "safetensors_consumption_path": str(
                view / "adapter/adapter_model.safetensors"
            ),
            "safetensors_consumption_is_explicit_executor_proc_fd_view": False,
        }
    else:
        ordered = [
            {"relative_path": item["relative_path"], "sha256": item["sha256"]}
            for item in files
        ]
        row = {
            **common,
            "model_root": str(root),
            "model_view_root": str(view),
            "manifest": {
                "path": "/dead/model.sha256",
                "sha256": authority.MODEL_MANIFEST_SHA256,
                "identity": identity(
                    directory=False, mode=0o644, inode=9000, size=100
                ),
                "row_count": authority.MODEL_FILE_COUNT,
                "ordered_rows_digest": aggregate.object_sha256(ordered),
            },
            "expected_uid": 1001,
            "expected_gid": 1002,
            "expected_device": 9,
            "expected_file_mode": 0o644,
        }
    row["initial_replay_digest"] = aggregate._expected_replay_digest(
        row,
        stage=f"adapter_capture:{task_id}" if adapter else "holder_capture",
        adapter=adapter,
    )
    return seal(row, "capture_digest")


def use(
    *, task_id: str, phase: str, capture: dict,
    pre_use_digest: str | None = None,
) -> dict:
    adapter = phase.startswith("adapter_")
    stage_by_phase = {
        "pre_use": f"task_pre:{task_id}",
        "post_use": f"task_post:{task_id}",
        "adapter_pre_use": f"adapter_pre:{task_id}",
        "adapter_post_use": f"adapter_post:{task_id}",
    }
    current_parent = capture["private_parent"]["identity"]
    row = {
        "schema_version": authority.MODEL_REPLAY_SCHEMA,
        "task_id": task_id,
        "phase": phase,
        "adapter_capture_digest" if adapter else "model_capture_digest": (
            capture["capture_digest"]
        ),
        "replay_digest": aggregate._expected_replay_digest(
            capture,
            stage=stage_by_phase[phase],
            adapter=adapter,
            private_parent_current_identity=current_parent,
        ),
        "private_parent_current_identity": current_parent,
    }
    if pre_use_digest is not None:
        row["pre_use_digest"] = pre_use_digest
    return seal(row, "use_digest")


def final_adapter(
    *, task_id: str, capture: dict, post_use_digest: str
) -> dict:
    current_parent = capture["private_parent"]["identity"]
    return seal(
        {
            "schema_version": authority.ADAPTER_FINAL_SCHEMA,
            "task_id": task_id,
            "adapter_capture_digest": capture["capture_digest"],
            "post_use_digest": post_use_digest,
            "final_rehash_digest": aggregate._expected_final_rehash_digest(
                capture,
                stage=f"adapter_final:{task_id}",
                adapter=True,
                private_parent_current_identity=current_parent,
            ),
            "private_parent_current_identity": current_parent,
            "all_adapter_bytes_rehashed_after_decoder_exit": True,
            "all_adapter_file_and_directory_fds_retained_through_rehash": True,
        },
        "adapter_final_digest",
    )


def fixture(*, adapter_enabled: bool) -> dict:
    task_id = "candidate-A" if adapter_enabled else "control-A"
    task_kind = "adapter_candidate" if adapter_enabled else "frozen_base_control"
    model = capture(adapter=False, task_id=task_id)
    adapter = capture(adapter=True, task_id=task_id) if adapter_enabled else None
    task_publication_root = {
        "fd": 999,
        "path": f"/evidence/{task_id}",
        "identity": identity(
            directory=True, mode=0o700, inode=9999
        ),
    }
    model_capture_path = Path(task_publication_root["path"]) / "model-capture.json"
    model_capture_sha = sha("model capture file")
    adapter_capture_path = (
        Path(f"/evidence/{task_id}/adapter-capture.json")
        if adapter_enabled
        else None
    )
    adapter_capture_sha = sha("adapter capture file") if adapter_enabled else None
    model_pre = use(
        task_id=task_id,
        phase="pre_use",
        capture=model,
    )
    adapter_pre = (
        use(
            task_id=task_id,
            phase="adapter_pre_use",
            capture=adapter,
        )
        if adapter is not None
        else None
    )
    inherited = aggregate._expected_fd_binding(
        task_id=task_id,
        model_capture=model,
        adapter_capture=adapter,
        task_publication_root=task_publication_root,
    )
    physical_digest = sha("physical bindings")
    d0 = {
        "schema_version": authority.CONSUMPTION_INPUT_SCHEMA,
        "task_id": task_id,
        "physical_bindings_digest": physical_digest,
        "model": {
            "capture_receipt_path": str(model_capture_path),
            "capture_receipt_sha256": model_capture_sha,
            "capture_digest": model["capture_digest"],
            "pre_use_digest": model_pre["use_digest"],
            "view_root": model["model_view_root"],
        },
        "adapter": (
            None
            if adapter is None
            else {
                "capture_receipt_path": str(adapter_capture_path),
                "capture_receipt_sha256": adapter_capture_sha,
                "capture_digest": adapter["capture_digest"],
                "pre_use_digest": adapter_pre["use_digest"],
                "view_root": adapter["adapter_view_root"],
            }
        ),
        "inherited_fds": inherited,
        "production_mode": False,
        "task_member_path_kind": "injected_named_test_root",
        "base_model_and_adapter_consumed_only_from_fd_views": True,
        "training_loss_read_or_used": False,
    }
    seal(d0, "consumption_input_digest")
    d0_path = Path(f"/evidence/{task_id}/consumption-input.json")
    d0_sha = sha("D0 file:" + task_id)
    task_input = {
        "schema_version": executor.TASK_INPUT_SCHEMA,
        "task_id": task_id,
        "task_kind": task_kind,
        "model_consumption_input": {
            "path": str(d0_path),
            "sha256": d0_sha,
            "consumption_input_digest": d0["consumption_input_digest"],
        },
    }
    seal(task_input, "input_digest")
    rank_evidence = {
        "consumption_input_digest": d0["consumption_input_digest"],
        "task_input_digest": task_input["input_digest"],
        "model_capture_digest": model["capture_digest"],
        "model_view_root": d0["model"]["view_root"],
        "adapter_capture_digest": (
            None if adapter is None else adapter["capture_digest"]
        ),
        "adapter_view_root": (
            None if adapter is None else d0["adapter"]["view_root"]
        ),
        "fd_view_files_authorized": (
            model["file_count"]
            + (0 if adapter is None else adapter["file_count"])
        ),
        "inherited_fd_binding_digest": inherited["fd_binding_digest"],
        "inherited_fd_count": inherited["fd_count"],
        "ptrace_authorization_used": False,
    }
    rank_digest = aggregate.object_sha256(rank_evidence)
    native = {
        "schema_version": decoder.INFERENCE_RECEIPT_SCHEMA,
        "consumption_input_digest": d0["consumption_input_digest"],
        "task_input_digest": task_input["input_digest"],
        "model_consumption": {
            **rank_evidence,
            "four_rank_attestation": {
                "world_size": 4,
                "all_ranks_replayed_exact_fd_views": True,
                "rank_evidence_digest": rank_digest,
                "ordered_rank_evidence_digests": [rank_digest] * 4,
            },
        },
    }
    seal(native, "receipt_digest")
    model_post = use(
        task_id=task_id,
        phase="post_use",
        capture=model,
        pre_use_digest=model_pre["use_digest"],
    )
    adapter_post = (
        use(
            task_id=task_id,
            phase="adapter_post_use",
            capture=adapter,
            pre_use_digest=adapter_pre["use_digest"],
        )
        if adapter is not None
        else None
    )
    adapter_final = (
        final_adapter(
            task_id=task_id,
            capture=adapter,
            post_use_digest=adapter_post["use_digest"],
        )
        if adapter is not None
        else None
    )
    chain = authority.build_consumption_chain(
        task_id=task_id,
        model_capture_digest=model["capture_digest"],
        model_pre_use_digest=model_pre["use_digest"],
        model_post_use_digest=model_post["use_digest"],
        adapter_capture_digest=(
            None if adapter is None else adapter["capture_digest"]
        ),
        adapter_pre_use_digest=(
            None if adapter_pre is None else adapter_pre["use_digest"]
        ),
        adapter_post_use_digest=(
            None if adapter_post is None else adapter_post["use_digest"]
        ),
        adapter_final_digest=(
            None
            if adapter_final is None
            else adapter_final["adapter_final_digest"]
        ),
        native_inference_receipt_digest=native["receipt_digest"],
        consumption_input_digest=d0["consumption_input_digest"],
    )
    fd_evidence = {
        "schema_version": executor.FD_INHERITANCE_SCHEMA,
        "fd_binding": inherited,
        "fd_binding_digest": inherited["fd_binding_digest"],
        "fd_count": inherited["fd_count"],
        "production_mode": True,
        "decoder_spawn_performed": True,
        "decoder_spawn_close_fds_true": True,
        "exact_pass_fds_only": True,
        "executor_parent_fds_cloexec_before_spawn": True,
        "executor_parent_fds_cloexec_after_wait": True,
        "unrelated_child_inherits_authority_fds": False,
        "ptrace_authorization_used": False,
        "injected_fixture": False,
    }
    seal(fd_evidence, "inheritance_digest")
    process = {
        "schema_version": executor.PROCESS_SCHEMA,
        "task_id": task_id,
        "input_digest": task_input["input_digest"],
        "consumption_digest": chain["consumption_digest"],
        "return_code": 0,
        "fd_inheritance": fd_evidence,
    }
    seal(process, "process_digest")
    staging = Path(f"/evidence/{task_id}/candidate.staging.mp4")
    video_sha = sha("video:" + task_id)
    video_size = 12345
    published_inode_identity = identity(
        directory=False, mode=0o444, inode=8000, size=video_size
    )
    published_inode_identity["nlink"] = 2
    gate = {
        "schema_version": authority.PUBLICATION_GATE_SCHEMA,
        "task_id": task_id,
        "consumption_digest": chain["consumption_digest"],
        "staging_path": str(staging),
        "staging_sha256": video_sha,
        "staging_size": video_size,
        "model_post_use_verified": True,
        "adapter_post_use_verified_or_base_control": True,
        "adapter_fds_closed_or_base_control": True,
        "publication_authorized": True,
        "publication_has_occurred": False,
    }
    seal(gate, "publication_gate_digest")
    output = {
        "schema_version": executor.TASK_OUTPUT_SCHEMA,
        "task_id": task_id,
        "task_kind": task_kind,
        "input_digest": task_input["input_digest"],
        "process_digest": process["process_digest"],
        "consumption_chain": chain,
        "consumption_digest": chain["consumption_digest"],
        "publication_gate": gate,
        "publication_gate_digest": gate["publication_gate_digest"],
        "native_inference_receipt": {"receipt_digest": native["receipt_digest"]},
        "output_relpath": f"decoded/{task_id}.mp4",
        "output_video_sha256": video_sha,
        "output_byte_size": video_size,
        "published_inode_identity": published_inode_identity,
    }
    seal(output, "output_digest")
    result = {
        "task_id": task_id,
        "status": "success",
        "terminal_receipt_digest": output["output_digest"],
        "consumption_digest": chain["consumption_digest"],
        "publication_gate_digest": gate["publication_gate_digest"],
        "output_relpath": output["output_relpath"],
    }
    seal(result, "result_digest")
    return {
        "task_id": task_id,
        "task_kind": task_kind,
        "physical_bindings_digest": physical_digest,
        "model_capture": model,
        "model_capture_path": model_capture_path,
        "model_capture_sha256": model_capture_sha,
        "model_pre": model_pre,
        "model_post": model_post,
        "adapter_capture": adapter,
        "adapter_capture_path": adapter_capture_path,
        "adapter_capture_sha256": adapter_capture_sha,
        "adapter_pre": adapter_pre,
        "adapter_post": adapter_post,
        "adapter_final": adapter_final,
        "consumption_input": d0,
        "consumption_input_path": d0_path,
        "consumption_input_sha256": d0_sha,
        "task_input": task_input,
        "native_receipt": native,
        "consumption_chain": chain,
        "process_receipt": process,
        "publication_gate": gate,
        "output_receipt": output,
        "result": result,
        "shard_summary_digest": sha("summary"),
        "staging_path": staging,
    }


def holder_rows() -> list[dict]:
    rows = []
    for index, holder in enumerate(plan.HOLDER_ROWS):
        completion_path = (
            "/evidence/"
            + plan.holder_completion_reservation_relative(holder["job_id"])
        )
        completion_anchor = {
            "schema_version": executor.HOLDER_COMPLETION_ANCHOR_SCHEMA,
            "holder_job_id": holder["job_id"],
            "completion_path": completion_path,
            "initial_inode_identity": {
                "device": 9,
                "inode": 7000 + index,
                "uid": 1001,
                "gid": 1002,
                "rdev": 0,
            },
            "completion_sha256": sha(f"holder completion file:{index}"),
            "completion_size": 123 + index,
            "completion_mode": 0o444,
            "completion_digest": sha(f"holder completion:{index}"),
            "holder_summary_digest": sha(f"summary:{index}"),
        }
        completion_anchor["anchor_digest"] = aggregate.object_sha256(
            completion_anchor
        )
        authority_row = {
            "job_id": holder["job_id"],
            "holder_completion_anchor_digest": completion_anchor[
                "anchor_digest"
            ],
            "holder_directory_completion_digest": sha(
                f"holder completion:{index}"
            ),
            "model_capture_digest": sha(f"model capture:{index}"),
            "model_final_digest": sha(f"model final:{index}"),
            "task_consumption_set_digest": sha(f"task set:{index}"),
            "ordered_chain_digests_digest": sha(f"ordered chains:{index}"),
        }
        rows.append(
            {
                "job_id": holder["job_id"],
                "node": holder["node"],
                "summary_path": f"/evidence/{holder['job_id']}/summary.json",
                "summary_sha256": sha(f"summary file:{index}"),
                "summary_digest": sha(f"summary:{index}"),
                "holder_execution_digest": sha(f"holder execution:{index}"),
                "holder_directory_completion_path": (
                    completion_path
                ),
                "holder_directory_completion_sha256": sha(
                    f"holder completion file:{index}"
                ),
                "holder_directory_completion_digest": authority_row[
                    "holder_directory_completion_digest"
                ],
                "holder_completion_anchor": completion_anchor,
                "executor_verified_release_capture": {
                    "receipt_path": f"/evidence/{holder['job_id']}/capture.json",
                    "receipt_sha256": sha(f"executor capture file:{index}"),
                    "capture_digest": sha(f"executor capture:{index}"),
                    "target": "action_preservation_decoded_eval_executor_v2.py",
                    "target_arguments_sha256": sha(f"executor arguments:{index}"),
                },
                "model_capture_path": (
                    f"/evidence/{holder['job_id']}/model-capture.json"
                ),
                "model_capture_sha256": sha(f"model capture file:{index}"),
                "model_capture_digest": authority_row["model_capture_digest"],
                "model_final_path": f"/evidence/{holder['job_id']}/model-final.json",
                "model_final_sha256": sha(f"model final file:{index}"),
                "model_final_digest": authority_row["model_final_digest"],
                "task_consumption_set_digest": authority_row[
                    "task_consumption_set_digest"
                ],
                "ordered_chain_digests_digest": authority_row[
                    "ordered_chain_digests_digest"
                ],
                "holder_authority_digest": aggregate.object_sha256(authority_row),
                "all_task_fd_inheritance_evidence_verified": True,
            }
        )
    return rows


class OfflineAuthorityAggregateTests(unittest.TestCase):
    def test_base_chain_verifies_after_fd_views_are_gone(self) -> None:
        evidence = fixture(adapter_enabled=False)
        self.assertEqual(
            evidence["consumption_input"]["inherited_fds"]["fd_count"], 25
        )
        self.assertFalse(Path(evidence["model_capture"]["model_view_root"]).exists())
        projection = aggregate.validate_offline_authority_chain(**evidence)
        self.assertEqual(
            projection["d0_consumption_input_digest"],
            evidence["consumption_input"]["consumption_input_digest"],
        )
        self.assertEqual(
            projection["consumption_digest"],
            evidence["consumption_chain"]["consumption_digest"],
        )

    def test_offline_chain_never_calls_live_view_resolvers(self) -> None:
        evidence = fixture(adapter_enabled=True)
        with mock.patch.object(
            authority,
            "load_consumption_input",
            side_effect=AssertionError("live consumption resolver called"),
        ), mock.patch.object(
            decoder,
            "resolve_request",
            side_effect=AssertionError("decoder request resolver called"),
        ):
            aggregate.validate_offline_authority_chain(**evidence)

    def test_adapter_chain_closes_exact_29_fd_allowlist(self) -> None:
        evidence = fixture(adapter_enabled=True)
        self.assertEqual(
            evidence["consumption_input"]["inherited_fds"]["fd_count"], 29
        )
        projection = aggregate.validate_offline_authority_chain(**evidence)
        self.assertRegex(projection["authority_chain_digest"], r"^[0-9a-f]{64}$")

    def test_production_chain_rejects_injected_named_roots(self) -> None:
        evidence = fixture(adapter_enabled=False)
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "production FD-view authority",
        ):
            aggregate.validate_offline_authority_chain(
                **evidence, production_required=True
            )

    def test_private_parent_immutable_swap_is_rejected(self) -> None:
        evidence = fixture(adapter_enabled=False)
        hostile = copy.deepcopy(evidence)
        hostile_parent = dict(
            hostile["model_pre"]["private_parent_current_identity"]
        )
        hostile_parent["inode"] += 1
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "private-parent immutable identity",
        ):
            aggregate._expected_replay_digest(
                hostile["model_capture"],
                stage=f"task_pre:{hostile['task_id']}",
                adapter=False,
                private_parent_current_identity=hostile_parent,
            )

    def test_mixed_consumption_digest_is_rejected(self) -> None:
        evidence = fixture(adapter_enabled=True)
        hostile = copy.deepcopy(evidence)
        hostile["process_receipt"]["consumption_digest"] = sha("other C")
        seal(hostile["process_receipt"], "process_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError, "C to process"
        ):
            aggregate.validate_offline_authority_chain(**hostile)

    def test_mixed_staging_evidence_is_rejected(self) -> None:
        evidence = fixture(adapter_enabled=False)
        hostile = copy.deepcopy(evidence)
        hostile["publication_gate"]["staging_path"] = "/other/staging.mp4"
        seal(hostile["publication_gate"], "publication_gate_digest")
        hostile["output_receipt"]["publication_gate"] = hostile[
            "publication_gate"
        ]
        hostile["output_receipt"]["publication_gate_digest"] = hostile[
            "publication_gate"
        ]["publication_gate_digest"]
        seal(hostile["output_receipt"], "output_digest")
        hostile["result"]["terminal_receipt_digest"] = hostile[
            "output_receipt"
        ]["output_digest"]
        hostile["result"]["publication_gate_digest"] = hostile[
            "publication_gate"
        ]["publication_gate_digest"]
        seal(hostile["result"], "result_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError, "staging binding"
        ):
            aggregate.validate_offline_authority_chain(**hostile)

    def test_mixed_task_adapter_evidence_is_rejected(self) -> None:
        evidence = fixture(adapter_enabled=True)
        hostile = copy.deepcopy(evidence)
        hostile["adapter_capture"]["task_id"] = "candidate-B"
        seal(hostile["adapter_capture"], "capture_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError, "adapter capture task"
        ):
            aggregate.validate_offline_authority_chain(**hostile)

    def test_model_final_requires_exact_ordered_task_set(self) -> None:
        model = capture(adapter=False)
        ordered = [sha("C0"), sha("C1")]
        current_parent = model["private_parent"]["identity"]
        final = {
            "schema_version": authority.MODEL_FINAL_SCHEMA,
            "model_capture_digest": model["capture_digest"],
            "task_count": 2,
            "task_consumption_digests": ordered,
            "task_consumption_set_digest": aggregate.object_sha256(ordered),
            "final_rehash_digest": aggregate._expected_final_rehash_digest(
                model,
                stage="holder_final",
                adapter=False,
                private_parent_current_identity=current_parent,
            ),
            "private_parent_current_identity": current_parent,
            "all_model_bytes_rehashed_after_last_task": True,
            "all_model_file_and_directory_fds_retained_through_final_rehash": True,
        }
        seal(final, "model_final_digest")
        aggregate._validate_model_final(
            final,
            model_capture=model,
            ordered_consumption_digests=ordered,
        )
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError, "ordered task set"
        ):
            aggregate._validate_model_final(
                final,
                model_capture=model,
                ordered_consumption_digests=list(reversed(ordered)),
            )

        forged = copy.deepcopy(final)
        forged["final_rehash_digest"] = sha("forged model final rehash")
        seal(forged, "model_final_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "model final rehash digest differs",
        ):
            aggregate._validate_model_final(
                forged,
                model_capture=model,
                ordered_consumption_digests=ordered,
            )

    def test_forged_capture_use_and_adapter_final_replays_are_rejected(self) -> None:
        model = capture(adapter=False)
        expected_model_files = {
            item["relative_path"]: item["sha256"] for item in model["files"]
        }
        expected_model_manifest = copy.deepcopy(model["manifest"])
        forged_model_files = copy.deepcopy(model)
        forged_model_files["files"][0]["sha256"] = sha("forged model member")
        forged_model_files["files_digest"] = aggregate.object_sha256(
            forged_model_files["files"]
        )
        forged_model_files["manifest"]["ordered_rows_digest"] = (
            aggregate.object_sha256(
                [
                    {
                        "relative_path": item["relative_path"],
                        "sha256": item["sha256"],
                    }
                    for item in forged_model_files["files"]
                ]
            )
        )
        forged_model_files["initial_replay_digest"] = (
            aggregate._expected_replay_digest(
                forged_model_files, stage="holder_capture", adapter=False
            )
        )
        seal(forged_model_files, "capture_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "model manifest/full expected file binding differs",
        ):
            aggregate._validate_capture_offline(
                forged_model_files,
                adapter=False,
                production_required=False,
                expected_model_files=expected_model_files,
                expected_model_manifest=expected_model_manifest,
            )

        forged_capture = copy.deepcopy(model)
        forged_capture["initial_replay_digest"] = sha("forged initial replay")
        seal(forged_capture, "capture_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "capture initial replay digest differs",
        ):
            aggregate._validate_capture_offline(
                forged_capture, adapter=False, production_required=False
            )

        model_pre = use(task_id="control-A", phase="pre_use", capture=model)
        forged_pre = copy.deepcopy(model_pre)
        forged_pre["replay_digest"] = sha("forged task pre replay")
        seal(forged_pre, "use_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "pre_use replay digest differs",
        ):
            aggregate._validate_use_receipt(
                forged_pre,
                task_id="control-A",
                phase="pre_use",
                capture=model,
            )

        adapter = capture(adapter=True, task_id="candidate-A")
        adapter_pre = use(
            task_id="candidate-A", phase="adapter_pre_use", capture=adapter
        )
        adapter_post = use(
            task_id="candidate-A",
            phase="adapter_post_use",
            capture=adapter,
            pre_use_digest=adapter_pre["use_digest"],
        )
        adapter_final = final_adapter(
            task_id="candidate-A",
            capture=adapter,
            post_use_digest=adapter_post["use_digest"],
        )
        adapter_final["final_rehash_digest"] = sha("forged adapter rehash")
        seal(adapter_final, "adapter_final_digest")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "adapter final rehash digest differs",
        ):
            aggregate._validate_adapter_final(
                adapter_final,
                task_id="candidate-A",
                capture=adapter,
                post_use_digest=adapter_post["use_digest"],
            )

    def test_noncanonical_stored_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "receipt.json"
            path.write_text(json.dumps({"b": 1, "a": 2}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                aggregate.DecodedEvaluationAggregateError, "canonical JSON"
            ):
                aggregate._json(path, label="hostile receipt")

    def test_sealed_publication_pair_rejects_swap_during_fd_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            staging = root / "candidate.staging.mp4"
            final = root / "candidate.mp4"
            raw = b"same-published-video-bytes"
            staging.write_bytes(raw)
            staging.chmod(0o444)
            os.link(staging, final)
            expected_identity = executor._stat_identity_row(staging.lstat())
            expected_sha = hashlib.sha256(raw).hexdigest()
            aggregate._validate_sealed_publication_pair(
                staging_path=staging,
                final_path=final,
                expected_identity=expected_identity,
                expected_sha256=expected_sha,
                expected_size=len(raw),
                label="stable fixture publication",
            )

            original_hash_fd = executor._hash_fd
            swapped = False

            def swap_names_before_hash(descriptor: int) -> tuple[str, int]:
                nonlocal swapped
                if not swapped:
                    swapped = True
                    staging.unlink()
                    final.unlink()
                    staging.write_bytes(raw)
                    staging.chmod(0o444)
                    os.link(staging, final)
                return original_hash_fd(descriptor)

            with mock.patch.object(
                executor, "_hash_fd", side_effect=swap_names_before_hash
            ), self.assertRaisesRegex(
                aggregate.DecodedEvaluationAggregateError,
                "held-FD bytes/inode or named replay differs",
            ):
                aggregate._validate_sealed_publication_pair(
                    staging_path=staging,
                    final_path=final,
                    expected_identity=expected_identity,
                    expected_sha256=expected_sha,
                    expected_size=len(raw),
                    label="hostile publication",
                )
            self.assertTrue(swapped)

    def test_holder_rows_bind_all_four_authority_digests(self) -> None:
        normalized, authorities = aggregate._validate_holder_rows(holder_rows())
        self.assertEqual(len(normalized), 4)
        self.assertEqual(len(authorities), 4)
        self.assertEqual(
            authorities[0]["ordered_chain_digests_digest"],
            normalized[0]["ordered_chain_digests_digest"],
        )

    def test_mixed_holder_authority_digest_is_rejected(self) -> None:
        hostile = holder_rows()
        hostile[2]["model_final_digest"] = sha("foreign model final")
        with self.assertRaisesRegex(
            aggregate.DecodedEvaluationAggregateError,
            "holder authority differs",
        ):
            aggregate._validate_holder_rows(hostile)


class RetainedPublicationRootTests(unittest.TestCase):
    def test_blinding_key_is_read_from_retained_work_root_with_literal_sha(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "work"
            work.mkdir(mode=0o700)
            key = work / "blinding.key"
            raw = b"k" * 32
            key.write_bytes(raw)
            key.chmod(0o400)
            root_fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            binding = {
                "path": str(work),
                "root_fd": root_fd,
                "parent_fd": parent_fd,
            }
            try:
                with mock.patch.object(
                    aggregate.bridge.verified_release,
                    "validate_inherited_work_root_binding",
                    return_value=binding,
                ) as validate:
                    observed = aggregate._read_inherited_work_root_member(
                        binding,
                        path=key,
                        expected_sha256=hashlib.sha256(raw).hexdigest(),
                        expected_mode=0o400,
                        label="blinding key",
                    )
                self.assertEqual(observed, raw)
                self.assertEqual(validate.call_count, 2)
                with mock.patch.object(
                    aggregate.bridge.verified_release,
                    "validate_inherited_work_root_binding",
                    return_value=binding,
                ), self.assertRaisesRegex(
                    aggregate.DecodedEvaluationAggregateError,
                    "same-FD replay differs",
                ):
                    aggregate._read_inherited_work_root_member(
                        binding,
                        path=key,
                        expected_sha256="0" * 64,
                        expected_mode=0o400,
                        label="blinding key",
                    )
            finally:
                os.close(root_fd)
                os.close(parent_fd)

    def _fixture(
        self, parent: Path
    ) -> tuple[Path, dict, dict, dict, list[dict], dict]:
        source = parent / "source.mp4"
        candidate = parent / "candidate.mp4"
        source.write_bytes(b"sealed-source-video")
        candidate.write_bytes(b"sealed-candidate-video")
        source.chmod(0o444)
        candidate.chmod(0o444)
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        candidate_info = candidate.stat()
        retained_candidate = aggregate._RetainedMediaFile.capture(
            candidate,
            expected_sha256=candidate_sha,
            expected_size=candidate_info.st_size,
            expected_identity=executor._stat_identity_row(candidate_info),
            expected_nlink={1},
        )
        root = parent / "aggregate"
        aggregate_row = {"evaluation_id": "aggregate-anchor-fixture"}
        aggregate_row["aggregate_digest"] = aggregate.object_sha256(
            aggregate_row
        )
        private: dict[str, object] = {}
        private["private_mapping_digest"] = aggregate.object_sha256(private)
        public: dict[str, object] = {}
        public["public_packet_digest"] = aggregate.object_sha256(public)
        outputs = [
            {
                "output_video_sha256": candidate_sha,
                "output_path": str(candidate),
                "_retained_media": retained_candidate,
            }
        ]
        source_info = source.stat()
        bindings = {
            "sources": [
                {
                    "source_video": {
                        "sha256": source_sha,
                        "path": str(source),
                        "size": source_info.st_size,
                        "mode": stat.S_IMODE(source_info.st_mode),
                        "device": source_info.st_dev,
                        "inode": source_info.st_ino,
                        "uid": source_info.st_uid,
                        "gid": source_info.st_gid,
                        "nlink": source_info.st_nlink,
                        "rdev": source_info.st_rdev,
                        "blocks": getattr(source_info, "st_blocks", 0),
                        "mtime_ns": source_info.st_mtime_ns,
                        "ctime_ns": source_info.st_ctime_ns,
                    }
                }
            ]
        }
        return root, aggregate_row, private, public, outputs, bindings

    def test_retained_media_borrows_prevalidated_parent_fd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            media = parent / "media.mp4"
            raw = b"borrowed-parent-media"
            media.write_bytes(raw)
            media.chmod(0o444)
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                retained = aggregate._RetainedMediaFile.capture(
                    media,
                    expected_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_size=len(raw),
                    expected_identity=executor._stat_identity_row(media.stat()),
                    expected_nlink={1},
                    parent_descriptor=parent_fd,
                )
                self.assertEqual(retained.parent_descriptor, parent_fd)
                self.assertFalse(retained.owns_parent_descriptor)
                retained.close()
                self.assertTrue(stat.S_ISDIR(os.fstat(parent_fd).st_mode))
            finally:
                os.close(parent_fd)

    def test_publish_content_addressed_media_deduplicates_distinct_inodes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, aggregate_row, private, public, outputs, bindings = self._fixture(
                parent
            )
            original = Path(outputs[0]["output_path"])
            duplicate = parent / "candidate-duplicate.mp4"
            duplicate.write_bytes(original.read_bytes())
            duplicate.chmod(0o444)
            duplicate_info = duplicate.stat()
            duplicate_retained = aggregate._RetainedMediaFile.capture(
                duplicate,
                expected_sha256=outputs[0]["output_video_sha256"],
                expected_size=duplicate_info.st_size,
                expected_identity=executor._stat_identity_row(duplicate_info),
                expected_nlink={1},
            )
            outputs.append(
                {
                    "output_video_sha256": outputs[0]["output_video_sha256"],
                    "output_path": str(duplicate),
                    "_retained_media": duplicate_retained,
                }
            )
            aggregate.publish(
                aggregate_root=root,
                aggregate=aggregate_row,
                private=private,
                public=public,
                outputs=outputs,
                bindings=bindings,
            )
            self.assertEqual(
                len(list((root / aggregate.MEDIA_DIRECTORY).iterdir())), 2
            )

    def test_publish_uses_retained_directories_and_seals_exact_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, aggregate_row, private, public, outputs, bindings = self._fixture(
                parent
            )
            output = aggregate.publish(
                aggregate_root=root,
                aggregate=aggregate_row,
                private=private,
                public=public,
                outputs=outputs,
                bindings=bindings,
            )
            self.assertEqual(output, root / aggregate.AGGREGATE_FILENAME)
            self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o555)
            media = root / aggregate.MEDIA_DIRECTORY
            self.assertEqual(stat.S_IMODE(media.lstat().st_mode), 0o555)
            self.assertEqual(
                {item.name for item in root.iterdir()},
                {
                    aggregate.MEDIA_DIRECTORY,
                    aggregate.PRIVATE_FILENAME,
                    aggregate.PUBLIC_FILENAME,
                    aggregate.AGGREGATE_FILENAME,
                },
            )
            self.assertEqual(len(list(media.iterdir())), 2)
            self.assertTrue(
                all(item.lstat().st_nlink == 1 for item in media.iterdir())
            )
            self.assertEqual(
                stat.S_IMODE((root / aggregate.PRIVATE_FILENAME).lstat().st_mode),
                0o400,
            )

    def test_root_same_name_replacement_cannot_redirect_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, aggregate_row, private, public, outputs, bindings = self._fixture(
                parent
            )
            original = aggregate._HeldPublicationDirectory.mkdir
            attacked = False

            def replace_before_first_child(
                held: aggregate._HeldPublicationDirectory,
                name: str,
                *,
                mode: int = 0o700,
            ) -> aggregate._HeldPublicationDirectory:
                nonlocal attacked
                if not attacked:
                    attacked = True
                    moved = held.path.with_name(held.path.name + ".moved")
                    held.path.rename(moved)
                    held.path.mkdir(mode=0o700)
                return original(held, name, mode=mode)

            with mock.patch.object(
                aggregate._HeldPublicationDirectory,
                "mkdir",
                replace_before_first_child,
            ), self.assertRaisesRegex(
                aggregate.DecodedEvaluationAggregateError,
                "identity or closure differs",
            ):
                aggregate.publish(
                    aggregate_root=root,
                    aggregate=aggregate_row,
                    private=private,
                    public=public,
                    outputs=outputs,
                    bindings=bindings,
                )
            self.assertTrue(attacked)
            self.assertEqual(list(root.iterdir()), [])

    def test_media_root_replacement_cannot_receive_copied_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root, aggregate_row, private, public, outputs, bindings = self._fixture(
                parent
            )
            original = aggregate._RetainedMediaFile.copy_to
            attacked = False

            def replace_media(
                retained: aggregate._RetainedMediaFile,
                *,
                destination_directory: aggregate._HeldPublicationDirectory,
                basename: str,
            ) -> None:
                nonlocal attacked
                if not attacked:
                    attacked = True
                    moved = destination_directory.path.with_name("media.moved")
                    destination_directory.path.rename(moved)
                    destination_directory.path.mkdir(mode=0o700)
                original(
                    retained,
                    destination_directory=destination_directory,
                    basename=basename,
                )

            with mock.patch.object(
                aggregate._RetainedMediaFile,
                "copy_to",
                replace_media,
            ), self.assertRaisesRegex(
                aggregate.DecodedEvaluationAggregateError,
                "identity or closure differs",
            ):
                aggregate.publish(
                    aggregate_root=root,
                    aggregate=aggregate_row,
                    private=private,
                    public=public,
                    outputs=outputs,
                    bindings=bindings,
                )
            self.assertTrue(attacked)
            self.assertEqual(list((root / aggregate.MEDIA_DIRECTORY).iterdir()), [])

    def test_aggregate_root_is_created_below_inherited_work_fd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "work"
            work.mkdir(mode=0o700)
            root, aggregate_row, private, public, outputs, bindings = self._fixture(
                work
            )
            root_fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            binding = {
                "path": str(work),
                "root_fd": root_fd,
                "parent_fd": parent_fd,
            }
            anchors: list[dict] = []
            try:
                with mock.patch.object(
                    aggregate.bridge.verified_release,
                    "validate_inherited_work_root_binding",
                    return_value=binding,
                ):
                    output = aggregate.publish(
                        aggregate_root=root,
                        aggregate=aggregate_row,
                        private=private,
                        public=public,
                        outputs=outputs,
                        bindings=bindings,
                        work_root_binding=binding,
                        completion_anchor_sink=anchors.append,
                    )
                self.assertEqual(output, root / aggregate.AGGREGATE_FILENAME)
                self.assertEqual(len(anchors), 1)
                anchor = aggregate.validate_aggregate_completion_anchor(
                    anchors[0]
                )
                self.assertEqual(
                    anchor["aggregate_root"], str(root)
                )
                self.assertEqual(anchor["media_file_count"], 2)
            finally:
                os.close(root_fd)
                os.close(parent_fd)

    def test_inherited_work_root_replacement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "work"
            work.mkdir(mode=0o700)
            root_fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            binding = {
                "path": str(work),
                "root_fd": root_fd,
                "parent_fd": parent_fd,
            }
            moved = parent / "work.moved"
            work.rename(moved)
            work.mkdir(mode=0o700)
            try:
                with mock.patch.object(
                    aggregate.bridge.verified_release,
                    "validate_inherited_work_root_binding",
                    return_value=binding,
                ), self.assertRaisesRegex(
                    aggregate.DecodedEvaluationAggregateError,
                    "parent identity differs",
                ):
                    aggregate._HeldPublicationDirectory.create_root_from_work_binding(
                        work / "aggregate", work_root_binding=binding
                    )
            finally:
                os.close(root_fd)
                os.close(parent_fd)


if __name__ == "__main__":
    unittest.main()
