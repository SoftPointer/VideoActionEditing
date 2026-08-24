from __future__ import annotations

import copy
import hashlib
import json
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

import action_preservation_decoded_eval_bridge_v1 as bridge
import action_preservation_decoded_eval_plan_v1 as plan


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write(path: pathlib.Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_json(path: pathlib.Path, value: dict, *, newline: bool = True) -> str:
    raw = bridge.canonical_json_bytes(value) + (b"\n" if newline else b"")
    return write(path, raw)


def producer_audit_bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def action_contract(iid: str) -> dict:
    description = f"Complete the fitted action for source {iid}, then hold the terminal pose."
    row = {
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
    row["contract_digest"] = plan.object_sha256(row)
    return row


class PhysicalBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = pathlib.Path(self.temporary.name).resolve()
        deployment_root = self.parent / "deployment-work"
        deployment_root.mkdir(mode=0o700)
        deployment_root.chmod(0o700)

        def directory_identity(info: os.stat_result) -> dict[str, int]:
            return {
                "device": info.st_dev, "inode": info.st_ino,
                "uid": info.st_uid, "gid": info.st_gid,
                "mode": info.st_mode, "nlink": info.st_nlink,
                "rdev": info.st_rdev, "size": info.st_size,
                "blocks": getattr(info, "st_blocks", 0),
                "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            }

        creation = directory_identity(deployment_root.stat())
        parent_identity = directory_identity(self.parent.stat())
        immutable_fields = ("device", "inode", "uid", "gid", "mode", "rdev")
        work_root_authority = {
            "schema_version": bridge.verified_release.WORK_ROOT_AUTHORITY_SCHEMA,
            "path": str(deployment_root),
            "parent_path": str(self.parent),
            "creation_identity": creation,
            "immutable_identity": {
                key: creation[key] for key in immutable_fields
            },
            "parent_immutable_identity": {
                key: parent_identity[key] for key in immutable_fields
            },
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        work_root_authority["authority_digest"] = bridge.object_sha256(
            work_root_authority
        )
        deployment_value = {
            "work_root_authority": work_root_authority,
            "receipt_digest": digest("test deployment receipt object"),
        }
        deployment_path = deployment_root / "deployment-receipt.json"
        deployment_sha = write_json(deployment_path, deployment_value)
        deployment_path.chmod(0o444)
        source_authority_value = {
            "work_root_authority": work_root_authority,
            "deployment_receipt_digest": deployment_value["receipt_digest"],
            "receipt_digest": digest("test source authority object"),
        }
        source_authority_path = deployment_root / "source-authority.json"
        source_authority_sha = write_json(
            source_authority_path, source_authority_value
        )
        source_authority_path.chmod(0o444)
        self.deployment_authority = {
            "schema_version": bridge.DEPLOYMENT_AUTHORITY_SCHEMA,
            "work_root_authority": work_root_authority,
            "deployment_receipt": {
                "path": str(deployment_path), "sha256": deployment_sha,
            },
            "source_spec_authority": {
                "path": str(source_authority_path),
                "sha256": source_authority_sha,
            },
            "deployment_receipt_digest": deployment_value["receipt_digest"],
            "source_spec_authority_digest": source_authority_value[
                "receipt_digest"
            ],
        }
        self.deployment_authority["authority_digest"] = bridge.object_sha256(
            self.deployment_authority
        )
        self.experiment = self.parent / "training"
        self.experiment.mkdir()
        (self.experiment / "logs").mkdir()
        (self.experiment / "runs").mkdir()

        receipt_rows = []
        for arm in plan.ARMS:
            for step in plan.CHECKPOINT_STEPS:
                root = self.experiment / "runs" / arm / f"checkpoint-{step:08d}"
                (root / "adapter").mkdir(parents=True)
                receipt = {
                    "training_contract": {"arm": arm},
                    "global_step": step,
                }
                receipt["receipt_digest"] = bridge.object_sha256(receipt)
                receipt_sha = write_json(root / "receipt.json", receipt)
                adapter_sha = write(
                    root / "adapter" / "adapter_model.safetensors",
                    f"adapter:{arm}:{step}".encode(),
                )
                config_sha = write_json(
                    root / "adapter" / "adapter_config.json", {"arm": arm}
                )
                optimizer_sha = write(
                    root / "optimizer.pt", f"optimizer:{arm}:{step}".encode()
                )
                receipt_rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "receipt_sha256": receipt_sha,
                        "adapter_sha256": adapter_sha,
                        "adapter_config_sha256": config_sha,
                        "optimizer_sha256": optimizer_sha,
                        "loss": 1.0,
                        "preclip_gradient_norm": 0.0 if step == 0 else 1.0,
                    }
                )
        audit = {
            "training_audit_go": True,
            "arm_count": 8,
            "checkpoint_count": 32,
            "checkpoint_steps": list(plan.CHECKPOINT_STEPS),
            "route_scopes": ["all_attention", "cross_attn2_qo"],
            "initialization_digest_by_scope": {},
            "checkpoint_zero_adapter_sha256_by_scope": {},
            "adapter_config_sha256_by_scope": {},
            "receipt_rows": receipt_rows,
            "decoded_evaluation_complete": False,
            "scientific_promotion_authorized": False,
        }
        self.audit = audit
        audit_sha = write(
            self.experiment / "logs" / "training-audit.json",
            producer_audit_bytes(audit),
        )
        pin_root = self.parent / "pins"
        pin_root.mkdir()
        self.pin_paths = {
            "source_manifest": pin_root / "source-manifest.json",
            "adapter_release_manifest": pin_root / "adapter-release.json",
            "model_release_manifest": pin_root / "model-release.json",
            "inference_release_manifest": self.parent / "eval-release-authority" / bridge.EVAL_RELEASE_MANIFEST_FILENAME,
            "inference_config": pin_root / "inference-config.json",
            "source_preprocessing": pin_root / "source-preprocessing.json",
            "calibration": pin_root / "calibration.json",
        }
        self.pin_shas = {
            key: write_json(path, {"authority": key})
            for key, path in self.pin_paths.items()
            if key not in {"inference_release_manifest", "source_preprocessing"}
        }
        release_root = self.parent / "eval-release"
        release_member_root = release_root / bridge.EVAL_RELEASE_MEMBER_ROOT
        release_authority_root = self.parent / "eval-release-authority"
        release_authority_root.mkdir()
        self.release_root = release_root
        self.release_member_root = release_member_root
        (release_member_root / "tools").mkdir(parents=True)
        release_rows = []
        release_payloads = {}
        for relative in bridge.EVAL_RELEASE_MEMBERS:
            member = release_member_root / relative
            payload = f"sealed eval member:{relative}\n".encode()
            release_payloads[relative] = payload
            member_sha = write(member, payload)
            mode = (
                0o555
                if relative
                == "action_preservation_decoded_eval_decoder_adapter_v1.py"
                else 0o444
            )
            member.chmod(mode)
            release_rows.append(
                {
                    "path": relative,
                    "sha256": member_sha,
                    "size": member.stat().st_size,
                    "mode": mode,
                }
            )
        release_manifest = {
            "schema_version": bridge.EVAL_RELEASE_SCHEMA,
            "release_generation": bridge.verified_release.RELEASE_GENERATION,
            "archive_format": bridge.verified_release.ARCHIVE_FORMAT,
            "member_root": bridge.verified_release.MEMBER_ROOT,
            "exact_member_closure": True,
            "file_count": len(bridge.EVAL_RELEASE_MEMBERS),
            "files": release_rows,
            "content_revision": bridge.verified_release.content_revision(
                release_rows
            ),
            "allowed_entrypoints": sorted(
                bridge.verified_release.ALLOWED_PYTHON_TARGETS
            ),
            "authority": bridge.verified_release.AUTHORITY,
            "component_sha256": {
                item["path"]: item["sha256"] for item in release_rows
            },
        }
        release_manifest["manifest_digest"] = bridge.object_sha256(
            release_manifest
        )
        self.pin_shas["inference_release_manifest"] = write_json(
            self.pin_paths["inference_release_manifest"], release_manifest
        )
        self.pin_paths["inference_release_manifest"].chmod(0o444)
        release_archive = release_authority_root / "source.tar"
        release_archive_sha = write(
            release_archive,
            bridge.verified_release.fixed_ustar_archive(
                release_rows, release_payloads
            ),
        )
        release_archive.chmod(0o444)
        release_envelope = release_authority_root / "deployment-envelope.json"
        release_envelope_value = {
            "schema_version": bridge.verified_release.ENVELOPE_SCHEMA,
            "release_generation": bridge.verified_release.RELEASE_GENERATION,
            "remote_release_exact_entries": [
                "deployment-envelope.json", "source.manifest.json", "source.tar"
            ],
            "source_archive": {
                "basename": "source.tar", "sha256": release_archive_sha,
                "mode": 0o444,
            },
            "source_manifest": {
                "basename": "source.manifest.json",
                "sha256": self.pin_shas["inference_release_manifest"],
                "manifest_digest": release_manifest["manifest_digest"],
                "content_revision": release_manifest["content_revision"],
                "file_count": len(bridge.EVAL_RELEASE_MEMBERS),
                "mode": 0o444,
            },
            "create_only_deployment_required": True,
            "fresh_materialized_root_required": True,
            "verified_runtime_required": True,
            "detached_controller_authority_receipt_required": True,
            "automatic_scientific_promotion_authorized": False,
        }
        release_envelope_value["envelope_digest"] = bridge.object_sha256(
            release_envelope_value
        )
        release_envelope_sha = write_json(
            release_envelope, release_envelope_value
        )
        release_envelope.chmod(0o444)
        self.release_archive = release_archive
        self.release_archive_sha = release_archive_sha
        self.release_envelope = release_envelope
        self.release_envelope_sha = release_envelope_sha
        self.release_manifest = release_manifest
        self.trusted_release_patch = mock.patch.object(
            bridge.verified_release, "TRUSTED_EXACT15", {}
        )
        self.trusted_release_patch.start()
        self.addCleanup(self.trusted_release_patch.stop)
        for directory in (
            release_member_root / "tools", release_member_root,
            release_root / "methods", release_root,
        ):
            directory.chmod(0o555)
        self.source_manifest_sha = self.pin_shas["source_manifest"]
        self.release_manifest_sha = self.pin_shas["adapter_release_manifest"]
        self.source_revision = hashlib.sha1(b"source revision").hexdigest()
        self.source_archive_sha = digest("source archive")
        completion = {
            "schema_version": "bernini-action-preservation-v2-training-complete-v3",
            "seed": 20260818,
            "cache_sha256": digest("cache"),
            "source_archive_sha256": self.source_archive_sha,
            "source_revision": self.source_revision,
            "source_data_manifest_sha256": self.source_manifest_sha,
            "source_data_manifest_digest": digest("source manifest digest"),
            "release_manifest_sha256": self.release_manifest_sha,
            "controller_sha256": digest("controller"),
            "deployment_envelope_sha256": digest("envelope"),
            "cache_audit_sha256": digest("cache audit"),
            "training_audit_sha256": audit_sha,
            "cache_receipt_sha256": digest("cache receipt"),
            "retained_tree_digest": digest("retained tree"),
            "retained_tree_file_count": 1,
            "retained_tree_stable_double_read_before_commit": True,
            "retained_tree_held_fd_identity_replay": True,
            "optimizer_updates_per_arm": 20,
            "arm_count": 8,
            "decoded_evaluation_complete": False,
            "scientific_promotion_authorized": False,
            "parent_allocations_cancelled": False,
            "automatic_retry": False,
        }
        completion["completion_digest"] = bridge.object_sha256(completion)
        self.completion = completion
        self.completion_sha = write_json(
            self.experiment / "TRAINING_COMPLETE.json", completion, newline=False
        )

        runtime_root = self.parent / "runtime"
        runtime_root.mkdir()
        for name in ("bernini", "veomni", "model"):
            (runtime_root / name).mkdir()
        root_python_sha = write(runtime_root / "root-python", b"root-python")
        python_sha = write(runtime_root / "python", b"python")
        infer_path = release_member_root / "infer_lora.py"
        decoder_path = release_member_root / (
            "action_preservation_decoded_eval_decoder_adapter_v1.py"
        )
        infer_sha = bridge._stable_file(infer_path, label="fixture infer")[1]["sha256"]
        decoder_sha = bridge._stable_file(decoder_path, label="fixture decoder")[1]["sha256"]
        ffprobe_sha = write(runtime_root / "ffprobe", b"ffprobe")
        for executable in (
            runtime_root / "root-python", runtime_root / "python",
            runtime_root / "ffprobe",
        ):
            executable.chmod(0o755)
        site_packages = runtime_root / "site-packages"
        torchrun_path = site_packages / "torch" / "distributed" / "run.py"
        torchrun_sha = write(torchrun_path, b"# captured fixture torchrun\n")
        handler_path = (
            site_packages
            / bridge.verified_release.TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
        )
        handler_sha = write(
            handler_path, b"# captured fixture subprocess handler\n"
        )
        for patcher in (
            mock.patch.object(
                bridge.verified_release, "TORCHRUN_SOURCE_SHA256", torchrun_sha
            ),
            mock.patch.object(
                bridge.verified_release,
                "TORCHRUN_SOURCE_SIZE",
                torchrun_path.stat().st_size,
            ),
            mock.patch.object(
                bridge.verified_release,
                "TORCHRUN_SUBPROCESS_HANDLER_SHA256",
                handler_sha,
            ),
            mock.patch.object(
                bridge.verified_release,
                "TORCHRUN_SUBPROCESS_HANDLER_SIZE",
                handler_path.stat().st_size,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        controller_path = runtime_root / "detached-eval-controller.py"
        controller_sha = write(
            controller_path, b"# detached fixture controller\n"
        )
        controller_path.chmod(0o444)
        release_binding = bridge.load_eval_release_manifest(
            self.pin_paths["inference_release_manifest"],
            expected_sha256=self.pin_shas["inference_release_manifest"],
            release_root=release_root,
            archive_path=release_archive,
            expected_archive_sha256=release_archive_sha,
            envelope_path=release_envelope,
            expected_envelope_sha256=release_envelope_sha,
            expected_content_revision=release_manifest["content_revision"],
            expected_manifest_digest=release_manifest["manifest_digest"],
            expected_envelope_digest=release_envelope_value["envelope_digest"],
            verify_files=True,
        )
        root_python_binding = bridge.verified_release.capture_executable_binding(
            runtime_root / "root-python", label="fixture root Python"
        )
        frozen_python_binding = bridge.verified_release.capture_executable_binding(
            runtime_root / "python", label="fixture frozen Python"
        )
        torchrun_binding = bridge.verified_release.capture_torchrun_binding(
            site_packages, label="fixture torchrun"
        )
        controller_binding = bridge.verified_release.capture_file_binding(
            controller_path, label="fixture detached controller",
            expected_sha256=controller_sha, expected_mode=0o444,
        )
        controller_authority = (
            bridge.verified_release.publish_controller_authority_receipt(
                runtime_root / "controller-authority.json",
                controller_binding=controller_binding,
                root_python_binding=root_python_binding,
                frozen_python_binding=frozen_python_binding,
                site_packages_binding=torchrun_binding["site_packages"],
                release_binding=bridge.eval_release_runtime_binding(
                    release_binding
                ),
                torchrun_binding=torchrun_binding,
            )
        )
        self.root_python_patch = mock.patch.object(
            bridge, "ROOT_PYTHON_PATH", runtime_root / "root-python"
        )
        self.root_python_uid_patch = mock.patch.object(
            bridge, "ROOT_PYTHON_UID", os.getuid()
        )
        self.root_python_gid_patch = mock.patch.object(
            bridge, "ROOT_PYTHON_GID", os.getgid()
        )
        self.ffprobe_path_patch = mock.patch.object(
            bridge, "FFPROBE_PATH", runtime_root / "ffprobe"
        )
        self.ffprobe_uid_patch = mock.patch.object(
            bridge, "FFPROBE_UID", os.getuid()
        )
        self.ffprobe_gid_patch = mock.patch.object(
            bridge, "FFPROBE_GID", os.getgid()
        )
        for patcher in (
            self.root_python_patch, self.root_python_uid_patch,
            self.root_python_gid_patch, self.ffprobe_path_patch,
            self.ffprobe_uid_patch, self.ffprobe_gid_patch,
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        sources = []
        for index, iid in enumerate(plan.FITTED_IIDS):
            video = self.parent / "sources" / iid / "source.mp4"
            receipt = self.parent / "sources" / iid / "receipt.json"
            video_sha = write(video, f"video:{iid}".encode())
            receipt_sha = write_json(receipt, {"iid": iid})
            instruction = f"Perform the fitted action for source {iid}."
            sources.append(
                {
                    "iid": iid,
                    "source_video_path": str(video),
                    "source_video_sha256": video_sha,
                    "source_receipt_path": str(receipt),
                    "source_receipt_sha256": receipt_sha,
                    "instruction": instruction,
                    "instruction_sha256": plan.text_sha256(instruction),
                    "action_review_contract": action_contract(iid),
                    "seed": 2026081801 + index,
                }
            )
        source_preprocessing = {
            "schema_version": bridge.SOURCE_PREPROCESSING_AUTHORITY_SCHEMA,
            "serialization": bridge.SOURCE_PREPROCESSING_SERIALIZATION,
            "source_manifest_sha256": self.source_manifest_sha,
            "source_manifest_digest": (
                "2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503"
            ),
            "source_order": list(plan.FITTED_IIDS),
            "sources": copy.deepcopy(sources),
            "source_video_bytes_consumed_directly": True,
            "precomputed_transformed_source_artifact_used": False,
            "runtime_decode_bound_by_inference_release": True,
            "target_video_available_to_inference": False,
            "training_loss_read_or_used_for_selection": False,
            "remote_launch_performed": False,
            "scientific_promotion_authorized": False,
        }
        source_preprocessing["authority_digest"] = bridge.object_sha256(
            source_preprocessing
        )
        self.source_preprocessing = source_preprocessing
        self.pin_shas["source_preprocessing"] = write_json(
            self.pin_paths["source_preprocessing"], source_preprocessing
        )
        for patcher in (
            mock.patch.object(
                bridge,
                "SOURCE_PREPROCESSING_AUTHORITY_SHA256",
                self.pin_shas["source_preprocessing"],
            ),
            mock.patch.object(
                bridge,
                "SOURCE_PREPROCESSING_AUTHORITY_DIGEST",
                source_preprocessing["authority_digest"],
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        pins = {key: digest(key) for key in plan.PIN_FIELDS}
        pins["source_manifest_sha256"] = self.source_manifest_sha
        pins["adapter_release_manifest_sha256"] = self.release_manifest_sha
        pins["model_release_manifest_sha256"] = self.pin_shas[
            "model_release_manifest"
        ]
        pins["inference_source_sha256"] = infer_sha
        pins["inference_release_manifest_sha256"] = self.pin_shas[
            "inference_release_manifest"
        ]
        pins["inference_config_sha256"] = self.pin_shas["inference_config"]
        pins["source_preprocessing_sha256"] = self.pin_shas[
            "source_preprocessing"
        ]
        pins["calibration_digest"] = None
        spec = {
            "schema_version": bridge.SOURCE_RUNTIME_SCHEMA,
            "pins": pins,
            "pin_files": {
                key: (
                    None
                    if key == "calibration"
                    else {"path": str(path), "sha256": self.pin_shas[key]}
                )
                for key, path in self.pin_paths.items()
            },
            "sources": sources,
            "runtime": {
                "root_python": {
                    "path": str(runtime_root / "root-python"),
                    "sha256": root_python_sha,
                },
                "python": {"path": str(runtime_root / "python"), "sha256": python_sha},
                "site_packages": str(site_packages),
                "torchrun": {
                    "path": str(torchrun_path), "sha256": torchrun_sha,
                },
                "deployment_controller": {
                    "path": str(controller_path), "sha256": controller_sha,
                },
                "controller_authority": {
                    "receipt": {
                        "path": controller_authority["receipt"]["path"],
                        "sha256": controller_authority["receipt"]["sha256"],
                    },
                    "authority_digest": controller_authority[
                        "authority_digest"
                    ],
                },
                "infer_lora": {
                    "path": str(infer_path),
                    "sha256": infer_sha,
                },
                "decoder_adapter": {
                    "path": str(decoder_path),
                    "sha256": decoder_sha,
                },
                "ffprobe": {
                    "path": str(runtime_root / "ffprobe"),
                    "sha256": ffprobe_sha,
                },
                "eval_release_root": str(release_root),
                "eval_release_archive": {
                    "path": str(release_archive),
                    "sha256": release_archive_sha,
                },
                "eval_release_envelope": {
                    "path": str(release_envelope),
                    "sha256": release_envelope_sha,
                },
                "eval_release_manifest_digest": release_manifest[
                    "manifest_digest"
                ],
                "eval_release_content_revision": release_manifest[
                    "content_revision"
                ],
                "eval_release_envelope_digest": release_envelope_value[
                    "envelope_digest"
                ],
                "bernini_root": str(runtime_root / "bernini"),
                "veomni_root": str(runtime_root / "veomni"),
                "model_checkpoint_root": str(runtime_root / "model"),
                "expected_bernini_commit": hashlib.sha1(b"bernini").hexdigest(),
                "expected_veomni_commit": hashlib.sha1(b"veomni").hexdigest(),
                "expected_checkpoint_tree_sha256": digest("model tree"),
                "method_source_revision": self.source_revision,
                "method_source_archive_sha256": self.source_archive_sha,
                "num_inference_steps": 40,
            },
        }
        spec["spec_digest"] = bridge.object_sha256(spec)
        self.spec = spec
        source_spec_path = deployment_root / "source-runtime-spec.json"
        source_spec_sha = write_json(source_spec_path, spec)
        source_spec_path.chmod(0o444)
        _, source_spec_file = bridge._stable_file(
            source_spec_path,
            label="test authorized source/runtime spec",
            expected_sha256=source_spec_sha,
        )
        deployment_value = {
            "schema_version": (
                "bernini-action-preservation-decoded-eval-deployment-receipt-v3"
            ),
            "release_generation": "test-exact15-r4",
            "work_root_authority": work_root_authority,
            "work_root_capture_before_receipt": {},
            "work_root_expected_phase_a_entries": [],
            "work_root_held_fd_through_controller_publication": True,
            "deployment_request": {},
            "deployment_request_digest": digest("test deployment request"),
            "controller": {},
            "root_python": {},
            "frozen_python": {},
            "site_packages": {},
            "torchrun": {},
            "release": {},
            "verified_runtime_source": {},
            "verified_runtime": {},
            "source_runtime_spec_path": str(source_spec_path),
            "source_spec_authority_receipt_path": str(source_authority_path),
            "controller_authority": controller_authority,
            "literal_request_sha_required": True,
            "controller_executed_from_same_fd_captured_bytes": True,
            "verified_runtime_executed_from_same_fd_captured_bytes": True,
            "automatic_retry": False,
            "network_used": False,
            "scientific_promotion_authorized": False,
        }
        deployment_value["receipt_digest"] = bridge.object_sha256(
            deployment_value
        )
        deployment_path.chmod(0o644)
        deployment_sha = write_json(deployment_path, deployment_value)
        deployment_path.chmod(0o444)
        _, deployment_file = bridge._stable_file(
            deployment_path,
            label="test deployment receipt",
            expected_sha256=deployment_sha,
        )
        source_authority_value = {
            "schema_version": bridge.SOURCE_SPEC_AUTHORITY_SCHEMA,
            "release_generation": deployment_value["release_generation"],
            "deployment_receipt": deployment_file,
            "work_root_authority": work_root_authority,
            "work_root_capture_before_receipt": {},
            "work_root_expected_source_spec_entries": sorted(
                {
                    deployment_path.name,
                    source_spec_path.name,
                    source_authority_path.name,
                }
            ),
            "work_root_held_fd_through_source_spec_publication": True,
            "deployment_receipt_digest": deployment_value["receipt_digest"],
            "controller_authority": controller_authority,
            "source_runtime_spec": source_spec_file,
            "source_runtime_spec_digest": spec["spec_digest"],
            "receipt_path": str(source_authority_path),
            "literal_source_runtime_spec_sha_required": True,
            "runtime_authority_continuity_verified": True,
            "automatic_retry": False,
            "network_used": False,
            "scientific_promotion_authorized": False,
        }
        source_authority_value["receipt_digest"] = bridge.object_sha256(
            source_authority_value
        )
        source_authority_path.chmod(0o644)
        source_authority_sha = write_json(
            source_authority_path, source_authority_value
        )
        source_authority_path.chmod(0o444)
        self.deployment_authority = {
            "schema_version": bridge.DEPLOYMENT_AUTHORITY_SCHEMA,
            "work_root_authority": work_root_authority,
            "deployment_receipt": {
                "path": str(deployment_path), "sha256": deployment_sha,
            },
            "source_spec_authority": {
                "path": str(source_authority_path),
                "sha256": source_authority_sha,
            },
            "deployment_receipt_digest": deployment_value["receipt_digest"],
            "source_spec_authority_digest": source_authority_value[
                "receipt_digest"
            ],
        }
        self.deployment_authority["authority_digest"] = bridge.object_sha256(
            self.deployment_authority
        )
        self.deployment_root = deployment_root
        self.evaluation_root = self.parent / "evaluation"
        self.bridge_root = self.parent / "bridge"

    def build(self):
        return bridge.build_bridge(
            experiment_root=self.experiment,
            completion_sha256=self.completion_sha,
            source_runtime_spec=self.spec,
            evaluation_id="apv2-eval-physical-r1",
            evaluation_root=self.evaluation_root,
            bridge_root=self.bridge_root,
            deployment_authority=self.deployment_authority,
        )

    def open_work_root_binding(
        self, *, capture_name: str = "bridge-runtime-capture.json",
    ) -> tuple[dict, tuple[int, int]]:
        authority = self.deployment_authority["work_root_authority"]
        flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_fd = os.open(self.deployment_root.parent, flags)
        root_fd = os.open(
            self.deployment_root.name, flags, dir_fd=parent_fd
        )
        os.set_inheritable(parent_fd, False)
        os.set_inheritable(root_fd, False)
        parent_identity = bridge.verified_release._work_root_identity_value(
            os.fstat(parent_fd)
        )
        root_identity = bridge.verified_release._work_root_identity_value(
            os.fstat(root_fd)
        )
        value = {
            "schema_version": bridge.verified_release.WORK_ROOT_BINDING_SCHEMA,
            "path": str(self.deployment_root),
            "parent_path": str(self.deployment_root.parent),
            "parent_fd": parent_fd,
            "root_fd": root_fd,
            "parent_identity": parent_identity,
            "root_identity": root_identity,
            "parent_immutable_identity": authority[
                "parent_immutable_identity"
            ],
            "root_immutable_identity": authority["immutable_identity"],
            "entries": sorted(os.listdir(root_fd)),
            "work_root_authority_digest": authority["authority_digest"],
            "work_root_authority": authority,
            "deployment_receipt": self.deployment_authority[
                "deployment_receipt"
            ],
            "source_spec_authority": self.deployment_authority[
                "source_spec_authority"
            ],
            "deployment_receipt_digest": self.deployment_authority[
                "deployment_receipt_digest"
            ],
            "source_spec_authority_digest": self.deployment_authority[
                "source_spec_authority_digest"
            ],
            "target": "action_preservation_decoded_eval_bridge_v1.py",
            "capture_receipt_path": str(
                self.deployment_root / capture_name
            ),
            "exact_two_directory_fds": True,
            "fds_inheritable_only_across_verified_exec": True,
        }
        value["binding_digest"] = bridge.object_sha256(value)
        bridge.verified_release.validate_inherited_work_root_binding(
            value,
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=True,
        )
        return value, (parent_fd, root_fd)

    def resign_training_audit(self, raw: bytes) -> None:
        audit_path = self.experiment / "logs" / "training-audit.json"
        audit_sha = write(audit_path, raw)
        self.completion["training_audit_sha256"] = audit_sha
        self.completion.pop("completion_digest", None)
        self.completion["completion_digest"] = bridge.object_sha256(
            self.completion
        )
        self.completion_sha = write_json(
            self.experiment / "TRAINING_COMPLETE.json",
            self.completion,
            newline=False,
        )

    def test_exact32_physical_bridge_and_create_only_publication(self) -> None:
        bundle, bindings, receipt = self.build()
        self.assertEqual(len(bindings["checkpoints"]), 32)
        self.assertEqual(len(bindings["sources"]), 4)
        self.assertEqual(bindings["eval_release"]["member_count"], 15)
        self.assertEqual(
            [item["relative_path"] for item in bindings["eval_release"]["members"]],
            list(bridge.EVAL_RELEASE_MEMBERS),
        )
        self.assertEqual(bundle["manifest"]["matrix"]["total_decode_count"], 264)
        self.assertFalse(bindings["training_loss_read_or_copied"])
        self.assertEqual(
            bindings["training_audit_digest"],
            bridge.object_sha256(self.audit),
        )
        self.assertEqual(
            bindings["training_audit_serialization"],
            bridge.TRAINING_AUDIT_SERIALIZATION,
        )
        self.assertEqual(
            receipt["training_audit_digest"],
            bindings["training_audit_digest"],
        )
        output = bridge.publish_bridge(bundle=bundle, bindings=bindings, receipt=receipt)
        self.assertEqual(output, self.bridge_root / bridge.BRIDGE_RECEIPT_FILENAME)
        bindings_path = self.bridge_root / bridge.PHYSICAL_BINDINGS_FILENAME
        loaded = bridge.load_physical_bindings(
            bindings_path,
            expected_sha256=hashlib.sha256(bindings_path.read_bytes()).hexdigest(),
            verify_files=True,
        )
        self.assertNotEqual(
            loaded["physical_bindings_digest"],
            bindings["physical_bindings_digest"],
        )
        publication = loaded["evaluation_publication"]
        self.assertTrue(publication["materialized"])
        self.assertEqual(
            publication["publication_receipt"]["publication_digest"],
            json.loads(
                (self.evaluation_root / plan.PUBLICATION_FILENAME).read_text()
            )["publication_digest"],
        )
        self.assertEqual(stat.S_IMODE(self.bridge_root.stat().st_mode), 0o555)
        final_receipt = json.loads(output.read_text())
        self.assertEqual(
            bridge.validate_bridge_receipt(
                final_receipt,
                bundle=bundle,
                bindings=loaded,
                materialized_required=True,
            ),
            final_receipt,
        )
        self.assertTrue((self.evaluation_root / plan.PUBLICATION_FILENAME).is_file())
        with self.assertRaisesRegex(bridge.DecodedEvaluationBridgeError, "not fresh"):
            bridge.publish_bridge(bundle=bundle, bindings=bindings, receipt=receipt)

    def test_bridge_publication_uses_real_inherited_work_root_fds(self) -> None:
        work_root, descriptors = self.open_work_root_binding()
        evaluation_root = self.deployment_root / "held-evaluation"
        bridge_root = self.deployment_root / "held-bridge"
        try:
            bundle, bindings, receipt = bridge.build_bridge(
                experiment_root=self.experiment,
                completion_sha256=self.completion_sha,
                source_runtime_spec=self.spec,
                evaluation_id="apv2-held-work-root-positive",
                evaluation_root=evaluation_root,
                bridge_root=bridge_root,
                deployment_authority=self.deployment_authority,
                work_root_binding=work_root,
            )
            output = bridge.publish_bridge(
                bundle=bundle,
                bindings=bindings,
                receipt=receipt,
                work_root_binding=work_root,
            )
            self.assertEqual(output, bridge_root / bridge.BRIDGE_RECEIPT_FILENAME)
            self.assertTrue((evaluation_root / plan.PUBLICATION_FILENAME).is_file())
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def test_inherited_work_root_rename_replacement_cannot_redirect_bridge(
        self,
    ) -> None:
        work_root, descriptors = self.open_work_root_binding(
            capture_name="bridge-hostile-runtime-capture.json"
        )
        evaluation_root = self.deployment_root / "hostile-evaluation"
        bridge_root = self.deployment_root / "hostile-bridge"
        displaced = self.parent / "deployment-work-displaced"
        try:
            bundle, bindings, receipt = bridge.build_bridge(
                experiment_root=self.experiment,
                completion_sha256=self.completion_sha,
                source_runtime_spec=self.spec,
                evaluation_id="apv2-held-work-root-hostile",
                evaluation_root=evaluation_root,
                bridge_root=bridge_root,
                deployment_authority=self.deployment_authority,
                work_root_binding=work_root,
            )

            def barrier(event: str, root: pathlib.Path, relative: str) -> None:
                if event == "after-root-create" and relative == ".":
                    os.rename(self.deployment_root, displaced)
                    os.mkdir(self.deployment_root, 0o700)

            with self.assertRaisesRegex(
                bridge.DecodedEvaluationBridgeError,
                "parent|identity|drift|closure",
            ):
                bridge.publish_bridge(
                    bundle=bundle,
                    bindings=bindings,
                    receipt=receipt,
                    evaluation_barrier=barrier,
                    work_root_binding=work_root,
                )
            self.assertEqual(list(self.deployment_root.iterdir()), [])
            self.assertFalse(bridge_root.exists())
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def test_resigned_bridge_receipt_extra_field_fails_before_publication(self) -> None:
        bundle, bindings, receipt = self.build()
        hostile = copy.deepcopy(receipt)
        hostile["unexpected"] = True
        hostile.pop("bridge_receipt_digest", None)
        hostile["bridge_receipt_digest"] = bridge.object_sha256(hostile)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "bridge receipt field closure differs",
        ):
            bridge.publish_bridge(
                bundle=bundle, bindings=bindings, receipt=hostile
            )
        self.assertFalse(self.evaluation_root.exists())
        self.assertFalse(self.bridge_root.exists())

    def test_bridge_retained_root_barrier_rejects_rename_replacement(self) -> None:
        bundle, bindings, receipt = self.build()
        displaced = self.parent / "bridge-displaced"

        def barrier(event: str, root: pathlib.Path, relative: str) -> None:
            if event == "after-root-create" and relative == ".":
                os.rename(root, displaced)
                os.mkdir(root, 0o700)

        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "identity|drift|closure",
        ):
            bridge.publish_bridge(
                bundle=bundle, bindings=bindings, receipt=receipt,
                bridge_barrier=barrier,
            )
        self.assertTrue(displaced.is_dir())
        self.assertEqual(list(self.bridge_root.iterdir()), [])
        self.assertFalse(
            (self.bridge_root / bridge.BRIDGE_RECEIPT_FILENAME).exists()
        )

    def test_resigned_top_level_evaluation_pins_cannot_escape_publication(self) -> None:
        bundle, bindings, receipt = self.build()
        bridge.publish_bridge(bundle=bundle, bindings=bindings, receipt=receipt)
        published = json.loads(
            (self.bridge_root / bridge.PHYSICAL_BINDINGS_FILENAME).read_text()
        )
        for field, hostile_value in (
            ("evaluation_id", "hostile-resigned-evaluation"),
            ("input_digest", "0" * 64),
            ("manifest_digest", "1" * 64),
        ):
            with self.subTest(field=field):
                hostile = copy.deepcopy(published)
                hostile[field] = hostile_value
                hostile.pop("physical_bindings_digest", None)
                hostile["physical_bindings_digest"] = bridge.object_sha256(
                    hostile
                )
                with self.assertRaisesRegex(
                    bridge.DecodedEvaluationBridgeError,
                    "physical/publication evaluation binding differs",
                ):
                    bridge.validate_physical_bindings(
                        hostile,
                        verify_files=True,
                        require_evaluation_publication=True,
                    )

    def test_final_retained_replay_rejects_publication_file_replacement(self) -> None:
        bundle, bindings, receipt = self.build()
        bridge.publish_bridge(bundle=bundle, bindings=bindings, receipt=receipt)
        published = json.loads(
            (self.bridge_root / bridge.PHYSICAL_BINDINGS_FILENAME).read_text()
        )
        hostile_directory = tempfile.TemporaryDirectory()
        self.addCleanup(hostile_directory.cleanup)
        displaced = pathlib.Path(hostile_directory.name) / "input-displaced.json"

        def barrier(stage: str, root: pathlib.Path) -> None:
            if stage == "before-final-retained-replay":
                source = root / plan.INPUT_FILENAME
                os.rename(source, displaced)
                source.write_bytes(displaced.read_bytes())
                source.chmod(0o400)

        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "identity|closure|drift",
        ):
            bridge.validate_physical_bindings(
                published,
                verify_files=True,
                require_evaluation_publication=True,
                evaluation_validation_barrier=barrier,
            )
        self.assertTrue(displaced.is_file())

    def test_resigned_training_audit_whitespace_variant_is_rejected(self) -> None:
        self.resign_training_audit(bridge.canonical_json_bytes(self.audit) + b"\n")
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "producer-exact serialization differs",
        ):
            self.build()

    def test_resigned_training_audit_duplicate_key_is_rejected(self) -> None:
        raw = producer_audit_bytes(self.audit)
        hostile = raw.replace(
            b"{\n",
            b'{\n  "training_audit_go": true,\n',
            1,
        )
        self.resign_training_audit(hostile)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "contains a duplicate key",
        ):
            self.build()

    def test_resigned_training_audit_nan_is_rejected(self) -> None:
        raw = producer_audit_bytes(self.audit)
        hostile = raw.replace(b'"loss": 1.0', b'"loss": NaN', 1)
        self.assertNotEqual(hostile, raw)
        self.resign_training_audit(hostile)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "cannot decode training audit",
        ):
            self.build()

    def test_eval_release_exact15_rejects_missing_extra_and_symlink(self) -> None:
        missing = self.release_member_root / "tools" / "materialize_vae.py"
        held = self.parent / "held-materialize-vae.py"
        missing.parent.chmod(0o755)
        missing.rename(held)
        missing.parent.chmod(0o555)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "links, extras, or missing entries|exact entry closure",
        ):
            self.build()
        missing.parent.chmod(0o755)
        held.rename(missing)
        missing.parent.chmod(0o555)

        self.release_member_root.chmod(0o755)
        extra = self.release_member_root / "unregistered_eval.py"
        extra.write_bytes(b"hostile extra member\n")
        extra.chmod(0o444)
        self.release_member_root.chmod(0o555)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "links, extras, or missing entries|exact entry closure",
        ):
            self.build()
        self.release_member_root.chmod(0o755)
        extra.unlink()
        self.release_member_root.chmod(0o555)

        victim = self.release_member_root / "action_preservation_gate_v1.py"
        held = self.parent / "held-gate.py"
        self.release_member_root.chmod(0o755)
        victim.rename(held)
        victim.symlink_to(held)
        self.release_member_root.chmod(0o555)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "not canonical|plain file|symlink is forbidden",
        ):
            self.build()
        self.release_member_root.chmod(0o755)
        victim.unlink()
        held.rename(victim)
        self.release_member_root.chmod(0o555)

    def test_eval_release_member_with_external_hardlink_is_rejected(self) -> None:
        member = self.release_member_root / "action_preservation_gate_v1.py"
        os.link(member, self.parent / "hostile-external-alias.py")
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "hard link|physical identity changed or differs",
        ):
            self.build()

    def test_self_recaptured_torchrun_after_authority_is_rejected(self) -> None:
        hostile = copy.deepcopy(self.spec)
        torchrun_path = pathlib.Path(hostile["runtime"]["torchrun"]["path"])
        torchrun_path.write_bytes(b"# hostile replacement torchrun\n")
        hostile["runtime"]["torchrun"]["sha256"] = hashlib.sha256(
            torchrun_path.read_bytes()
        ).hexdigest()
        hostile["spec_digest"] = bridge.object_sha256(
            {
                key: value
                for key, value in hostile.items()
                if key != "spec_digest"
            }
        )
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "authority|torchrun|continuity",
        ):
            bridge.build_bridge(
                experiment_root=self.experiment,
                completion_sha256=self.completion_sha,
                source_runtime_spec=hostile,
                evaluation_id="hostile-torchrun",
                evaluation_root=self.evaluation_root,
                bridge_root=self.bridge_root,
                deployment_authority=self.deployment_authority,
            )

    def test_same_fd_double_read_rejects_named_path_exchange(self) -> None:
        target = self.parent / "stable-authority.json"
        replacement = self.parent / "replacement-authority.json"
        held = self.parent / "held-authority.json"
        payload = b'{"authority":"same-bytes"}\n'
        target.write_bytes(payload)
        replacement.write_bytes(payload)
        original_reader = bridge._read_file_descriptor
        call_count = 0

        def exchange_after_first_read(descriptor: int) -> bytes:
            nonlocal call_count
            captured = original_reader(descriptor)
            call_count += 1
            if call_count == 1:
                target.rename(held)
                replacement.rename(target)
            return captured

        with mock.patch.object(
            bridge, "_read_file_descriptor", side_effect=exchange_after_first_read
        ), self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError, "stable double read"
        ):
            bridge._stable_file(
                target,
                label="hostile named exchange",
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )

    def test_source_runtime_spec_requires_external_file_sha(self) -> None:
        path = self.parent / "externally-pinned-source-runtime-spec.json"
        value = {"authority": "literal-detached-pin"}
        payload = bridge.canonical_json_bytes(value) + b"\n"
        path.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(
            bridge._load(
                path, label="source/runtime spec", expected_sha256=expected
            ),
            value,
        )
        resigned = {"authority": "same-uid-self-resigned-replacement"}
        path.write_bytes(bridge.canonical_json_bytes(resigned) + b"\n")
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError, "SHA differs"
        ):
            bridge._load(
                path, label="source/runtime spec", expected_sha256=expected
            )

    def test_verified_runtime_capture_positive_replay_and_frozen_identity(self) -> None:
        _, bindings, _ = self.build()
        target = "action_preservation_decoded_eval_bridge_v1.py"
        member = next(
            item for item in bindings["eval_release"]["members"]
            if item["relative_path"] == target
        )
        release = bindings["eval_release"]
        arguments = ["--positive-capture-replay"]
        work_root = self.parent / "runtime-capture-work"
        work_root.mkdir(mode=0o700)
        receipt_path = work_root / "positive-runtime-capture.json"

        def identity(value: os.stat_result) -> dict[str, int]:
            return {
                "device": value.st_dev, "inode": value.st_ino,
                "uid": value.st_uid, "gid": value.st_gid,
                "mode": value.st_mode, "nlink": value.st_nlink,
                "rdev": value.st_rdev, "size": value.st_size,
                "blocks": getattr(value, "st_blocks", 0),
                "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns,
            }

        parent_identity = identity(work_root.parent.stat())
        root_identity = identity(work_root.stat())
        immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
        work_root_authority = {
            "schema_version": bridge.verified_release.WORK_ROOT_AUTHORITY_SCHEMA,
            "path": str(work_root),
            "parent_path": str(work_root.parent),
            "creation_identity": root_identity,
            "immutable_identity": {
                key: root_identity[key] for key in immutable_fields
            },
            "parent_immutable_identity": {
                key: parent_identity[key] for key in immutable_fields
            },
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        work_root_authority["authority_digest"] = bridge.object_sha256(
            work_root_authority
        )
        deployment_pair = {
            "path": str(work_root / "deployment-receipt.json"),
            "sha256": digest("runtime deployment receipt file"),
        }
        source_pair = {
            "path": str(work_root / "source-authority.json"),
            "sha256": digest("runtime source authority file"),
        }
        work_root_binding = {
            "schema_version": bridge.verified_release.WORK_ROOT_BINDING_SCHEMA,
            "path": str(work_root),
            "parent_path": str(work_root.parent),
            "parent_fd": 101,
            "root_fd": 102,
            "parent_identity": parent_identity,
            "root_identity": root_identity,
            "parent_immutable_identity": {
                key: parent_identity[key] for key in immutable_fields
            },
            "root_immutable_identity": {
                key: root_identity[key] for key in immutable_fields
            },
            "entries": sorted(
                {pathlib.Path(deployment_pair["path"]).name,
                 pathlib.Path(source_pair["path"]).name}
            ),
            "work_root_authority": work_root_authority,
            "deployment_receipt": deployment_pair,
            "source_spec_authority": source_pair,
            "work_root_authority_digest": work_root_authority[
                "authority_digest"
            ],
            "deployment_receipt_digest": digest("deployment-receipt"),
            "source_spec_authority_digest": digest("source-spec-authority"),
            "target": target,
            "capture_receipt_path": str(receipt_path),
            "exact_two_directory_fds": True,
            "fds_inheritable_only_across_verified_exec": True,
        }
        work_root_binding["binding_digest"] = bridge.object_sha256(
            work_root_binding
        )
        receipt = {
            "schema_version": bridge.verified_release.CAPTURE_RECEIPT_SCHEMA,
            "release_generation": bridge.verified_release.RELEASE_GENERATION,
            "archive_sha256": release["archive_file"]["sha256"],
            "manifest_sha256": release["manifest_file"]["sha256"],
            "manifest_digest": release["manifest_digest"],
            "content_revision": release["content_revision"],
            "envelope_sha256": release["envelope_file"]["sha256"],
            "envelope_digest": release["envelope_digest"],
            "all_members_capture_digest": release[
                "all_members_capture_digest"
            ],
            "member_count": len(bridge.EVAL_RELEASE_MEMBERS),
            "target": target,
            "target_sha256": member["sha256"],
            "target_size": member["size"],
            "target_mode": member["mode"],
            "target_arguments_sha256": bridge.object_sha256(arguments),
            "root_python": {
                key: bindings["runtime"]["root_python"][key]
                for key in bridge._CAPTURED_FILE_FIELDS
            },
            "frozen_python": {
                key: bindings["runtime"]["python"][key]
                for key in bridge._CAPTURED_FILE_FIELDS
            },
            "site_packages": {
                key: bindings["runtime"]["site_packages"][key]
                for key in bridge._CAPTURED_DIRECTORY_FIELDS
            },
            "release_artifacts": {
                name: {
                    key: release[f"{name}_file"][key]
                    for key in bridge._CAPTURED_FILE_FIELDS
                }
                for name in ("archive", "manifest", "envelope")
            },
            "controller_authority": copy.deepcopy(
                bindings["runtime"]["controller_authority"]
            ),
            "captured_torchrun": None,
            "work_root": work_root_binding,
            "task_fd_binding": None,
            "publication_authority_kind": "work_root",
            "receipt_path": str(receipt_path),
            "publication_policy": {
                "create_only_o_excl": True,
                "mode": 0o444,
                "rank_zero_only": True,
                "same_fd_double_read_after_fsync": True,
                "named_identity_replay_after_write": True,
                "post_close_stable_double_read": True,
                "parent_directory_fsync": True,
            },
        }
        receipt["capture_digest"] = bridge.object_sha256(receipt)
        write_json(receipt_path, receipt)
        receipt_path.chmod(0o444)
        with mock.patch.dict(
            os.environ,
            {
                bridge.verified_release.CAPTURE_DIGEST_ENV: receipt[
                    "capture_digest"
                ],
                bridge.verified_release.CAPTURE_RECEIPT_ENV: str(receipt_path),
            },
        ):
            observed = bridge.validate_running_verified_capture(
                bindings, target=target, expected_arguments=arguments,
                verify_file=True,
            )
        self.assertEqual(observed["capture_digest"], receipt["capture_digest"])

        hostile = copy.deepcopy(receipt)
        hostile["frozen_python"]["ctime_ns"] += 1
        hostile_path = work_root / "hostile-runtime-capture.json"
        hostile["receipt_path"] = str(hostile_path)
        hostile["work_root"]["capture_receipt_path"] = str(hostile_path)
        hostile["work_root"]["binding_digest"] = bridge.object_sha256(
            {
                key: value
                for key, value in hostile["work_root"].items()
                if key != "binding_digest"
            }
        )
        hostile["capture_digest"] = bridge.object_sha256(
            {key: value for key, value in hostile.items() if key != "capture_digest"}
        )
        write_json(hostile_path, hostile)
        hostile_path.chmod(0o444)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError, "capture differs"
        ):
            bridge.validate_verified_capture_receipt(
                bindings, receipt_path=hostile_path, target=target,
                expected_arguments=arguments, verify_file=True,
            )

    def test_verified_target_and_captured_torchrun_argv_are_isolated(self) -> None:
        _, bindings, _ = self.build()
        ordinary = bridge.verified_target_argv(
            bindings,
            target="action_preservation_decoded_eval_bridge_v1.py",
            arguments=["--help"],
            capture_receipt_path=self.parent / "ordinary-capture.json",
        )
        rank = bridge.verified_target_argv(
            bindings,
            target="infer_lora.py",
            arguments=[
                "--model-consumption-input",
                str(self.parent / "consumption-input.json"),
                "--model-consumption-input-sha256",
                "1" * 64,
                "--model-consumption-input-digest",
                "2" * 64,
                "--task-input-digest",
                "3" * 64,
            ],
            capture_receipt_path=self.parent / "rank-capture.json",
        )
        outer = bridge.captured_torchrun_argv(
            bindings,
            torchrun_arguments=["--standalone", "--nproc_per_node=4"],
            rank_target_argv=rank,
        )
        for command in (ordinary, rank, outer):
            self.assertEqual(command[1:4], ["-I", "-S", "-B"])
        self.assertNotIn("torch.distributed.run", outer)
        self.assertNotIn("-m", outer)
        self.assertIn("--max-restarts=0", outer)
        self.assertIn("--no-python", outer)
        self.assertEqual(outer[-len(rank):], rank)

    def test_tampered_checkpoint_bytes_fail_before_plan_publication(self) -> None:
        path = (
            self.experiment / "runs" / plan.ARMS[0] / "checkpoint-00000000"
            / "adapter" / "adapter_model.safetensors"
        )
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(bridge.DecodedEvaluationBridgeError, "SHA differs"):
            self.build()
        self.assertFalse(self.evaluation_root.exists())
        self.assertFalse(self.bridge_root.exists())

    def test_self_resigned_physical_binding_cannot_escape_training_audit(self) -> None:
        _, bindings, _ = self.build()
        hostile = copy.deepcopy(bindings)
        adapter_path = pathlib.Path(hostile["checkpoints"][0]["adapter_model"]["path"])
        adapter_path.write_bytes(b"self-resigned hostile adapter")
        _, recaptured = bridge._stable_file(adapter_path, label="hostile adapter")
        hostile["checkpoints"][0]["adapter_model"] = recaptured
        hostile.pop("physical_bindings_digest")
        hostile["physical_bindings_digest"] = bridge.object_sha256(hostile)
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "differs from replayed training audit",
        ):
            bridge.validate_physical_bindings(hostile, verify_files=True)

    def test_deployment_authority_pair_swap_sha_and_resign_are_rejected(self) -> None:
        _, bindings, _ = self.build()

        def resign(value: dict) -> dict:
            value["deployment_authority"].pop("authority_digest", None)
            value["deployment_authority"]["authority_digest"] = (
                bridge.object_sha256(value["deployment_authority"])
            )
            value.pop("physical_bindings_digest", None)
            value["physical_bindings_digest"] = bridge.object_sha256(value)
            return value

        swapped = copy.deepcopy(bindings)
        deployment = swapped["deployment_authority"]
        deployment["deployment_receipt"], deployment[
            "source_spec_authority"
        ] = (
            deployment["source_spec_authority"],
            deployment["deployment_receipt"],
        )
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "deployment receipt field closure|deployment authority receipt continuity",
        ):
            bridge.validate_physical_bindings(
                resign(swapped), verify_files=True
            )

        wrong_sha = copy.deepcopy(bindings)
        wrong_sha["deployment_authority"]["deployment_receipt"][
            "sha256"
        ] = digest("self-resigned wrong deployment file SHA")
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "SHA differs|SHA-256 differs|same-FD capture",
        ):
            bridge.validate_physical_bindings(
                resign(wrong_sha), verify_files=True
            )

        resigned = copy.deepcopy(bindings)
        resigned["deployment_authority"]["deployment_receipt_digest"] = (
            digest("self-resigned deployment object")
        )
        resigned["deployment_authority"][
            "source_spec_authority_digest"
        ] = digest("self-resigned source authority object")
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "deployment receipt authority|deployment authority receipt continuity",
        ):
            bridge.validate_physical_bindings(
                resign(resigned), verify_files=True
            )

    def test_completion_and_release_bindings_are_mandatory(self) -> None:
        hostile = copy.deepcopy(self.spec)
        hostile["pins"]["adapter_release_manifest_sha256"] = digest("wrong release")
        hostile["spec_digest"] = bridge.object_sha256(
            {key: value for key, value in hostile.items() if key != "spec_digest"}
        )
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "source/runtime spec differs from held source spec authority|release_manifest physical pin",
        ):
            bridge.build_bridge(
                experiment_root=self.experiment,
                completion_sha256=self.completion_sha,
                source_runtime_spec=hostile,
                evaluation_id="hostile",
                evaluation_root=self.evaluation_root,
                bridge_root=self.bridge_root,
                deployment_authority=self.deployment_authority,
            )

    def test_missing_calibration_is_allowed_for_decode_but_not_claimed(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["pins"]["calibration_digest"] = None
        spec["pin_files"]["calibration"] = None
        spec["spec_digest"] = bridge.object_sha256(
            {key: value for key, value in spec.items() if key != "spec_digest"}
        )
        bundle, bindings, _ = bridge.build_bridge(
            experiment_root=self.experiment,
            completion_sha256=self.completion_sha,
            source_runtime_spec=spec,
            evaluation_id="no-calibration",
            evaluation_root=self.evaluation_root,
            bridge_root=self.bridge_root,
            deployment_authority=self.deployment_authority,
        )
        self.assertIsNone(bundle["input_spec"]["pins"]["calibration_digest"])
        self.assertIsNone(bindings["calibration_digest"])
        self.assertFalse(bindings["scientific_promotion_authorized"])

    def test_fully_resigned_spec_cannot_swap_sources_behind_same_authority(self) -> None:
        hostile = copy.deepcopy(self.spec)
        replacement = "Perform a fully re-signed but unauthorized different action."
        hostile["sources"][0]["instruction"] = replacement
        hostile["sources"][0]["instruction_sha256"] = plan.text_sha256(
            replacement
        )
        hostile["spec_digest"] = bridge.object_sha256(
            {key: value for key, value in hostile.items() if key != "spec_digest"}
        )
        self.assertEqual(
            hostile["pins"]["source_preprocessing_sha256"],
            self.pin_shas["source_preprocessing"],
        )
        with self.assertRaisesRegex(
            bridge.DecodedEvaluationBridgeError,
            "sources differ from source preprocessing authority",
        ):
            bridge.validate_source_runtime_spec(hostile)


class ProductionSourcePreprocessingAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        path = ROOT / (
            "audits/action_preservation_decoded_eval_r7_"
            "source_preprocessing_authority_20260816.json"
        )
        self.value = json.loads(path.read_text(encoding="utf-8"))

    def test_real_r7_authority_positive(self) -> None:
        observed = bridge._validate_source_preprocessing_authority(
            copy.deepcopy(self.value),
            source_manifest_sha256=(
                "62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8"
            ),
        )
        self.assertEqual(observed, self.value)

    def test_authority_extra_and_missing_fields_are_rejected(self) -> None:
        for label, hostile in (
            ("extra", {**copy.deepcopy(self.value), "unexpected": False}),
            (
                "missing",
                {
                    key: value
                    for key, value in copy.deepcopy(self.value).items()
                    if key != "runtime_decode_bound_by_inference_release"
                },
            ),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                bridge.DecodedEvaluationBridgeError, "field closure differs"
            ):
                bridge._validate_source_preprocessing_authority(
                    hostile,
                    source_manifest_sha256=self.value["source_manifest_sha256"],
                )

    def test_authority_reordered_extra_and_missing_source_rows_are_rejected(self) -> None:
        hostiles = []
        reordered = copy.deepcopy(self.value)
        reordered["sources"][0], reordered["sources"][1] = (
            reordered["sources"][1], reordered["sources"][0]
        )
        hostiles.append(reordered)
        extra = copy.deepcopy(self.value)
        extra["sources"][0]["unexpected"] = False
        hostiles.append(extra)
        missing = copy.deepcopy(self.value)
        missing["sources"][0].pop("source_receipt_sha256")
        hostiles.append(missing)
        for index, hostile in enumerate(hostiles):
            hostile["authority_digest"] = bridge.object_sha256(
                {
                    key: value
                    for key, value in hostile.items()
                    if key != "authority_digest"
                }
            )
            with mock.patch.object(
                bridge,
                "SOURCE_PREPROCESSING_AUTHORITY_DIGEST",
                hostile["authority_digest"],
            ), self.subTest(index=index), self.assertRaises(
                bridge.DecodedEvaluationBridgeError
            ):
                bridge._validate_source_preprocessing_authority(
                    hostile,
                    source_manifest_sha256=self.value["source_manifest_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
