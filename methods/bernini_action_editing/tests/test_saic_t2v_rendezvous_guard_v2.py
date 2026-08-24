from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import saic_t2v_rendezvous_guard_v2 as guard


def mkdir(path: Path, mode: int = 0o700) -> Path:
    path.mkdir(mode=mode)
    os.chmod(path, mode)
    return path


def write_json(path: Path, value: dict) -> None:
    guard.write_create_only(path, value)


def rewrite_json(path: Path, value: dict) -> None:
    os.chmod(path, 0o600)
    path.unlink()
    guard.write_create_only(path, value)


class Fixture:
    def __init__(self, root: Path, *, one_retry: bool = False) -> None:
        root = root.resolve(strict=True)
        self.root = mkdir(root / "output")
        self.logs = mkdir(self.root / "logs")
        self.attempts = mkdir(self.root / "attempts")
        self.plan = mkdir(self.root / "plan")
        self.rendezvous = mkdir(self.logs / "rendezvous")
        self.claims = mkdir(self.rendezvous / "port-claims")
        self.job_id = "765432"
        self.one_retry = one_retry
        self.completions: dict[tuple[str, int], Path] = {}
        self.collisions: dict[tuple[str, int], Path] = {}
        self._build()

    @property
    def claim_identity(self) -> str:
        info = self.claims.lstat()
        return f"{info.st_dev}:{info.st_ino}"

    def _build(self) -> None:
        prior_claim = None
        for group_number, group_id in enumerate(("sp4-a", "sp4-b")):
            plan_dir = mkdir(self.plan / group_id)
            group_root = mkdir(self.rendezvous / group_id)
            plan_rows = []
            for index in range(30):
                candidate_id = f"candidate-{group_id}-{index:02d}"
                envelope_path = plan_dir / f"{index:02d}.json"
                envelope = {"candidate": {"candidate_id": candidate_id}}
                write_json(envelope_path, envelope)
                envelope_raw = guard.read_ready_bytes(
                    envelope_path, label="fixture envelope"
                )
                candidate_digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
                prefix = (
                    f"saic-{self.job_id}-{group_id}-c{index:02d}-{candidate_digest[:16]}"
                )
                plan_rows.append(
                    {
                        "candidate_index": index,
                        "candidate_id": candidate_id,
                        "envelope_path": str(envelope_path),
                        "envelope_sha256": hashlib.sha256(envelope_raw).hexdigest(),
                        "candidate_id_sha256": candidate_digest,
                        "rdzv_id_prefix": prefix,
                        "requested_rendezvous_endpoint": "127.0.0.1:0",
                        "numeric_port_preregistered": False,
                    }
                )
                candidate_root = mkdir(
                    group_root / f"candidate-{index:02d}-{candidate_digest[:16]}"
                )
                if self.one_retry and group_id == "sp4-a" and index == 1:
                    collision_dir = mkdir(candidate_root / "launch-01")
                    assert prior_claim is not None
                    collision = guard.seal(
                        {
                            "schema_version": guard.COLLISION_SCHEMA_VERSION,
                            "status": "kernel_port_already_claimed_in_this_job_before_runtime",
                            "slurm_job_id": self.job_id,
                            "group_id": group_id,
                            "candidate_index": index,
                            "candidate_id": candidate_id,
                            "launch_ordinal": 1,
                            "rdzv_id": f"{prefix}-l01",
                            "actual_master_port": prior_claim["actual_master_port"],
                            "existing_claim_receipt_digest": prior_claim["receipt_digest"],
                            "existing_claim_sha256": guard.ready_file_sha256(
                                self.claims / f"port-{prior_claim['actual_master_port']}.json",
                                label="fixture prior claim",
                            ),
                            "existing_admission_receipt_path": prior_claim[
                                "admission_receipt_path"
                            ],
                            "existing_admission_receipt_digest": guard.load_sealed(
                                Path(prior_claim["admission_receipt_path"]),
                                schema_version=guard.DECISION_SCHEMA_VERSION,
                                exact_fields=guard.DECISION_FIELDS,
                            )["receipt_digest"],
                            "generation_runtime_entered": False,
                            "candidate_output_reuse_authorized": False,
                            "authority": guard.AUTHORITY,
                        }
                    )
                    write_json(collision_dir / "collision.json", collision)
                    self.collisions[(group_id, index)] = collision_dir / "collision.json"
                    success_ordinal = 2
                else:
                    success_ordinal = 1
                lifecycle = mkdir(candidate_root / f"launch-{success_ordinal:02d}")
                port = 30000 + group_number * 100 + index
                rdzv_id = f"{prefix}-l{success_ordinal:02d}"
                lifecycle_info = lifecycle.lstat()
                claim = guard.seal(
                    {
                        "schema_version": guard.CLAIM_SCHEMA_VERSION,
                        "status": "reserved_before_generation_runtime",
                        "slurm_job_id": self.job_id,
                        "group_id": group_id,
                        "candidate_index": index,
                        "candidate_id": candidate_id,
                        "launch_ordinal": success_ordinal,
                        "rdzv_id": rdzv_id,
                        "rdzv_backend": "c10d",
                        "rdzv_endpoint_request": "127.0.0.1:0",
                        "actual_master_addr": "127.0.0.1",
                        "actual_master_port": port,
                        "lifecycle_dir": str(lifecycle),
                        "lifecycle_dir_identity": f"{lifecycle_info.st_dev}:{lifecycle_info.st_ino}",
                        "admission_receipt_path": str(lifecycle / "admission.json"),
                        "torch_disable_share_rdzv_tcp_store": "0",
                        "shared_tcp_store_bootstrap": True,
                        "kernel_selected_free_port": True,
                        "port_claim_create_only_across_both_groups_for_this_job": True,
                        "generation_runtime_entered": False,
                        "scientific_spec_changed": False,
                        "authority": guard.AUTHORITY,
                    }
                )
                claim_path = self.claims / f"port-{port}.json"
                write_json(claim_path, claim)
                packets = []
                for rank in range(4):
                    packet = guard.seal(
                        {
                            "schema_version": guard.RANK_SCHEMA_VERSION,
                            "status": "prepared_before_generation_runtime",
                            "slurm_job_id": self.job_id,
                            "group_id": group_id,
                            "candidate_index": index,
                            "candidate_id": candidate_id,
                            "launch_ordinal": success_ordinal,
                            "rdzv_id": rdzv_id,
                            "rdzv_backend": "c10d",
                            "rdzv_endpoint_request": "127.0.0.1:0",
                            "actual_master_addr": "127.0.0.1",
                            "actual_master_port": port,
                            "rank": rank,
                            "local_rank": rank,
                            "world_size": 4,
                            "local_world_size": 4,
                            "port_claim_receipt_digest": claim["receipt_digest"],
                            "runtime_sha256": guard.EXPECTED_RUNTIME_SHA256,
                            "torch_disable_share_rdzv_tcp_store": "0",
                            "shared_tcp_store_bootstrap": True,
                            "generation_runtime_entered_before_admission": False,
                            "scientific_spec_changed": False,
                            "authority": guard.AUTHORITY,
                        }
                    )
                    write_json(lifecycle / f"rank-{rank}.json", packet)
                    packets.append(packet)
                decision = guard.seal(
                    {
                        "schema_version": guard.DECISION_SCHEMA_VERSION,
                        "status": "exact_world4_admitted_before_generation_runtime",
                        "slurm_job_id": self.job_id,
                        "group_id": group_id,
                        "candidate_index": index,
                        "candidate_id": candidate_id,
                        "launch_ordinal": success_ordinal,
                        "rdzv_id": rdzv_id,
                        "actual_master_addr": "127.0.0.1",
                        "actual_master_port": port,
                        "world_size": 4,
                        "rank_order": [0, 1, 2, 3],
                        "rank_packet_digests": [p["receipt_digest"] for p in packets],
                        "port_claim_receipt_digest": claim["receipt_digest"],
                        "runtime_sha256": guard.EXPECTED_RUNTIME_SHA256,
                        "torch_disable_share_rdzv_tcp_store": "0",
                        "shared_tcp_store_bootstrap": True,
                        "all_four_ranks_admitted": True,
                        "generation_runtime_entry_authorized": True,
                        "scientific_spec_changed": False,
                        "authority": guard.AUTHORITY,
                    }
                )
                write_json(lifecycle / "admission.json", decision)
                output = mkdir(self.attempts / candidate_id)
                attempt = guard.seal(
                    {
                        "schema_version": guard.ATTEMPT_SCHEMA_VERSION,
                        "group_id": group_id,
                        "candidate": {"candidate_id": candidate_id},
                        "event_verified": False,
                        "identity_preservation_verified": False,
                        "seed_selection_authorized": False,
                        "training_target_authorized": False,
                        "optimizer_or_parameter_update_authorized": False,
                    }
                )
                attempt_path = output / guard.ATTEMPT_RECEIPT_BASENAME
                write_json(attempt_path, attempt)
                completion = guard.seal(
                    {
                        "schema_version": guard.COMPLETION_SCHEMA_VERSION,
                        "status": "generation_runtime_completed_after_exact_world4_admission",
                        "slurm_job_id": self.job_id,
                        "group_id": group_id,
                        "candidate_index": index,
                        "candidate_id": candidate_id,
                        "launch_ordinal": success_ordinal,
                        "rdzv_id": rdzv_id,
                        "rdzv_backend": "c10d",
                        "rdzv_endpoint_request": "127.0.0.1:0",
                        "actual_master_addr": "127.0.0.1",
                        "actual_master_port": port,
                        "kernel_selected_and_atomically_bound": True,
                        "permanent_job_local_port_claim_path": str(claim_path),
                        "permanent_job_local_port_claim_sha256": guard.ready_file_sha256(
                            claim_path, label="fixture claim"
                        ),
                        "port_claim_receipt_digest": claim["receipt_digest"],
                        "exact_world4_rank_order": [0, 1, 2, 3],
                        "rank_packet_digests": [p["receipt_digest"] for p in packets],
                        "admission_receipt_digest": decision["receipt_digest"],
                        "runtime_sha256": guard.EXPECTED_RUNTIME_SHA256,
                        "torch_disable_share_rdzv_tcp_store": "0",
                        "shared_tcp_store_bootstrap": True,
                        "candidate_output": str(output),
                        "attempt_receipt_path": str(attempt_path),
                        "attempt_receipt_sha256": guard.ready_file_sha256(
                            attempt_path, label="fixture attempt"
                        ),
                        "attempt_receipt_digest": attempt["receipt_digest"],
                        "collision_retry_used_for_scientific_selection": False,
                        "candidate_output_reused": False,
                        "scientific_spec_changed": False,
                        "authority": guard.AUTHORITY,
                    }
                )
                completion_path = lifecycle / "completion.json"
                write_json(completion_path, completion)
                assemble_stdout = lifecycle / "assemble.stdout"
                descriptor = os.open(assemble_stdout, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
                os.chmod(assemble_stdout, 0o444)
                self.completions[(group_id, index)] = completion_path
                prior_claim = claim
            plan_receipt = guard.seal(
                {
                    "schema_version": guard.PLAN_SCHEMA_VERSION,
                    "group_id": group_id,
                    "slurm_job_id": self.job_id,
                    "candidate_count": 30,
                    "fixed_order": "lexicographic_envelope_basename",
                    "rdzv_backend": "c10d",
                    "rdzv_endpoint_request": "127.0.0.1:0",
                    "kernel_atomic_port_allocation": True,
                    "numeric_ports_preregistered": False,
                    "torch_disable_share_rdzv_tcp_store": "0",
                    "shared_tcp_store_bootstrap_required": True,
                    "permanent_create_only_claim_across_both_groups": True,
                    "maximum_operational_launches_per_candidate": 16,
                    "retry_condition": "sealed_same_job_prior_admitted_port_claim_and_candidate_output_absent_only",
                    "scientific_candidate_set_or_order_changed_by_retry": False,
                    "candidate_rows": plan_rows,
                    "authority": guard.AUTHORITY,
                }
            )
            write_json(
                self.logs / f"{group_id}-rendezvous-dynamic-plan-v1.json",
                plan_receipt,
            )

    def audit(self, receipt_name: str = "audit.json") -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return guard.audit_job(
                argparse.Namespace(
                    expected_runtime_sha256=guard.EXPECTED_RUNTIME_SHA256,
                    slurm_job_id=self.job_id,
                    output_root=str(self.root),
                    claim_root=str(self.claims),
                    claim_root_identity=self.claim_identity,
                    receipt=str(self.logs / receipt_name),
                )
            )

    def collision_args(self) -> argparse.Namespace:
        collision_path = self.collisions[("sp4-a", 1)]
        lifecycle = collision_path.parent
        info = lifecycle.lstat()
        candidate_id = "candidate-sp4-a-01"
        digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
        return argparse.Namespace(
            expected_runtime_sha256=guard.EXPECTED_RUNTIME_SHA256,
            slurm_job_id=self.job_id,
            group_id="sp4-a",
            candidate_index=1,
            candidate_id=candidate_id,
            launch_ordinal=1,
            expected_rdzv_id=f"saic-{self.job_id}-sp4-a-c01-{digest[:16]}-l01",
            claim_root=str(self.claims),
            claim_root_identity=self.claim_identity,
            lifecycle_dir=str(lifecycle),
            lifecycle_dir_identity=f"{info.st_dev}:{info.st_ino}",
            candidate_output=str(self.attempts / candidate_id),
        )


class SAICT2VRendezvousGuardV2Tests(unittest.TestCase):
    def test_runtime_pin_is_the_unchanged_scientific_runtime(self) -> None:
        runtime = METHOD_ROOT / "generate_saic_pure_t2v_event_bank_topup_v2.py"
        self.assertEqual(guard.file_sha256(runtime), guard.EXPECTED_RUNTIME_SHA256)

    def test_terminal_directory_audit_allows_only_device_remap_same_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve(strict=True)
            os.chmod(path, 0o700)
            observed = path.lstat()
            expected = f"{observed.st_dev + 1}:{observed.st_ino}"
            with self.assertRaisesRegex(
                guard.SAICT2VRendezvousGuardError, "inode changed"
            ):
                guard.exact_directory(
                    path, label="strict mount identity", expected_identity=expected
                )
            self.assertEqual(
                guard.exact_directory(
                    path,
                    label="terminal cross-mount identity",
                    expected_identity=expected,
                    allow_device_remap_same_inode=True,
                ),
                path,
            )
            wrong_inode = f"{observed.st_dev + 1}:{observed.st_ino + 1}"
            with self.assertRaisesRegex(
                guard.SAICT2VRendezvousGuardError, "inode changed"
            ):
                guard.exact_directory(
                    path,
                    label="terminal wrong inode",
                    expected_identity=wrong_inode,
                    allow_device_remap_same_inode=True,
                )

    def test_completion_uses_explicit_cross_mount_terminal_scope(self) -> None:
        source = Path(guard.__file__).read_text(encoding="utf-8")
        completion_start = source.index("def _validate_completion(")
        collision_start = source.index("def _validate_collision_for_job_audit(")
        completion = source[completion_start:collision_start]
        self.assertIn("allow_cross_mount_device_remap=True", completion)
        self.assertIn("canonical path/inode", completion)

    def test_publication_reader_waits_for_same_inode_0600_to_0444(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "delayed.json"
            value = guard.seal({"schema_version": "delay-test-v1", "value": 7})
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

            def publish() -> None:
                time.sleep(0.05)
                payload = guard.canonical_json_bytes(value) + b"\n"
                os.write(descriptor, payload)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
                os.close(descriptor)

            thread = threading.Thread(target=publish)
            thread.start()
            observed = guard.wait_load_sealed(
                path,
                schema_version="delay-test-v1",
                exact_fields={"schema_version", "value", "receipt_digest"},
                label="delayed receipt",
            )
            thread.join()
            self.assertEqual(observed, value)
            self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o444)

    def test_open_first_wait_accepts_stale_leaf_0444_for_same_inode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stale-leaf.json"
            value = guard.seal({"schema_version": "stale-leaf-v1", "value": 9})
            payload = guard.canonical_json_bytes(value) + b"\n"
            descriptor = os.open(
                path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            real_lstat = Path.lstat

            def stale_ready_lstat(candidate: Path):
                observed = real_lstat(candidate)
                if candidate == path and stat.S_IMODE(observed.st_mode) == 0o600:
                    fields = list(observed)
                    fields[0] = stat.S_IFREG | 0o444
                    return os.stat_result(fields)
                return observed

            def publish() -> None:
                time.sleep(0.05)
                os.write(descriptor, payload)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.close(descriptor)

            thread = threading.Thread(target=publish)
            thread.start()
            with mock.patch.object(Path, "lstat", new=stale_ready_lstat):
                observed = guard.wait_load_sealed(
                    path,
                    schema_version="stale-leaf-v1",
                    exact_fields={"schema_version", "value", "receipt_digest"},
                    label="stale leaf same-inode receipt",
                )
            thread.join()
            self.assertEqual(observed, value)

    def test_wait_rejects_provisional_inode_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "receipt.json"
            replaced = root / "replaced-0600.json"
            descriptor = os.open(
                path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            os.close(descriptor)
            replacement = guard.seal(
                {"schema_version": "replacement-v1", "value": 3}
            )

            def replace_inode() -> None:
                time.sleep(0.05)
                path.rename(replaced)
                guard.write_create_only(path, replacement)

            thread = threading.Thread(target=replace_inode)
            thread.start()
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.wait_load_sealed(
                    path,
                    schema_version="replacement-v1",
                    label="replaced provisional receipt",
                )
            thread.join()

    def test_wait_rejects_disappearing_pinned_provisional_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            descriptor = os.open(
                path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            os.close(descriptor)

            def remove_inode() -> None:
                time.sleep(0.05)
                path.unlink()

            thread = threading.Thread(target=remove_inode)
            thread.start()
            started = time.monotonic()
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.wait_load_sealed(
                    path,
                    schema_version="missing-after-pin-v1",
                    label="disappearing provisional receipt",
                )
            thread.join()
            self.assertLess(time.monotonic() - started, 0.5)

    def test_ready_reader_rejects_bad_mode_nlink_inode_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = guard.seal({"schema_version": "bad-ready-v1", "value": 1})
            payload = guard.canonical_json_bytes(value) + b"\n"

            wrong_mode = root / "wrong-mode.json"
            wrong_mode.write_bytes(payload)
            os.chmod(wrong_mode, 0o644)
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.load_sealed(wrong_mode, schema_version="bad-ready-v1")

            empty_wrong_mode = root / "empty-wrong-mode.json"
            empty_wrong_mode.touch(mode=0o644)
            os.chmod(empty_wrong_mode, 0o644)
            started = time.monotonic()
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.wait_load_sealed(
                    empty_wrong_mode,
                    schema_version="bad-ready-v1",
                    label="empty wrong-mode receipt",
                )
            self.assertLess(time.monotonic() - started, 0.5)

            linked = root / "linked.json"
            guard.write_create_only(linked, value)
            os.link(linked, root / "linked-again.json")
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.load_sealed(linked, schema_version="bad-ready-v1")

            other = root / "other.json"
            guard.write_create_only(other, value)
            real_lstat = Path.lstat

            def swapped_lstat(candidate: Path):
                if candidate == linked:
                    return real_lstat(other)
                return real_lstat(candidate)

            with mock.patch.object(Path, "lstat", new=swapped_lstat):
                with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                    guard.load_sealed(linked, schema_version="bad-ready-v1")

            malformed = root / "malformed.json"
            malformed.write_bytes(b"{\n")
            os.chmod(malformed, 0o444)
            started = time.monotonic()
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.wait_load_sealed(
                    malformed,
                    schema_version="bad-ready-v1",
                    label="stable malformed receipt",
                )
            self.assertLess(time.monotonic() - started, 0.5)

    def test_terminal_publish_ignores_close_error_after_fchmod(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            value = guard.seal({"schema_version": "terminal-close-v1", "value": 5})
            real_close = os.close

            def close_then_raise_for_published_regular(descriptor: int) -> None:
                observed = os.fstat(descriptor)
                published = stat.S_ISREG(observed.st_mode) and (
                    stat.S_IMODE(observed.st_mode) == 0o444
                )
                real_close(descriptor)
                if published:
                    raise OSError("injected close error after terminal fchmod")

            with mock.patch.object(os, "close", side_effect=close_then_raise_for_published_regular):
                guard.write_create_only(path, value)
            self.assertEqual(
                guard.load_sealed(path, schema_version="terminal-close-v1"), value
            )

    def _pending_prior_claim(
        self,
        root: Path,
        *,
        existing_job: str = "765432",
        duplicate_identity: bool = False,
    ):
        root = root.resolve(strict=True)
        claims = mkdir(root / "claims")
        prior_lifecycle = mkdir(root / "prior-lifecycle")
        current_lifecycle = mkdir(root / "current-lifecycle")
        port = 32123
        prior_candidate = "prior-sp4-a-candidate-00"
        prior_digest = hashlib.sha256(prior_candidate.encode("ascii")).hexdigest()
        prior_rdzv = (
            f"saic-{existing_job}-sp4-a-c00-{prior_digest[:16]}-l01"
        )
        prior_info = prior_lifecycle.lstat()
        claim = guard.seal(
            {
                "schema_version": guard.CLAIM_SCHEMA_VERSION,
                "status": "reserved_before_generation_runtime",
                "slurm_job_id": existing_job,
                "group_id": "sp4-a",
                "candidate_index": 0,
                "candidate_id": prior_candidate,
                "launch_ordinal": 1,
                "rdzv_id": prior_rdzv,
                "rdzv_backend": "c10d",
                "rdzv_endpoint_request": "127.0.0.1:0",
                "actual_master_addr": "127.0.0.1",
                "actual_master_port": port,
                "lifecycle_dir": str(prior_lifecycle),
                "lifecycle_dir_identity": f"{prior_info.st_dev}:{prior_info.st_ino}",
                "admission_receipt_path": str(prior_lifecycle / "admission.json"),
                "torch_disable_share_rdzv_tcp_store": "0",
                "shared_tcp_store_bootstrap": True,
                "kernel_selected_free_port": True,
                "port_claim_create_only_across_both_groups_for_this_job": True,
                "generation_runtime_entered": False,
                "scientific_spec_changed": False,
                "authority": guard.AUTHORITY,
            }
        )
        guard.write_create_only(claims / f"port-{port}.json", claim)

        current_job = "765432"
        if duplicate_identity:
            current_group = "sp4-a"
            current_index = 0
            current_candidate = prior_candidate
            current_rdzv = prior_rdzv
        else:
            current_group = "sp4-b"
            current_index = 0
            current_candidate = "current-sp4-b-candidate-00"
            current_digest = hashlib.sha256(
                current_candidate.encode("ascii")
            ).hexdigest()
            current_rdzv = (
                f"saic-{current_job}-sp4-b-c00-{current_digest[:16]}-l01"
            )
        current_info = current_lifecycle.lstat()
        args = argparse.Namespace(
            slurm_job_id=current_job,
            group_id=current_group,
            candidate_index=current_index,
            candidate_id=current_candidate,
            launch_ordinal=1,
            expected_rdzv_id=current_rdzv,
            lifecycle_dir_identity=(
                f"{current_info.st_dev}:{current_info.st_ino}"
            ),
        )

        def publish_valid_prior_admission() -> dict:
            packets = []
            for rank in range(guard.WORLD_SIZE):
                packet = guard.seal(
                    {
                        "schema_version": guard.RANK_SCHEMA_VERSION,
                        "status": "prepared_before_generation_runtime",
                        "slurm_job_id": existing_job,
                        "group_id": "sp4-a",
                        "candidate_index": 0,
                        "candidate_id": prior_candidate,
                        "launch_ordinal": 1,
                        "rdzv_id": prior_rdzv,
                        "rdzv_backend": "c10d",
                        "rdzv_endpoint_request": "127.0.0.1:0",
                        "actual_master_addr": "127.0.0.1",
                        "actual_master_port": port,
                        "rank": rank,
                        "local_rank": rank,
                        "world_size": guard.WORLD_SIZE,
                        "local_world_size": guard.WORLD_SIZE,
                        "port_claim_receipt_digest": claim["receipt_digest"],
                        "runtime_sha256": guard.EXPECTED_RUNTIME_SHA256,
                        "torch_disable_share_rdzv_tcp_store": "0",
                        "shared_tcp_store_bootstrap": True,
                        "generation_runtime_entered_before_admission": False,
                        "scientific_spec_changed": False,
                        "authority": guard.AUTHORITY,
                    }
                )
                guard.write_create_only(
                    prior_lifecycle / f"rank-{rank}.json", packet
                )
                packets.append(packet)
            decision = guard.seal(
                {
                    "schema_version": guard.DECISION_SCHEMA_VERSION,
                    "status": "exact_world4_admitted_before_generation_runtime",
                    "slurm_job_id": existing_job,
                    "group_id": "sp4-a",
                    "candidate_index": 0,
                    "candidate_id": prior_candidate,
                    "launch_ordinal": 1,
                    "rdzv_id": prior_rdzv,
                    "actual_master_addr": "127.0.0.1",
                    "actual_master_port": port,
                    "world_size": guard.WORLD_SIZE,
                    "rank_order": list(range(guard.WORLD_SIZE)),
                    "rank_packet_digests": [
                        packet["receipt_digest"] for packet in packets
                    ],
                    "port_claim_receipt_digest": claim["receipt_digest"],
                    "runtime_sha256": guard.EXPECTED_RUNTIME_SHA256,
                    "torch_disable_share_rdzv_tcp_store": "0",
                    "shared_tcp_store_bootstrap": True,
                    "all_four_ranks_admitted": True,
                    "generation_runtime_entry_authorized": True,
                    "scientific_spec_changed": False,
                    "authority": guard.AUTHORITY,
                }
            )
            guard.write_create_only(prior_lifecycle / "admission.json", decision)
            return decision

        return (
            args,
            claims,
            current_lifecycle,
            port,
            claim,
            publish_valid_prior_admission,
        )

    def test_same_job_claim_loser_waits_for_full_prior_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                args,
                claims,
                lifecycle,
                port,
                claim,
                publish,
            ) = self._pending_prior_claim(Path(temporary))
            published = []

            def delayed_publish() -> None:
                time.sleep(0.05)
                published.append(publish())

            thread = threading.Thread(target=delayed_publish)
            thread.start()
            started = time.monotonic()
            observed_claim, collided = guard._claim_for_worker(
                args=args,
                claim_root=claims,
                lifecycle=lifecycle,
                port=port,
                address="127.0.0.1",
                shared_store_environment_value="0",
            )
            thread.join()
            self.assertIsNone(observed_claim)
            self.assertTrue(collided)
            self.assertGreaterEqual(time.monotonic() - started, 0.04)
            self.assertEqual(len(published), 1)
            collision = guard.load_sealed(
                lifecycle / "collision.json",
                schema_version=guard.COLLISION_SCHEMA_VERSION,
                exact_fields=guard.COLLISION_FIELDS,
            )
            self.assertEqual(
                collision["existing_claim_receipt_digest"], claim["receipt_digest"]
            )
            self.assertEqual(
                collision["existing_admission_receipt_digest"],
                published[0]["receipt_digest"],
            )

    def test_unadmitted_foreign_duplicate_or_tampered_claim_never_collides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, claims, lifecycle, port, _, _ = self._pending_prior_claim(
                mkdir(root / "never")
            )
            with mock.patch.object(guard, "WAIT_SECONDS", 0.1):
                with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                    guard._claim_for_worker(
                        args=args,
                        claim_root=claims,
                        lifecycle=lifecycle,
                        port=port,
                        address="127.0.0.1",
                        shared_store_environment_value="0",
                    )
            self.assertFalse((lifecycle / "collision.json").exists())

            args, claims, lifecycle, port, _, _ = self._pending_prior_claim(
                mkdir(root / "foreign"), existing_job="999999"
            )
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard._claim_for_worker(
                    args=args,
                    claim_root=claims,
                    lifecycle=lifecycle,
                    port=port,
                    address="127.0.0.1",
                    shared_store_environment_value="0",
                )
            self.assertFalse((lifecycle / "collision.json").exists())

            args, claims, lifecycle, port, _, _ = self._pending_prior_claim(
                mkdir(root / "duplicate"), duplicate_identity=True
            )
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard._claim_for_worker(
                    args=args,
                    claim_root=claims,
                    lifecycle=lifecycle,
                    port=port,
                    address="127.0.0.1",
                    shared_store_environment_value="0",
                )
            self.assertFalse((lifecycle / "collision.json").exists())

            args, claims, lifecycle, port, _, _ = self._pending_prior_claim(
                mkdir(root / "tampered")
            )
            prior = Path(temporary) / "tampered" / "prior-lifecycle"

            def publish_tampered() -> None:
                time.sleep(0.05)
                guard.write_create_only(
                    prior / "admission.json",
                    guard.seal(
                        {
                            "schema_version": guard.DECISION_SCHEMA_VERSION,
                            "status": "forged",
                        }
                    ),
                )

            thread = threading.Thread(target=publish_tampered)
            thread.start()
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard._claim_for_worker(
                    args=args,
                    claim_root=claims,
                    lifecycle=lifecycle,
                    port=port,
                    address="127.0.0.1",
                    shared_store_environment_value="0",
                )
            thread.join()
            self.assertFalse((lifecycle / "collision.json").exists())

    def test_exact60_positive_and_collision_retry_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), one_retry=True)
            self.assertEqual(fixture.audit(), 0)
            receipt = guard.load_sealed(
                fixture.logs / "audit.json",
                schema_version=guard.JOB_AUDIT_SCHEMA_VERSION,
            )
            self.assertEqual(receipt["completion_receipt_count"], 60)
            self.assertEqual(receipt["rank_packet_count"], 240)
            self.assertEqual(receipt["all_launch_rdzv_id_count"], 61)

    def test_terminal_audit_waits_for_delayed_completion_and_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), one_retry=True)
            completion_path = fixture.completions[("sp4-a", 0)]
            collision_path = fixture.collisions[("sp4-a", 1)]
            completion_payload = completion_path.read_bytes()
            collision_payload = collision_path.read_bytes()
            os.chmod(completion_path, 0o600)
            os.chmod(collision_path, 0o600)
            completion_path.write_bytes(b"")
            collision_path.write_bytes(b"")

            def publish() -> None:
                time.sleep(0.05)
                with completion_path.open("wb") as handle:
                    handle.write(completion_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(completion_path, 0o444)
                time.sleep(0.10)
                with collision_path.open("wb") as handle:
                    handle.write(collision_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(collision_path, 0o444)

            thread = threading.Thread(target=publish)
            thread.start()
            self.assertEqual(fixture.audit("delayed-terminal-audit.json"), 0)
            thread.join()

    def _assert_audit_rejects(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), one_retry=True)
            mutate(fixture)
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                fixture.audit()

    def test_rejects_resigned_candidate_index_swap(self) -> None:
        def mutate(fixture: Fixture) -> None:
            path = fixture.completions[("sp4-a", 2)]
            value = guard.load_sealed(
                path,
                schema_version=guard.COMPLETION_SCHEMA_VERSION,
                exact_fields=guard.COMPLETION_FIELDS,
            )
            value.pop("receipt_digest")
            value["candidate_index"] = 3
            rewrite_json(path, guard.seal(value))

        self._assert_audit_rejects(mutate)

    def test_rejects_coordinated_resign_of_entire_identity_chain_against_plan(self) -> None:
        def mutate(fixture: Fixture) -> None:
            group_id, index = "sp4-a", 2
            old_id = "candidate-sp4-a-02"
            new_id = "coordinated-forged-candidate-02"
            old_completion_path = fixture.completions[(group_id, index)]
            completion = guard.load_sealed(
                old_completion_path,
                schema_version=guard.COMPLETION_SCHEMA_VERSION,
                exact_fields=guard.COMPLETION_FIELDS,
            )
            old_output = fixture.attempts / old_id
            new_output = fixture.attempts / new_id
            old_output.rename(new_output)
            attempt_path = new_output / guard.ATTEMPT_RECEIPT_BASENAME
            attempt = guard.load_sealed(
                attempt_path,
                schema_version=guard.ATTEMPT_SCHEMA_VERSION,
            )
            attempt.pop("receipt_digest")
            attempt["candidate"]["candidate_id"] = new_id
            rewrite_json(attempt_path, guard.seal(attempt))

            old_candidate_root = old_completion_path.parent.parent
            new_digest = hashlib.sha256(new_id.encode("ascii")).hexdigest()
            new_candidate_root = old_candidate_root.parent / f"candidate-{index:02d}-{new_digest[:16]}"
            old_candidate_root.rename(new_candidate_root)
            lifecycle = new_candidate_root / "launch-01"
            completion_path = lifecycle / "completion.json"
            new_rdzv_id = f"saic-{fixture.job_id}-{group_id}-c{index:02d}-{new_digest[:16]}-l01"
            claim_path = Path(completion["permanent_job_local_port_claim_path"])
            claim = guard.load_sealed(
                claim_path,
                schema_version=guard.CLAIM_SCHEMA_VERSION,
                exact_fields=guard.CLAIM_FIELDS,
            )
            claim.pop("receipt_digest")
            claim["candidate_id"] = new_id
            claim["rdzv_id"] = new_rdzv_id
            claim["lifecycle_dir"] = str(lifecycle)
            claim["admission_receipt_path"] = str(lifecycle / "admission.json")
            claim = guard.seal(claim)
            rewrite_json(claim_path, claim)

            packets = []
            for rank in range(4):
                packet_path = lifecycle / f"rank-{rank}.json"
                packet = guard.load_sealed(
                    packet_path,
                    schema_version=guard.RANK_SCHEMA_VERSION,
                    exact_fields=guard.RANK_FIELDS,
                )
                packet.pop("receipt_digest")
                packet["candidate_id"] = new_id
                packet["rdzv_id"] = new_rdzv_id
                packet["port_claim_receipt_digest"] = claim["receipt_digest"]
                packet = guard.seal(packet)
                rewrite_json(packet_path, packet)
                packets.append(packet)

            decision_path = lifecycle / "admission.json"
            decision = guard.load_sealed(
                decision_path,
                schema_version=guard.DECISION_SCHEMA_VERSION,
                exact_fields=guard.DECISION_FIELDS,
            )
            decision.pop("receipt_digest")
            decision["candidate_id"] = new_id
            decision["rdzv_id"] = new_rdzv_id
            decision["port_claim_receipt_digest"] = claim["receipt_digest"]
            decision["rank_packet_digests"] = [p["receipt_digest"] for p in packets]
            decision = guard.seal(decision)
            rewrite_json(decision_path, decision)

            completion.pop("receipt_digest")
            completion["candidate_id"] = new_id
            completion["rdzv_id"] = new_rdzv_id
            completion["permanent_job_local_port_claim_sha256"] = guard.ready_file_sha256(
                claim_path, label="coordinated forged claim"
            )
            completion["port_claim_receipt_digest"] = claim["receipt_digest"]
            completion["rank_packet_digests"] = [p["receipt_digest"] for p in packets]
            completion["admission_receipt_digest"] = decision["receipt_digest"]
            completion["candidate_output"] = str(new_output)
            completion["attempt_receipt_path"] = str(attempt_path)
            completion["attempt_receipt_sha256"] = guard.ready_file_sha256(
                attempt_path, label="coordinated forged attempt"
            )
            completion["attempt_receipt_digest"] = guard.load_sealed(
                attempt_path, schema_version=guard.ATTEMPT_SCHEMA_VERSION
            )["receipt_digest"]
            rewrite_json(completion_path, guard.seal(completion))

        self._assert_audit_rejects(mutate)

    def test_rejects_resigned_arbitrary_collision_run_id(self) -> None:
        def mutate(fixture: Fixture) -> None:
            path = fixture.collisions[("sp4-a", 1)]
            value = guard.load_sealed(
                path,
                schema_version=guard.COLLISION_SCHEMA_VERSION,
                exact_fields=guard.COLLISION_FIELDS,
            )
            value.pop("receipt_digest")
            value["rdzv_id"] = "arbitrary-retry-run-id"
            rewrite_json(path, guard.seal(value))

        self._assert_audit_rejects(mutate)

    def test_rejects_reused_collision_run_id(self) -> None:
        def mutate(fixture: Fixture) -> None:
            path = fixture.collisions[("sp4-a", 1)]
            completion = guard.load_sealed(
                fixture.completions[("sp4-a", 0)],
                schema_version=guard.COMPLETION_SCHEMA_VERSION,
                exact_fields=guard.COMPLETION_FIELDS,
            )
            value = guard.load_sealed(
                path,
                schema_version=guard.COLLISION_SCHEMA_VERSION,
                exact_fields=guard.COLLISION_FIELDS,
            )
            value.pop("receipt_digest")
            value["rdzv_id"] = completion["rdzv_id"]
            rewrite_json(path, guard.seal(value))

        self._assert_audit_rejects(mutate)

    def test_rejects_external_attempt_receipt_even_when_resigned(self) -> None:
        def mutate(fixture: Fixture) -> None:
            path = fixture.completions[("sp4-a", 2)]
            value = guard.load_sealed(
                path,
                schema_version=guard.COMPLETION_SCHEMA_VERSION,
                exact_fields=guard.COMPLETION_FIELDS,
            )
            other = guard.load_sealed(
                fixture.completions[("sp4-a", 3)],
                schema_version=guard.COMPLETION_SCHEMA_VERSION,
                exact_fields=guard.COMPLETION_FIELDS,
            )
            value.pop("receipt_digest")
            for field in (
                "candidate_output", "attempt_receipt_path", "attempt_receipt_sha256",
                "attempt_receipt_digest",
            ):
                value[field] = other[field]
            rewrite_json(path, guard.seal(value))

        self._assert_audit_rejects(mutate)

    def test_rejects_extra_success_evidence(self) -> None:
        def mutate(fixture: Fixture) -> None:
            extra = fixture.completions[("sp4-a", 2)].parent / "extra.json"
            write_json(extra, guard.seal({"schema_version": "extra-v1"}))

        self._assert_audit_rejects(mutate)

    def test_rejects_resigned_rank_packet_with_wrong_status(self) -> None:
        def mutate(fixture: Fixture) -> None:
            completion_path = fixture.completions[("sp4-a", 2)]
            packet_path = completion_path.parent / "rank-1.json"
            packet = guard.load_sealed(
                packet_path,
                schema_version=guard.RANK_SCHEMA_VERSION,
                exact_fields=guard.RANK_FIELDS,
            )
            packet.pop("receipt_digest")
            packet["status"] = "forged"
            packet = guard.seal(packet)
            rewrite_json(packet_path, packet)
            decision_path = completion_path.parent / "admission.json"
            decision = guard.load_sealed(
                decision_path,
                schema_version=guard.DECISION_SCHEMA_VERSION,
                exact_fields=guard.DECISION_FIELDS,
            )
            decision.pop("receipt_digest")
            decision["rank_packet_digests"][1] = packet["receipt_digest"]
            decision = guard.seal(decision)
            rewrite_json(decision_path, decision)
            completion = guard.load_sealed(
                completion_path,
                schema_version=guard.COMPLETION_SCHEMA_VERSION,
                exact_fields=guard.COMPLETION_FIELDS,
            )
            completion.pop("receipt_digest")
            completion["rank_packet_digests"][1] = packet["receipt_digest"]
            completion["admission_receipt_digest"] = decision["receipt_digest"]
            rewrite_json(completion_path, guard.seal(completion))

        self._assert_audit_rejects(mutate)

    def test_rejects_coordinated_shared_store_disable_one_chain(self) -> None:
        def mutate(fixture: Fixture) -> None:
            completion_path = fixture.completions[("sp4-a", 2)]
            completion = guard.load_sealed(
                completion_path,
                schema_version=guard.COMPLETION_SCHEMA_VERSION,
                exact_fields=guard.COMPLETION_FIELDS,
            )
            claim_path = Path(completion["permanent_job_local_port_claim_path"])
            claim = guard.load_sealed(
                claim_path,
                schema_version=guard.CLAIM_SCHEMA_VERSION,
                exact_fields=guard.CLAIM_FIELDS,
            )
            claim.pop("receipt_digest")
            claim["torch_disable_share_rdzv_tcp_store"] = "1"
            claim["shared_tcp_store_bootstrap"] = False
            claim = guard.seal(claim)
            rewrite_json(claim_path, claim)
            packets = []
            for rank in range(4):
                path = completion_path.parent / f"rank-{rank}.json"
                packet = guard.load_sealed(
                    path,
                    schema_version=guard.RANK_SCHEMA_VERSION,
                    exact_fields=guard.RANK_FIELDS,
                )
                packet.pop("receipt_digest")
                packet["port_claim_receipt_digest"] = claim["receipt_digest"]
                packet["torch_disable_share_rdzv_tcp_store"] = "1"
                packet["shared_tcp_store_bootstrap"] = False
                packet = guard.seal(packet)
                rewrite_json(path, packet)
                packets.append(packet)
            decision_path = completion_path.parent / "admission.json"
            decision = guard.load_sealed(
                decision_path,
                schema_version=guard.DECISION_SCHEMA_VERSION,
                exact_fields=guard.DECISION_FIELDS,
            )
            decision.pop("receipt_digest")
            decision["port_claim_receipt_digest"] = claim["receipt_digest"]
            decision["rank_packet_digests"] = [p["receipt_digest"] for p in packets]
            decision["torch_disable_share_rdzv_tcp_store"] = "1"
            decision["shared_tcp_store_bootstrap"] = False
            decision = guard.seal(decision)
            rewrite_json(decision_path, decision)
            completion.pop("receipt_digest")
            completion["permanent_job_local_port_claim_sha256"] = guard.ready_file_sha256(
                claim_path, label="shared store poisoned claim"
            )
            completion["port_claim_receipt_digest"] = claim["receipt_digest"]
            completion["rank_packet_digests"] = [p["receipt_digest"] for p in packets]
            completion["admission_receipt_digest"] = decision["receipt_digest"]
            completion["torch_disable_share_rdzv_tcp_store"] = "1"
            completion["shared_tcp_store_bootstrap"] = False
            rewrite_json(completion_path, guard.seal(completion))

        self._assert_audit_rejects(mutate)

    def test_rejects_symlink_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            value = guard.seal({"schema_version": "canonical-v1"})
            write_json(target, value)
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.load_sealed(link, schema_version="canonical-v1")

    def test_rejects_noncanonical_resigned_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "noncanonical.json"
            value = guard.seal({"schema_version": "canonical-v1", "value": 4})
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
            os.chmod(path, 0o444)
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.load_sealed(path, schema_version="canonical-v1")

    def test_collision_classifier_allows_only_valid_sealed_absent_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), one_retry=True)
            args = fixture.collision_args()
            held_output = fixture.root.parent / "held-candidate-output"
            Path(args.candidate_output).rename(held_output)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(guard.admit_collision(args), 0)

    def test_collision_classifier_rejects_output_present_generic_and_foreign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), one_retry=True)
            args = fixture.collision_args()
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.admit_collision(args)
            held_output = fixture.root.parent / "held-candidate-output"
            Path(args.candidate_output).rename(held_output)
            collision_path = Path(args.lifecycle_dir) / "collision.json"
            hidden_collision = collision_path.with_name("hidden-collision.json")
            collision_path.rename(hidden_collision)
            with mock.patch.object(guard, "WAIT_SECONDS", 0.1):
                with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                    guard.admit_collision(args)
            hidden_collision.rename(collision_path)
            args.slurm_job_id = "999999"
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.admit_collision(args)

    def test_retry_budget_identity_accepts_16_and_rejects_17(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), one_retry=True)
            args = fixture.collision_args()
            args.launch_ordinal = 16
            args.expected_rdzv_id = args.expected_rdzv_id[:-2] + "16"
            guard.validate_identity(args)
            args.launch_ordinal = 17
            args.expected_rdzv_id = args.expected_rdzv_id[:-2] + "17"
            with self.assertRaises(guard.SAICT2VRendezvousGuardError):
                guard.validate_identity(args)


if __name__ == "__main__":
    unittest.main()
