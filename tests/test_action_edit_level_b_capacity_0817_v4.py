from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CAPACITY_PATH = (
    ROOT / "methods" / "bernini_action_editing"
    / "action_edit_level_b_p2_00435_capacity_0817_v4.py"
)
IDLE_RAW_PATH = Path(
    "/private/tmp/level-b-v4-rocm-combined-single-20260817/stdout"
)
BUSY_RAW_PATH = Path(
    "/private/tmp/level-b-v4-direct-node-ssh-20260817/"
    "rocm-controlled-allflags-v3.stdout"
)


def load_capacity():
    name = "action_edit_level_b_p2_00435_capacity_0817_v4_test"
    spec = importlib.util.spec_from_file_location(name, CAPACITY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


capacity = load_capacity()


def report_value(*, used: int = 13_078_528, use: int = 0):
    value = {}
    for index, identity in enumerate(capacity.CARD_IDENTITIES):
        unique_id, serial, bus, node_id, guid = identity
        value[f"card{index}"] = {
            **capacity.FIXED_CARD_FIELDS,
            "Unique ID": unique_id,
            "Serial Number": serial,
            "PCI Bus": bus,
            "Node ID": node_id,
            "GUID": guid,
            "GPU use (%)": str(use),
            "GFX Activity": str(1_000_000 + index),
            "VRAM Total Used Memory (B)": str(used),
        }
    value["system"] = {"PID2726801": "gpuagent, 0, 0, 0, 0"}
    return value


def report_raw(value=None):
    if value is None:
        value = report_value()
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def closure_value():
    def plain(path, pin, number):
        mode, nlink, size, digest = pin
        return {"path": str(path), "mode": mode, "nlink": nlink, "size": size,
                "sha256": digest, "uid": 0, "gid": 0}

    def link(path, text, target, number):
        return {"path": str(path), "mode": 0o777, "nlink": 1,
                "size": len(text), "link_text": text,
                "resolved_target": str(target), "uid": 0, "gid": 0}

    return {
        "python": plain(capacity.PYTHON, capacity.PYTHON_PIN, 1),
        "source_directory": {
            "path": str(capacity.SOURCE_DIR), "mode": 0o755, "uid": 0, "gid": 0,
            "entries": ["__pycache__", "rocm_smi.py", "rsmiBindings.py",
                        "rsmiBindingsInit.py"],
        },
        "sources": [
            plain(path, capacity.SOURCE_PINS[path], index + 3)
            for index, path in enumerate(capacity.SOURCE_PINS)
        ],
        "library_link": link(
            capacity.ROCM_LIBRARY_LINK, capacity.ROCM_LIBRARY_LINK_TEXT,
            capacity.ROCM_LIBRARY_TARGET, 6,
        ),
        "library_target": plain(
            capacity.ROCM_LIBRARY_TARGET, capacity.ROCM_LIBRARY_TARGET_PIN, 7
        ),
        "shell_link": link(
            capacity.SHELL_LINK, capacity.SHELL_LINK_TEXT, capacity.SHELL_TARGET, 8
        ),
        "shell_target": plain(capacity.SHELL_TARGET, capacity.SHELL_TARGET_PIN, 9),
        "ps": plain(capacity.PS, capacity.PS_PIN, 10),
    }


def remote_shell_value():
    def plain(path, pin, uid=0, gid=0):
        mode, nlink, size, digest = pin[:4]
        return {"path": str(path), "mode": mode, "nlink": nlink, "size": size,
                "sha256": digest, "uid": uid, "gid": gid}

    return {
        "account_mapping": {
            "name": "guangyi.chen", "uid": 2012, "gid": 2000,
            "home": str(capacity.REMOTE_HOME), "shell": "/bin/bash",
        },
        "bin_link": {
            "path": str(capacity.BIN_LINK), "mode": 0o777, "nlink": 1,
            "size": len(capacity.BIN_LINK_TEXT), "link_text": capacity.BIN_LINK_TEXT,
            "resolved_target": str(capacity.USR_BIN), "uid": 0, "gid": 0,
        },
        "usr_bin_directory": {
            "path": str(capacity.USR_BIN), "mode": 0o755, "uid": 0, "gid": 0,
        },
        "bash": plain(capacity.BASH, capacity.BASH_PIN),
        "env_tool": plain(capacity.ENV_TOOL, capacity.ENV_TOOL_PIN),
        "bashrc": plain(capacity.BASHRC, capacity.BASHRC_PIN, 2012, 2000),
        "absent_startup_paths": [str(path) for path in capacity.REMOTE_STARTUP_ABSENT],
        "environment_after_absolute_env_i": capacity.remote_target_environment(),
        "same_user_login_shell_startup_and_transitive_conda_are_trusted_boundary": True,
        "remote_shell_was_entered_before_absolute_env_i": True,
        "nss_and_passwd_resolution_are_system_trust_boundary": True,
    }


def ssh_authority_value():
    def plain(path, pin, uid=0, gid=0):
        mode, nlink, size, digest = pin[:4]
        return {"path": str(path), "mode": mode, "nlink": nlink, "size": size,
                "sha256": digest, "uid": uid, "gid": gid}

    return {
        "ssh_client": plain(capacity.SSH, capacity.SSH_PIN),
        "sealed_known_hosts": plain(
            capacity.KNOWN_HOSTS, capacity.KNOWN_HOSTS_PIN, 2012, 2000
        ),
        "capacity_member": {
            "path": str(capacity.CAPACITY_SELF), "mode": 0o444, "nlink": 1,
            "size": 50_000, "sha256": "f" * 64, "uid": 2012, "gid": 2000,
        },
    }


def receipt(raw=None, phase="foreground", challenge="a" * 64, now=None):
    raw = report_raw() if raw is None else raw
    now = time.time_ns() if now is None else now
    if phase == "step":
        return capacity.build_receipt(raw, phase, challenge, now, closure_value())
    target = capacity.build_receipt(
        raw, phase, challenge, now, closure_value(), {
            "mode": "direct-node-ssh-target",
            "remote_shell_boundary": remote_shell_value(),
            "remote_shell_boundary_stable_pre_and_post": True,
            "capacity_member": ssh_authority_value()["capacity_member"],
            "capacity_member_stable_pre_and_post": True,
            "capacity_member_sha256_must_be_bound_by_launch_authority": True,
        },
    )
    return capacity.finalize_direct_node_receipt(
        target, capacity.base64.b64encode(target).decode("ascii"), phase,
        challenge, ssh_authority_value(), now_ns=now,
    )


class LevelBV4CapacityTests(unittest.TestCase):
    def assert_refused(self, raw: bytes):
        with self.assertRaises(capacity.CapacityError):
            capacity.parse_report(raw)

    def test_challenge_is_fresh_fixed_32_byte_getrandom_hex_without_newline(self):
        with mock.patch.object(
            capacity.os, "getrandom", return_value=b"\xab" * 32, create=True
        ) as get:
            with mock.patch.object(capacity.sys.stdout, "write") as write:
                self.assertEqual(capacity.main(["challenge"]), 0)
        get.assert_called_once_with(32)
        write.assert_called_once_with("ab" * 32)

    def test_probe_base64_is_single_process_ascii_framing_without_lf(self):
        raw = receipt(phase="step", challenge="7" * 64)
        with mock.patch.object(capacity, "probe", return_value=raw) as probe, \
                mock.patch.object(capacity.sys.stdout.buffer, "write") as write:
            self.assertEqual(
                capacity.main(["probe-base64", "step", "7" * 64]), 0
            )
        probe.assert_called_once_with("step", "7" * 64)
        encoded = capacity.base64.b64encode(raw)
        self.assertNotIn(b"\n", encoded)
        write.assert_called_once_with(encoded)

    def test_remote_probe_is_one_hardened_ssh_and_binds_target_receipt(self):
        phase = "foreground"; challenge = "2" * 64; now = time.time_ns()
        target = capacity.build_receipt(
            report_raw(), phase, challenge, now, closure_value(), {
                "mode": "direct-node-ssh-target",
                "remote_shell_boundary": remote_shell_value(),
                "remote_shell_boundary_stable_pre_and_post": True,
                "capacity_member": ssh_authority_value()["capacity_member"],
                "capacity_member_stable_pre_and_post": True,
                "capacity_member_sha256_must_be_bound_by_launch_authority": True,
            },
        )
        encoded = capacity.base64.b64encode(target)
        completed = subprocess.CompletedProcess(
            capacity.ssh_argv(phase, challenge), 0, stdout=encoded, stderr=b""
        )
        authority = ssh_authority_value()
        with mock.patch.object(
            capacity, "ssh_authority_snapshot", return_value=authority
        ) as snapshot, mock.patch.object(
            capacity.subprocess, "run", return_value=completed
        ) as run, mock.patch.object(capacity.time, "time_ns", return_value=now):
            raw = capacity.direct_node_probe(phase, challenge)
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        self.assertEqual(args, (capacity.ssh_argv(phase, challenge),))
        self.assertEqual(kwargs["cwd"], "/")
        self.assertEqual(kwargs["env"], capacity.SSH_ENV)
        self.assertEqual(kwargs["timeout"], 30)
        value = capacity.validate_receipt_bytes(
            raw, hashlib.sha256(raw).hexdigest(), phase, challenge, now_ns=now
        )
        self.assertEqual(value["transport"]["mode"], "direct-node-ssh")
        self.assertEqual(
            capacity.base64.b64decode(
                value["transport"]["remote_target_receipt_base64"]
            ), target,
        )

    def test_remote_probe_rejects_nonzero_stderr_newline_and_transport_tamper(self):
        phase = "controller"; challenge = "3" * 64; now = time.time_ns()
        target = capacity.build_receipt(
            report_raw(), phase, challenge, now, closure_value(), {
                "mode": "direct-node-ssh-target",
                "remote_shell_boundary": remote_shell_value(),
                "remote_shell_boundary_stable_pre_and_post": True,
                "capacity_member": ssh_authority_value()["capacity_member"],
                "capacity_member_stable_pre_and_post": True,
                "capacity_member_sha256_must_be_bound_by_launch_authority": True,
            },
        )
        authority = ssh_authority_value()
        for returncode, stderr in ((1, b""), (0, b"\n"), (0, b"warning\n")):
            completed = subprocess.CompletedProcess(
                capacity.ssh_argv(phase, challenge), returncode,
                stdout=capacity.base64.b64encode(target), stderr=stderr,
            )
            with mock.patch.object(
                capacity, "ssh_authority_snapshot", return_value=authority
            ), mock.patch.object(capacity.subprocess, "run", return_value=completed):
                with self.assertRaises(capacity.CapacityError):
                    capacity.direct_node_probe(phase, challenge)
        raw = capacity.finalize_direct_node_receipt(
            target, capacity.base64.b64encode(target).decode("ascii"), phase,
            challenge, authority, now_ns=now,
        )
        value = json.loads(raw)
        value["transport"]["ssh_argv"][1] = "-evil"
        unsigned = dict(value); del unsigned["receipt_digest"]
        value["receipt_digest"] = hashlib.sha256(
            capacity.canonical_json_bytes(unsigned)
        ).hexdigest()
        tampered = capacity.canonical_json_bytes(value)
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                tampered, hashlib.sha256(tampered).hexdigest(), phase,
                challenge, now_ns=now,
            )

    def test_remote_ssh_argv_and_startup_trust_boundary_are_exact(self):
        challenge = "4" * 64
        argv = capacity.ssh_argv("foreground", challenge)
        self.assertEqual(argv[0:5], (
            "/usr/bin/ssh", "-F", "/dev/null", "-n", "-T"
        ))
        joined = "\n".join(argv)
        for token in (
            "BatchMode=yes", "ConnectTimeout=10", "ConnectionAttempts=1",
            "StrictHostKeyChecking=yes", "HostKeyAlgorithms=ssh-ed25519",
            "UpdateHostKeys=no", "IdentitiesOnly=yes", "GSSAPIAuthentication=no",
            "KbdInteractiveAuthentication=no", "PasswordAuthentication=no",
            "ForwardAgent=no", "ForwardX11=no", "ClearAllForwardings=yes",
            "PermitLocalCommand=no", "ControlMaster=no", "ControlPath=none",
            "CanonicalizeHostname=no", "Hostname=auh7-1b-gpu-279",
            "KnownHostsCommand=none", "VerifyHostKeyDNS=no",
            "ProxyCommand=none", "ProxyJump=none",
        ):
            self.assertIn(token, joined)
        self.assertIn(str(capacity.KNOWN_HOSTS), joined)
        self.assertEqual(argv[-2], capacity.REMOTE_USER_HOST)
        self.assertIn("/usr/bin/env -i", argv[-1])
        self.assertIn("direct-node-target-base64 foreground " + challenge, argv[-1])
        self.assertNotIn("~/.ssh/known_hosts", joined)
        shell = remote_shell_value()
        capacity.validate_remote_shell_boundary_receipt(shell)
        self.assertTrue(
            shell["same_user_login_shell_startup_and_transitive_conda_are_trusted_boundary"]
        )

    def test_frozen_live_idle_fixture_is_accepted(self):
        if not IDLE_RAW_PATH.is_file():
            self.skipTest("shared AUH read-only fixture is absent")
        raw = IDLE_RAW_PATH.read_bytes()
        self.assertEqual(len(raw), 3928)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "e1bbe3310e6e5918833574eca6eb2604b428797ce3b1a4e1c72bba9d39b400cf",
        )
        cards, processes = capacity.parse_report(raw)
        self.assertEqual([row["index"] for row in cards], list(range(8)))
        self.assertEqual({row["gpu_use_percent"] for row in cards}, {0})
        self.assertEqual({row["vram_total_bytes"] for row in cards}, {68_702_699_520})
        self.assertEqual([row["name"] for row in processes], ["gpuagent"])

    def test_frozen_live_busy_fixture_is_rejected(self):
        if not BUSY_RAW_PATH.is_file():
            self.skipTest("shared AUH busy fixture is absent")
        raw = BUSY_RAW_PATH.read_bytes()
        self.assertEqual(len(raw), 4443)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "91f4d8b18b572040c5b2f58a28989af0f2e3c297ade05426d49a0c25beee74d1",
        )
        self.assert_refused(raw)

    def test_exact_95_percent_boundary_passes_and_one_byte_below_fails(self):
        boundary_used = capacity.EXPECTED_TOTAL_BYTES // 20
        cards, _ = capacity.parse_report(report_raw(report_value(used=boundary_used)))
        self.assertEqual({row["free_basis_points_floor"] for row in cards}, {9500})
        self.assert_refused(report_raw(report_value(used=boundary_used + 1)))

    def test_use_must_be_exact_integer_zero_on_every_card(self):
        for bad in ("1", "00", "-1", "0.0", "true"):
            value = report_value()
            value["card4"]["GPU use (%)"] = bad
            self.assert_refused(report_raw(value))

    def test_total_is_exact_and_used_is_canonical_bounded_uint(self):
        for bad in (str(capacity.EXPECTED_TOTAL_BYTES - 1),
                    str(capacity.EXPECTED_TOTAL_BYTES + 1)):
            value = report_value()
            value["card3"]["VRAM Total Memory (B)"] = bad
            self.assert_refused(report_raw(value))
        for bad in ("-1", "+1", "01", "1.0", str(capacity.EXPECTED_TOTAL_BYTES + 1)):
            value = report_value()
            value["card3"]["VRAM Total Used Memory (B)"] = bad
            self.assert_refused(report_raw(value))

    def test_average_cannot_mask_one_bad_card(self):
        value = report_value(used=0)
        value["card7"]["VRAM Total Used Memory (B)"] = str(
            capacity.EXPECTED_TOTAL_BYTES // 20 + 1
        )
        self.assert_refused(report_raw(value))

    def test_card_topology_and_full_identity_are_exact(self):
        mutations = []
        missing = report_value(); del missing["card7"]; mutations.append(missing)
        extra = report_value(); extra["card8"] = copy.deepcopy(extra["card7"]); mutations.append(extra)
        wrong_uid = report_value(); wrong_uid["card0"]["Unique ID"] = "0x0"; mutations.append(wrong_uid)
        wrong_bus = report_value(); wrong_bus["card1"]["PCI Bus"] = "0000:09:00.0"; mutations.append(wrong_bus)
        wrong_serial = report_value(); wrong_serial["card2"]["Serial Number"] = "1"; mutations.append(wrong_serial)
        wrong_name = report_value(); wrong_name["card3"]["Card Series"] = "MI210"; mutations.append(wrong_name)
        unknown = report_value(); unknown["card4"]["unknown"] = "0"; mutations.append(unknown)
        for value in mutations:
            self.assert_refused(report_raw(value))

    def test_recursive_duplicate_keys_nonfinite_nul_and_lf_are_rejected(self):
        raw = report_raw()
        top_duplicate = raw[:-2] + b',"card0":{} }\n'
        nested_duplicate = raw.replace(
            b'"GPU use (%)":"0"', b'"GPU use (%)":"0","GPU use (%)":"0"', 1
        )
        nonfinite = raw.replace(b'"GFX Activity":"1000000"', b'"GFX Activity":NaN', 1)
        for bad in (top_duplicate, nested_duplicate, nonfinite, raw + b"\n",
                    raw[:-1], raw[:-1] + b"\x00\n", b" " + raw,
                    b"\t" + raw, raw[:-1] + b"\r\n"):
            self.assert_refused(bad)

    def test_system_is_exactly_one_zero_gpuagent_with_canonical_pid(self):
        for system in (
            {},
            {"PID0": "gpuagent, 0, 0, 0, 0"},
            {"PID01": "gpuagent, 0, 0, 0, 0"},
            {"PID1": "python3.12, 0, 0, 0, 0"},
            {"PID1": "gpuagent, 1, 0, 0, 0"},
            {"PID1": "gpuagent, 0, 0, 0, 0", "PID2": "gpuagent, 0, 0, 0, 0"},
        ):
            value = report_value(); value["system"] = system
            self.assert_refused(report_raw(value))

    def test_receipt_embeds_and_reparses_exact_raw_report(self):
        challenge = "b" * 64
        now = time.time_ns()
        raw = receipt(challenge=challenge, now=now)
        value = capacity.validate_receipt_bytes(
            raw, hashlib.sha256(raw).hexdigest(), "foreground", challenge, now_ns=now
        )
        embedded = value["producer"]["rocm_smi_stdout_base64"]
        self.assertEqual(capacity.base64.b64decode(embedded), report_raw())

    def test_receipt_rejects_phase_challenge_staleness_and_summary_tamper(self):
        now = time.time_ns(); challenge = "c" * 64
        raw = receipt(challenge=challenge, now=now)
        digest = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(raw, digest, "controller", challenge, now_ns=now)
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(raw, digest, "foreground", "d" * 64, now_ns=now)
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                raw, digest, "foreground", challenge,
                now_ns=now + capacity.MAX_RECEIPT_AGE_NS + 1,
            )
        step_raw = receipt(phase="step", challenge=challenge, now=now)
        step_digest = hashlib.sha256(step_raw).hexdigest()
        archived = capacity.validate_receipt_bytes(
            step_raw, step_digest, "step", challenge,
            now_ns=now + capacity.MAX_RECEIPT_AGE_NS + 1,
            _enforce_max_age=False,
        )
        self.assertEqual(archived["sample_phase"], "step")
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                raw, digest, "foreground", challenge,
                now_ns=now + capacity.MAX_RECEIPT_AGE_NS + 1,
                _enforce_max_age=False,
            )
        value = json.loads(raw)
        value["cards"][0]["vram_used_bytes"] += 1
        unsigned = dict(value); del unsigned["receipt_digest"]
        value["receipt_digest"] = hashlib.sha256(
            capacity.canonical_json_bytes(unsigned)
        ).hexdigest()
        tampered = capacity.canonical_json_bytes(value)
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                tampered, hashlib.sha256(tampered).hexdigest(),
                "foreground", challenge, now_ns=now,
            )

    def test_archival_file_mode_revalidates_old_bytes_without_reauthorizing_freshness(self):
        challenge = "d" * 64
        issued = time.time_ns() - capacity.MAX_RECEIPT_AGE_NS - 1
        raw = receipt(phase="step", challenge=challenge, now=issued)
        digest = hashlib.sha256(raw).hexdigest()
        with mock.patch.object(
            capacity, "stable_receipt_file", return_value=raw
        ), mock.patch.object(capacity.sys.stdout.buffer, "write") as write:
            self.assertEqual(capacity.main([
                "validate-file-archival", "/absolute/receipt.json", digest,
                "step", challenge,
            ]), 0)
        write.assert_called_once_with(raw)
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(raw, digest, "step", challenge)
        future = time.time_ns() + capacity.MAX_FUTURE_SKEW_NS + 1
        future_raw = receipt(phase="step", challenge=challenge, now=future)
        future_digest = hashlib.sha256(future_raw).hexdigest()
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                future_raw, future_digest, "step", challenge,
                now_ns=future - capacity.MAX_FUTURE_SKEW_NS - 1,
            )
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                future_raw, future_digest, "step", challenge,
                now_ns=future - capacity.MAX_FUTURE_SKEW_NS - 1,
                _enforce_max_age=False,
            )
        with mock.patch.object(capacity, "stable_receipt_file") as stable:
            self.assertEqual(capacity.main([
                "validate-file-archival", "/absolute/receipt.json", digest,
                "controller", challenge,
            ]), 98)
            stable.assert_not_called()

    def test_receipt_rejects_substituted_executable_closure_even_if_redigested(self):
        now = time.time_ns(); challenge = "9" * 64
        raw = receipt(challenge=challenge, now=now)
        value = json.loads(raw)
        value["producer"]["selected_producer_closure"] = {"attacker": "substituted"}
        unsigned = dict(value); del unsigned["receipt_digest"]
        value["receipt_digest"] = hashlib.sha256(
            capacity.canonical_json_bytes(unsigned)
        ).hexdigest()
        tampered = capacity.canonical_json_bytes(value)
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                tampered, hashlib.sha256(tampered).hexdigest(),
                "foreground", challenge, now_ns=now,
            )

    def test_receipt_rejects_noncanonical_base64_unused_pad_bits(self):
        now = time.time_ns(); challenge = "4" * 64
        value = json.loads(receipt(challenge=challenge, now=now))
        encoded = value["producer"]["rocm_smi_stdout_base64"]
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        self.assertTrue(encoded.endswith("="))
        position = len(encoded.rstrip("=")) - 1
        original = alphabet.index(encoded[position])
        replacement = alphabet[(original & 0b111100) | ((original + 1) & 0b11)]
        self.assertNotEqual(replacement, encoded[position])
        value["producer"]["rocm_smi_stdout_base64"] = (
            encoded[:position] + replacement + encoded[position + 1:]
        )
        unsigned = dict(value); del unsigned["receipt_digest"]
        value["receipt_digest"] = hashlib.sha256(
            capacity.canonical_json_bytes(unsigned)
        ).hexdigest()
        raw = capacity.canonical_json_bytes(value)
        with self.assertRaises(capacity.CapacityError):
            capacity.validate_receipt_bytes(
                raw, hashlib.sha256(raw).hexdigest(),
                "foreground", challenge, now_ns=now,
            )

    def test_receipt_rejects_bool_for_every_integer_authority_and_dynamic_inode(self):
        now = time.time_ns(); challenge = "8" * 64
        base = json.loads(receipt(challenge=challenge, now=now))
        mutations = (
            ("producer", "top_level_invocation_count"),
            ("producer", "rocm_smi_exit_code"),
            ("producer", "rocm_smi_stderr_size"),
            ("thresholds", "gpu_use_percent_required"),
        )
        values = []
        for parent, key in mutations:
            value = copy.deepcopy(base); value[parent][key] = bool(value[parent][key])
            values.append(value)
        value = copy.deepcopy(base); value["cards"][0]["gpu_use_percent"] = False
        values.append(value)
        value = copy.deepcopy(base)
        value["producer"]["selected_producer_closure"]["python"]["device"] = 48
        values.append(value)
        for value in values:
            unsigned = dict(value); del unsigned["receipt_digest"]
            value["receipt_digest"] = hashlib.sha256(
                capacity.canonical_json_bytes(unsigned)
            ).hexdigest()
            raw = capacity.canonical_json_bytes(value)
            with self.assertRaises(capacity.CapacityError):
                capacity.validate_receipt_bytes(
                    raw, hashlib.sha256(raw).hexdigest(),
                    "foreground", challenge, now_ns=now,
                )

    def test_receipt_publisher_is_create_only_nofollow_fsync_and_nlink_one(self):
        now = time.time_ns(); challenge = "6" * 64
        raw = receipt(phase="step", challenge=challenge, now=now)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            started = Path(temporary) / "STARTED"
            started.mkdir(mode=0o700)
            os.chmod(started, 0o700)
            final = started / "step-capacity-receipt.json"
            with mock.patch.object(capacity, "ATTEMPT_STARTED", started):
                capacity.publish_receipt(
                    raw, digest, "step", challenge, final, now_ns=now
                )
            info = final.lstat()
            self.assertTrue(stat.S_ISREG(info.st_mode))
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(info.st_nlink, 1)
            self.assertEqual(final.read_bytes(), raw)
            self.assertEqual([path.name for path in started.iterdir()], [final.name])

    def test_foreground_receipt_publishes_only_beneath_fresh_attempt_root(self):
        now = time.time_ns(); challenge = "1" * 64
        raw = receipt(phase="foreground", challenge=challenge, now=now)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            attempt = Path(temporary) / "attempt"
            attempt.mkdir(mode=0o700); os.chmod(attempt, 0o700)
            final = attempt / "foreground-capacity-receipt.json"
            with mock.patch.object(capacity, "ATTEMPT_ROOT", attempt):
                capacity.publish_receipt(
                    raw, digest, "foreground", challenge, final, now_ns=now
                )
            self.assertEqual(final.read_bytes(), raw)
            self.assertEqual(stat.S_IMODE(final.lstat().st_mode), 0o444)
            self.assertEqual(final.lstat().st_nlink, 1)
            self.assertEqual([path.name for path in attempt.iterdir()], [final.name])

    def test_receipt_publisher_rejects_preexisting_final_and_temp_symlink(self):
        now = time.time_ns(); challenge = "5" * 64
        raw = receipt(phase="controller", challenge=challenge, now=now)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            started = Path(temporary) / "STARTED"
            started.mkdir(mode=0o700); os.chmod(started, 0o700)
            final = started / "controller-capacity-receipt.json"
            final.write_bytes(b"existing")
            with mock.patch.object(capacity, "ATTEMPT_STARTED", started):
                with self.assertRaises(capacity.CapacityError):
                    capacity.publish_receipt(
                        raw, digest, "controller", challenge, final, now_ns=now
                    )
            final.unlink()
            victim = Path(temporary) / "victim"; victim.write_bytes(b"safe")
            temp = started / f".{final.name}.{challenge}.tmp"
            temp.symlink_to(victim)
            with mock.patch.object(capacity, "ATTEMPT_STARTED", started):
                with self.assertRaises(capacity.CapacityError):
                    capacity.publish_receipt(
                        raw, digest, "controller", challenge, final, now_ns=now
                    )
            self.assertEqual(victim.read_bytes(), b"safe")
            self.assertFalse(final.exists())

    def test_invalid_phase_or_challenge_never_reaches_hostname_or_producer(self):
        for phase, challenge in (("evil", "1" * 64), ("step", "not-64hex")):
            with mock.patch.object(capacity.socket, "gethostname") as hostname, \
                    mock.patch.object(capacity.subprocess, "run") as run:
                with self.assertRaises(capacity.CapacityError):
                    capacity.probe(phase, challenge)
            hostname.assert_not_called()
            run.assert_not_called()
            raw = receipt(phase="step", challenge="1" * 64)
            with self.assertRaises(capacity.CapacityError):
                capacity.validate_receipt_bytes(
                    raw, hashlib.sha256(raw).hexdigest(), phase, challenge
                )
        with mock.patch.object(capacity, "probe") as probe:
            self.assertEqual(
                capacity.main(["probe-base64", "foreground", "1" * 64]), 98
            )
        probe.assert_not_called()

    def test_probe_is_one_bounded_top_level_invocation_without_retry(self):
        challenge = "e" * 64
        completed = subprocess.CompletedProcess(
            capacity.PRODUCER_ARGV, 0, stdout=report_raw(), stderr=b""
        )
        closure = closure_value()
        with mock.patch.object(
            capacity.socket, "gethostname", return_value=capacity.NODE
        ), mock.patch.object(
            capacity, "lexists", return_value=False
        ), mock.patch.object(
            capacity, "executable_closure_snapshot", return_value=closure
        ), mock.patch.object(
            capacity.subprocess, "run", return_value=completed
        ) as run:
            raw = capacity.probe("step", challenge)
        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        self.assertEqual(args, (capacity.PRODUCER_ARGV,))
        self.assertEqual(kwargs["cwd"], "/")
        self.assertEqual(kwargs["env"], capacity.PRODUCER_ENV)
        self.assertEqual(kwargs["timeout"], 30)
        capacity.validate_receipt_bytes(
            raw, hashlib.sha256(raw).hexdigest(), "step", challenge
        )

    def test_wrong_host_nonzero_and_any_stderr_fail_before_receipt(self):
        with mock.patch.object(capacity.socket, "gethostname", return_value="auh-login"):
            with self.assertRaises(capacity.CapacityError):
                capacity.probe("foreground", "f" * 64)
        for returncode, stderr in ((1, b""), (0, b"\n"), (0, b"warning\n")):
            completed = subprocess.CompletedProcess(
                capacity.PRODUCER_ARGV, returncode, stdout=report_raw(), stderr=stderr
            )
            with mock.patch.object(
                capacity.socket, "gethostname", return_value=capacity.NODE
            ), mock.patch.object(
                capacity, "lexists", return_value=False
            ), mock.patch.object(
                capacity, "executable_closure_snapshot", return_value={}
            ), mock.patch.object(
                capacity.subprocess, "run", return_value=completed
            ):
                with self.assertRaises(capacity.CapacityError):
                    capacity.probe("controller", "f" * 64)

    def test_producer_is_isolated_source_owned_and_pyc_excluded(self):
        self.assertEqual(capacity.PRODUCER_ARGV[1:6], (
            "-I", "-S", "-B", "-X", f"pycache_prefix={capacity.PY_CACHE_PREFIX}"
        ))
        self.assertNotIn("-P", capacity.PRODUCER_ARGV)
        self.assertNotIn(str(capacity.ENTRY), capacity.PRODUCER_ARGV[:7])
        self.assertIn("tuple(sys.path)==expected", capacity.WRAPPER)
        self.assertNotIn("runpy", capacity.WRAPPER)
        self.assertIn("builtins.compile", capacity.WRAPPER)
        self.assertIn("builtins.exec", capacity.WRAPPER)
        self.assertIn("builtins.exit=sys.exit", capacity.WRAPPER)
        self.assertEqual(set(capacity.PRODUCER_ENV), {
            "HOME", "LANG", "LC_ALL", "PATH", "ROCM_SMI_LIB_PATH"
        })
        self.assertNotIn("LD_LIBRARY_PATH", capacity.PRODUCER_ENV)
        self.assertNotIn("PYTHONPATH", capacity.PRODUCER_ENV)

    def test_tool_closure_pins_entry_sources_native_python_shell_and_ps(self):
        self.assertEqual(len(capacity.SOURCE_PINS), 3)
        self.assertEqual(capacity.ROCM_LIBRARY_LINK_TEXT, "librocm_smi64.so.7.8.70000")
        self.assertEqual(capacity.ROCM_LIBRARY_TARGET_PIN[3],
                         "c08a51ffa7051a67264e9e7bf899abb0c5adee0366d200b452152a40c74b45f0")
        self.assertEqual(capacity.SHELL_LINK_TEXT, "dash")
        self.assertEqual(capacity.SHELL_TARGET_PIN[3],
                         "4f291296e89b784cd35479fca606f228126e3641f5bcaee68dee36583d7c9483")
        self.assertEqual(capacity.PS_PIN[3],
                         "207df9d438f75185ab3af2ab1173d104831a6631c28ef40d38b2ab43de27b40f")


if __name__ == "__main__":
    unittest.main()
