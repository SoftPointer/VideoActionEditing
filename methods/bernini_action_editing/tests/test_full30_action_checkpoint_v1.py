from __future__ import annotations

import base64
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "full30_action_checkpoint_v1.py"
SPEC = importlib.util.spec_from_file_location(
    "full30_action_checkpoint_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


def sha(character: str) -> str:
    return character * 64


def bindings(arm: str = "action+retain"):
    return checkpoint.CheckpointBindings(
        arm=arm,
        release_sha256=sha("1"),
        model_sha256=sha("2"),
        data_sha256=sha("3"),
        teacher_sha256=sha("4"),
        nuisance_sha256=sha("5"),
        noise_sha256=sha("6"),
        runtime_sha256=sha("7"),
        objective_sha256=sha("8"),
    )


def schedule():
    result = []
    for update in range(checkpoint.MAX_UPDATES):
        for microbatch in range(4):
            source = f"source-{(update * 4 + microbatch) % 64:03d}"
            branches = (
                ("action", "incomplete")
                if (update // 16 + microbatch) % 2 == 0
                else ("incomplete", "action")
            )
            for dp_rank, branch in enumerate(branches):
                result.append(
                    {
                        "global_index": len(result),
                        "epoch": update // 16,
                        "update": update,
                        "microbatch": microbatch,
                        "dp_rank": dp_rank,
                        "sigma_index": (4, 12, 20, 28, 35, 38)[
                            (update * 4 + microbatch) % 6
                        ],
                        "noise_seed": 10_000 + update * 4 + microbatch,
                        "row": {
                            "row_id": f"{source}--{branch}",
                            "source_id": source,
                            "branch": branch,
                            "teacher_cell_id": f"cell-{int(source[-3:]) // 8}",
                        },
                    }
                )
    return result


def rng_state(tag: bytes = b"zero"):
    encode = lambda value: base64.b64encode(value).decode("ascii")
    return {
        "schema_version": checkpoint.RNG_SCHEMA_VERSION,
        "python_rank_state_b64": [
            encode(b"python-rank-" + str(rank).encode("ascii") + b"-" + tag)
            for rank in range(8)
        ],
        "torch_cpu_rank_state_b64": [
            encode(b"cpu-rank-" + str(rank).encode("ascii") + b"-" + tag)
            for rank in range(8)
        ],
        "torch_cuda_rank_state_b64": [
            encode(b"rank-" + str(rank).encode("ascii") + b"-" + tag)
            for rank in range(8)
        ],
    }


def rng_sha(value) -> str:
    return hashlib.sha256(checkpoint.canonical_json_bytes(value) + b"\n").hexdigest()


class StreamTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        values: list[float],
        *,
        requires_grad: bool,
    ) -> None:
        self.shape = shape
        self.dtype = "torch.float32"
        self.requires_grad = requires_grad
        self.values = list(values)
        self.chunk_requests: list[int] = []
        count = 1
        for item in shape:
            count *= item
        assert count == len(values)

    def numel(self):
        return len(self.values)

    def is_contiguous(self):
        return True

    def iter_checkpoint_fp32_chunks(self, maximum: int):
        self.chunk_requests.append(maximum)
        for start in range(0, len(self.values), maximum):
            values = self.values[start : start + maximum]
            yield struct.pack(f"<{len(values)}f", *values)


class FakeOptimizer:
    def __init__(self, *, moment_values=None):
        names = ("alpha.weight", "zeta.weight")
        self.canonical_parameter_names = names
        self._parameters = {
            "alpha.weight": StreamTensor((2, 2), [1.0, 2.0, 3.0, 4.0], requires_grad=True),
            "zeta.weight": StreamTensor((1, 2), [-1.0, 0.5], requires_grad=True),
        }
        moments = moment_values or {
            "alpha.weight": [0.0, 0.0, 0.0, 0.0],
            "zeta.weight": [0.0, 0.0],
        }
        self._second_moments = {
            "alpha.weight": StreamTensor((2, 2), moments["alpha.weight"], requires_grad=False),
            "zeta.weight": StreamTensor((1, 2), moments["zeta.weight"], requires_grad=False),
        }
        self._update_count = 0

    @property
    def update_count(self):
        return self._update_count


def inventory_sha(optimizer) -> str:
    return checkpoint.inventory_identity_v2(
        optimizer, test_only_allow_small_capacity=True
    )["inventory_sha256"]


def status_consensus(local):
    if local["ok"]:
        return {"ok": True, "digest": local["digest"], "participant_count": 8}
    return {"ok": False, "error": "one-or-more-ranks-failed", "participant_count": 8}


def declared_full30_inventory_rows():
    rows = []
    for block in range(30):
        for attention in (1, 2):
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                for factor in ("A", "B"):
                    shape = [256, 1536] if factor == "A" else [1536, 256]
                    rows.append(
                        {
                            "name": f"blocks.{block}.attn{attention}.{projection}.lora_{factor}.default.weight",
                            "shape": shape,
                            "runtime_dtype": "torch.float32",
                            "stored_dtype": "float32-le",
                            "numel": 256 * 1536,
                        }
                    )
    for name, shape in (
        ("patch.source_delta.weight", [1536, 16, 1, 2, 2]),
        ("patch.source_delta.bias", [1536]),
        ("patch.target_delta.weight", [1536, 16, 1, 2, 2]),
        ("patch.target_delta.bias", [1536]),
        ("patch.role_embedding", [2, 1536]),
    ):
        count = 1
        for item in shape:
            count *= item
        rows.append(
            {
                "name": name,
                "shape": shape,
                "runtime_dtype": "torch.float32",
                "stored_dtype": "float32-le",
                "numel": count,
            }
        )
    rows.sort(key=lambda row: row["name"].encode("utf-8"))
    return rows


class StreamingCheckpointTests(unittest.TestCase):
    def save_zero(self, parent: Path, name="checkpoint-00000000", **changes):
        optimizer = changes.pop("optimizer", FakeOptimizer())
        target = parent / name
        arguments = {
            "optimizer": optimizer,
            "completed_updates": 0,
            "full_schedule": schedule(),
            "history": [],
            "rng_state": rng_state(),
            "bindings": bindings(),
            "authoritative_inventory_sha256": inventory_sha(optimizer),
            "test_only_allow_small_capacity": True,
        }
        arguments.update(changes)
        reference = checkpoint.save_checkpoint(target, **arguments)
        return target, optimizer, reference

    def load_zero(self, target: Path, optimizer, reference, **changes):
        arguments = {
            "optimizer": optimizer,
            "expected_bindings": bindings(),
            "expected_full_schedule": schedule(),
            "expected_completed_updates": 0,
            "expected_previous_checkpoint": None,
            "expected_reference": reference,
            "authoritative_inventory_sha256": inventory_sha(optimizer),
            "test_only_allow_small_capacity": True,
        }
        arguments.update(changes)
        return checkpoint.load_checkpoint(target, **arguments)

    def test_streaming_roundtrip_deep_freeze_and_mechanical_cursor(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            target, optimizer, reference = self.save_zero(
                Path(directory), stream_observer=lambda event, size: events.append((event, size))
            )
            loaded = self.load_zero(
                target,
                optimizer,
                reference,
                stream_observer=lambda event, size: events.append((event, size)),
            )
            self.assertEqual(loaded.reference, reference)
            self.assertEqual(loaded.next_cursor["completed_flat_rows"], 0)
            self.assertEqual(loaded.next_cursor["next_global_index"], 0)
            self.assertEqual(loaded.next_cursor["next_update"], 0)
            self.assertEqual(len(loaded.schedule), 1280)
            self.assertTrue(events)
            self.assertLessEqual(max(size for _, size in events), checkpoint.STREAM_CHUNK_BYTES)
            self.assertTrue(any(event.startswith("write:") for event, _ in events))
            self.assertTrue(any(event.startswith("read:") for event, _ in events))
            with self.assertRaises(TypeError):
                loaded.manifest["progress"]["completed_updates"] = 7
            with self.assertRaises(TypeError):
                loaded.schedule[0]["global_index"] = 9
            with self.assertRaises(TypeError):
                loaded.rng_state["torch_cuda_rank_state_b64"][0] = "forged"
            self.assertEqual(
                hashlib.sha256(loaded.manifest_bytes).hexdigest(),
                reference.manifest_sha256,
            )
            self.assertEqual(stat.S_IMODE(os.lstat(target).st_mode), 0o750)
            for name in checkpoint.ALL_FILE_NAMES:
                self.assertEqual(stat.S_IMODE(os.lstat(target / name).st_mode), 0o600)

    def test_byte_determinism_and_create_only_exact_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first, optimizer, first_reference = self.save_zero(parent, "checkpoint-a")
            second, _, second_reference = self.save_zero(
                parent, "checkpoint-b", optimizer=optimizer
            )
            self.assertEqual(first_reference, second_reference)
            for name in checkpoint.ALL_FILE_NAMES:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            recovered = checkpoint.save_checkpoint(
                first,
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )
            self.assertEqual(recovered, first_reference)

    def test_nonrankzero_stream_hashes_but_never_materializes_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            committed, optimizer, reference = self.save_zero(parent, "reference")
            target = parent / "rank-one-must-not-exist"
            events = []
            returned = checkpoint.save_checkpoint(
                target,
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                rank=1,
                world_size=8,
                status_consensus=status_consensus,
                result_broadcast=lambda local: {
                    "ok": True,
                    "status": "committed",
                    "reference": reference.as_dict(),
                },
                stream_observer=lambda event, size: events.append((event, size)),
                test_only_allow_small_capacity=True,
            )
            self.assertEqual(returned, reference)
            self.assertFalse(target.exists())
            self.assertTrue(any(event.startswith("scan:") for event, _ in events))
            self.assertFalse(any(event.startswith("write:") for event, _ in events))
            self.assertTrue(committed.is_dir())

    def test_consensus_digest_mismatch_reaches_result_broadcast_without_publish(self):
        optimizer = FakeOptimizer()
        broadcasts = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "consensus-mismatch"
            with self.assertRaises(checkpoint.Full30CheckpointTransactionError):
                checkpoint.save_checkpoint(
                    target,
                    optimizer=optimizer,
                    completed_updates=0,
                    full_schedule=schedule(),
                    history=[],
                    rng_state=rng_state(),
                    bindings=bindings(),
                    authoritative_inventory_sha256=inventory_sha(optimizer),
                    rank=0,
                    world_size=8,
                    status_consensus=lambda _local: {
                        "ok": True,
                        "digest": sha("f"),
                        "participant_count": 8,
                    },
                    result_broadcast=lambda local: broadcasts.append(local) or local,
                    test_only_allow_small_capacity=True,
                )
            self.assertEqual(len(broadcasts), 1)
            self.assertEqual(broadcasts[0]["status"], "not_committed")
            self.assertFalse(target.exists())

    def test_rankzero_setup_failure_is_broadcast_and_definitely_not_committed(self):
        optimizer = FakeOptimizer()
        broadcasts = []
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            checkpoint.tempfile, "mkdtemp", side_effect=OSError("injected mkdir failure")
        ):
            target = Path(directory) / "setup-failure"
            with self.assertRaises(checkpoint.Full30CheckpointTransactionError):
                checkpoint.save_checkpoint(
                    target,
                    optimizer=optimizer,
                    completed_updates=0,
                    full_schedule=schedule(),
                    history=[],
                    rng_state=rng_state(),
                    bindings=bindings(),
                    authoritative_inventory_sha256=inventory_sha(optimizer),
                    rank=0,
                    world_size=8,
                    status_consensus=status_consensus,
                    result_broadcast=lambda local: broadcasts.append(local) or local,
                    test_only_allow_small_capacity=True,
                )
            self.assertEqual(len(broadcasts), 1)
            self.assertEqual(broadcasts[0]["status"], "not_committed")
            self.assertFalse(target.exists())

    def test_preflight_error_still_enters_one_fixed_consensus_and_never_broadcasts(self):
        optimizer = FakeOptimizer(
            moment_values={
                "alpha.weight": [-1.0, 0.0, 0.0, 0.0],
                "zeta.weight": [0.0, 0.0],
            }
        )
        calls = []
        broadcasts = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                checkpoint.Full30CheckpointError, "failed consistently"
            ):
                checkpoint.save_checkpoint(
                    Path(directory) / "invalid",
                    optimizer=optimizer,
                    completed_updates=0,
                    full_schedule=schedule(),
                    history=[],
                    rng_state=rng_state(),
                    bindings=bindings(),
                    authoritative_inventory_sha256=inventory_sha(optimizer),
                    rank=3,
                    world_size=8,
                    status_consensus=lambda local: (
                        calls.append(local) or status_consensus(local)
                    ),
                    result_broadcast=lambda local: broadcasts.append(local),
                    test_only_allow_small_capacity=True,
                )
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["ok"])
        self.assertEqual(broadcasts, [])

    def test_invalid_save_path_and_invalid_load_sha_still_enter_status_consensus(self):
        optimizer = FakeOptimizer()
        save_calls = []
        with self.assertRaisesRegex(checkpoint.Full30CheckpointError, "failed consistently"):
            checkpoint.save_checkpoint(
                Path("relative-checkpoint"),
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                rank=3,
                world_size=8,
                status_consensus=lambda local: save_calls.append(local) or status_consensus(local),
                result_broadcast=lambda _local: self.fail("preflight failure must not broadcast"),
                test_only_allow_small_capacity=True,
            )
        self.assertEqual(len(save_calls), 1)
        self.assertFalse(save_calls[0]["ok"])

        with tempfile.TemporaryDirectory() as directory:
            target, optimizer, reference = self.save_zero(Path(directory))
            load_calls = []
            with self.assertRaisesRegex(checkpoint.Full30CheckpointError, "failed consistently"):
                self.load_zero(
                    target,
                    optimizer,
                    reference,
                    authoritative_inventory_sha256="bad",
                    rank=3,
                    world_size=8,
                    status_consensus=lambda local: load_calls.append(local) or status_consensus(local),
                )
            self.assertEqual(len(load_calls), 1)
            self.assertFalse(load_calls[0]["ok"])

    def test_u0_rejects_nonzero_and_negative_v_and_arbitrary_cursor_api_is_absent(self):
        for value in (1.0, -1.0):
            optimizer = FakeOptimizer(
                moment_values={
                    "alpha.weight": [value, 0.0, 0.0, 0.0],
                    "zeta.weight": [0.0, 0.0],
                }
            )
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(checkpoint.Full30CheckpointError):
                    checkpoint.save_checkpoint(
                        Path(directory) / "invalid",
                        optimizer=optimizer,
                        completed_updates=0,
                        full_schedule=schedule(),
                        history=[],
                        rng_state=rng_state(),
                        bindings=bindings(),
                        authoritative_inventory_sha256=inventory_sha(optimizer),
                        test_only_allow_small_capacity=True,
                    )
        optimizer = FakeOptimizer()
        optimizer._parameters["alpha.weight"].grad = object()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            checkpoint.Full30CheckpointError, "segment boundary"
        ):
            checkpoint.save_checkpoint(
                Path(directory) / "active-gradient",
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )
        optimizer = FakeOptimizer()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(TypeError):
            checkpoint.save_checkpoint(
                Path(directory) / "invalid-api",
                optimizer=optimizer,
                completed_updates=0,
                next_cursor=None,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )

    def test_reference_only_predecessor_recomputes_real_history_prefix_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            zero_path, optimizer, zero_reference = self.save_zero(parent)
            loaded_zero = self.load_zero(zero_path, optimizer, zero_reference)
            before = checkpoint.optimizer_state_identity_v2(
                optimizer, test_only_allow_small_capacity=True
            )
            for tensor in optimizer._parameters.values():
                tensor.values = [value + 0.25 for value in tensor.values]
            for tensor in optimizer._second_moments.values():
                tensor.values = [0.5 for _ in tensor.values]
            optimizer._update_count = 1
            after = checkpoint.optimizer_state_identity_v2(
                optimizer, test_only_allow_small_capacity=True
            )
            prefix1 = checkpoint.schedule_digests_v2(schedule(), 1)[1]
            row1 = checkpoint.build_history_row_v2(
                update_count=1,
                optimizer_receipt_digest=sha("a"),
                parameters_before_sha256=before["trainable_state_sha256"],
                parameters_after_sha256=after["trainable_state_sha256"],
                optimizer_v_before_sha256=before["optimizer_v_state_sha256"],
                optimizer_v_after_sha256=after["optimizer_v_state_sha256"],
                rng_before_sha256=rng_sha(rng_state()),
                rng_after_sha256=rng_sha(rng_state(b"one")),
                schedule_prefix_sha256=prefix1,
            )
            one_path = parent / "checkpoint-00000001"
            one_reference = checkpoint.save_checkpoint(
                one_path,
                optimizer=optimizer,
                completed_updates=1,
                full_schedule=schedule(),
                history=[row1],
                rng_state=rng_state(b"one"),
                bindings=bindings(),
                previous_checkpoint=loaded_zero,
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )
            loaded_one = checkpoint.load_checkpoint(
                one_path,
                optimizer=optimizer,
                expected_bindings=bindings(),
                expected_full_schedule=schedule(),
                expected_completed_updates=1,
                expected_previous_checkpoint=zero_reference,
                expected_reference=one_reference,
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )
            self.assertEqual(loaded_one.previous_checkpoint, zero_reference)

            optimizer._update_count = 2
            prefix2 = checkpoint.schedule_digests_v2(schedule(), 2)[1]
            row2 = checkpoint.build_history_row_v2(
                update_count=2,
                optimizer_receipt_digest=sha("b"),
                parameters_before_sha256=after["trainable_state_sha256"],
                parameters_after_sha256=after["trainable_state_sha256"],
                optimizer_v_before_sha256=after["optimizer_v_state_sha256"],
                optimizer_v_after_sha256=after["optimizer_v_state_sha256"],
                rng_before_sha256=rng_sha(rng_state(b"one")),
                rng_after_sha256=rng_sha(rng_state(b"two")),
                schedule_prefix_sha256=prefix2,
            )
            tampered = dict(row1)
            tampered["optimizer_receipt_digest"] = sha("c")
            with self.assertRaisesRegex(
                checkpoint.Full30CheckpointError, "predecessor history prefix SHA"
            ):
                checkpoint.save_checkpoint(
                    parent / "checkpoint-00000002",
                    optimizer=optimizer,
                    completed_updates=2,
                    full_schedule=schedule(),
                    history=[tampered, row2],
                    rng_state=rng_state(b"two"),
                    bindings=bindings(),
                    previous_checkpoint=one_reference,
                    authoritative_inventory_sha256=inventory_sha(optimizer),
                    test_only_allow_small_capacity=True,
                )
            forged_state_reference = replace(
                one_reference, trainable_state_sha256=sha("d")
            )
            with self.assertRaisesRegex(
                checkpoint.Full30CheckpointError, "predecessor reference"
            ):
                checkpoint.save_checkpoint(
                    parent / "checkpoint-forged-predecessor-state",
                    optimizer=optimizer,
                    completed_updates=2,
                    full_schedule=schedule(),
                    history=[row1, row2],
                    rng_state=rng_state(b"two"),
                    bindings=bindings(),
                    previous_checkpoint=forged_state_reference,
                    authoritative_inventory_sha256=inventory_sha(optimizer),
                    test_only_allow_small_capacity=True,
                )

    def test_postrename_fsync_failure_is_indeterminate_then_exact_recovery_converges(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            optimizer = FakeOptimizer()
            target = parent / "checkpoint-fsync"
            original = checkpoint._fsync_directory
            calls = []

            def fail_parent(path):
                calls.append(Path(path))
                if len(calls) == 2:
                    raise OSError("injected parent fsync failure")
                return original(path)

            with mock.patch.object(checkpoint, "_fsync_directory", side_effect=fail_parent):
                with self.assertRaises(checkpoint.Full30CheckpointCommitIndeterminate):
                    checkpoint.save_checkpoint(
                        target,
                        optimizer=optimizer,
                        completed_updates=0,
                        full_schedule=schedule(),
                        history=[],
                        rng_state=rng_state(),
                        bindings=bindings(),
                        authoritative_inventory_sha256=inventory_sha(optimizer),
                        test_only_allow_small_capacity=True,
                    )
            self.assertTrue(target.is_dir())
            recovered = checkpoint.save_checkpoint(
                target,
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )
            loaded = self.load_zero(target, optimizer, recovered)
            self.assertEqual(loaded.reference, recovered)

    def test_prerename_stream_write_failure_cleans_stage_and_never_publishes(self):
        optimizer = FakeOptimizer()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            checkpoint, "_write_tensor_file", side_effect=OSError("injected stream write failure")
        ):
            parent = Path(directory)
            target = parent / "checkpoint-write-failure"
            with self.assertRaises(checkpoint.Full30CheckpointTransactionError):
                checkpoint.save_checkpoint(
                    target,
                    optimizer=optimizer,
                    completed_updates=0,
                    full_schedule=schedule(),
                    history=[],
                    rng_state=rng_state(),
                    bindings=bindings(),
                    authoritative_inventory_sha256=inventory_sha(optimizer),
                    test_only_allow_small_capacity=True,
                )
            self.assertFalse(target.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_result_broadcast_failure_after_commit_is_indeterminate_and_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            optimizer = FakeOptimizer()
            target = parent / "checkpoint-broadcast"
            with self.assertRaises(checkpoint.Full30CheckpointCommitIndeterminate):
                checkpoint.save_checkpoint(
                    target,
                    optimizer=optimizer,
                    completed_updates=0,
                    full_schedule=schedule(),
                    history=[],
                    rng_state=rng_state(),
                    bindings=bindings(),
                    authoritative_inventory_sha256=inventory_sha(optimizer),
                    rank=0,
                    world_size=8,
                    status_consensus=status_consensus,
                    result_broadcast=lambda local: (_ for _ in ()).throw(
                        RuntimeError("injected broadcast failure")
                    ),
                    test_only_allow_small_capacity=True,
                )
            self.assertTrue(target.is_dir())
            reference = checkpoint.save_checkpoint(
                target,
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )
            self.assertEqual(self.load_zero(target, optimizer, reference).reference, reference)

    def test_partial_extra_symlink_mode_corruption_and_load_consensus(self):
        cases = ("partial", "extra", "symlink", "hardlink", "mode", "corrupt")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                target, optimizer, reference = self.save_zero(Path(directory))
                if case == "partial":
                    (target / "optimizer_v.f32").unlink()
                elif case == "extra":
                    extra = target / "extra"
                    extra.write_bytes(b"x")
                    os.chmod(extra, 0o600)
                elif case == "symlink":
                    copy_path = Path(directory) / "copy"
                    copy_path.write_bytes((target / "rng.json").read_bytes())
                    (target / "rng.json").unlink()
                    (target / "rng.json").symlink_to(copy_path)
                elif case == "hardlink":
                    copy_path = Path(directory) / "hardlink-copy"
                    os.link(target / "rng.json", copy_path)
                elif case == "mode":
                    os.chmod(target / "history.json", 0o644)
                else:
                    path = target / "trainables.f32"
                    raw = bytearray(path.read_bytes())
                    raw[-1] ^= 1
                    path.write_bytes(raw)
                    os.chmod(path, 0o600)
                calls = []
                with self.assertRaisesRegex(
                    checkpoint.Full30CheckpointError, "failed consistently"
                ):
                    self.load_zero(
                        target,
                        optimizer,
                        reference,
                        rank=2,
                        world_size=8,
                        status_consensus=lambda local: (
                            calls.append(local) or status_consensus(local)
                        ),
                    )
                self.assertEqual(len(calls), 1)
                self.assertFalse(calls[0]["ok"])

    def test_schedule_prefix_is_eight_rows_per_update_and_cursor_is_derived(self):
        full0, prefix0 = checkpoint.schedule_digests_v2(schedule(), 0)
        full1, prefix1 = checkpoint.schedule_digests_v2(schedule(), 1)
        self.assertEqual(full0, full1)
        self.assertNotEqual(prefix0, prefix1)
        cursor = checkpoint.next_cursor_v2(schedule(), 1)
        self.assertEqual(cursor["completed_flat_rows"], 8)
        self.assertEqual(cursor["next_global_index"], 8)
        self.assertEqual(cursor["next_update"], 1)
        altered = schedule()
        first_seed = altered[6]["noise_seed"]
        second_seed = altered[8]["noise_seed"]
        altered[6]["noise_seed"] = altered[7]["noise_seed"] = second_seed
        altered[8]["noise_seed"] = altered[9]["noise_seed"] = first_seed
        self.assertNotEqual(
            prefix1, checkpoint.schedule_digests_v2(altered, 1)[1]
        )
        terminal = checkpoint.next_cursor_v2(schedule(), checkpoint.MAX_UPDATES)
        self.assertEqual(terminal["next_global_index"], 1280)
        self.assertEqual(terminal["next_update"], 160)
        self.assertEqual(terminal["next_epoch"], 10)
        self.assertTrue(terminal["terminal"])

        wrong_branch_order = schedule()
        wrong_branch_order[0]["row"], wrong_branch_order[1]["row"] = (
            wrong_branch_order[1]["row"],
            wrong_branch_order[0]["row"],
        )
        with self.assertRaisesRegex(checkpoint.Full30CheckpointError, "paired source/noise"):
            checkpoint.canonical_schedule_v2(wrong_branch_order)

    def test_declared_189m_capacity_plan_has_bounded_zero_retained_payload(self):
        rows = declared_full30_inventory_rows()
        checkpoint._validate_production_capacity(rows)
        named = [(row["name"], row["numel"]) for row in rows]
        tracemalloc.start()
        try:
            plan = checkpoint.streaming_allocation_plan_v2(named)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(plan.tensor_count, 485)
        self.assertEqual(plan.total_numel, checkpoint.EXACT_TRAINABLE_NUMEL)
        self.assertLessEqual(plan.maximum_chunk_bytes, checkpoint.STREAM_CHUNK_BYTES)
        self.assertFalse(plan.whole_state_payload_materialized)
        self.assertEqual(plan.per_rank_tensor_file_bytes_retained, 0)
        self.assertLess(peak, 8 * 1024 * 1024)
        malformed = copy.deepcopy(rows)
        malformed[0]["shape"] = [1, malformed[0]["numel"]]
        with self.assertRaises(checkpoint.Full30CheckpointError):
            checkpoint._validate_production_capacity(malformed)

    def test_main_arm_name_only_and_authoritative_inventory_sha_required(self):
        with self.assertRaises(checkpoint.Full30CheckpointError):
            bindings("main")
        optimizer = FakeOptimizer()
        with self.assertRaisesRegex(
            checkpoint.Full30CheckpointError, "actual Full30ActionFirstOptimizerV1"
        ):
            checkpoint.inventory_identity_v2(optimizer)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(
            checkpoint.Full30CheckpointError
        ):
            checkpoint.save_checkpoint(
                Path(directory) / "wrong-inventory",
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=sha("f"),
                test_only_allow_small_capacity=True,
            )
        invalid_rng = rng_state()
        invalid_rng["torch_cpu_rank_state_b64"].pop()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(
            checkpoint.Full30CheckpointError
        ):
            checkpoint.save_checkpoint(
                Path(directory) / "incomplete-rng",
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=invalid_rng,
                bindings=bindings(),
                authoritative_inventory_sha256=inventory_sha(optimizer),
                test_only_allow_small_capacity=True,
            )


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable")
class RealOptimizerRestoreTests(unittest.TestCase):
    def setUp(self):
        import torch

        sys.path.insert(0, str(ROOT))
        import full30_action_optimizer_v1 as optimizer_module

        self.torch = torch
        self.optimizer_module = optimizer_module

    def test_cpu_narrow_stream_never_serializes_base_storage_outside_chunk(self):
        tensor = self.torch.arange(5, dtype=self.torch.float32).contiguous()
        observed = []
        with mock.patch.object(checkpoint, "STREAM_CHUNK_ELEMENTS", 2), mock.patch.object(
            checkpoint, "STREAM_CHUNK_BYTES", 8
        ):
            chunks = list(
                checkpoint._iter_tensor_chunks(
                    tensor,
                    observer=lambda event, size: observed.append((event, size)),
                    event="narrow-test",
                )
            )
        self.assertEqual([len(chunk) for chunk in chunks], [8, 8, 4])
        values = [number for chunk in chunks for (number,) in struct.iter_unpack("<f", chunk)]
        self.assertEqual(values, [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(observed, [("narrow-test", 8), ("narrow-test", 8), ("narrow-test", 4)])

    def test_real_optimizer_fp32_roundtrip_preserves_parameter_and_hook_identity(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
        hook = parameter.register_hook(lambda gradient: gradient)
        optimizer = self.optimizer_module.Full30ActionFirstOptimizerV1({"p": parameter})
        with self.assertRaisesRegex(
            checkpoint.Full30CheckpointError, "authoritative trainable capacity"
        ):
            checkpoint.inventory_identity_v2(optimizer)
        inventory = checkpoint.inventory_identity_v2(
            optimizer, test_only_allow_small_capacity=True
        )["inventory_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint-real"
            reference = checkpoint.save_checkpoint(
                target,
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            loaded = checkpoint.load_checkpoint(
                target,
                optimizer=optimizer,
                expected_bindings=bindings(),
                expected_full_schedule=schedule(),
                expected_completed_updates=0,
                expected_previous_checkpoint=None,
                expected_reference=reference,
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            parameter_id = id(parameter)
            hooks_id = id(parameter._backward_hooks)
            with torch.no_grad():
                parameter.add_(10.0)

            class RNGTransaction:
                def commit(self):
                    return reference.rng_sha256

                def rollback(self):
                    return None

            receipt = checkpoint.restore_checkpoint_state(
                loaded,
                optimizer=optimizer,
                rng_transaction_factory=lambda state: RNGTransaction(),
                test_only_allow_small_capacity=True,
            )
            self.assertEqual(receipt["status"], "committed")
            self.assertEqual(id(parameter), parameter_id)
            self.assertEqual(id(parameter._backward_hooks), hooks_id)
            self.assertTrue(torch.equal(parameter, torch.tensor([1.0, -2.0])))
            self.assertEqual(optimizer.update_count, 0)
            self.assertTrue(torch.equal(optimizer.second_moment("p"), torch.zeros(2)))
            hook.remove()

    def test_nonzero_optimizer_v_and_update_count_restore_exactly(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
        optimizer = self.optimizer_module.Full30ActionFirstOptimizerV1({"p": parameter})
        inventory = checkpoint.inventory_identity_v2(
            optimizer, test_only_allow_small_capacity=True
        )["inventory_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            zero_path = parent / "checkpoint-u000"
            zero_reference = checkpoint.save_checkpoint(
                zero_path,
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            loaded_zero = checkpoint.load_checkpoint(
                zero_path,
                optimizer=optimizer,
                expected_bindings=bindings(),
                expected_full_schedule=schedule(),
                expected_completed_updates=0,
                expected_previous_checkpoint=None,
                expected_reference=zero_reference,
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            before = checkpoint.optimizer_state_identity_v2(
                optimizer, test_only_allow_small_capacity=True
            )
            with torch.no_grad():
                parameter.copy_(torch.tensor([1.25, -1.75], dtype=torch.float32))
                optimizer._second_moments["p"].copy_(
                    torch.tensor([0.5, 0.75], dtype=torch.float32)
                )
            optimizer._update_count = 1
            saved_parameter = parameter.detach().clone()
            saved_moment = optimizer.second_moment("p")
            after = checkpoint.optimizer_state_identity_v2(
                optimizer, test_only_allow_small_capacity=True
            )
            row = checkpoint.build_history_row_v2(
                update_count=1,
                optimizer_receipt_digest=sha("a"),
                parameters_before_sha256=before["trainable_state_sha256"],
                parameters_after_sha256=after["trainable_state_sha256"],
                optimizer_v_before_sha256=before["optimizer_v_state_sha256"],
                optimizer_v_after_sha256=after["optimizer_v_state_sha256"],
                rng_before_sha256=zero_reference.rng_sha256,
                rng_after_sha256=rng_sha(rng_state(b"one")),
                schedule_prefix_sha256=checkpoint.schedule_digests_v2(schedule(), 1)[1],
            )
            one_path = parent / "checkpoint-u001"
            one_reference = checkpoint.save_checkpoint(
                one_path,
                optimizer=optimizer,
                completed_updates=1,
                full_schedule=schedule(),
                history=[row],
                rng_state=rng_state(b"one"),
                bindings=bindings(),
                previous_checkpoint=loaded_zero,
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            loaded_one = checkpoint.load_checkpoint(
                one_path,
                optimizer=optimizer,
                expected_bindings=bindings(),
                expected_full_schedule=schedule(),
                expected_completed_updates=1,
                expected_previous_checkpoint=loaded_zero,
                expected_reference=one_reference,
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            with torch.no_grad():
                parameter.add_(9.0)
                optimizer._second_moments["p"].add_(4.0)
            optimizer._update_count = 7

            class RNGTransaction:
                def commit(self):
                    return one_reference.rng_sha256

                def rollback(self):
                    return None

            checkpoint.restore_checkpoint_state(
                loaded_one,
                optimizer=optimizer,
                rng_transaction_factory=lambda _state: RNGTransaction(),
                test_only_allow_small_capacity=True,
            )
            self.assertTrue(torch.equal(parameter, saved_parameter))
            self.assertTrue(torch.equal(optimizer.second_moment("p"), saved_moment))
            self.assertEqual(optimizer.update_count, 1)

    def test_commit_injection_rolls_back_live_parameter_optimizer_and_hook(self):
        torch = self.torch
        parameters = {
            "a": torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float32)),
            "b": torch.nn.Parameter(torch.tensor([4.0], dtype=torch.float32)),
        }
        hook = parameters["a"].register_hook(lambda gradient: gradient)
        optimizer = self.optimizer_module.Full30ActionFirstOptimizerV1(parameters)
        inventory = checkpoint.inventory_identity_v2(
            optimizer, test_only_allow_small_capacity=True
        )["inventory_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint-rollback"
            reference = checkpoint.save_checkpoint(
                target,
                optimizer=optimizer,
                completed_updates=0,
                full_schedule=schedule(),
                history=[],
                rng_state=rng_state(),
                bindings=bindings(),
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            loaded = checkpoint.load_checkpoint(
                target,
                optimizer=optimizer,
                expected_bindings=bindings(),
                expected_full_schedule=schedule(),
                expected_completed_updates=0,
                expected_previous_checkpoint=None,
                expected_reference=reference,
                authoritative_inventory_sha256=inventory,
                test_only_allow_small_capacity=True,
            )
            before = {name: value.detach().clone() for name, value in parameters.items()}
            ids = {name: id(value) for name, value in parameters.items()}
            hook_id = id(parameters["a"]._backward_hooks)

            class RNGTransaction:
                def commit(self):
                    return reference.rng_sha256

                def rollback(self):
                    return None

            with self.assertRaisesRegex(checkpoint.Full30CheckpointError, "rolled back"):
                checkpoint.restore_checkpoint_state(
                    loaded,
                    optimizer=optimizer,
                    rng_transaction_factory=lambda state: RNGTransaction(),
                    test_only_allow_small_capacity=True,
                    test_only_commit_hook=lambda name, index: (
                        (_ for _ in ()).throw(RuntimeError("injected swap failure"))
                        if index == 0
                        else None
                    ),
                )
            for name, value in parameters.items():
                self.assertEqual(id(value), ids[name])
                self.assertTrue(torch.equal(value, before[name]))
            self.assertEqual(id(parameters["a"]._backward_hooks), hook_id)
            self.assertEqual(optimizer.update_count, 0)

            with torch.no_grad():
                for value in parameters.values():
                    value.add_(5.0)
                for value in optimizer._second_moments.values():
                    value.fill_(2.0)
            optimizer._update_count = 3
            before_rng_failure = {
                name: value.detach().clone() for name, value in parameters.items()
            }
            before_moments = {
                name: value.detach().clone()
                for name, value in optimizer._second_moments.items()
            }
            rng_events = []

            class BadRNGTransaction:
                def commit(self):
                    rng_events.append("commit")
                    return sha("f")

                def rollback(self):
                    rng_events.append("rollback")

            with self.assertRaisesRegex(checkpoint.Full30CheckpointError, "rolled back"):
                checkpoint.restore_checkpoint_state(
                    loaded,
                    optimizer=optimizer,
                    rng_transaction_factory=lambda _state: BadRNGTransaction(),
                    test_only_allow_small_capacity=True,
                )
            self.assertEqual(rng_events, ["commit", "rollback"])
            self.assertEqual(optimizer.update_count, 3)
            for name, value in parameters.items():
                self.assertTrue(torch.equal(value, before_rng_failure[name]))
                self.assertTrue(
                    torch.equal(optimizer._second_moments[name], before_moments[name])
                )
            hook.remove()


if __name__ == "__main__":
    unittest.main()
