from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
from typing import Any, Mapping
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import full30_action_source7_reencode_plan_v1 as plan  # noqa: E402
import materialize_full30_action_source7_reencode_v1 as materializer  # noqa: E402
import full30_action_source7_reencode_controller_v1 as controller  # noqa: E402
import build_full30_action_source7_reencode_release_v1 as release  # noqa: E402
import full30_action_source7_reencode_runtime_cache_v1 as runtime_cache  # noqa: E402


LAUNCHER = METHOD_ROOT / "scripts/auh_full30_action_source7_reencode_136141_v1.sh"
CARD = (
    METHOD_ROOT.parents[1]
    / "md/action_editing/20260814_box/cards/"
    "20260815_BOX-EXP-014_full30-source-only-exact7-reencode.md"
)


class _FiniteBool:
    def all(self):
        return self

    def item(self):
        return True


class _FakeTensor:
    layout = "strided"
    device = types.SimpleNamespace(type="cpu")
    dtype = "float32"

    def __init__(self, shape=(1, 32, 21, 2, 2)):
        self.shape = shape

    def is_contiguous(self):
        return True


def fake_torch_module(*, load_value=None):
    return types.SimpleNamespace(
        Tensor=_FakeTensor,
        strided="strided",
        float32="float32",
        load=lambda *args, **kwargs: (
            _FakeTensor() if load_value is None else load_value
        ),
        isfinite=lambda value: _FiniteBool(),
    )


class _PhysicalTensorBytes:
    def __init__(
        self, *, shape: list[int], raw: bytes, dtype: str = "torch.float32",
        layout: str = "torch.strided", finite: bool = True,
    ) -> None:
        self.shape = tuple(shape)
        self._raw = raw
        self.dtype = dtype
        self.layout = layout
        self.device = types.SimpleNamespace(type="cpu")
        self._finite = finite

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def is_contiguous(self):
        return True

    def to(self, *, device, dtype):
        if device != "cpu" or dtype != "torch.float32":
            raise AssertionError("test tensor CPU/FP32 conversion differs")
        return self

    def view(self, dtype):
        if dtype != "torch.uint8":
            raise AssertionError("test tensor byte view differs")
        return self

    def reshape(self, *shape):
        if shape != (-1,):
            raise AssertionError("test tensor flatten differs")
        return self

    def numpy(self):
        return self

    def tobytes(self, *, order):
        if order != "C":
            raise AssertionError("test tensor byte order differs")
        return self._raw


class _PhysicalFiniteBool:
    def __init__(self, value: bool) -> None:
        self._value = value

    def all(self):
        return self

    def item(self):
        return self._value


def _physical_test_torch() -> Any:
    def load(buffer, *, map_location, weights_only):
        if map_location != "cpu" or weights_only is not True:
            raise AssertionError("post-srun loader is not safe CPU weights-only")
        blob = buffer.read()
        if len(blob) < 512:
            raise ValueError("truncated test tensor")
        header = json.loads(blob[:512].rstrip(b"\0").decode("ascii"))
        if set(header) != {"container", "dtype", "finite", "layout", "shape"}:
            raise ValueError("test tensor header closure differs")
        if header["container"] != "bare-tensor":
            return {"posterior_parameters": object()}
        raw = blob[512:]
        expected = 4
        for dimension in header["shape"]:
            expected *= dimension
        if len(raw) != expected:
            raise ValueError("test tensor payload size differs")
        return _PhysicalTensorBytes(
            shape=header["shape"],
            raw=raw,
            dtype=header["dtype"],
            layout=header["layout"],
            finite=header["finite"],
        )

    return types.SimpleNamespace(
        Tensor=_PhysicalTensorBytes,
        uint8="torch.uint8",
        float32="torch.float32",
        strided="torch.strided",
        load=load,
        isfinite=lambda value: _PhysicalFiniteBool(value._finite),
    )


def _physical_test_tensor_blob(
    iid: str, shape: list[int], *, container: str = "bare-tensor",
    dtype: str = "torch.float32", layout: str = "torch.strided",
    finite: bool = True,
) -> tuple[bytes, str, str]:
    header = {
        "container": container,
        "dtype": dtype,
        "finite": finite,
        "layout": layout,
        "shape": shape,
    }
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    if len(encoded) > 512:
        raise AssertionError("test tensor header is too large")
    size = 4
    for dimension in shape:
        size *= dimension
    byte = hashlib.sha256(iid.encode("ascii")).digest()[0]
    tensor_raw = bytes([byte]) * size
    blob = encoded + b"\0" * (512 - len(encoded)) + tensor_raw
    tensor_header = runtime_cache.canonical_json_bytes(
        {"dtype": dtype, "shape": shape}
    )
    tensor_sha = sha(tensor_header + b"\0" + tensor_raw)
    return blob, tensor_sha, sha(tensor_raw)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_plan(root: Path) -> tuple[Path, str, dict]:
    value = plan.canonical_plan()
    raw = plan.canonical_json_bytes(value) + b"\n"
    path = root / "source7-plan.json"
    path.write_bytes(raw)
    return path.resolve(strict=True), sha(raw), value


class Source7PlanTests(unittest.TestCase):
    def test_exact_canonical_inventory_and_external_binding(self) -> None:
        value = plan.validate_plan(plan.canonical_plan())
        self.assertEqual(value["experiment_id"], "BOX-EXP-014")
        self.assertEqual(len(value["rows"]), 7)
        self.assertEqual(
            [row["iid"] for row in value["rows"]],
            [
                "57cda7597d924dbb",
                "6d4a7f95a52e47e9",
                "a0b66487ab68498a",
                "38b113317af14f01",
                "5ae60e8417244e6e",
                "1149c58e43e54add",
                "a535e13301e448d7",
            ],
        )
        self.assertEqual(
            [row["expected_posterior_shape"] for row in value["rows"]],
            [
                [1, 32, 21, 68, 54],
                [1, 32, 21, 68, 54],
                [1, 32, 21, 82, 46],
                [1, 32, 21, 74, 50],
                [1, 32, 21, 68, 54],
                [1, 32, 21, 68, 54],
                [1, 32, 21, 70, 52],
            ],
        )
        external = value["external_existing_index0"]
        self.assertEqual(external["iid"], "2d2e28871a5a4856")
        self.assertFalse(external["reencoded"])
        self.assertNotIn(external["iid"], {row["iid"] for row in value["rows"]})
        self.assertFalse(value["exact8_authority_go_claimed"])
        self.assertTrue(value["teacher_cross_disjointness_pending"])

    def test_source_paths_and_shas_are_frozen(self) -> None:
        value = plan.canonical_plan()
        expected_shas = {
            "57cda7597d924dbb": "6409f59896c50f0d19dff7ac1e67f37362aa57968bc64a21a9a8271f5a85fec8",
            "6d4a7f95a52e47e9": "d8ea67b2f1ada75894cd3d2b55f877fd7ffd3b0f127119288fbdec37c5925b1a",
            "a0b66487ab68498a": "62dbdf50c385233087919b686a0c5d064ce1ae47170ac477f6eeb320a885afa7",
            "38b113317af14f01": "eece8b2a298a5488fb689c736ceb074842ceb61e07090aacd7ea9a34d48e2fd6",
            "5ae60e8417244e6e": "88d7bd4601f6f8c16e8a9d0bbdb5cb75f0c77538061e17e30095ecf1a620ea99",
            "1149c58e43e54add": "d24e32daf499850b33b40f13cd537c11fd8a230eda6fa121a24c33e0a79dfe7a",
            "a535e13301e448d7": "70ccabb237fd8a2a159d0cbf40fb53d21fb070c3d280d974aa28ec58d0d1130c",
        }
        for row in value["rows"]:
            iid = row["iid"]
            self.assertEqual(row["source_video_sha256"], expected_shas[iid])
            self.assertEqual(
                row["source_video_path"],
                str(plan.SOURCE_BASE / iid / "samples" / iid / "source_video.mp4"),
            )
            self.assertEqual(row["vae_encode_calls"], 1)

    def test_plan_tamper_is_rejected(self) -> None:
        value = copy.deepcopy(plan.canonical_plan())
        value["rows"][0]["source_video_sha256"] = "0" * 64
        with self.assertRaisesRegex(plan.Source7ReencodePlanError, "frozen exact7"):
            plan.validate_plan(value)

    def test_plan_cli_emits_canonical_value(self) -> None:
        result = subprocess.run(
            [sys.executable, str(METHOD_ROOT / "full30_action_source7_reencode_plan_v1.py"), "emit"],
            check=True,
            capture_output=True,
        )
        self.assertEqual(result.stdout, plan.canonical_json_bytes(plan.canonical_plan()) + b"\n")


class Source7RuntimeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._hostname_patch = mock.patch.object(
            runtime_cache,
            "_short_hostname",
            return_value=runtime_cache.EXPECTED_COMPUTE_NODE,
        )
        self._hostname_patch.start()

    def tearDown(self) -> None:
        self._hostname_patch.stop()

    @staticmethod
    def _environment(
        job_id: str, step_id: str, home: Path, cache_parent: Path
    ) -> dict[str, str]:
        cache_root = cache_parent / f"BOX-EXP-014-r4-{job_id}-{step_id}"
        return {
            "SLURM_JOB_ID": job_id,
            "SLURM_STEP_ID": step_id,
            "HOME": str(home),
            "MIOPEN_USER_DB_PATH": str(cache_root / "user-db"),
            "MIOPEN_CUSTOM_CACHE_DIR": str(cache_root / "kernel-cache"),
            "XDG_CACHE_HOME": str(cache_root / "xdg-cache"),
            "TMPDIR": str(cache_root / "tmp"),
        }

    @staticmethod
    def _filesystem(cache_parent: Path) -> dict[str, object]:
        return {
            "statfs_type": "ext2/ext3",
            "statfs_magic_hex": "0xef53",
            "mount_fstype": "ext4",
            "mount_point": "/",
            "source": "/dev/mapper/vgroot-lvroot",
            "source_is_local_block_device": True,
            "local_filesystem": True,
        }

    @staticmethod
    def _create_kernel_db(cache_root: Path, *, wrong_schema: bool = False) -> Path:
        path = cache_root / "kernel-cache/gfx90a68.ukdb"
        connection = sqlite3.connect(path)
        if wrong_schema:
            connection.execute("CREATE TABLE kern_db (id INTEGER PRIMARY KEY, wrong TEXT)")
        else:
            connection.execute(
                "CREATE TABLE `kern_db` (`id` INTEGER PRIMARY KEY ASC,"
                "`kernel_name` TEXT NOT NULL,`kernel_args` TEXT NOT NULL,"
                "`kernel_blob` BLOB NOT NULL,`kernel_hash` TEXT NOT NULL,"
                "`uncompressed_size` INT NOT NULL)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX idx_kern_db ON kern_db(kernel_name,kernel_args)"
            )
            connection.execute(
                "INSERT INTO kern_db(kernel_name,kernel_args,kernel_blob,kernel_hash,uncompressed_size) "
                "VALUES (?,?,?,?,?)",
                ("fixture-kernel", "fixture-args", b"fixture", "fixture-hash", 7),
            )
        connection.commit()
        connection.close()
        path.chmod(0o600)
        return path

    @staticmethod
    def _create_scoped_lock(cache_root: Path, *, mode: int = 0o777) -> Path:
        lock_root = cache_root / "tmp/miopen-lockfiles"
        lock_root.mkdir(mode=0o700)
        lock_root.chmod(mode)
        identity = runtime_cache.expected_miopen_lock_file_basenames(cache_root)
        path = lock_root / identity["expected_lock_basenames"][0]
        path.write_bytes(b"")
        path.chmod(mode)
        return path

    @staticmethod
    def _create_user_db(
        cache_root: Path, *, main_mode: int = 0o777, time_mode: int = 0o644
    ) -> tuple[Path, Path]:
        main = (
            cache_root
            / "user-db/gfx90a68.HIP.3_3_0_a85ca8a54-dirty.ufdb.txt"
        )
        main.write_bytes(b"fixture-user-find-db\n")
        main.chmod(main_mode)
        sidecar = Path(f"{main}.time")
        sidecar.write_bytes(b"fixture-time\n")
        sidecar.chmod(time_mode)
        return main, sidecar

    @staticmethod
    def _shared_exact7_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        plan_value = plan.validate_plan(plan.canonical_plan())
        plan_path = root / "source7-plan.json"
        plan_raw = plan.canonical_json_bytes(plan_value) + b"\n"
        plan_path.write_bytes(plan_raw)
        plan_path.chmod(0o400)
        plan_binding = {
            "path": str(plan_path),
            "file_sha256": sha(plan_raw),
            "plan_digest": plan_value["plan_digest"],
        }

        release_manifest, _ = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        release_path = root / "source.manifest.json"
        release_raw = release.canonical_json_bytes(release_manifest) + b"\n"
        release_path.write_bytes(release_raw)
        release_path.chmod(0o444)
        release_binding = {
            "manifest_path": str(release_path),
            "manifest_file_sha256": sha(release_raw),
            "manifest_digest": release_manifest["manifest_digest"],
            "content_closure_sha1": release_manifest["content_closure_sha1"],
            "release_generation": "r4",
        }

        output = root / "physical_source_posterior_index0_exact7"
        output.mkdir(mode=0o700)
        receipt_rows = []
        physical_rows = []
        for index, planned in enumerate(plan_value["rows"], 1):
            iid = planned["iid"]
            shape = planned["expected_posterior_shape"]
            path = output / planned["output_filename"]
            blob, tensor_sha, tensor_raw_sha = _physical_test_tensor_blob(
                iid, shape
            )
            path.write_bytes(blob)
            path.chmod(0o400)
            file_sha = sha(blob)
            row = {
                "schema_version": materializer.ROW_SCHEMA,
                "iid": iid,
                "analysis_split": planned["analysis_split"],
                "event_id": planned["event_id"],
                "actor_kind": planned["actor_kind"],
                "q0_id": planned["q0_id"],
                "group_id": planned["group_id"],
                "actor_id": planned["actor_id"],
                "scene_id": planned["scene_id"],
                "source_video_path": planned["source_video_path"],
                "source_video_sha256": planned["source_video_sha256"],
                "source_video_stat_identity": [1, index, 1024 + index, 2000 + index],
                "source_video_sha256_before_decode": planned["source_video_sha256"],
                "source_video_sha256_after_decode": planned["source_video_sha256"],
                "source_video_pre_post_stat_and_hash_stable": True,
                "frame_count": 81,
                "expected_fps": 25.0,
                "reported_fps": 25.0,
                "input_hw": [shape[3] * 8, shape[4] * 8],
                "source_aspect_bucket_hw": [shape[3] * 8, shape[4] * 8],
                "posterior_parameters_path": str(path),
                "posterior_parameters_file_sha256": file_sha,
                "posterior_parameters_tensor_sha256": tensor_sha,
                "posterior_parameters_tensor_raw_sha256": tensor_raw_sha,
                "posterior_parameters_shape": shape,
                "posterior_parameters_dtype": "torch.float32",
                "posterior_parameters_device": "cpu",
                "posterior_parameters_layout": "torch.strided",
                "posterior_parameters_contiguous": True,
                "posterior_parameters_finite": True,
                "posterior_parameters_bare_tensor": True,
                "posterior_sample_materialized": False,
                "physical_file_reopened_after_write": True,
                "physical_tensor_reopened_after_write": True,
                "physical_tensor_equal_to_encoded_tensor": True,
                "peak_allocated_bytes": 1,
                **materializer._negative_access_closure(),
            }
            if set(row) != materializer.ROW_RECEIPT_FIELDS:
                raise AssertionError("physical materialization row fixture differs")
            receipt_rows.append(row)
            physical_rows.append(
                {
                    "iid": iid,
                    "path": str(path),
                    "file_sha256": file_sha,
                    "tensor_sha256": tensor_sha,
                    "tensor_raw_sha256": tensor_raw_sha,
                    "shape": shape,
                    "physical_file_and_tensor_reopened_post_publish": True,
                }
            )
        vae_unsigned = {
            "checkpoint_root": str(runtime_cache.EXPECTED_CHECKPOINT_ROOT),
            "checkpoint_content_manifest_path": str(
                runtime_cache.EXPECTED_CHECKPOINT_CONTENT_MANIFEST
            ),
            "checkpoint_content_manifest_sha256": (
                runtime_cache.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
            ),
            "vae_config_sha256": runtime_cache.EXPECTED_VAE_CONFIG_SHA256,
            "vae_files": {
                "vae/config.json": "1" * 64,
                "vae/diffusion_pytorch_model.safetensors": "2" * 64,
            },
            "every_vae_file_sha256_verified": True,
            "posterior_representation": "latent_dist.parameters_fp32",
            "posterior_sample_materialized": False,
        }
        vae_identity = {
            **vae_unsigned,
            "vae_identity_digest": materializer.object_sha256(vae_unsigned),
        }
        materialization_unsigned = {
            "schema_version": materializer.SCHEMA_VERSION,
            "method": materializer.METHOD_NAME,
            "experiment_id": runtime_cache.EXPERIMENT_ID,
            "complete": True,
            "plan": plan_binding,
            "output_root": str(output),
            "row_count": 7,
            "rows": receipt_rows,
            "external_existing_index0": {
                **plan_value["external_existing_index0"],
                "opened_by_materializer": False,
                "included_in_exact7_output_files": False,
                "reencoded": False,
            },
            "vae_identity": vae_identity,
            "output_filenames": list(runtime_cache.EXACT7_OUTPUT_FILENAMES),
            "output_exact_member_closure": True,
            "distinct_source_mp4_count": 7,
            "total_vae_encode_calls": 7,
            "posterior_sample_materialized": False,
            "external_existing_index0_opened": False,
            "external_existing_index0_reencoded": False,
            "inventory_snapshot_only": True,
            "exact8_authority_go_claimed": False,
            "teacher_cross_disjointness_pending": True,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "training_authorized": False,
            **materializer._negative_access_closure(),
        }
        materialization_receipt = {
            **materialization_unsigned,
            "receipt_digest": materializer.object_sha256(materialization_unsigned),
        }
        if set(materialization_receipt) != materializer.RECEIPT_FIELDS:
            raise AssertionError("physical materialization fixture differs")
        materialization_path = output / "materialization_receipt.json"
        materialization_raw = materializer.canonical_json_bytes(
            materialization_receipt
        ) + b"\n"
        materialization_path.write_bytes(materialization_raw)
        materialization_path.chmod(0o400)
        materialization_binding = {
            "receipt_path": str(materialization_path),
            "receipt_file_sha256": sha(materialization_raw),
            "receipt_digest": materialization_receipt["receipt_digest"],
            "post_publish_rows": physical_rows,
            "all_seven_physical_files_and_tensors_reopened": True,
        }
        return plan_binding, release_binding, materialization_binding

    @staticmethod
    def _smoke_fixture(
        receipt: Mapping[str, Any],
        inventory: Mapping[str, list[Mapping[str, Any]]],
        kernel_db: Mapping[str, Any],
        prepare_path: Path,
        prepare_sha: str,
    ) -> dict[str, Any]:
        geometries = []
        boundaries = {
            96: [864.0, 576.0, 576.0, 384.0],
            192: [1728.0, 1152.0, 1152.0, 768.0],
            384: [3456.0, 2304.0, 2304.0, 1536.0],
        }
        for index, values in enumerate(runtime_cache.WAN_RESAMPLE_CANDIDATE_GEOMETRIES):
            channels, height, width, padded_height, padded_width, out_height, out_width = values
            geometries.append(
                {
                    "candidate_index": index,
                    "pre_pad_input_shape": [1, channels, height, width],
                    "conv_input_shape": [1, channels, padded_height, padded_width],
                    "weight_shape": [channels, channels, 3, 3],
                    "bias_shape": [channels],
                    "stride": [2, 2],
                    "padding": [0, 0],
                    "output_shape": [1, channels, out_height, out_width],
                    "module_and_input_declared_dtype": "torch.float32",
                    "cuda_autocast_dtype": "torch.bfloat16",
                    "output_dtype": "torch.bfloat16",
                    "boundary_samples_fp32": boundaries[channels],
                    "finite": True,
                    "cuda_synchronized": True,
                }
            )
        cache_root = Path(receipt["cache_root"])
        lock_inventory, temp_lock_evidence = (
            runtime_cache.validate_scoped_miopen_temp_lock_activity(cache_root)
        )
        if lock_inventory != inventory:
            raise AssertionError("fixture inventory changed during lock validation")
        user_db_evidence = runtime_cache.miopen_user_db_evidence(inventory)
        user_db_claim = (
            "path-bound;expected-plaintext-write-observed-not-required"
            if user_db_evidence["plaintext_main_write_observed"]
            else "path-bound;no-write-observed-and-no-write-claim"
        )
        unsigned = {
            "schema_version": "bernini-full30-action-source7-reencode-miopen-conv-smoke-v3",
            "experiment_id": runtime_cache.EXPERIMENT_ID,
            "run_generation": "r4",
            "complete": True,
            "prepare_receipt_path": str(prepare_path),
            "prepare_receipt_file_sha256": prepare_sha,
            "prepare_digest": receipt["prepare_digest"],
            "cache_root": receipt["cache_root"],
            "environment": receipt["environment"],
            "miopen_user_db_path_kind": "directory",
            "miopen_custom_cache_dir_kind": "directory",
            "miopen_cache_paths_remained_canonical_0700_directories": True,
            "pre_conv_cache_inventory": receipt["post_probe_empty_inventory"],
            "post_conv_cache_inventory": inventory,
            "miopen_kernel_db_activity_required": True,
            "miopen_kernel_db_activity_observed": True,
            "miopen_kernel_db_evidence": kernel_db,
            "kernel_cache_claim": "path-bound;fresh-ukdb-write-required-and-observed",
            "miopen_user_db_claim": user_db_claim,
            "miopen_user_db_evidence": user_db_evidence,
            "scoped_miopen_temp_lock_activity_required": True,
            "scoped_miopen_temp_lock_activity_observed": True,
            "scoped_miopen_temp_lock_evidence": temp_lock_evidence,
            "tmpdir_cpp_temp_directory_path_redirect_observed": True,
            "global_miopen_lock_root_before_torch": receipt[
                "global_miopen_lock_root_before_torch"
            ],
            "global_miopen_lock_root_after_smoke": receipt[
                "global_miopen_lock_root_before_torch"
            ],
            "global_miopen_lock_root_metadata_unchanged": True,
            "global_miopen_lock_root_members_scanned": False,
            "global_miopen_lock_root_mutation_attempted": False,
            "torch_import_after_cache_validation": True,
            "pinned_runtime": {
                "torch_version": runtime_cache.EXPECTED_TORCH_VERSION,
                "torch_hip_version": runtime_cache.EXPECTED_TORCH_HIP_VERSION,
                "miopen_backend_version": runtime_cache.EXPECTED_MIOPEN_BACKEND_VERSION,
                "miopen_library_resolved_path": runtime_cache.EXPECTED_MIOPEN_LIBRARY_PATH,
                "miopen_library_size": runtime_cache.EXPECTED_MIOPEN_LIBRARY_SIZE,
                "miopen_library_sha256": runtime_cache.EXPECTED_MIOPEN_LIBRARY_SHA256,
                "miopen_embedded_version": runtime_cache.EXPECTED_MIOPEN_EMBEDDED_VERSION,
            },
            "backend": "ROCm-MIOpen-via-torch.backends.cudnn",
            "device_index": 0,
            "device_name": "fixture-gpu",
            "operation": "WanResample-downsample-ZeroPad2d-right-bottom-1-then-Conv2d",
            "r2_failure_step": "136141.115",
            "r2_failure_stack_location": "diffusers/models/autoencoders/autoencoder_kl_wan.py:298",
            "r2_failure_stack_candidate_closure": "all-three-first-temporal-chunk-spatial-downsample-convs",
            "vae_config_sha256": runtime_cache.EXPECTED_VAE_CONFIG_SHA256,
            "geometry_count": 3,
            "geometries": geometries,
            "module_and_input_declared_dtype": "torch.float32",
            "cuda_autocast_dtype": "torch.bfloat16",
            "loaded_miopen_library_paths": [runtime_cache.EXPECTED_MIOPEN_LIBRARY_PATH],
            "loaded_miopen_library_unique_exact_path": True,
            "peak_allocated_bytes": 1,
            "gpu_cache_cleared": True,
            "gpu_memory_allocated_after_clear": 0,
            "source_video_opened": False,
            "source_video_decoded": False,
            "vae_encode_calls": 0,
        }
        value = {
            **unsigned,
            "smoke_digest": runtime_cache.object_sha256(unsigned),
        }
        if set(value) != runtime_cache.SMOKE_FIELDS:
            raise AssertionError("fixture smoke field closure differs")
        return value

    def test_gpu299_statfs_and_mountinfo_identity_fixture_is_exact_and_hostile_closed(self) -> None:
        class FakeLibC:
            def __init__(self, magic: int):
                self.magic = magic

            def statfs(self, path, output):
                output._obj.f_type = self.magic
                return 0

        mountinfo = (
            "41 32 253:0 / / rw,relatime - ext4 "
            "/dev/mapper/vgroot-lvroot rw,errors=remount-ro\n"
        )
        with mock.patch.object(runtime_cache.ctypes, "CDLL", return_value=FakeLibC(0xEF53)), mock.patch.object(
            runtime_cache.Path, "read_text", return_value=mountinfo
        ):
            observed = runtime_cache._filesystem_identity(Path("/tmp"))
        self.assertEqual(observed, self._filesystem(Path("/tmp")))
        hostile_mountinfo = mountinfo.replace("/dev/mapper/vgroot-lvroot", "/dev/other")
        with mock.patch.object(runtime_cache.ctypes, "CDLL", return_value=FakeLibC(0xEF53)), mock.patch.object(
            runtime_cache.Path, "read_text", return_value=hostile_mountinfo
        ):
            hostile_source = runtime_cache._filesystem_identity(Path("/tmp"))
        self.assertFalse(hostile_source["local_filesystem"])
        with mock.patch.object(runtime_cache.ctypes, "CDLL", return_value=FakeLibC(0x58465342)), mock.patch.object(
            runtime_cache.Path, "read_text", return_value=mountinfo
        ):
            hostile_magic = runtime_cache._filesystem_identity(Path("/tmp"))
        self.assertEqual(hostile_magic["statfs_type"], "unknown")
        self.assertFalse(hostile_magic["local_filesystem"])

    def test_prepare_tool_is_torch_free_and_controller_conv_is_deferred_before_materializer(self) -> None:
        cache_source = (
            TOOLS_ROOT / "full30_action_source7_reencode_runtime_cache_v1.py"
        ).read_text(encoding="utf-8")
        cache_tree = ast.parse(cache_source)
        top_level_imports = {
            alias.name
            for node in cache_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("import torch", cache_source)
        controller_source = (
            METHOD_ROOT / "full30_action_source7_reencode_controller_v1.py"
        ).read_text(encoding="utf-8")
        controller_tree = ast.parse(controller_source)
        controller_top_level_imports = {
            alias.name
            for node in controller_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", controller_top_level_imports)
        self.assertIn("torch.nn.functional.conv2d", controller_source)
        run_source = controller_source[controller_source.index("def run(") :]
        self.assertLess(
            run_source.index("runtime_cache.validate_prepare_receipt"),
            run_source.index("_run_cuda_miopen_conv_smoke"),
        )
        self.assertLess(
            run_source.index("_run_cuda_miopen_conv_smoke"),
            run_source.index("materializer.materialize"),
        )

    def test_prepare_proves_paths_modes_fsync_sqlite_and_cleanup_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_parent = root / "tmp"
            logs = root / "logs"
            home = root / "home"
            for path in (cache_parent, logs, home):
                path.mkdir(mode=0o700)
            env = self._environment("136141", "901", home, cache_parent)
            prepare_path = logs / "prepare.json"
            physical_reader = runtime_cache._read_shared_plain_file_at

            def development_release_reader(path, **kwargs):
                try:
                    path.relative_to(METHOD_ROOT.resolve(strict=True))
                except ValueError:
                    pass
                else:
                    kwargs["expected_mode"] = stat.S_IMODE(path.stat().st_mode)
                return physical_reader(path, **kwargs)

            with mock.patch.object(runtime_cache, "CACHE_PARENT", cache_parent), mock.patch.object(
                runtime_cache,
                "_filesystem_identity",
                return_value=self._filesystem(cache_parent),
            ), mock.patch.object(
                runtime_cache,
                "_load_pinned_cpu_torch",
                return_value=_physical_test_torch(),
            ), mock.patch.object(
                runtime_cache,
                "_read_shared_plain_file_at",
                side_effect=development_release_reader,
            ):
                receipt = runtime_cache.prepare_runtime_cache(
                    receipt_output=prepare_path,
                    environ=env,
                    cache_parent=cache_parent,
                )
                cache_root = cache_parent / "BOX-EXP-014-r4-136141-901"
                self.assertEqual(receipt["cache_root"], str(cache_root))
                self.assertEqual(stat.S_IMODE(cache_root.stat().st_mode), 0o700)
                for name in runtime_cache.SUBDIRECTORY_NAMES:
                    self.assertEqual(stat.S_IMODE((cache_root / name).stat().st_mode), 0o700)
                self.assertTrue(receipt["created_fresh_create_only"])
                self.assertTrue(receipt["exclusive_fsync_probe"]["o_excl"])
                self.assertTrue(receipt["exclusive_fsync_probe"]["physical_reopen"])
                self.assertTrue(receipt["sqlite_commit_reopen_probe"]["transaction_committed"])
                self.assertTrue(receipt["sqlite_commit_reopen_probe"]["readonly_reopen"])
                self.assertFalse(receipt["cache_reusable"])
                self.assertNotIn("torch", sys.modules)
                bound_env = {**env, **receipt["environment"]}
                prepare_sha = sha(prepare_path.read_bytes())
                validated = runtime_cache.validate_prepare_receipt(
                    receipt_path=prepare_path,
                    expected_sha256=prepare_sha,
                    environ=bound_env,
                )
                self.assertEqual(validated["prepare_digest"], receipt["prepare_digest"])
                self.assertEqual(
                    receipt["post_probe_empty_inventory"],
                    {name: [] for name in runtime_cache.SUBDIRECTORY_NAMES},
                )
                self._create_kernel_db(cache_root)
                self._create_scoped_lock(cache_root)
                self._create_user_db(cache_root)
                post_inventory, kernel_db = (
                    runtime_cache.validate_miopen_kernel_cache_activity(cache_root)
                )
                smoke_fixture = self._smoke_fixture(
                    receipt, post_inventory, kernel_db, prepare_path, prepare_sha
                )
                hostile_smoke = copy.deepcopy(smoke_fixture)
                hostile_smoke["unregistered_field"] = False
                hostile_unsigned = dict(hostile_smoke)
                hostile_unsigned.pop("smoke_digest")
                hostile_smoke["smoke_digest"] = runtime_cache.object_sha256(
                    hostile_unsigned
                )
                with self.assertRaisesRegex(
                    runtime_cache.Source7RuntimeCacheError,
                    "field closure differs",
                ):
                    runtime_cache._validate_smoke_for_cleanup(
                        hostile_smoke,
                        prepare=receipt,
                        prepare_receipt_path=prepare_path,
                        expected_prepare_sha256=prepare_sha,
                    )
                connection = sqlite3.connect(
                    cache_root / "kernel-cache/gfx90a68.ukdb"
                )
                connection.execute(
                    "INSERT INTO kern_db(kernel_name,kernel_args,kernel_blob,kernel_hash,uncompressed_size) "
                    "VALUES (?,?,?,?,?)",
                    ("second-kernel", "second-args", b"second", "second-hash", 6),
                )
                connection.commit()
                connection.close()
                final_inventory, final_kernel_db = (
                    runtime_cache.validate_miopen_kernel_cache_activity(cache_root)
                )
                lock_inventory, final_temp_lock = (
                    runtime_cache.validate_scoped_miopen_temp_lock_activity(
                        cache_root
                    )
                )
                self.assertEqual(lock_inventory, final_inventory)
                final_user_db = runtime_cache.miopen_user_db_evidence(
                    final_inventory
                )
                self.assertGreater(
                    final_kernel_db["kern_db_row_count"],
                    kernel_db["kern_db_row_count"],
                )

                plan_binding, release_binding, materialization_binding = (
                    self._shared_exact7_fixture(root)
                )
                frozen_plan = plan.canonical_plan()
                completion_unsigned = {
                    "schema_version": runtime_cache.CONTROLLER_COMPLETION_SCHEMA,
                    "experiment_id": runtime_cache.EXPERIMENT_ID,
                    "complete": True,
                    "run_generation": "r4",
                    "purpose": frozen_plan["purpose"],
                    "scientific_target": frozen_plan["scientific_target"],
                    "learning_target": frozen_plan["learning_target"],
                    "numeric_target": frozen_plan["numeric_target"],
                    "dataset": frozen_plan["dataset"],
                    "steps": frozen_plan["steps"],
                    "baseline": frozen_plan["baseline"],
                    "core_validation": frozen_plan["core_validation"],
                    "holder": {
                        "job_id": 136141,
                        "step_id": 901,
                        "node": runtime_cache.EXPECTED_COMPUTE_NODE,
                        "parent_retained": True,
                        "parent_cancelled": False,
                        "parent_released": False,
                        "parent_requeued": False,
                    },
                    "release": release_binding,
                    "runtime_cache": {
                        "prepare_receipt_path": str(prepare_path),
                        "prepare_receipt_file_sha256": prepare_sha,
                        "prepare_digest": receipt["prepare_digest"],
                        "cache_root": str(cache_root),
                        "hostname": receipt["hostname"],
                        "cache_root_device": receipt["cache_root_device"],
                        "cache_root_inode": receipt["cache_root_inode"],
                        "filesystem": receipt["filesystem"],
                        "directories": receipt["directories"],
                        "environment": receipt["environment"],
                        "home_unchanged": True,
                        "created_fresh_create_only": True,
                        "exclusive_fsync_probe": receipt[
                            "exclusive_fsync_probe"
                        ],
                        "sqlite_commit_reopen_probe": receipt[
                            "sqlite_commit_reopen_probe"
                        ],
                        "post_probe_empty_inventory": receipt[
                            "post_probe_empty_inventory"
                        ],
                        "global_miopen_lock_root_before_torch": receipt[
                            "global_miopen_lock_root_before_torch"
                        ],
                        "validated_before_torch_import": True,
                        "cache_reusable": False,
                        "cleanup_policy": receipt["cleanup_policy"],
                    },
                    "cuda_miopen_smoke": smoke_fixture,
                    "runtime_cache_post_materialization": {
                        "captured_after_exact7_materialization": True,
                        "cache_root": str(cache_root),
                        "inventory": final_inventory,
                        "miopen_kernel_db_evidence": final_kernel_db,
                        "miopen_user_db_evidence": final_user_db,
                        "scoped_miopen_temp_lock_evidence": final_temp_lock,
                        "global_miopen_lock_root_after_exact7": receipt[
                            "global_miopen_lock_root_before_torch"
                        ],
                        "global_miopen_lock_root_metadata_unchanged": True,
                        "global_miopen_lock_root_members_scanned": False,
                        "global_miopen_lock_root_mutation_attempted": False,
                    },
                    "plan": plan_binding,
                    "materialization": materialization_binding,
                    "external_existing_index0": frozen_plan[
                        "external_existing_index0"
                    ],
                    "external_existing_index0_reencoded": False,
                    "inventory_snapshot_only": True,
                    "exact8_authority_go_claimed": False,
                    "teacher_cross_disjointness_pending": True,
                    "optimizer_created": False,
                    "optimizer_updates": 0,
                    "training_authorized": False,
                    **materializer._negative_access_closure(),
                }
                completion = {
                    **completion_unsigned,
                    "completion_digest": runtime_cache.object_sha256(completion_unsigned),
                }
                completion_path = root / "controller-completion.json"
                completion_path.write_bytes(runtime_cache.canonical_json_bytes(completion) + b"\n")
                completion_path.chmod(0o400)
                completion_sha = sha(completion_path.read_bytes())
                cleanup_path = logs / "cleanup.json"
                cleanup = runtime_cache.cleanup_runtime_cache(
                    prepare_receipt_path=prepare_path,
                    expected_prepare_sha256=prepare_sha,
                    controller_completion_path=completion_path,
                    expected_controller_completion_sha256=completion_sha,
                    cleanup_receipt_output=cleanup_path,
                    controller_exit_status=0,
                    environ=bound_env,
                )
                self.assertTrue(cleanup["cleanup_after_controller_exit"])
                self.assertTrue(cleanup["cleanup_before_numbered_step_exit"])
                self.assertEqual(cleanup["cleanup_node"], "auh7-1b-gpu-299")
                self.assertTrue(cleanup["cache_root_removed"])
                self.assertFalse(cache_root.exists())
                audited = runtime_cache.validate_cleanup_receipt(
                    cleanup_receipt_path=cleanup_path,
                    expected_cleanup_sha256=sha(cleanup_path.read_bytes()),
                    prepare_receipt_path=prepare_path,
                    expected_prepare_sha256=prepare_sha,
                    controller_completion_path=completion_path,
                    expected_controller_completion_sha256=completion_sha,
                )
                self.assertFalse(audited["cache_root_reusable"])

                stale_prepare_digest = copy.deepcopy(receipt)
                stale_prepare_digest["sqlite_commit_reopen_probe"][
                    "sqlite_version"
                ] += "-tampered-without-inner-resign"
                prepare_path.write_bytes(
                    runtime_cache.canonical_json_bytes(stale_prepare_digest)
                    + b"\n"
                )
                with self.assertRaisesRegex(
                    runtime_cache.Source7RuntimeCacheError,
                    "prepare receipt digest differs",
                ):
                    runtime_cache.validate_cleanup_receipt(
                        cleanup_receipt_path=cleanup_path,
                        expected_cleanup_sha256=sha(cleanup_path.read_bytes()),
                        prepare_receipt_path=prepare_path,
                        expected_prepare_sha256=sha(prepare_path.read_bytes()),
                        controller_completion_path=completion_path,
                        expected_controller_completion_sha256=completion_sha,
                    )
                prepare_path.write_bytes(
                    runtime_cache.canonical_json_bytes(receipt) + b"\n"
                )

                stale_completion_digest = copy.deepcopy(completion)
                stale_completion_digest["post_srun_tamper"] = True
                completion_path.chmod(0o600)
                completion_path.write_bytes(
                    runtime_cache.canonical_json_bytes(stale_completion_digest)
                    + b"\n"
                )
                completion_path.chmod(0o400)
                with self.assertRaisesRegex(
                    runtime_cache.Source7RuntimeCacheError,
                    "controller completion digest differs",
                ):
                    runtime_cache.validate_cleanup_receipt(
                        cleanup_receipt_path=cleanup_path,
                        expected_cleanup_sha256=sha(cleanup_path.read_bytes()),
                        prepare_receipt_path=prepare_path,
                        expected_prepare_sha256=sha(prepare_path.read_bytes()),
                        controller_completion_path=completion_path,
                        expected_controller_completion_sha256=sha(
                            completion_path.read_bytes()
                        ),
                    )
                completion_path.chmod(0o600)
                completion_path.write_bytes(
                    runtime_cache.canonical_json_bytes(completion) + b"\n"
                )
                completion_path.chmod(0o400)

                materialization_path = Path(
                    completion["materialization"]["receipt_path"]
                )
                baseline_materialization_receipt = json.loads(
                    materialization_path.read_bytes()
                )

                def reset_physical_tensors() -> None:
                    for planned in frozen_plan["rows"]:
                        tensor_path = (
                            materialization_path.parent / planned["output_filename"]
                        )
                        blob, _, _ = _physical_test_tensor_blob(
                            planned["iid"], planned["expected_posterior_shape"]
                        )
                        tensor_path.chmod(0o600)
                        tensor_path.write_bytes(blob)
                        tensor_path.chmod(0o400)

                def write_canonical(path: Path, value: Mapping[str, Any], mode: int) -> str:
                    path.chmod(0o600)
                    raw = runtime_cache.canonical_json_bytes(value) + b"\n"
                    path.write_bytes(raw)
                    path.chmod(mode)
                    return sha(raw)

                def fully_resign_hostile(mutator) -> tuple[str, str, str]:
                    reset_physical_tensors()
                    hostile_prepare = copy.deepcopy(receipt)
                    unsigned_prepare = dict(hostile_prepare)
                    unsigned_prepare.pop("prepare_digest")
                    hostile_prepare["prepare_digest"] = runtime_cache.object_sha256(
                        unsigned_prepare
                    )
                    hostile_prepare_sha = write_canonical(
                        prepare_path, hostile_prepare, 0o600
                    )

                    hostile_materialization = copy.deepcopy(
                        baseline_materialization_receipt
                    )
                    hostile_completion = copy.deepcopy(completion)
                    mutator(hostile_materialization, hostile_completion)
                    unsigned_materialization = dict(hostile_materialization)
                    unsigned_materialization.pop("receipt_digest")
                    hostile_materialization["receipt_digest"] = (
                        runtime_cache.object_sha256(unsigned_materialization)
                    )
                    hostile_materialization_sha = write_canonical(
                        materialization_path, hostile_materialization, 0o400
                    )
                    hostile_completion["materialization"][
                        "receipt_file_sha256"
                    ] = hostile_materialization_sha
                    hostile_completion["materialization"]["receipt_digest"] = (
                        hostile_materialization["receipt_digest"]
                    )
                    hostile_completion["runtime_cache"][
                        "prepare_receipt_file_sha256"
                    ] = hostile_prepare_sha
                    hostile_completion["runtime_cache"]["prepare_digest"] = (
                        hostile_prepare["prepare_digest"]
                    )
                    hostile_smoke = hostile_completion["cuda_miopen_smoke"]
                    hostile_smoke["prepare_receipt_file_sha256"] = (
                        hostile_prepare_sha
                    )
                    hostile_smoke["prepare_digest"] = hostile_prepare[
                        "prepare_digest"
                    ]
                    unsigned_smoke = dict(hostile_smoke)
                    unsigned_smoke.pop("smoke_digest")
                    hostile_smoke["smoke_digest"] = runtime_cache.object_sha256(
                        unsigned_smoke
                    )
                    unsigned_completion = dict(hostile_completion)
                    unsigned_completion.pop("completion_digest")
                    hostile_completion["completion_digest"] = (
                        runtime_cache.object_sha256(unsigned_completion)
                    )
                    hostile_completion_sha = write_canonical(
                        completion_path, hostile_completion, 0o400
                    )

                    hostile_cleanup = copy.deepcopy(cleanup)
                    hostile_cleanup["prepare_receipt_file_sha256"] = (
                        hostile_prepare_sha
                    )
                    hostile_cleanup["prepare_digest"] = hostile_prepare[
                        "prepare_digest"
                    ]
                    hostile_cleanup["controller_completion_file_sha256"] = (
                        hostile_completion_sha
                    )
                    hostile_cleanup["controller_completion_digest"] = (
                        hostile_completion["completion_digest"]
                    )
                    unsigned_cleanup = dict(hostile_cleanup)
                    unsigned_cleanup.pop("cleanup_digest")
                    hostile_cleanup["cleanup_digest"] = runtime_cache.object_sha256(
                        unsigned_cleanup
                    )
                    hostile_cleanup_sha = write_canonical(
                        cleanup_path, hostile_cleanup, 0o600
                    )
                    return (
                        hostile_prepare_sha,
                        hostile_completion_sha,
                        hostile_cleanup_sha,
                    )

                first_iid = frozen_plan["rows"][0]["iid"]
                first_shape = frozen_plan["rows"][0]["expected_posterior_shape"]

                def mutate_path(materialization_receipt, completion_value):
                    wrong = str(
                        materialization_path.parent
                        / f"{first_iid}.wrong-source-posterior-index0.pt"
                    )
                    materialization_receipt["rows"][0][
                        "posterior_parameters_path"
                    ] = wrong
                    completion_value["materialization"]["post_publish_rows"][0][
                        "path"
                    ] = wrong

                def mutate_file_sha(materialization_receipt, completion_value):
                    materialization_receipt["rows"][0][
                        "posterior_parameters_file_sha256"
                    ] = "a" * 64
                    completion_value["materialization"]["post_publish_rows"][0][
                        "file_sha256"
                    ] = "a" * 64

                def mutate_tensor_sha(materialization_receipt, completion_value):
                    materialization_receipt["rows"][0][
                        "posterior_parameters_tensor_sha256"
                    ] = "b" * 64
                    completion_value["materialization"]["post_publish_rows"][0][
                        "tensor_sha256"
                    ] = "b" * 64

                def mutate_tensor_raw_sha(materialization_receipt, completion_value):
                    materialization_receipt["rows"][0][
                        "posterior_parameters_tensor_raw_sha256"
                    ] = "c" * 64
                    completion_value["materialization"]["post_publish_rows"][0][
                        "tensor_raw_sha256"
                    ] = "c" * 64

                def content_mutator(**blob_options):
                    def mutate(materialization_receipt, completion_value):
                        blob, tensor_sha, tensor_raw_sha = _physical_test_tensor_blob(
                            first_iid,
                            blob_options.pop("shape", first_shape),
                            **blob_options,
                        )
                        tensor_path = materialization_path.parent / (
                            f"{first_iid}.source-posterior-index0.pt"
                        )
                        tensor_path.chmod(0o600)
                        tensor_path.write_bytes(blob)
                        tensor_path.chmod(0o400)
                        file_sha = sha(blob)
                        receipt_row = materialization_receipt["rows"][0]
                        post_row = completion_value["materialization"][
                            "post_publish_rows"
                        ][0]
                        receipt_row["posterior_parameters_file_sha256"] = file_sha
                        receipt_row["posterior_parameters_tensor_sha256"] = tensor_sha
                        receipt_row["posterior_parameters_tensor_raw_sha256"] = (
                            tensor_raw_sha
                        )
                        post_row["file_sha256"] = file_sha
                        post_row["tensor_sha256"] = tensor_sha
                        post_row["tensor_raw_sha256"] = tensor_raw_sha

                    return mutate

                def mutate_row_extra(materialization_receipt, completion_value):
                    materialization_receipt["rows"][0]["unregistered"] = False

                def mutate_release_extra(materialization_receipt, completion_value):
                    completion_value["release"]["unregistered"] = False

                def mutate_completion_extra(materialization_receipt, completion_value):
                    completion_value["unregistered"] = False

                hostile_cases = {
                    "canonical_path": mutate_path,
                    "physical_file_sha": mutate_file_sha,
                    "producer_tensor_sha": mutate_tensor_sha,
                    "producer_raw_sha": mutate_tensor_raw_sha,
                    "pickle_object_container": content_mutator(
                        container="mapping-with-tensor-key"
                    ),
                    "wrong_dtype": content_mutator(dtype="torch.float16"),
                    "wrong_shape": content_mutator(
                        shape=[1, 32, 21, first_shape[4], first_shape[3]]
                    ),
                    "nonfinite": content_mutator(finite=False),
                    "row_extra_field": mutate_row_extra,
                    "release_extra_field": mutate_release_extra,
                    "completion_extra_field": mutate_completion_extra,
                }
                marker = logs / "BOX-EXP-014_R4_COMPLETE-physical-hostile"
                for label, mutator in hostile_cases.items():
                    with self.subTest(post_srun_hostile=label):
                        marker.unlink(missing_ok=True)
                        prepare_case_sha, completion_case_sha, cleanup_case_sha = (
                            fully_resign_hostile(mutator)
                        )
                        admitted = False
                        try:
                            runtime_cache.validate_cleanup_receipt(
                                cleanup_receipt_path=cleanup_path,
                                expected_cleanup_sha256=cleanup_case_sha,
                                prepare_receipt_path=prepare_path,
                                expected_prepare_sha256=prepare_case_sha,
                                controller_completion_path=completion_path,
                                expected_controller_completion_sha256=(
                                    completion_case_sha
                                ),
                            )
                            admitted = True
                        except runtime_cache.Source7RuntimeCacheError:
                            pass
                        if admitted:
                            marker.write_bytes(b"incorrectly-admitted\n")
                        self.assertFalse(admitted)
                        self.assertFalse(marker.exists())

                reset_physical_tensors()
                write_canonical(
                    materialization_path,
                    baseline_materialization_receipt,
                    0o400,
                )
                write_canonical(prepare_path, receipt, 0o600)
                write_canonical(completion_path, completion, 0o400)
                write_canonical(cleanup_path, cleanup, 0o600)

                # Simulate a post-srun shared-filesystem attacker that changes
                # all three receipts, recomputes every internal digest and
                # updates every cross-file SHA/digest binding.  Semantic
                # authority must still reject paired access before a marker.
                hostile_prepare = copy.deepcopy(receipt)
                hostile_prepare["sqlite_commit_reopen_probe"][
                    "sqlite_version"
                ] += "-resigned"
                hostile_prepare_unsigned = dict(hostile_prepare)
                hostile_prepare_unsigned.pop("prepare_digest")
                hostile_prepare["prepare_digest"] = runtime_cache.object_sha256(
                    hostile_prepare_unsigned
                )
                prepare_path.write_bytes(
                    runtime_cache.canonical_json_bytes(hostile_prepare) + b"\n"
                )
                prepare_path.chmod(0o600)
                hostile_prepare_sha = sha(prepare_path.read_bytes())

                hostile_completion = copy.deepcopy(completion)
                hostile_completion["paired_dataset_accessed"] = True
                hostile_completion["runtime_cache"][
                    "prepare_receipt_file_sha256"
                ] = hostile_prepare_sha
                hostile_completion["runtime_cache"][
                    "prepare_digest"
                ] = hostile_prepare["prepare_digest"]
                hostile_smoke = hostile_completion["cuda_miopen_smoke"]
                hostile_smoke["prepare_receipt_file_sha256"] = hostile_prepare_sha
                hostile_smoke["prepare_digest"] = hostile_prepare["prepare_digest"]
                hostile_smoke_unsigned = dict(hostile_smoke)
                hostile_smoke_unsigned.pop("smoke_digest")
                hostile_smoke["smoke_digest"] = runtime_cache.object_sha256(
                    hostile_smoke_unsigned
                )
                hostile_completion_unsigned = dict(hostile_completion)
                hostile_completion_unsigned.pop("completion_digest")
                hostile_completion["completion_digest"] = (
                    runtime_cache.object_sha256(hostile_completion_unsigned)
                )
                completion_path.chmod(0o600)
                completion_path.write_bytes(
                    runtime_cache.canonical_json_bytes(hostile_completion) + b"\n"
                )
                completion_path.chmod(0o400)
                hostile_completion_sha = sha(completion_path.read_bytes())

                hostile_cleanup = copy.deepcopy(cleanup)
                hostile_cleanup["prepare_receipt_file_sha256"] = (
                    hostile_prepare_sha
                )
                hostile_cleanup["prepare_digest"] = hostile_prepare[
                    "prepare_digest"
                ]
                hostile_cleanup["controller_completion_file_sha256"] = (
                    hostile_completion_sha
                )
                hostile_cleanup["controller_completion_digest"] = (
                    hostile_completion["completion_digest"]
                )
                hostile_cleanup_unsigned = dict(hostile_cleanup)
                hostile_cleanup_unsigned.pop("cleanup_digest")
                hostile_cleanup["cleanup_digest"] = runtime_cache.object_sha256(
                    hostile_cleanup_unsigned
                )
                cleanup_path.write_bytes(
                    runtime_cache.canonical_json_bytes(hostile_cleanup) + b"\n"
                )
                cleanup_path.chmod(0o600)
                hostile_cleanup_sha = sha(cleanup_path.read_bytes())
                marker = logs / "BOX-EXP-014_R4_COMPLETE"
                audit_admitted = False
                try:
                    runtime_cache.validate_cleanup_receipt(
                        cleanup_receipt_path=cleanup_path,
                        expected_cleanup_sha256=hostile_cleanup_sha,
                        prepare_receipt_path=prepare_path,
                        expected_prepare_sha256=hostile_prepare_sha,
                        controller_completion_path=completion_path,
                        expected_controller_completion_sha256=(
                            hostile_completion_sha
                        ),
                    )
                    audit_admitted = True
                except runtime_cache.Source7RuntimeCacheError as error:
                    self.assertIn("negative-access authority differs", str(error))
                if audit_admitted:
                    marker.write_bytes(b"incorrectly-admitted\n")
                self.assertFalse(audit_admitted)
                self.assertFalse(marker.exists())

    def test_prepare_rejects_unsafe_token_and_every_existing_root_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_parent = root / "tmp"
            logs = root / "logs"
            home = root / "home"
            for path in (cache_parent, logs, home):
                path.mkdir(mode=0o700)
            filesystem = self._filesystem(cache_parent)
            with mock.patch.object(runtime_cache, "CACHE_PARENT", cache_parent), mock.patch.object(
                runtime_cache, "_filesystem_identity", return_value=filesystem
            ):
                with self.assertRaisesRegex(runtime_cache.Source7RuntimeCacheError, "token differs"):
                    runtime_cache.prepare_runtime_cache(
                        receipt_output=logs / "unsafe.json",
                        environ=self._environment("136141", "../115", home, cache_parent),
                        cache_parent=cache_parent,
                    )
                for step_id, kind in (("902", "file"), ("903", "directory"), ("904", "symlink")):
                    with self.subTest(kind=kind):
                        expected = cache_parent / f"BOX-EXP-014-r4-136141-{step_id}"
                        if kind == "file":
                            expected.write_bytes(b"occupied")
                        elif kind == "directory":
                            expected.mkdir()
                        else:
                            expected.symlink_to(home, target_is_directory=True)
                        with self.assertRaisesRegex(runtime_cache.Source7RuntimeCacheError, "not fresh"):
                            runtime_cache.prepare_runtime_cache(
                                receipt_output=logs / f"{step_id}.json",
                                environ=self._environment("136141", step_id, home, cache_parent),
                                cache_parent=cache_parent,
                            )

    def test_probe_failure_retains_fresh_cache_and_publishes_no_success_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_parent = root / "tmp"
            logs = root / "logs"
            home = root / "home"
            for path in (cache_parent, logs, home):
                path.mkdir(mode=0o700)
            receipt_path = logs / "prepare.json"
            env = self._environment("136141", "905", home, cache_parent)
            with mock.patch.object(runtime_cache, "CACHE_PARENT", cache_parent), mock.patch.object(
                runtime_cache,
                "_filesystem_identity",
                return_value=self._filesystem(cache_parent),
            ), mock.patch.object(
                runtime_cache,
                "_sqlite_commit_reopen_probe",
                side_effect=runtime_cache.Source7RuntimeCacheError("forced SQLite probe failure"),
            ):
                with self.assertRaisesRegex(runtime_cache.Source7RuntimeCacheError, "forced SQLite"):
                    runtime_cache.prepare_runtime_cache(
                        receipt_output=receipt_path,
                        environ=env,
                        cache_parent=cache_parent,
                    )
            self.assertTrue((cache_parent / "BOX-EXP-014-r4-136141-905").is_dir())
            self.assertFalse(receipt_path.exists())

    def test_empty_or_wrong_schema_kernel_cache_is_rejected_after_conv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in runtime_cache.SUBDIRECTORY_NAMES:
                (root / name).mkdir(mode=0o700)
            root.chmod(0o700)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError, "gfx90a68.ukdb was not created"
            ):
                runtime_cache.validate_miopen_kernel_cache_activity(root)
            self._create_kernel_db(root, wrong_schema=True)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError, "column schema differs"
            ):
                runtime_cache.validate_miopen_kernel_cache_activity(root)

    def test_kernel_cache_gfx90a_ukdb_requires_nonempty_kern_db_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in runtime_cache.SUBDIRECTORY_NAMES:
                (root / name).mkdir(mode=0o700)
            root.chmod(0o700)
            path = self._create_kernel_db(root)
            connection = sqlite3.connect(path)
            connection.execute("DELETE FROM kern_db")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError, "no compiled kernel rows"
            ):
                runtime_cache.validate_miopen_kernel_cache_activity(root)
            connection = sqlite3.connect(path)
            connection.execute(
                "INSERT INTO kern_db(kernel_name,kernel_args,kernel_blob,kernel_hash,uncompressed_size) "
                "VALUES (?,?,?,?,?)",
                ("fixture-kernel", "fixture-args", b"fixture", "fixture-hash", 7),
            )
            connection.commit()
            connection.close()
            inventory, evidence = runtime_cache.validate_miopen_kernel_cache_activity(root)
            self.assertTrue(evidence["kern_db_nonempty"])
            self.assertEqual(evidence["kern_db_row_count"], 1)
            self.assertEqual(evidence["basename"], "gfx90a68.ukdb")
            self.assertTrue(inventory["kernel-cache"])

    def test_kernel_cache_rejects_mode_0644_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in runtime_cache.SUBDIRECTORY_NAMES:
                (root / name).mkdir(mode=0o700)
            root.chmod(0o700)
            path = self._create_kernel_db(root)
            path.chmod(0o644)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError, "name/kind/mode differs"
            ):
                runtime_cache.validate_miopen_kernel_cache_activity(root)

    def test_user_db_real_0777_main_and_0644_time_are_positive_but_hostiles_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in runtime_cache.SUBDIRECTORY_NAMES:
                (root / name).mkdir(mode=0o700)
            root.chmod(0o700)
            main, sidecar = self._create_user_db(root)
            inventory = runtime_cache.inventory_cache_directories(root)
            evidence = runtime_cache.miopen_user_db_evidence(inventory)
            self.assertTrue(evidence["plaintext_main_write_observed"])
            self.assertEqual(evidence["main_file_mode_required"], 0o777)
            self.assertEqual(
                {row["mode"] for row in evidence["files"]}, {0o777, 0o644}
            )
            main.chmod(0o644)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError,
                "PlainTextDb main file mode must be 0777",
            ):
                runtime_cache.inventory_cache_directories(root)
            main.chmod(0o777)
            foreign = root / "user-db/foreign.ufdb.txt"
            foreign.write_bytes(b"foreign")
            foreign.chmod(0o777)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError,
                "user DB inventory member name/kind differs",
            ):
                runtime_cache.inventory_cache_directories(root)
            foreign.unlink()
            nested = root / "user-db/nested"
            nested.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError,
                "user DB inventory member name/kind differs",
            ):
                runtime_cache.inventory_cache_directories(root)
            nested.rmdir()
            sidecar.unlink()
            os.link(main, sidecar)
            with self.assertRaises(runtime_cache.Source7RuntimeCacheError):
                runtime_cache.inventory_cache_directories(root)
            sidecar.unlink()
            self.assertEqual(main.stat().st_nlink, 1)
            sidecar.symlink_to(main)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError, "identity differs"
            ):
                runtime_cache.inventory_cache_directories(root)
            sidecar.unlink()
            foreign_root_member = root / "foreign-top-level"
            foreign_root_member.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError,
                "root exact subdirectory closure differs",
            ):
                runtime_cache.inventory_cache_directories(root)

    def test_scoped_lock_md5_path_binding_and_global_root_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as global_directory:
            root = Path(directory).resolve()
            for name in runtime_cache.SUBDIRECTORY_NAMES:
                (root / name).mkdir(mode=0o700)
            root.chmod(0o700)
            global_root = Path(global_directory).resolve()
            global_root.chmod(0o777)
            sentinel = global_root / "foreign-user.lock"
            sentinel.write_bytes(b"do-not-touch")
            sentinel.chmod(0o777)
            before_sentinel = (
                sentinel.read_bytes(),
                sentinel.stat().st_ino,
                sentinel.stat().st_mtime_ns,
            )
            with mock.patch.object(
                runtime_cache, "GLOBAL_MIOPEN_LOCK_ROOT", global_root
            ):
                global_before = runtime_cache.observe_global_miopen_lock_root()
                lock = self._create_scoped_lock(root)
                inventory, evidence = (
                    runtime_cache.validate_scoped_miopen_temp_lock_activity(root)
                )
                global_after = runtime_cache.observe_global_miopen_lock_root()
            identity = runtime_cache.expected_miopen_lock_file_basenames(root)
            self.assertEqual(global_before, global_after)
            self.assertFalse(global_before["members_scanned"])
            self.assertFalse(global_before["mutation_attempted"])
            self.assertEqual(
                evidence["user_db_parent_md5"],
                identity["user_db_parent_md5"],
            )
            self.assertIn(lock.name, identity["expected_lock_basenames"])
            self.assertTrue(inventory["tmp"])
            self.assertEqual(
                before_sentinel,
                (
                    sentinel.read_bytes(),
                    sentinel.stat().st_ino,
                    sentinel.stat().st_mtime_ns,
                ),
            )
            observer_source = inspect.getsource(
                runtime_cache.observe_global_miopen_lock_root
            )
            self.assertNotIn("iterdir", observer_source)
            self.assertNotIn("os.walk", observer_source)
            lock.unlink()
            wrong = lock.parent / (
                "0" * 32
                + "_gfx90a68.HIP.3_3_0_a85ca8a54-dirty.ufdb.txt.lock"
            )
            wrong.write_bytes(b"")
            wrong.chmod(0o777)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError,
                "lock file name/kind/mode differs",
            ):
                runtime_cache.inventory_cache_directories(root)
            wrong.unlink()
            lock.write_bytes(b"")
            lock.chmod(0o644)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError,
                "lock file name/kind/mode differs",
            ):
                runtime_cache.inventory_cache_directories(root)

    def test_kernel_db_immutable_validation_accepts_shm_and_rejects_nonempty_wal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in runtime_cache.SUBDIRECTORY_NAMES:
                (root / name).mkdir(mode=0o700)
            root.chmod(0o700)
            self._create_kernel_db(root)
            shm = root / "kernel-cache/gfx90a68.ukdb-shm"
            shm.write_bytes(b"fixture-shm")
            shm.chmod(0o600)
            before = (shm.read_bytes(), shm.stat().st_mtime_ns)
            _, evidence = runtime_cache.validate_miopen_kernel_cache_activity(root)
            self.assertTrue(evidence["sqlite_immutable_reopen"])
            self.assertTrue(evidence["shm_observed"])
            self.assertEqual(before, (shm.read_bytes(), shm.stat().st_mtime_ns))
            wal = root / "kernel-cache/gfx90a68.ukdb-wal"
            wal.write_bytes(b"nonempty-wal")
            wal.chmod(0o600)
            with self.assertRaisesRegex(
                runtime_cache.Source7RuntimeCacheError,
                "absent-or-empty WAL",
            ):
                runtime_cache.validate_miopen_kernel_cache_activity(root)

    def test_prepare_and_cleanup_reject_wrong_compute_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_parent = root / "tmp"
            logs = root / "logs"
            home = root / "home"
            for path in (cache_parent, logs, home):
                path.mkdir(mode=0o700)
            with mock.patch.object(runtime_cache, "_short_hostname", return_value="login1"):
                with self.assertRaisesRegex(
                    runtime_cache.Source7RuntimeCacheError, "prepare node differs"
                ):
                    runtime_cache.prepare_runtime_cache(
                        receipt_output=logs / "prepare.json",
                        environ=self._environment("136141", "919", home, cache_parent),
                        cache_parent=cache_parent,
                    )

    def test_validation_rejects_environment_and_mode_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_parent = root / "tmp"
            logs = root / "logs"
            home = root / "home"
            for path in (cache_parent, logs, home):
                path.mkdir(mode=0o700)
            env = self._environment("136141", "906", home, cache_parent)
            receipt_path = logs / "prepare.json"
            with mock.patch.object(runtime_cache, "CACHE_PARENT", cache_parent), mock.patch.object(
                runtime_cache,
                "_filesystem_identity",
                return_value=self._filesystem(cache_parent),
            ):
                receipt = runtime_cache.prepare_runtime_cache(
                    receipt_output=receipt_path,
                    environ=env,
                    cache_parent=cache_parent,
                )
                receipt_sha = sha(receipt_path.read_bytes())
                bad_env = {**env, **receipt["environment"], "XDG_CACHE_HOME": str(root / "wrong")}
                with self.assertRaisesRegex(runtime_cache.Source7RuntimeCacheError, "environment differs"):
                    runtime_cache.validate_prepare_receipt(
                        receipt_path=receipt_path,
                        expected_sha256=receipt_sha,
                        environ=bad_env,
                    )
                cache_root = Path(receipt["cache_root"])
                cache_root.chmod(0o755)
                with self.assertRaisesRegex(runtime_cache.Source7RuntimeCacheError, "mode differs"):
                    runtime_cache.validate_prepare_receipt(
                        receipt_path=receipt_path,
                        expected_sha256=receipt_sha,
                        environ={**env, **receipt["environment"]},
                    )

    def test_retained_failure_observes_scoped_lock_root_absent_before_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_parent = root / "tmp"
            logs = root / "logs"
            home = root / "home"
            for path in (cache_parent, logs, home):
                path.mkdir(mode=0o700)
            env = self._environment("136141", "918", home, cache_parent)
            prepare_path = logs / "prepare.json"
            with mock.patch.object(
                runtime_cache, "CACHE_PARENT", cache_parent
            ), mock.patch.object(
                runtime_cache,
                "_filesystem_identity",
                return_value=self._filesystem(cache_parent),
            ):
                prepare = runtime_cache.prepare_runtime_cache(
                    receipt_output=prepare_path,
                    environ=env,
                    cache_parent=cache_parent,
                )
                prepare_sha = sha(prepare_path.read_bytes())
                cache_root = Path(prepare["cache_root"])
                scoped_lock_root = cache_root / "tmp/miopen-lockfiles"
                self.assertFalse(scoped_lock_root.exists())
                absent_path = logs / "retained-absent.json"
                absent = runtime_cache.record_retained_failure(
                    prepare_receipt_path=prepare_path,
                    expected_prepare_sha256=prepare_sha,
                    retained_failure_receipt_output=absent_path,
                    controller_exit_status=42,
                    environ=env,
                )
                self.assertTrue(absent["cache_root_retained"])
                self.assertFalse(absent["scoped_miopen_temp_lock_root_present"])
                self.assertFalse(absent["scoped_miopen_temp_lock_root_retained"])
                self.assertEqual(
                    absent["scoped_miopen_temp_lock_root_observation"]["kind"],
                    "absent",
                )
                audited_absent = runtime_cache.validate_retained_failure_receipt(
                    retained_failure_receipt_path=absent_path,
                    expected_retained_failure_sha256=sha(absent_path.read_bytes()),
                    prepare_receipt_path=prepare_path,
                    expected_prepare_sha256=prepare_sha,
                )
                self.assertFalse(
                    audited_absent["scoped_miopen_temp_lock_root_retained"]
                )

                self._create_scoped_lock(cache_root)
                present_path = logs / "retained-present.json"
                present = runtime_cache.record_retained_failure(
                    prepare_receipt_path=prepare_path,
                    expected_prepare_sha256=prepare_sha,
                    retained_failure_receipt_output=present_path,
                    controller_exit_status=43,
                    environ=env,
                )
                self.assertTrue(present["cache_root_retained"])
                self.assertTrue(present["scoped_miopen_temp_lock_root_present"])
                self.assertTrue(present["scoped_miopen_temp_lock_root_retained"])
                self.assertTrue(
                    present["scoped_miopen_temp_lock_root_observation"][
                        "canonical_nofollow_directory"
                    ]
                )
                runtime_cache.validate_retained_failure_receipt(
                    retained_failure_receipt_path=present_path,
                    expected_retained_failure_sha256=sha(present_path.read_bytes()),
                    prepare_receipt_path=prepare_path,
                    expected_prepare_sha256=prepare_sha,
                )

    def test_phase_failure_terminal_covers_retained_and_already_removed_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_parent = root / "tmp"
            logs = root / "logs"
            home = root / "home"
            for path in (cache_parent, logs, home):
                path.mkdir(mode=0o700)
            env = self._environment("136141", "920", home, cache_parent)
            cache_root = cache_parent / "BOX-EXP-014-r4-136141-920"
            cache_root.mkdir(mode=0o700)
            retained_output = logs / "retained-phase.json"
            with mock.patch.object(runtime_cache, "CACHE_PARENT", cache_parent):
                retained = runtime_cache.record_phase_failure(
                    phase_failure_receipt_output=retained_output,
                    phase="prepare-or-precontroller",
                    failure_exit_status=44,
                    environ=env,
                )
                self.assertTrue(retained["cache_root_present"])
                self.assertTrue(retained["cache_root_retained"])
                self.assertFalse(retained["success_claimed"])
                self.assertFalse(retained["final_marker_authorized"])
                with self.assertRaisesRegex(
                    runtime_cache.Source7RuntimeCacheError,
                    "receipt output must be fresh",
                ):
                    runtime_cache.record_phase_failure(
                        phase_failure_receipt_output=retained_output,
                        phase="prepare-or-precontroller",
                        failure_exit_status=44,
                        environ=env,
                    )
                cache_root.rmdir()
                removed_output = logs / "removed-phase.json"
                removed = runtime_cache.record_phase_failure(
                    phase_failure_receipt_output=removed_output,
                    phase="cleanup-or-cleanup-receipt-publication",
                    failure_exit_status=45,
                    environ=env,
                )
            self.assertFalse(removed["cache_root_present"])
            self.assertTrue(removed["cache_root_absent_at_terminal"])
            self.assertTrue(removed["cleanup_may_have_removed_cache_before_terminal"])
            self.assertFalse(removed["success_claimed"])


class Source7MaterializerTests(unittest.TestCase):
    def test_source_code_has_one_encode_site_and_no_legacy_reader(self) -> None:
        path = TOOLS_ROOT / "materialize_full30_action_source7_reencode_v1.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        encode_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "encode"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "model"
        ]
        self.assertEqual(len(encode_calls), 1)
        self.assertIn("source_only.base._decode_exact_video", source.replace("base._decode_exact_video", "source_only.base._decode_exact_video"))
        self.assertIn("base.source_aspect_bucket", source)
        self.assertIn("pinned.PinnedBerniniWanPosteriorEncoder", source)
        self.assertIn("base.tensor_to_bytes", source)
        self.assertNotIn("video_vae_latents", source)
        self.assertNotIn("pyarrow", source)
        self.assertNotIn(".parquet", source)
        self.assertNotIn("_materialize_row(", source)
        self.assertNotIn("_style_transform(", source)

    def test_negative_access_receipt_closure_is_literal(self) -> None:
        expected = {
                "source_only_reencode_from_source_video": True,
                "vae_encode_calls_per_source": 1,
                "paired_dataset_accessed": False,
                "legacy_source_target_container_opened": False,
                "synthetic_target_index1_path_read": False,
                "synthetic_target_index1_bytes_read": False,
                "synthetic_target_index1_decoded": False,
                "synthetic_target_index1_filtered_on": False,
                "synthetic_target_index1_hashed": False,
                "target_video_path_present": False,
                "target_video_accessed": False,
            }
        self.assertEqual(materializer._negative_access_closure(), expected)
        self.assertEqual(materializer.NEGATIVE_ACCESS_FIELDS, set(expected))
        self.assertEqual(controller.FORMAL_NEGATIVE_ACCESS_FIELDS, set(expected))
        canonical_plan = plan.canonical_plan()
        release_authority = release._authority()
        card = CARD.read_text(encoding="utf-8")
        for field, value in expected.items():
            self.assertEqual(canonical_plan[field], value)
            self.assertEqual(release_authority[field], value)
            rendered = str(value).lower() if type(value) is bool else str(value)
            self.assertIn(f"{field}={rendered}", card)

    def test_bare_fp32_tensor_roundtrip_and_non_tensor_rejection(self) -> None:
        tensor = _FakeTensor()
        with mock.patch.dict(sys.modules, {"torch": fake_torch_module(load_value=tensor)}):
            reopened = materializer._decode_bare_tensor(
                b"bare-fixture", list(tensor.shape), label="fixture"
            )
        self.assertIs(reopened, tensor)
        self.assertTrue(reopened.is_contiguous())
        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch_module(load_value={"posterior_parameters": tensor})},
        ), self.assertRaisesRegex(
            materializer.Source7ReencodeMaterializationError, "must be a tensor"
        ):
            materializer._decode_bare_tensor(
                b"mapping-fixture", list(tensor.shape), label="fixture"
            )

    def test_materialize_orchestration_schedules_exactly_seven_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan_path, plan_sha, value = write_plan(root)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            manifest = root / "checkpoint.sha256"
            manifest.write_bytes(b"fixture\n")
            output = root / "output"
            calls: list[str] = []

            class FakeEncoder:
                identity = {"vae_identity_digest": "f" * 64}

                def __init__(self, *args, **kwargs):
                    pass

            def fake_encode(row, *, encoder, stage, final_root):
                calls.append(row["iid"])
                (stage / row["output_filename"]).write_bytes(row["iid"].encode("ascii"))
                receipt = {field: None for field in materializer.ROW_RECEIPT_FIELDS}
                receipt.update(
                    {"iid": row["iid"], **materializer._negative_access_closure()}
                )
                return receipt

            with mock.patch.object(
                materializer.pinned,
                "PinnedBerniniWanPosteriorEncoder",
                FakeEncoder,
            ), mock.patch.object(
                materializer, "_bind_official_source_self_primitives"
            ), mock.patch.object(materializer, "_encode_one", side_effect=fake_encode):
                receipt = materializer.materialize(
                    plan_path=plan_path,
                    expected_plan_sha256=plan_sha,
                    checkpoint=checkpoint,
                    checkpoint_content_manifest=manifest,
                    expected_checkpoint_content_manifest_sha256="0" * 64,
                    output_root=output,
                )
            self.assertEqual(calls, [row["iid"] for row in value["rows"]])
            self.assertEqual(receipt["total_vae_encode_calls"], 7)
            self.assertEqual(receipt["distinct_source_mp4_count"], 7)
            self.assertFalse(receipt["external_existing_index0_opened"])
            self.assertFalse(receipt["external_existing_index0_reencoded"])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {row["output_filename"] for row in value["rows"]}
                | {"materialization_receipt.json"},
            )

    def test_atomic_noreplace_rejects_competing_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            stage = root / ".staging"
            destination = root / "published"
            stage.mkdir()
            (stage / "sealed.bin").write_bytes(b"sealed")
            self.assertFalse(destination.exists())
            # Deterministic publication race: a competitor claims the exact
            # destination after staging but before the atomic rename.
            destination.mkdir()
            with self.assertRaisesRegex(
                materializer.Source7ReencodeMaterializationError,
                "target already exists",
            ):
                materializer._rename_noreplace(stage, destination)
            self.assertTrue(stage.is_dir())
            self.assertEqual((stage / "sealed.bin").read_bytes(), b"sealed")
            self.assertTrue(destination.is_dir())
            self.assertEqual(list(destination.iterdir()), [])

    def test_create_only_output_is_enforced_before_encoder_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan_path, plan_sha, _ = write_plan(root)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            manifest = root / "checkpoint.sha256"
            manifest.write_bytes(b"fixture\n")
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(materializer.Source7ReencodeMaterializationError, "create-only"):
                materializer.materialize(
                    plan_path=plan_path,
                    expected_plan_sha256=plan_sha,
                    checkpoint=checkpoint,
                    checkpoint_content_manifest=manifest,
                    expected_checkpoint_content_manifest_sha256="0" * 64,
                    output_root=output,
                )


class Source7ControllerTests(unittest.TestCase):
    def test_pinned_torch_miopen_runtime_accepts_exact_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = root / "torch"
            library = package / "lib/libMIOpen.so"
            library.parent.mkdir(parents=True)
            torch_init = package / "__init__.py"
            torch_init.write_text("# fixture\n", encoding="ascii")
            raw = b"fixture\0" + controller.EXPECTED_MIOPEN_EMBEDDED_VERSION.encode("ascii") + b"\0"
            library.write_bytes(raw)
            library.chmod(0o755)
            fake_cudnn = types.SimpleNamespace(
                enabled=True,
                is_available=lambda: True,
                version=lambda: controller.EXPECTED_MIOPEN_BACKEND_VERSION,
            )
            fake = types.SimpleNamespace(
                __version__=controller.EXPECTED_TORCH_VERSION,
                __file__=str(torch_init),
                version=types.SimpleNamespace(hip=controller.EXPECTED_TORCH_HIP_VERSION),
                backends=types.SimpleNamespace(cudnn=fake_cudnn),
            )
            with mock.patch.object(controller, "EXPECTED_MIOPEN_LIBRARY_PATH", library), mock.patch.object(
                controller, "EXPECTED_MIOPEN_LIBRARY_SIZE", len(raw)
            ), mock.patch.object(
                controller, "EXPECTED_MIOPEN_LIBRARY_OWNER_UID", os.getuid()
            ), mock.patch.object(
                controller, "EXPECTED_MIOPEN_LIBRARY_SHA256", sha(raw)
            ):
                pinned = controller._validate_pinned_torch_miopen_runtime(fake)
                self.assertEqual(pinned["miopen_library_sha256"], sha(raw))
                drifted = types.SimpleNamespace(**{**fake.__dict__, "__version__": "drift"})
                with self.assertRaisesRegex(
                    controller.Source7ReencodeControllerError, "torch version differs"
                ):
                    controller._validate_pinned_torch_miopen_runtime(drifted)
                with mock.patch.object(
                    controller, "EXPECTED_MIOPEN_LIBRARY_SHA256", "0" * 64
                ):
                    with self.assertRaisesRegex(
                        controller.Source7ReencodeControllerError, "SHA-256 differs"
                    ):
                        controller._validate_pinned_torch_miopen_runtime(fake)

    def _published_fixture(self, root: Path) -> tuple[Path, dict]:
        value = plan.canonical_plan()
        plan_path = root / "plan.json"
        plan_raw = plan.canonical_json_bytes(value) + b"\n"
        plan_path.write_bytes(plan_raw)
        output = root / "published"
        output.mkdir()
        rows = []
        tensor_sha = "1" * 64
        tensor_raw_sha = "2" * 64
        for planned in value["rows"]:
            path = output / planned["output_filename"]
            raw = planned["iid"].encode("ascii")
            path.write_bytes(raw)
            rows.append(
                {
                    "schema_version": materializer.ROW_SCHEMA,
                    "iid": planned["iid"],
                    "analysis_split": planned["analysis_split"],
                    "event_id": planned["event_id"],
                    "actor_kind": planned["actor_kind"],
                    "q0_id": planned["q0_id"],
                    "group_id": planned["group_id"],
                    "actor_id": planned["actor_id"],
                    "scene_id": planned["scene_id"],
                    "source_video_path": planned["source_video_path"],
                    "source_video_sha256": planned["source_video_sha256"],
                    "source_video_stat_identity": [1, 2, len(raw), 4],
                    "source_video_sha256_before_decode": planned["source_video_sha256"],
                    "source_video_sha256_after_decode": planned["source_video_sha256"],
                    "source_video_pre_post_stat_and_hash_stable": True,
                    "frame_count": planned["frame_count"],
                    "expected_fps": planned["fps"],
                    "reported_fps": planned["fps"],
                    "input_hw": [720, 576],
                    "source_aspect_bucket_hw": [
                        planned["expected_posterior_shape"][3] * 8,
                        planned["expected_posterior_shape"][4] * 8,
                    ],
                    "posterior_parameters_path": str(path),
                    "posterior_parameters_file_sha256": sha(raw),
                    "posterior_parameters_tensor_sha256": tensor_sha,
                    "posterior_parameters_tensor_raw_sha256": tensor_raw_sha,
                    "posterior_parameters_shape": planned["expected_posterior_shape"],
                    "posterior_parameters_dtype": "torch.float32",
                    "posterior_parameters_device": "cpu",
                    "posterior_parameters_layout": "torch.strided",
                    "posterior_parameters_contiguous": True,
                    "posterior_parameters_finite": True,
                    "posterior_parameters_bare_tensor": True,
                    "posterior_sample_materialized": False,
                    "physical_file_reopened_after_write": True,
                    "physical_tensor_reopened_after_write": True,
                    "physical_tensor_equal_to_encoded_tensor": True,
                    "peak_allocated_bytes": 1,
                    **materializer._negative_access_closure(),
                }
            )
        unsigned = {
            "schema_version": materializer.SCHEMA_VERSION,
            "method": materializer.METHOD_NAME,
            "experiment_id": plan.EXPERIMENT_ID,
            "complete": True,
            "row_count": 7,
            "total_vae_encode_calls": 7,
            "distinct_source_mp4_count": 7,
            "plan": {
                "path": str(plan_path),
                "file_sha256": sha(plan_raw),
                "plan_digest": value["plan_digest"],
            },
            "output_root": str(output),
            "vae_identity": {
                "checkpoint_content_manifest_sha256": (
                    controller.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
                ),
                "every_vae_file_sha256_verified": True,
                "posterior_representation": "latent_dist.parameters_fp32",
                "posterior_sample_materialized": False,
            },
            "output_filenames": [row["output_filename"] for row in value["rows"]],
            "output_exact_member_closure": True,
            "rows": rows,
            "external_existing_index0": {
                **value["external_existing_index0"],
                "opened_by_materializer": False,
                "included_in_exact7_output_files": False,
                "reencoded": False,
            },
            "posterior_sample_materialized": False,
            "external_existing_index0_opened": False,
            "external_existing_index0_reencoded": False,
            "inventory_snapshot_only": True,
            "exact8_authority_go_claimed": False,
            "teacher_cross_disjointness_pending": True,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "training_authorized": False,
            **materializer._negative_access_closure(),
        }
        receipt = {**unsigned, "receipt_digest": controller.object_sha256(unsigned)}
        (output / "materialization_receipt.json").write_bytes(
            controller.canonical_json_bytes(receipt) + b"\n"
        )
        return output, value

    def _resign_receipt(self, output: Path, mutate) -> None:
        path = output / "materialization_receipt.json"
        value = json.loads(path.read_text(encoding="ascii"))
        mutate(value)
        unsigned = dict(value)
        unsigned.pop("receipt_digest", None)
        value = {**unsigned, "receipt_digest": controller.object_sha256(unsigned)}
        path.write_bytes(controller.canonical_json_bytes(value) + b"\n")

    def _assert_receipt_rejected(self, output: Path, value: dict) -> None:
        tensor = _FakeTensor(shape=(1,))
        with mock.patch.dict(
            sys.modules, {"torch": fake_torch_module(load_value=tensor)}
        ), mock.patch.object(
            materializer, "_decode_bare_tensor", return_value=tensor
        ), mock.patch.object(
            materializer.base, "_tensor_sha256", return_value="1" * 64
        ), mock.patch.object(
            materializer, "_tensor_raw_sha256", return_value="2" * 64
        ), self.assertRaises(controller.Source7ReencodeControllerError):
            controller.validate_published_materialization(
                output_root=output, plan=value
            )

    def test_post_publish_validation_reopens_all_seven(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, value = self._published_fixture(Path(directory).resolve())
            tensor = _FakeTensor(shape=(1,))
            with mock.patch.dict(
                sys.modules, {"torch": fake_torch_module(load_value=tensor)}
            ), mock.patch.object(
                materializer, "_decode_bare_tensor", return_value=tensor
            ) as decoder, mock.patch.object(
                materializer.base, "_tensor_sha256", return_value="1" * 64
            ), mock.patch.object(
                materializer, "_tensor_raw_sha256", return_value="2" * 64
            ):
                result = controller.validate_published_materialization(
                    output_root=output, plan=value
                )
            self.assertEqual(decoder.call_count, 7)
            self.assertTrue(result["all_seven_physical_files_and_tensors_reopened"])
            self.assertEqual(len(result["post_publish_rows"]), 7)

    def test_resigned_formal_true_is_rejected_top_level_and_per_row(self) -> None:
        false_fields = sorted(
            field
            for field, value in materializer._negative_access_closure().items()
            if value is False
        )
        for scope in ("top", "row"):
            for field in false_fields:
                with self.subTest(scope=scope, field=field), tempfile.TemporaryDirectory() as directory:
                    output, value = self._published_fixture(Path(directory).resolve())

                    def mutate(receipt, *, scope=scope, field=field):
                        target = receipt if scope == "top" else receipt["rows"][0]
                        target[field] = True

                    self._resign_receipt(output, mutate)
                    self._assert_receipt_rejected(output, value)

    def test_resigned_extra_field_is_rejected_top_level_and_per_row(self) -> None:
        for scope in ("top", "row"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as directory:
                output, value = self._published_fixture(Path(directory).resolve())

                def mutate(receipt, *, scope=scope):
                    target = receipt if scope == "top" else receipt["rows"][0]
                    target["unregistered_field"] = False

                self._resign_receipt(output, mutate)
                self._assert_receipt_rejected(output, value)

    def test_controller_completion_keeps_exact7_as_inventory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan_output = root / "plan.json"
            completion_output = root / "completion.json"
            fake_manifest = {
                "manifest_digest": "a" * 64,
                "content_closure_sha1": "b" * 40,
                "release_generation": "r4",
            }
            fake_receipt = {"complete": True}
            fake_global_lock = {"exists": False, "path": "/tmp/miopen-lockfiles"}
            fake_cache = {
                "slurm_step_id": "907",
                "prepare_digest": "9" * 64,
                "cache_root": "/tmp/BOX-EXP-014-r4-136141-907",
                "hostname": "auh7-1b-gpu-299",
                "cache_root_device": 2049,
                "cache_root_inode": 314159,
                "filesystem": {"local_filesystem": True},
                "directories": {},
                "environment": {
                    "HOME": "/home/fixture",
                    "MIOPEN_USER_DB_PATH": "/tmp/BOX-EXP-014-r4-136141-907/user-db",
                    "MIOPEN_CUSTOM_CACHE_DIR": "/tmp/BOX-EXP-014-r4-136141-907/kernel-cache",
                    "XDG_CACHE_HOME": "/tmp/BOX-EXP-014-r4-136141-907/xdg-cache",
                    "TMPDIR": "/tmp/BOX-EXP-014-r4-136141-907/tmp",
                },
                "exclusive_fsync_probe": {"o_excl": True},
                "sqlite_commit_reopen_probe": {"transaction_committed": True},
                "post_probe_empty_inventory": {
                    name: [] for name in runtime_cache.SUBDIRECTORY_NAMES
                },
                "global_miopen_lock_root_before_torch": fake_global_lock,
                "cleanup_policy": "cleanup-scoped-cache-including-tmp-locks-on-compute-node-after-controller-success-before-numbered-step-exit;retain-on-failure;never-touch-global-lock-root;never-reuse",
            }
            fake_smoke = {
                "complete": True,
                "geometry_count": 3,
                "miopen_kernel_db_activity_required": True,
                "miopen_kernel_db_activity_observed": True,
                "miopen_kernel_db_evidence": {
                    "kern_db_nonempty": True,
                    "kern_db_row_count": 1,
                },
                "scoped_miopen_temp_lock_activity_required": True,
                "scoped_miopen_temp_lock_activity_observed": True,
                "global_miopen_lock_root_before_torch": fake_global_lock,
                "global_miopen_lock_root_metadata_unchanged": True,
                "loaded_miopen_library_unique_exact_path": True,
                "source_video_opened": False,
                "source_video_decoded": False,
                "vae_encode_calls": 0,
                "smoke_digest": "8" * 64,
            }
            fake_physical = {
                "receipt_path": str(root / "receipt.json"),
                "receipt_file_sha256": "c" * 64,
                "receipt_digest": "d" * 64,
                "post_publish_rows": [],
                "all_seven_physical_files_and_tensors_reopened": True,
            }
            ordering: list[str] = []

            def validate_cache(**kwargs):
                ordering.append("cache-validated-before-torch")
                return fake_cache

            def smoke(**kwargs):
                ordering.append("three-wan-resample-cuda-miopen-convs")
                return fake_smoke

            def materialize(**kwargs):
                ordering.append("first-source-open-decode-encode")
                return fake_receipt

            with mock.patch.object(controller, "validate_release_tree", return_value=fake_manifest), mock.patch.object(
                controller.runtime_cache, "validate_prepare_receipt", side_effect=validate_cache
            ), mock.patch.object(
                controller, "_run_cuda_miopen_conv_smoke", side_effect=smoke
            ), mock.patch.object(
                controller.materializer, "materialize", side_effect=materialize
            ), mock.patch.object(
                controller, "validate_published_materialization", return_value=fake_physical
            ), mock.patch.object(
                controller.runtime_cache,
                "validate_miopen_kernel_cache_activity",
                return_value=(
                    {name: [] for name in runtime_cache.SUBDIRECTORY_NAMES},
                    {"kern_db_nonempty": True, "kern_db_row_count": 1},
                ),
            ), mock.patch.object(
                controller.runtime_cache,
                "validate_scoped_miopen_temp_lock_activity",
                return_value=(
                    {name: [] for name in runtime_cache.SUBDIRECTORY_NAMES},
                    {"activity_observed": True},
                ),
            ), mock.patch.object(
                controller.runtime_cache,
                "miopen_user_db_evidence",
                return_value={
                    "write_required": False,
                    "plaintext_main_write_observed": False,
                },
            ), mock.patch.object(
                controller.runtime_cache,
                "observe_global_miopen_lock_root",
                return_value=fake_global_lock,
            ):
                result = controller.run(
                    method_root=root,
                    release_manifest=root / "release.json",
                    expected_release_manifest_sha256="e" * 64,
                    plan_output=plan_output,
                    checkpoint=root,
                    checkpoint_content_manifest=root / "checkpoint.sha256",
                    expected_checkpoint_content_manifest_sha256=(
                        controller.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
                    ),
                    materialization_output_root=root / "materialized",
                    completion_output=completion_output,
                    runtime_cache_receipt=root / "runtime-cache-prepare.json",
                    expected_runtime_cache_receipt_sha256="7" * 64,
                    enforce_live_holder=False,
                )
            self.assertEqual(
                ordering,
                [
                    "cache-validated-before-torch",
                    "three-wan-resample-cuda-miopen-convs",
                    "first-source-open-decode-encode",
                ],
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["run_generation"], "r4")
            self.assertTrue(result["runtime_cache"]["validated_before_torch_import"])
            self.assertFalse(result["cuda_miopen_smoke"]["source_video_opened"])
            self.assertFalse(result["exact8_authority_go_claimed"])
            self.assertTrue(result["teacher_cross_disjointness_pending"])
            self.assertFalse(result["external_existing_index0_reencoded"])
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertTrue(completion_output.is_file())


class Source7ReleaseAndLauncherTests(unittest.TestCase):
    def test_release_archive_is_deterministic_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / "source.tar"
            manifest = root / "source.manifest.json"
            built = release.build(METHOD_ROOT.resolve(strict=True), archive, manifest)
            audited = release.audit(
                archive_path=archive,
                expected_archive_sha256=built["archive_sha256"],
                manifest_path=manifest,
                expected_manifest_sha256=built["manifest_sha256"],
            )
            self.assertTrue(audited["static_audit_go"])
            self.assertFalse(audited["upload_authorized"])
            self.assertFalse(audited["launch_authorized"])
            value = json.loads(manifest.read_text(encoding="ascii"))
            self.assertEqual(value["topology"]["holder_job_id"], 136141)
            self.assertEqual(value["topology"]["holder_node"], "auh7-1b-gpu-299")
            self.assertEqual(value["topology"]["run_generation"], "r4")
            self.assertEqual(value["topology"]["runtime_cache_statfs_type"], "ext2/ext3")
            self.assertEqual(value["topology"]["runtime_cache_mount_fstype"], "ext4")
            self.assertTrue(value["topology"]["parent_must_not_inspect_or_remove_compute_node_tmp"])
            self.assertEqual(value["authority"]["vae_encode_calls_per_source"], 1)
            self.assertEqual(value["authority"]["torch_version"], "2.7.1+rocm6.3")
            self.assertEqual(value["authority"]["miopen_backend_version"], 3003000)
            self.assertTrue(value["authority"]["miopen_custom_cache_kern_db_nonempty_required"])
            self.assertTrue(value["authority"]["miopen_lock_basenames_path_hash_bound"])
            self.assertTrue(value["authority"]["ustar_header_fields_explicitly_normalized"])
            self.assertEqual(value["release_generation"], "r4")

    def test_ustar_is_identical_across_python38_and_python39_builders(self) -> None:
        interpreters = [Path(sys.executable).resolve(strict=True), Path("/usr/bin/python3")]
        versions = []
        for interpreter in interpreters:
            version_result = subprocess.run(
                [str(interpreter), "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            version = (version_result.stdout or version_result.stderr).strip()
            versions.append(version)
        self.assertNotEqual(versions[0], versions[1])
        builder = TOOLS_ROOT / "build_full30_action_source7_reencode_release_v1.py"
        outputs: list[tuple[bytes, bytes]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for index, interpreter in enumerate(interpreters):
                archive = root / f"py{index}/source.tar"
                manifest = root / f"py{index}/source.manifest.json"
                subprocess.run(
                    [
                        str(interpreter),
                        str(builder),
                        "build",
                        "--method-root",
                        str(METHOD_ROOT.resolve(strict=True)),
                        "--archive",
                        str(archive),
                        "--manifest",
                        str(manifest),
                    ],
                    check=True,
                    capture_output=True,
                )
                outputs.append((archive.read_bytes(), manifest.read_bytes()))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0][0][329:345], b"\x00" * 16)
        member = release._tar_info("fixture", 1, 0o444)
        self.assertEqual(member.devmajor, 0)
        self.assertEqual(member.devminor, 0)
        self.assertEqual(member.pax_headers, {})

    def test_archive_tamper_is_rejected(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        raw = bytearray(release.build_archive(manifest, payloads))
        raw[512] ^= 1
        with self.assertRaises(release.Source7ReencodeReleaseError):
            release.verify_archive_bytes(bytes(raw), manifest)

    def test_ustar_semantic_zero_octal_devfields_are_raw_byte_rejected(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT.resolve(strict=True))
        canonical_raw = release.build_archive(manifest, payloads)
        raw = bytearray(canonical_raw)
        raw[329:337] = b"0000000\x00"
        raw[337:345] = b"0000000\x00"
        header = bytearray(raw[:512])
        header[148:156] = b" " * 8
        checksum = sum(header)
        raw[148:156] = f"{checksum:06o}\x00 ".encode("ascii")
        with tarfile.open(fileobj=io.BytesIO(bytes(raw)), mode="r:") as archive:
            member = archive.getmembers()[0]
            self.assertEqual(member.devmajor, 0)
            self.assertEqual(member.devminor, 0)
        with self.assertRaisesRegex(
            release.Source7ReencodeReleaseError,
            "not sixteen NUL bytes",
        ):
            release.verify_archive_bytes(bytes(raw), manifest)
        with self.assertRaisesRegex(
            release.Source7ReencodeReleaseError,
            "zero trailer differs",
        ):
            release.verify_archive_bytes(canonical_raw + b"\x00" * 512, manifest)

    def test_launcher_is_one_gpu_retained_136141_and_non_destructive(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("holder_job=136141", source)
        self.assertIn("holder_node=auh7-1b-gpu-299", source)
        self.assertIn("readonly run_generation=r4", source)
        self.assertIn('if [[ "${role}" == child ]]; then', source)
        self.assertNotIn('if [[ "${role}" == __child ]]; then', source)
        self.assertIn(
            "full30-action-source7-reencode-r1-a5f7e159-j136141-r1", source
        )
        self.assertIn(
            "full30-action-source7-reencode-r2-4f188c71-j136141-r1", source
        )
        self.assertIn(
            "full30-action-source7-reencode-r1-a5f7e159-20260815-r1", source
        )
        self.assertIn(
            "full30-action-source7-reencode-r2-4f188c71-20260815-r1", source
        )
        self.assertIn('manifest["release_generation"] == "r4"', source)
        self.assertIn("MIOPEN_USER_DB_PATH", source)
        self.assertIn("MIOPEN_CUSTOM_CACHE_DIR", source)
        self.assertIn("XDG_CACHE_HOME", source)
        self.assertIn("export TMPDIR=", source)
        self.assertLess(source.index("export TMPDIR="), source.index('"${runtime_cache_tool}" prepare'))
        self.assertIn("HOME changed during runtime cache preparation", source)
        self.assertIn("audit-cleanup", source)
        self.assertIn("record-retained-failure", source)
        self.assertIn("record-phase-failure", source)
        self.assertIn("runtime-cache-phase-failure-r4.json", source)
        self.assertIn("cleanup_before_numbered_step_exit", (
            TOOLS_ROOT / "full30_action_source7_reencode_runtime_cache_v1.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("os.O_EXCL", source)
        self.assertIn("login_parent_compute_tmp_accessed=false", source)
        self.assertIn("release_archive_sha256", source)
        self.assertIn("runtime_cache_prepare_receipt_sha256", source)
        self.assertIn("controller_completion_digest", source)
        self.assertIn("runtime_cache_cleanup_digest", source)
        self.assertIn('header[329:345] == b"\\x00" * 16', source)
        self.assertIn('archive_raw[offset:] == b"\\x00" * (len(archive_raw) - offset)', source)
        self.assertIn("--gpus-per-task=1", source)
        self.assertNotIn("holder_job=136140", source)
        self.assertNotIn("scancel", source)
        self.assertNotIn("scontrol release", source)
        self.assertNotIn("scontrol requeue", source)
        self.assertIn("parent_retained=true", source)
        for field, value in materializer._negative_access_closure().items():
            rendered = str(value).lower() if type(value) is bool else str(value)
            self.assertIn(f"{field}={rendered}", source)
        card = CARD.read_text(encoding="utf-8")
        self.assertIn("step: 136141.114", card)
        self.assertIn("step state/exit: FAILED / 2:0", card)
        self.assertIn("posterior tensors published: 0", card)
        self.assertIn("VAE encode calls: 0", card)
        self.assertIn("optimizer updates: 0", card)
        self.assertIn("reuse of failed run root: permanently forbidden", card)
        self.assertIn("reuse or launch of r1 release: permanently forbidden", card)
        self.assertIn("step: 136141.115", card)
        self.assertIn("VAE encode calls attempted: 1", card)
        self.assertIn("VAE encode calls completed: 0", card)
        self.assertIn(
            "child log SHA-256: 5530c4739da1114a6201bb37244c488471540b8d4c07ad6a192898ebc8d76004",
            card,
        )
        self.assertIn("reuse of failed r2 run root: permanently forbidden", card)
        self.assertIn("reuse or launch of r2 release: permanently forbidden", card)
        self.assertIn(
            "R3_STATIC_RELEASE_FROZEN__LOCAL_AUDIT_GO__INDEPENDENT_AUDIT_PENDING",
            card,
        )
        self.assertIn("static unit tests: 30/30 passed", card)
        self.assertIn("r3 upload executed: false", card)
        self.assertIn("r3 launch executed: false", card)
        self.assertIn("r3 GPU work executed: false", card)
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_launcher_parent_stubbed_srun_dispatches_child_once_and_rejects_unknown_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root = root / "fixture/methods/bernini_action_editing"
            for relative in release.FILES_AND_MODES:
                source = METHOD_ROOT / relative
                destination = method_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

            trace = root / "trace.log"
            bin_root = root / "stub-bin"
            bin_root.mkdir()
            python_shim = bin_root / "python-shim"
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            checkpoint_manifest = root / "checkpoint.sha256"
            checkpoint_manifest.write_bytes(b"checkpoint fixture\n")
            checkpoint_manifest_sha = sha(checkpoint_manifest.read_bytes())
            run_scope = root / "allowed-run-roots"
            run_scope.mkdir()
            run_root = run_scope / "source7-r4"
            cache_parent = root / "local-tmp"
            cache_parent.mkdir(mode=0o700)
            test_home = root / "home"
            test_home.mkdir(mode=0o700)
            failed_r1_root = run_scope / "failed-r1"
            failed_r2_root = run_scope / "failed-r2"
            failed_r1_release = root / "failed-release-r1"
            failed_r2_release = root / "failed-release-r2"
            failed_r3_run_prefix = run_scope / "failed-r3-"
            failed_r3_release_prefix = root / "failed-release-r3-"

            launcher = method_root / LAUNCHER.relative_to(METHOD_ROOT)
            launcher_source = launcher.read_text(encoding="utf-8")
            replacements = {
                "readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12": (
                    f"readonly python_bin={python_shim}"
                ),
                "readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4": (
                    f"readonly checkpoint={checkpoint}"
                ),
                "readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256": (
                    f"readonly checkpoint_manifest={checkpoint_manifest}"
                ),
                "readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831": (
                    f"readonly checkpoint_manifest_sha={checkpoint_manifest_sha}"
                ),
                '[[ "${run_root}" == /vast/users/guangyi.chen/* ]] || fail "run root scope differs"': (
                    f'[[ "${{run_root}}" == {run_scope}/* ]] || fail "run root scope differs"'
                ),
                "readonly failed_r1_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-source7-reencode-r1-a5f7e159-j136141-r1": (
                    f"readonly failed_r1_run_root={failed_r1_root}"
                ),
                "readonly failed_r2_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-source7-reencode-r2-4f188c71-j136141-r1": (
                    f"readonly failed_r2_run_root={failed_r2_root}"
                ),
                "readonly failed_r1_release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-source7-reencode-r1-a5f7e159-20260815-r1": (
                    f"readonly failed_r1_release_root={failed_r1_release}"
                ),
                "readonly failed_r2_release_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-source7-reencode-r2-4f188c71-20260815-r1": (
                    f"readonly failed_r2_release_root={failed_r2_release}"
                ),
                "readonly failed_r3_run_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/data_prep/full30-action-source7-reencode-r3-": (
                    f"readonly failed_r3_run_prefix={failed_r3_run_prefix}"
                ),
                "readonly failed_r3_release_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_generic_source_anchored_action_v1_20260814/releases/full30-action-source7-reencode-r3-": (
                    f"readonly failed_r3_release_prefix={failed_r3_release_prefix}"
                ),
                'readonly cache_root="/tmp/BOX-EXP-014-r4-${child_job_token}-${child_step_token}"': (
                    f'readonly cache_root="{cache_parent}/BOX-EXP-014-r4-${{child_job_token}}-${{child_step_token}}"'
                ),
            }
            for old, new in replacements.items():
                self.assertEqual(launcher_source.count(old), 1)
                launcher_source = launcher_source.replace(old, new)
            launcher.write_text(launcher_source, encoding="utf-8")

            real_python = Path(sys.executable).resolve(strict=True)
            python_shim.write_text(
                "#!/usr/bin/env bash\n"
                "set -Eeuo pipefail\n"
                "runtime_tool=false controller=false command=\n"
                "for argument in \"$@\"; do\n"
                "  [[ \"${argument}\" == */full30_action_source7_reencode_runtime_cache_v1.py ]] && runtime_tool=true\n"
                "  [[ \"${argument}\" == */full30_action_source7_reencode_controller_v1.py ]] && controller=true\n"
                "  case \"${argument}\" in prepare|cleanup|audit-cleanup|record-retained-failure|audit-retained-failure|record-phase-failure) command=${argument} ;; esac\n"
                "done\n"
                "if [[ \"${runtime_tool}\" == true ]]; then\n"
                "  case \"${command}\" in\n"
                "    prepare)\n"
                "      receipt=\n"
                "      while (( $# )); do\n"
                "        [[ \"$1\" == --receipt-output ]] && { receipt=$2; shift 2; continue; }\n"
                "        shift\n"
                "      done\n"
                "      cache_root=\"${TEST_CACHE_PARENT}/BOX-EXP-014-r4-${SLURM_JOB_ID}-${SLURM_STEP_ID}\"\n"
                "      [[ -n \"${receipt}\" && \"${MIOPEN_USER_DB_PATH}\" == \"${cache_root}/user-db\" && \"${MIOPEN_CUSTOM_CACHE_DIR}\" == \"${cache_root}/kernel-cache\" && \"${XDG_CACHE_HOME}\" == \"${cache_root}/xdg-cache\" && \"${TMPDIR}\" == \"${cache_root}/tmp\" ]]\n"
                "      [[ ! -e \"${cache_root}\" && ! -L \"${cache_root}\" ]]\n"
                "      /bin/mkdir -m 0700 \"${cache_root}\" \"${cache_root}/user-db\" \"${cache_root}/kernel-cache\" \"${cache_root}/xdg-cache\" \"${cache_root}/tmp\"\n"
                "      printf 'cache-create-only-ext4-fixture\\nexclusive-fsync-probe\\nsqlite-commit-reopen-probe\\n' >>\"${TRACE_LOG}\"\n"
                "      [[ \"${FORCE_PREPARE_FAIL_AFTER_ROOT:-0}\" != 1 ]] || { printf 'prepare-failed-after-root\\n' >>\"${TRACE_LOG}\"; exit 44; }\n"
                "      printf '{\"prepare_digest\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"run_generation\":\"r4\"}\\n' >\"${receipt}\"\n"
                "      chmod 0600 \"${receipt}\"\n"
                "      exit 0\n"
                "      ;;\n"
                "    cleanup)\n"
                "      cleanup_receipt=\n"
                "      while (( $# )); do\n"
                "        [[ \"$1\" == --cleanup-receipt-output ]] && { cleanup_receipt=$2; shift 2; continue; }\n"
                "        shift\n"
                "      done\n"
                "      [[ -n \"${cleanup_receipt}\" && -n \"${SLURM_STEP_ID:-}\" ]]\n"
                "      printf 'compute-child-cleanup-after-controller-exit-before-step-exit\\n' >>\"${TRACE_LOG}\"\n"
                "      cache_root=\"${TEST_CACHE_PARENT}/BOX-EXP-014-r4-136141-${TEST_STEP_ID:-908}\"\n"
                "      /bin/rmdir \"${cache_root}/user-db\" \"${cache_root}/kernel-cache\" \"${cache_root}/xdg-cache\" \"${cache_root}/tmp\" \"${cache_root}\"\n"
                "      [[ \"${FORCE_CLEANUP_FAIL_AFTER_DELETE:-0}\" != 1 ]] || { printf 'cleanup-failed-after-delete-before-receipt\\n' >>\"${TRACE_LOG}\"; exit 45; }\n"
                "      printf '{\"cache_root_removed\":true,\"cleanup_digest\":\"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\",\"controller_complete\":true,\"controller_completion_digest\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",\"prepare_digest\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"run_generation\":\"r4\"}\\n' >\"${cleanup_receipt}\"\n"
                "      chmod 0600 \"${cleanup_receipt}\"\n"
                "      exit 0\n"
                "      ;;\n"
                "    audit-cleanup)\n"
                "      if [[ -n \"${SLURM_STEP_ID:-}\" ]]; then\n"
                "        printf 'compute-child-cleanup-audit\\n' >>\"${TRACE_LOG}\"\n"
                "        [[ \"${FORCE_POST_CLEANUP_AUDIT_FAIL:-0}\" != 1 ]] || { printf 'post-cleanup-audit-failed\\n' >>\"${TRACE_LOG}\"; exit 46; }\n"
                "      else\n"
                "        printf 'login-parent-receipt-only-cleanup-audit\\n' >>\"${TRACE_LOG}\"\n"
                "      fi\n"
                "      printf '{}\\n'\n"
                "      exit 0\n"
                "      ;;\n"
                "    record-retained-failure)\n"
                "      retained_receipt=\n"
                "      while (( $# )); do\n"
                "        [[ \"$1\" == --retained-failure-receipt-output ]] && { retained_receipt=$2; shift 2; continue; }\n"
                "        shift\n"
                "      done\n"
                "      [[ -n \"${retained_receipt}\" && -n \"${SLURM_STEP_ID:-}\" ]]\n"
                "      printf 'compute-child-cache-retained-failure\\n' >>\"${TRACE_LOG}\"\n"
                "      cache_root=\"${TEST_CACHE_PARENT}/BOX-EXP-014-r4-${SLURM_JOB_ID}-${SLURM_STEP_ID}\"\n"
                "      [[ ! -e \"${cache_root}/tmp/miopen-lockfiles\" && ! -L \"${cache_root}/tmp/miopen-lockfiles\" ]]\n"
                "      printf 'pre-smoke-scoped-lock-root-absent-honestly-recorded\\n' >>\"${TRACE_LOG}\"\n"
                "      printf '{}\\n' >\"${retained_receipt}\"\n"
                "      chmod 0600 \"${retained_receipt}\"\n"
                "      exit 0\n"
                "      ;;\n"
                "    audit-retained-failure)\n"
                "      printf 'compute-child-retained-failure-audit\\n' >>\"${TRACE_LOG}\"\n"
                "      printf '{}\\n'\n"
                "      exit 0\n"
                "      ;;\n"
                "    record-phase-failure)\n"
                "      [[ \"${FORCE_PHASE_TOOL_FAIL:-0}\" != 1 ]] || exit 94\n"
                "      phase_receipt= phase=\n"
                "      while (( $# )); do\n"
                "        [[ \"$1\" == --phase-failure-receipt-output ]] && { phase_receipt=$2; shift 2; continue; }\n"
                "        [[ \"$1\" == --phase ]] && { phase=$2; shift 2; continue; }\n"
                "        shift\n"
                "      done\n"
                "      [[ -n \"${phase_receipt}\" && -n \"${phase}\" ]]\n"
                "      printf 'phase-failure-terminal:%s\\n' \"${phase}\" >>\"${TRACE_LOG}\"\n"
                "      printf '{\"final_marker_authorized\":false,\"phase\":\"%s\",\"success_claimed\":false}\\n' \"${phase}\" >\"${phase_receipt}\"\n"
                "      chmod 0600 \"${phase_receipt}\"\n"
                "      exit 0\n"
                "      ;;\n"
                "    *) exit 96 ;;\n"
                "  esac\n"
                "fi\n"
                "if [[ \"${controller}\" == true ]]; then\n"
                "  output_root= completion= prepare_receipt=\n"
                "  while (( $# )); do\n"
                "    case \"$1\" in\n"
                "      --materialization-output-root) output_root=$2; shift 2 ;;\n"
                "      --completion-output) completion=$2; shift 2 ;;\n"
                "      --runtime-cache-receipt) prepare_receipt=$2; shift 2 ;;\n"
                "      *) shift ;;\n"
                "    esac\n"
                "  done\n"
                "  expected_root=\"${TEST_CACHE_PARENT}/BOX-EXP-014-r4-${SLURM_JOB_ID}-${SLURM_STEP_ID}\"\n"
                "  [[ -n \"${output_root}\" && -n \"${completion}\" && -f \"${prepare_receipt}\" ]]\n"
                "  [[ \"${HOME}\" == \"${ORIGINAL_TEST_HOME}\" ]]\n"
                "  [[ \"${MIOPEN_USER_DB_PATH}\" == \"${expected_root}/user-db\" ]]\n"
                "  [[ \"${MIOPEN_CUSTOM_CACHE_DIR}\" == \"${expected_root}/kernel-cache\" ]]\n"
                "  [[ \"${XDG_CACHE_HOME}\" == \"${expected_root}/xdg-cache\" ]]\n"
                "  [[ \"${TMPDIR}\" == \"${expected_root}/tmp\" ]]\n"
                "  if [[ \"${FORCE_CONTROLLER_FAIL:-0}\" == 1 ]]; then\n"
                "    printf 'controller-failed-before-source\\n' >>\"${TRACE_LOG}\"\n"
                "    exit 42\n"
                "  fi\n"
                "  printf 'cache-env-verified-before-torch\\nscoped-tmpdir-lock-verified\\nwan-resample-candidate-i0\\nwan-resample-candidate-i1\\nwan-resample-candidate-i2\\nmiopen-custom-cache-ukdb-kern-db-nonempty\\nfirst-source-open-decode\\ncontroller\\ncontroller-exit\\n' >>\"${TRACE_LOG}\"\n"
                "  /bin/mkdir -p \"${output_root}\"\n"
                "  printf '{\"complete\":true,\"completion_digest\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",\"run_generation\":\"r4\"}\\n' >\"${completion}\"\n"
                "  chmod 0400 \"${completion}\"\n"
                "  if [[ \"${PRECREATE_FINAL_MARKER:-0}\" == 1 ]]; then printf 'hostile-preexisting-marker\\n' >\"${S7_RUN_ROOT}/BOX-EXP-014_R4_COMPLETE\"; fi\n"
                "  exit 0\n"
                "fi\n"
                f'exec "{real_python}" "$@"\n',
                encoding="utf-8",
            )

            stubs = {
                "sha256sum": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    f'exec "{real_python}" -c \'import hashlib,sys; p=sys.argv[1]; print(hashlib.sha256(open(p,"rb").read()).hexdigest(), p)\' "$1"\n'
                ),
                "readlink": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    "[[ \"$1\" == -f ]] && shift\n"
                    "[[ \"${1:-}\" == -- ]] && shift\n"
                    f'exec "{real_python}" -c \'import os,sys; print(os.path.realpath(sys.argv[1]))\' "$1"\n'
                ),
                "realpath": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    "[[ \"$1\" == -m ]] && shift\n"
                    "[[ \"${1:-}\" == -- ]] && shift\n"
                    f'exec "{real_python}" -c \'import os,sys; print(os.path.realpath(sys.argv[1]))\' "$1"\n'
                ),
                "hostname": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    "printf 'hostname\\n' >>\"${TRACE_LOG}\"\n"
                    "printf 'auh7-1b-gpu-299\\n'\n"
                ),
                "scontrol": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    "printf 'scontrol\\n' >>\"${TRACE_LOG}\"\n"
                    "printf 'JobId=136141 JobState=RUNNING UserId=guangyi.chen NodeList=auh7-1b-gpu-299\\n'\n"
                ),
                "squeue": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    "printf 'squeue\\n' >>\"${TRACE_LOG}\"\n"
                ),
                "mkdir": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    "printf 'parent-mkdir\\n' >>\"${TRACE_LOG}\"\n"
                    "exec /bin/mkdir \"$@\"\n"
                ),
                "srun": (
                    "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
                    "printf 'srun\\n' >>\"${TRACE_LOG}\"\n"
                    "while (( $# )) && [[ \"$1\" != env ]]; do shift; done\n"
                    "[[ \"${1:-}\" == env ]] || exit 97\n"
                    "shift\n"
                    "set +e\n"
                    "env SLURM_JOB_ID=136141 SLURM_STEP_ID=\"${TEST_STEP_ID:-908}\" \"$@\"\n"
                    "status=$?\n"
                    "set -e\n"
                    "printf 'numbered-child-ended\\n' >>\"${TRACE_LOG}\"\n"
                    "exit \"${status}\"\n"
                ),
            }
            for name, source in stubs.items():
                path = bin_root / name
                path.write_text(source, encoding="utf-8")
                path.chmod(0o755)
            python_shim.chmod(0o755)

            archive = root / "sealed/source.tar"
            manifest = root / "sealed/source.manifest.json"
            built = release.build(method_root.resolve(strict=True), archive, manifest)
            environment = {
                **{
                    key: value
                    for key, value in os.environ.items()
                    if key
                    not in {
                        "MIOPEN_USER_DB_PATH",
                        "MIOPEN_CUSTOM_CACHE_DIR",
                        "XDG_CACHE_HOME",
                        "TMPDIR",
                    }
                },
                "PATH": f"{bin_root}{os.pathsep}{os.environ['PATH']}",
                "TRACE_LOG": str(trace),
                "TEST_CACHE_PARENT": str(cache_parent),
                "ORIGINAL_TEST_HOME": str(test_home),
                "HOME": str(test_home),
                "S7_CONFIRM": "launch-approved-BOX-EXP-014-source-only-exact7-reencode-r4-136141",
                "S7_RUN_ROOT": str(run_root),
                "S7_METHOD_ARCHIVE": str(archive),
                "S7_METHOD_ARCHIVE_SHA256": built["archive_sha256"],
                "S7_METHOD_MANIFEST": str(manifest),
                "S7_METHOD_MANIFEST_SHA256": built["manifest_sha256"],
            }
            parent = subprocess.run(
                ["bash", str(launcher)], env=environment, capture_output=True, text=True
            )
            self.assertEqual(parent.returncode, 0, parent.stderr)
            self.assertIn("BOX-EXP-014_R4_COMPLETE", parent.stdout)
            events = trace.read_text(encoding="utf-8").splitlines()
            self.assertEqual(events.count("scontrol"), 1)
            self.assertEqual(events.count("squeue"), 1)
            self.assertEqual(events.count("parent-mkdir"), 1)
            self.assertEqual(events.count("srun"), 1)
            self.assertEqual(events.count("hostname"), 1)
            self.assertEqual(events.count("controller"), 1)
            self.assertEqual(events.count("cache-create-only-ext4-fixture"), 1)
            self.assertEqual(events.count("exclusive-fsync-probe"), 1)
            self.assertEqual(events.count("sqlite-commit-reopen-probe"), 1)
            self.assertEqual(events.count("cache-env-verified-before-torch"), 1)
            self.assertEqual(events.count("scoped-tmpdir-lock-verified"), 1)
            for index in range(3):
                self.assertEqual(events.count(f"wan-resample-candidate-i{index}"), 1)
            self.assertEqual(events.count("miopen-custom-cache-ukdb-kern-db-nonempty"), 1)
            self.assertEqual(events.count("first-source-open-decode"), 1)
            self.assertEqual(
                events.count("compute-child-cleanup-after-controller-exit-before-step-exit"), 1
            )
            self.assertEqual(events.count("compute-child-cleanup-audit"), 1)
            self.assertEqual(events.count("login-parent-receipt-only-cleanup-audit"), 1)
            self.assertLess(events.index("sqlite-commit-reopen-probe"), events.index("cache-env-verified-before-torch"))
            self.assertLess(events.index("cache-env-verified-before-torch"), events.index("wan-resample-candidate-i0"))
            self.assertLess(events.index("wan-resample-candidate-i2"), events.index("miopen-custom-cache-ukdb-kern-db-nonempty"))
            self.assertLess(events.index("miopen-custom-cache-ukdb-kern-db-nonempty"), events.index("first-source-open-decode"))
            self.assertLess(events.index("controller-exit"), events.index("compute-child-cleanup-after-controller-exit-before-step-exit"))
            self.assertLess(events.index("compute-child-cleanup-after-controller-exit-before-step-exit"), events.index("numbered-child-ended"))
            self.assertLess(events.index("numbered-child-ended"), events.index("login-parent-receipt-only-cleanup-audit"))
            marker = run_root / "BOX-EXP-014_R4_COMPLETE"
            self.assertTrue(marker.is_file())
            marker_text = marker.read_text(encoding="ascii")
            self.assertIn(f"release_archive_sha256={built['archive_sha256']}", marker_text)
            self.assertIn(f"release_manifest_sha256={built['manifest_sha256']}", marker_text)
            self.assertIn("runtime_cache_prepare_receipt_sha256=", marker_text)
            self.assertIn("controller_completion_digest=" + "c" * 64, marker_text)
            self.assertIn("runtime_cache_cleanup_digest=" + "d" * 64, marker_text)
            self.assertFalse((cache_parent / "BOX-EXP-014-r4-136141-908").exists())

            octal_archive_raw = bytearray(archive.read_bytes())
            octal_archive_raw[329:337] = b"0000000\x00"
            octal_archive_raw[337:345] = b"0000000\x00"
            octal_header = bytearray(octal_archive_raw[:512])
            octal_header[148:156] = b" " * 8
            octal_archive_raw[148:156] = (
                f"{sum(octal_header):06o}\x00 ".encode("ascii")
            )
            octal_archive = root / "sealed/source-octal-devfields.tar"
            octal_archive.write_bytes(bytes(octal_archive_raw))
            octal_archive.chmod(0o444)
            octal_run_root = run_scope / "source7-r4-octal-devfield-hostile"
            octal_hostile = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(octal_run_root),
                    "S7_METHOD_ARCHIVE": str(octal_archive),
                    "S7_METHOD_ARCHIVE_SHA256": sha(bytes(octal_archive_raw)),
                },
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(octal_hostile.returncode, 0)
            self.assertFalse(
                (octal_run_root / "BOX-EXP-014_R4_COMPLETE").exists()
            )
            self.assertEqual(
                trace.read_text(encoding="utf-8").splitlines().count("srun"), 1
            )

            before = trace.read_text(encoding="utf-8").splitlines()
            hostile_root = run_scope / "hostile-unknown-role"
            hostile_environment = {**environment, "S7_RUN_ROOT": str(hostile_root)}
            hostile = subprocess.run(
                ["bash", str(launcher), "unknown-role"],
                env=hostile_environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(hostile.returncode, 2)
            self.assertIn("launcher arguments differ", hostile.stderr)
            self.assertEqual(trace.read_text(encoding="utf-8").splitlines(), before)

            failure_run_root = run_scope / "source7-r4-forced-controller-failure"
            failure = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(failure_run_root),
                    "TEST_STEP_ID": "909",
                    "FORCE_CONTROLLER_FAIL": "1",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(failure.returncode, 2)
            self.assertIn("source7 re-encode child failed status=42", failure.stderr)
            failure_events = trace.read_text(encoding="utf-8").splitlines()
            self.assertEqual(failure_events.count("controller-failed-before-source"), 1)
            self.assertEqual(
                failure_events.count("compute-child-cleanup-after-controller-exit-before-step-exit"),
                before.count("compute-child-cleanup-after-controller-exit-before-step-exit"),
            )
            self.assertEqual(failure_events.count("compute-child-cache-retained-failure"), 1)
            self.assertEqual(
                failure_events.count(
                    "pre-smoke-scoped-lock-root-absent-honestly-recorded"
                ),
                1,
            )
            self.assertEqual(failure_events.count("compute-child-retained-failure-audit"), 1)
            self.assertEqual(
                failure_events.count("first-source-open-decode"),
                before.count("first-source-open-decode"),
            )
            retained_cache = cache_parent / "BOX-EXP-014-r4-136141-909"
            self.assertTrue(retained_cache.is_dir())
            self.assertTrue(
                (failure_run_root / "logs/runtime-cache-prepare-r4.json").is_file()
            )
            self.assertTrue(
                (failure_run_root / "logs/runtime-cache-retained-failure-r4.json").is_file()
            )
            self.assertFalse((failure_run_root / "BOX-EXP-014_R4_COMPLETE").exists())
            self.assertFalse(hostile_root.exists())

            marker_run_root = run_scope / "source7-r4-preexisting-marker"
            marker_hostile = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(marker_run_root),
                    "TEST_STEP_ID": "910",
                    "PRECREATE_FINAL_MARKER": "1",
                },
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(marker_hostile.returncode, 0)
            self.assertEqual(
                (marker_run_root / "BOX-EXP-014_R4_COMPLETE").read_text(encoding="ascii"),
                "hostile-preexisting-marker\n",
            )
            self.assertFalse((cache_parent / "BOX-EXP-014-r4-136141-910").exists())
            prepare_failure_root = run_scope / "source7-r4-prepare-phase-failure"
            prepare_failure = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(prepare_failure_root),
                    "TEST_STEP_ID": "911",
                    "FORCE_PREPARE_FAIL_AFTER_ROOT": "1",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(prepare_failure.returncode, 2)
            prepare_terminal = (
                prepare_failure_root / "logs/runtime-cache-phase-failure-r4.json"
            )
            self.assertTrue(prepare_terminal.is_file())
            self.assertIn(
                '"phase":"prepare-or-precontroller"',
                prepare_terminal.read_text(encoding="ascii"),
            )
            self.assertTrue(
                (cache_parent / "BOX-EXP-014-r4-136141-911").is_dir()
            )
            self.assertFalse(
                (prepare_failure_root / "BOX-EXP-014_R4_COMPLETE").exists()
            )

            cleanup_failure_root = run_scope / "source7-r4-cleanup-phase-failure"
            cleanup_failure = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(cleanup_failure_root),
                    "TEST_STEP_ID": "912",
                    "FORCE_CLEANUP_FAIL_AFTER_DELETE": "1",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(cleanup_failure.returncode, 2)
            cleanup_terminal = (
                cleanup_failure_root / "logs/runtime-cache-phase-failure-r4.json"
            )
            self.assertTrue(cleanup_terminal.is_file())
            self.assertIn(
                '"phase":"cleanup-or-cleanup-receipt-publication"',
                cleanup_terminal.read_text(encoding="ascii"),
            )
            self.assertFalse(
                (cache_parent / "BOX-EXP-014-r4-136141-912").exists()
            )
            self.assertFalse(
                (cleanup_failure_root / "BOX-EXP-014_R4_COMPLETE").exists()
            )

            audit_failure_root = run_scope / "source7-r4-post-cleanup-audit-failure"
            audit_failure = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(audit_failure_root),
                    "TEST_STEP_ID": "913",
                    "FORCE_POST_CLEANUP_AUDIT_FAIL": "1",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(audit_failure.returncode, 2)
            audit_terminal = (
                audit_failure_root / "logs/runtime-cache-phase-failure-r4.json"
            )
            self.assertTrue(audit_terminal.is_file())
            self.assertIn(
                '"phase":"post-cleanup-audit"',
                audit_terminal.read_text(encoding="ascii"),
            )
            self.assertFalse(
                (cache_parent / "BOX-EXP-014-r4-136141-913").exists()
            )
            fallback_root = run_scope / "source7-r4-phase-tool-fallback"
            fallback_failure = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(fallback_root),
                    "TEST_STEP_ID": "914",
                    "FORCE_PREPARE_FAIL_AFTER_ROOT": "1",
                    "FORCE_PHASE_TOOL_FAIL": "1",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(fallback_failure.returncode, 2)
            fallback_terminal = (
                fallback_root / "logs/runtime-cache-phase-failure-r4.fallback"
            )
            self.assertTrue(fallback_terminal.is_file())
            fallback_text = fallback_terminal.read_text(encoding="ascii")
            self.assertIn("phase=prepare-or-precontroller", fallback_text)
            self.assertIn("success_claimed=false", fallback_text)
            self.assertIn("final_marker_authorized=false", fallback_text)
            after_failure = trace.read_text(encoding="utf-8").splitlines()

            for generation, failed_root in (("r1", failed_r1_root), ("r2", failed_r2_root)):
                for candidate in (failed_root, failed_root / "descendant"):
                    with self.subTest(denied_run_generation=generation, candidate=candidate):
                        denied = subprocess.run(
                            ["bash", str(launcher)],
                            env={**environment, "S7_RUN_ROOT": str(candidate)},
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(denied.returncode, 2)
                        self.assertIn(
                            f"failed {generation} run root or descendant is permanently forbidden",
                            denied.stderr,
                        )
                        self.assertFalse(candidate.exists())
            r3_denied_root = Path(f"{failed_r3_run_prefix}candidate")
            denied_r3 = subprocess.run(
                ["bash", str(launcher)],
                env={**environment, "S7_RUN_ROOT": str(r3_denied_root)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(denied_r3.returncode, 2)
            self.assertIn("failed r3 run generation is permanently forbidden", denied_r3.stderr)
            dot_alias = run_scope / "unused" / ".." / failed_r1_root.name
            denied_dot = subprocess.run(
                ["bash", str(launcher)],
                env={**environment, "S7_RUN_ROOT": str(dot_alias)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(denied_dot.returncode, 2)
            self.assertIn("must already be canonical", denied_dot.stderr)
            failed_r1_root.mkdir()
            symlink_alias = run_scope / "failed-r1-symlink-alias"
            symlink_alias.symlink_to(failed_r1_root, target_is_directory=True)
            denied_symlink = subprocess.run(
                ["bash", str(launcher)],
                env={**environment, "S7_RUN_ROOT": str(symlink_alias)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(denied_symlink.returncode, 2)
            self.assertIn("must already be canonical", denied_symlink.stderr)
            symlink_alias.unlink()
            failed_r1_root.rmdir()
            for generation, failed_release in (
                ("r1", failed_r1_release),
                ("r2", failed_r2_release),
            ):
                with self.subTest(denied_release_generation=generation):
                    denied_root = run_scope / f"denied-release-{generation}"
                    denied = subprocess.run(
                        ["bash", str(launcher)],
                        env={
                            **environment,
                            "S7_RUN_ROOT": str(denied_root),
                            "S7_METHOD_ARCHIVE": str(failed_release / "source.tar"),
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(denied.returncode, 2)
                    self.assertIn(f"failed {generation} release or descendant is permanently forbidden", denied.stderr)
                    self.assertFalse(denied_root.exists())
            r3_denied_release_root = run_scope / "denied-release-r3"
            denied_r3_release = subprocess.run(
                ["bash", str(launcher)],
                env={
                    **environment,
                    "S7_RUN_ROOT": str(r3_denied_release_root),
                    "S7_METHOD_ARCHIVE": str(
                        Path(f"{failed_r3_release_prefix}candidate") / "source.tar"
                    ),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(denied_r3_release.returncode, 2)
            self.assertIn(
                "failed r3 release generation is permanently forbidden",
                denied_r3_release.stderr,
            )
            self.assertEqual(trace.read_text(encoding="utf-8").splitlines(), after_failure)

    def test_release_member_closure_has_only_runtime_dependencies(self) -> None:
        self.assertEqual(
            set(release.FILES_AND_MODES),
            {
                "full30_action_source7_reencode_plan_v1.py",
                "full30_action_source7_reencode_controller_v1.py",
                "source_self_role_repaint.py",
                "tools/materialize_full30_action_source7_reencode_v1.py",
                "tools/materialize_source_self_role_repaint.py",
                "tools/materialize_ramp_motion_analogy_vae.py",
                "tools/materialize_vae.py",
                "tools/build_renderer_dataset.py",
                "tools/full30_action_source7_reencode_runtime_cache_v1.py",
                "tools/build_full30_action_source7_reencode_release_v1.py",
                "scripts/auh_full30_action_source7_reencode_136141_v1.sh",
            },
        )


if __name__ == "__main__":
    unittest.main()
