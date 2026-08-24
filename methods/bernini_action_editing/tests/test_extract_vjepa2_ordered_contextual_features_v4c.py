from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "extract_vjepa2_ordered_contextual_features_v4c.py"

if str(METHOD_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT.parent.parent))

from methods.bernini_action_editing import (  # noqa: E402
    extract_vjepa2_ordered_contextual_features_v4c as runtime,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _VersionStringSubclass(str):
    """Pickle-visible stand-in for torch.torch_version.TorchVersion."""


class VJepa2ExtractorV4CStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _definition(self, name: str, kind: type) -> ast.AST:
        matches = [
            node for node in ast.walk(self.tree)
            if isinstance(node, kind) and getattr(node, "name", None) == name
        ]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def test_authority_constants_are_exact(self) -> None:
        self.assertEqual(runtime.MANIFEST_SCHEMA, "semantic-moments-action-reward-audit-v1")
        self.assertEqual(runtime.FEATURE_SCHEMA, "vjepa2-ordered-contextual-anchor-shard-v4c")
        self.assertEqual(runtime.RECEIPT_SCHEMA, "vjepa2-ordered-contextual-exact644-receipt-v4c")
        self.assertEqual(
            runtime.FEATURE_MANIFEST_SHA256,
            "963cc02f9875048120fbea042ecbeac9b59e5e40d23121c52a9d2488556ca4e5",
        )
        self.assertEqual(
            runtime.FEATURE_MANIFEST_DIGEST,
            "51480da9c13f1dd060bead9309badf12c59a05810a83c1f6eab643f62418687a",
        )
        self.assertEqual(runtime.MODEL_REPO, "facebook/vjepa2-vitl-fpc64-256")
        self.assertEqual(runtime.MODEL_REVISION, "b3c1679b7c34d3255ef3547f27c7b226aefab26f")
        self.assertEqual(set(runtime.MODEL_FILES), {
            "config.json", "model.safetensors", "video_preprocessor_config.json",
        })
        self.assertEqual(runtime.VIEW_NAMES, (
            "original", "monotone_warp", "reverse", "block_shuffle", "phase_swap",
        ))

    def test_processor_is_one_keyword_call_on_pil_frames(self) -> None:
        function = ast.get_source_segment(
            self.source,
            self._definition("process_canonical_base64", ast.FunctionDef),
        ) or ""
        self.assertIn("pil_frames = [Image.fromarray", function)
        self.assertIn("self.processor(videos=pil_frames, return_tensors=\"pt\")", function)
        self.assertIn("self.processor_call_count += 1", function)
        self.assertNotIn("self.processor(sampled_rgb", function)

    def test_pool_promotes_hidden_to_float32_before_mean(self) -> None:
        function = ast.get_source_segment(
            self.source,
            self._definition("pool_time_major_hidden", ast.FunctionDef),
        ) or ""
        promote = function.index("to(dtype=torch.float32)")
        reduce = function.index(".mean(dim=2)")
        self.assertLess(promote, reduce)
        self.assertIn("hidden.dtype != torch.float16", function)
        self.assertNotIn("grid.mean(dim=2).detach().to(dtype=torch.float32", function)

    def test_views_are_pixel_space_and_backbone_calls_are_separate(self) -> None:
        views = ast.get_source_segment(
            self.source, self._definition("pixel_views", ast.FunctionDef)
        ) or ""
        extract = ast.get_source_segment(
            self.source, self._definition("extract_one", ast.FunctionDef)
        ) or ""
        self.assertIn("canonical_pixels.flip(1)", views)
        self.assertIn("block64_indices(iid)", views)
        self.assertIn("phase64_indices()", views)
        self.assertIn("for forward_ordinal, name in enumerate(VIEW_NAMES", extract)
        self.assertIn("sequence = frozen.forward_view(pixels)[0]", extract)
        self.assertIn('"model_forward_batching_across_views": False', extract)
        self.assertIn('"post_backbone_token_permutation_used": False', extract)

    def test_shard_writer_is_create_only_sealed_and_reloaded(self) -> None:
        function = ast.get_source_segment(
            self.source, self._definition("_save_torch_create_only", ast.FunctionDef)
        ) or ""
        self.assertIn('path.open("xb")', function)
        self.assertIn("os.fsync", function)
        self.assertIn("os.chmod(path, 0o444)", function)
        self.assertIn("st_nlink != 1", function)
        self.assertIn("torch.load(handle, map_location=\"cpu\", weights_only=True)", function)
        self.assertIn("_shard_semantic_sha256(reloaded)", function)

    def test_cli_surface_is_exactly_extract_and_aggregate(self) -> None:
        self.assertEqual(self.source.count("commands.add_parser("), 2)
        self.assertIn('commands.add_parser("extract-shard")', self.source)
        self.assertIn('commands.add_parser("aggregate-shards")', self.source)
        aggregate = ast.get_source_segment(
            self.source, self._definition("aggregate_shards", ast.FunctionDef)
        ) or ""
        self.assertIn("len(args.shard) != 6", aggregate)
        self.assertIn("len(args.expected_shard_sha256) != 6", aggregate)
        self.assertIn("set(by_index) != set(range(6))", aggregate)
        self.assertIn("set(records_by_ordinal) != set(range(644))", aggregate)
        self.assertIn("_write_json_create_only(Path(args.output), receipt)", aggregate)

    def test_manifest_is_parsed_from_one_open_fd_with_identity_and_reread_gates(self) -> None:
        function = ast.get_source_segment(
            self.source, self._definition("load_anchor_manifest", ast.FunctionDef)
        ) or ""
        self.assertEqual(function.count('with requested.open("rb") as handle:'), 1)
        self.assertIn("opened_before = os.fstat(handle.fileno())", function)
        self.assertIn("raw = handle.read()", function)
        self.assertIn("raw_after = handle.read()", function)
        self.assertIn("opened_after = os.fstat(handle.fileno())", function)
        self.assertIn('json.loads(raw.decode("utf-8"))', function)
        self.assertNotIn("read_text(", function)


class VJepa2ExtractorV4CTransformTests(unittest.TestCase):
    def test_processor_adapter_uses_one_named_videos_call(self) -> None:
        calls = []

        class FakeProcessor:
            def __call__(self, *args, **kwargs):
                calls.append((args, kwargs))
                return {
                    "pixel_values_videos": torch.zeros(
                        (1, 64, 3, 256, 256), dtype=torch.float32
                    )
                }

        frozen = object.__new__(runtime.FrozenVJepa2)
        frozen.processor = FakeProcessor()
        frozen.processor_call_count = 0
        sampled = torch.zeros((64, 2, 3, 3), dtype=torch.uint8)
        pixels = frozen.process_canonical_base64(sampled)
        self.assertEqual(frozen.processor_call_count, 1)
        self.assertEqual(len(calls), 1)
        positional, keyword = calls[0]
        self.assertEqual(positional, ())
        self.assertEqual(set(keyword), {"videos", "return_tensors"})
        self.assertEqual(keyword["return_tensors"], "pt")
        self.assertEqual(len(keyword["videos"]), 64)
        self.assertTrue(all(frame.__class__.__module__.startswith("PIL") for frame in keyword["videos"]))
        self.assertEqual(tuple(pixels.shape), (1, 64, 3, 256, 256))
        self.assertEqual(pixels.dtype, torch.float32)

    def test_sampling_and_transform_hashes(self) -> None:
        base = runtime.base64_indices()
        self.assertEqual(base.dtype, torch.long)
        self.assertEqual(tuple(base.shape), (64,))
        self.assertEqual(base.tolist(), [(80 * index) // 63 for index in range(64)])
        self.assertEqual(runtime.tensor_sha256(base), runtime.BASE64_INDICES_SHA256)

        warp = runtime.warp64_coordinates()
        self.assertEqual(warp.dtype, torch.float32)
        self.assertTrue(bool((warp[1:] > warp[:-1]).all()))
        self.assertEqual(float(warp[0]), 0.0)
        self.assertEqual(float(warp[-1]), 63.0)
        self.assertEqual(runtime.tensor_sha256(warp), runtime.WARP64_COORDINATES_SHA256)

        phase = runtime.phase64_indices()
        self.assertEqual(tuple(sorted(phase.tolist())), tuple(range(64)))
        self.assertFalse(torch.equal(phase, torch.arange(64)))
        for iid in ("iid-a", "iid-b", "iid-c"):
            block = runtime.block64_indices(iid)
            self.assertEqual(tuple(sorted(block.tolist())), tuple(range(64)))
            self.assertFalse(torch.equal(block, torch.arange(64)))
            self.assertFalse(torch.equal(block, phase))

    def test_pixel_views_apply_the_exact_temporal_maps(self) -> None:
        timeline = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1, 1, 1)
        pixels = timeline.expand(1, 64, 3, 256, 256)
        views = runtime.pixel_views(pixels, "view-contract-iid")
        self.assertEqual(tuple(views), runtime.VIEW_NAMES)
        for value in views.values():
            self.assertEqual(tuple(value.shape), (1, 64, 3, 256, 256))
            self.assertEqual(value.dtype, torch.float32)
        observed = lambda value: value[0, :, 0, 0, 0]
        self.assertTrue(torch.equal(observed(views["original"]), torch.arange(64)))
        self.assertTrue(torch.equal(observed(views["monotone_warp"]), runtime.warp64_coordinates()))
        self.assertTrue(torch.equal(observed(views["reverse"]), torch.arange(63, -1, -1)))
        self.assertTrue(torch.equal(
            observed(views["block_shuffle"]), runtime.block64_indices("view-contract-iid").float()
        ))
        self.assertTrue(torch.equal(observed(views["phase_swap"]), runtime.phase64_indices().float()))

    def test_pixel_views_reject_hostile_geometry_dtype_and_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            runtime.pixel_views(torch.zeros((64, 3, 256, 256)), "iid")
        with self.assertRaises(ValueError):
            runtime.pixel_views(torch.zeros((1, 64, 3, 256, 256), dtype=torch.float16), "iid")
        hostile = torch.zeros((1, 64, 3, 256, 256), dtype=torch.float32)
        hostile[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            runtime.pixel_views(hostile, "iid")

    def test_pool_is_float32_before_spatial_reduction(self) -> None:
        token_values = ((torch.arange(8192) % 257).float() / 100.0).half()
        hidden = token_values.reshape(1, 8192, 1).expand(1, 8192, 1024)
        pooled = runtime.pool_time_major_hidden(hidden)
        expected = hidden.float().reshape(1, 32, 256, 1024).mean(dim=2)
        half_reduction = hidden.reshape(1, 32, 256, 1024).mean(dim=2).float()
        self.assertEqual(pooled.dtype, torch.float32)
        self.assertEqual(pooled.device.type, "cpu")
        self.assertTrue(pooled.is_contiguous())
        self.assertTrue(torch.equal(pooled, expected))
        self.assertTrue(bool((pooled != half_reduction).any()))
        with self.assertRaises(ValueError):
            runtime.pool_time_major_hidden(hidden.float())


class VJepa2ExtractorV4CManifestTests(unittest.TestCase):
    def _manifest(self) -> dict:
        items = []
        for index in range(644):
            iid = "iid-%04d" % index
            items.append({"metadata": {"role": "source"}})
            items.append({
                "item_id": "exact644:%s:action_anchor" % iid,
                "group": "exact644_action_anchor",
                "path": "/sealed/anchors/%s.mp4" % iid,
                "sha256": _digest("media:%s" % iid),
                "metadata": {
                    "role": "action_anchor",
                    "iid": iid,
                    "family": "family-%02d" % (index % 28),
                    "group_id": _digest("group:%s" % iid),
                    "instruction_sha256": _digest("instruction:%s" % iid),
                    "strict_selection_gates_all_true": index < 359,
                    "source_manifest_digest": runtime.SOURCE_MANIFEST_DIGEST,
                    "paired_ground_truth_claimed": False,
                },
            })
        return {
            "schema_version": runtime.MANIFEST_SCHEMA,
            "formal_training_authorized": False,
            "paired_ground_truth_claimed": False,
            "manifest_digest": runtime.FEATURE_MANIFEST_DIGEST,
            "source_release": {
                "sha256": runtime.SOURCE_MANIFEST_FILE_SHA256,
                "manifest_digest": runtime.SOURCE_MANIFEST_DIGEST,
                "row_count": 644,
            },
            "counts": {
                "total": 1288,
                "unique_base_clips": 644,
                "by_group": {"exact644_action_anchor": 644, "exact644_source": 644},
            },
            "items": items,
        }

    def _load(self, value: dict):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "manifest.json"
            raw = json.dumps(value, sort_keys=True).encode("utf-8")
            path.write_bytes(raw)
            os.chmod(path, 0o444)
            digest = hashlib.sha256(raw).hexdigest()
            with mock.patch.object(runtime, "FEATURE_MANIFEST_SHA256", digest):
                return runtime.load_anchor_manifest(path, digest)

    def test_valid_exact644_manifest_is_joined_in_manifest_order(self) -> None:
        anchors, manifest = self._load(self._manifest())
        self.assertEqual(len(anchors), 644)
        self.assertEqual([row.ordinal for row in anchors], list(range(644)))
        self.assertEqual(len({row.iid for row in anchors}), 644)
        self.assertEqual(len({row.family for row in anchors}), 28)
        self.assertEqual(sum(row.strict for row in anchors), 359)
        self.assertEqual(manifest["counts"]["total"], 1288)

    def test_manifest_rejects_wrong_authority_before_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "manifest.json"
            path.write_text("{}", encoding="utf-8")
            os.chmod(path, 0o444)
            with self.assertRaises(ValueError):
                runtime.load_anchor_manifest(path, "0" * 64)

    def test_manifest_rejects_symlink_same_fd_drift_and_path_replacement(self) -> None:
        value = self._manifest()
        raw = json.dumps(value, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "manifest.json"
            path.write_bytes(raw)
            os.chmod(path, 0o444)

            link = root / "manifest-link.json"
            link.symlink_to(path)
            with mock.patch.object(runtime, "FEATURE_MANIFEST_SHA256", digest):
                with self.assertRaises(ValueError):
                    runtime.load_anchor_manifest(link, digest)

            info = path.stat()
            drifted = types.SimpleNamespace(
                st_dev=info.st_dev, st_ino=info.st_ino, st_size=info.st_size,
                st_nlink=info.st_nlink, st_mode=info.st_mode,
                st_mtime_ns=info.st_mtime_ns + 1, st_ctime_ns=info.st_ctime_ns,
            )
            with mock.patch.object(runtime, "FEATURE_MANIFEST_SHA256", digest), mock.patch.object(
                runtime.os, "fstat", side_effect=(info, drifted)
            ):
                with self.assertRaises(RuntimeError):
                    runtime.load_anchor_manifest(path, digest)

            replacement = root / "replacement.json"
            replacement.write_bytes(raw)
            os.chmod(replacement, 0o444)
            original_open = Path.open
            replaced = []

            def replacing_open(candidate, *args, **kwargs):
                handle = original_open(candidate, *args, **kwargs)
                if candidate == path and not replaced:
                    os.replace(str(replacement), str(path))
                    replaced.append(True)
                return handle

            with mock.patch.object(runtime, "FEATURE_MANIFEST_SHA256", digest), mock.patch.object(
                Path, "open", new=replacing_open
            ):
                with self.assertRaises(RuntimeError):
                    runtime.load_anchor_manifest(path, digest)

    def test_manifest_rejects_duplicate_relative_and_malformed_rows(self) -> None:
        cases = []
        duplicate = self._manifest()
        duplicate["items"][3]["metadata"]["iid"] = duplicate["items"][1]["metadata"]["iid"]
        duplicate["items"][3]["item_id"] = duplicate["items"][1]["item_id"]
        cases.append(duplicate)
        relative = self._manifest()
        relative["items"][1]["path"] = "relative.mp4"
        cases.append(relative)
        malformed = self._manifest()
        malformed["items"][1]["metadata"]["group_id"] = "not-a-sha"
        cases.append(malformed)
        wrong_role = self._manifest()
        wrong_role["items"][1]["metadata"]["paired_ground_truth_claimed"] = True
        cases.append(wrong_role)
        for value in cases:
            with self.subTest(case=len(cases)):
                with self.assertRaises((ValueError, TypeError)):
                    self._load(value)


class VJepa2ExtractorV4CModelClosureTests(unittest.TestCase):
    def _model_root(self, parent: Path):
        root = parent / "model"
        root.mkdir()
        payloads = {
            "config.json": b"config-v4c",
            "model.safetensors": b"weights-v4c",
            "video_preprocessor_config.json": b"processor-v4c",
        }
        expected = {}
        for name, payload in payloads.items():
            path = root / name
            path.write_bytes(payload)
            os.chmod(path, 0o444)
            expected[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        os.chmod(root, 0o555)
        return root, expected

    def _restore(self, root: Path) -> None:
        if root.exists():
            os.chmod(root, 0o755)
            for path in root.iterdir():
                if not path.is_symlink():
                    os.chmod(path, 0o644)

    def test_exact3_plain_readonly_model_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, expected = self._model_root(Path(temporary))
            try:
                with mock.patch.object(runtime, "MODEL_FILES", expected):
                    closure = runtime.sealed_model_closure(root)
                self.assertEqual(closure["exact_top_level_regular_file_count"], 3)
                self.assertEqual(closure["root_mode"], 0o555)
                self.assertEqual(len(closure["files"]), 3)
                self.assertEqual(len({(row["device"], row["inode"]) for row in closure["files"]}), 3)
                self.assertRegex(closure["closure_sha256"], r"^[0-9a-f]{64}$")
            finally:
                self._restore(root)

    def test_model_closure_rejects_extra_symlink_mode_and_content(self) -> None:
        mutators = (
            lambda root: (os.chmod(root, 0o755), (root / "extra").write_bytes(b"x"), os.chmod(root, 0o555)),
            lambda root: os.chmod(root / "config.json", 0o644),
            lambda root: (os.chmod(root / "config.json", 0o644), (root / "config.json").write_bytes(b"changed"), os.chmod(root / "config.json", 0o444)),
            lambda root: os.chmod(root, 0o755),
        )
        for ordinal, mutate in enumerate(mutators):
            with self.subTest(ordinal=ordinal), tempfile.TemporaryDirectory() as temporary:
                root, expected = self._model_root(Path(temporary))
                try:
                    mutate(root)
                    with mock.patch.object(runtime, "MODEL_FILES", expected):
                        with self.assertRaises(ValueError):
                            runtime.sealed_model_closure(root)
                finally:
                    self._restore(root)


class VJepa2ExtractorV4CSealAndShardTests(unittest.TestCase):
    def _minimal_payload(self) -> dict:
        return {
            "schema_version": runtime.FEATURE_SCHEMA,
            "record_count": 1,
            "records": [{
                "iid": "seal-iid",
                "view_sequences": {
                    name: torch.arange(12, dtype=torch.float32).reshape(3, 4)
                    for name in runtime.VIEW_NAMES
                },
            }],
        }

    def test_create_only_torch_save_is_sealed_and_semantically_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "shard.pt"
            payload = self._minimal_payload()
            binding = runtime._save_torch_create_only(output, payload)
            info = output.stat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(info.st_nlink, 1)
            self.assertEqual(binding["sha256"], runtime.file_sha256(output))
            self.assertEqual(binding["semantic_sha256"], runtime._shard_semantic_sha256(payload))
            self.assertTrue(binding["fresh_torch_load_readback_exact"])
            reloaded = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(
                runtime._shard_semantic_sha256(reloaded), binding["semantic_sha256"]
            )
            with self.assertRaises(ValueError):
                runtime._save_torch_create_only(output, payload)

    def test_shard_writer_rejects_relative_and_malformed_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            relative = Path("relative-v4c-shard.pt")
            with self.assertRaises(ValueError):
                runtime._save_torch_create_only(relative, self._minimal_payload())
            malformed = self._minimal_payload()
            malformed["records"][0]["view_sequences"].pop(runtime.VIEW_NAMES[-1])
            with self.assertRaises((KeyError, ValueError)):
                runtime._shard_semantic_sha256(malformed)

    def test_extract_shard_exact6_partition_is_exhaustive_and_disjoint(self) -> None:
        anchors = [
            runtime.AnchorItem(
                ordinal=index,
                iid="iid-%04d" % index,
                family="family-%02d" % (index % 28),
                group_id=_digest("group:%d" % index),
                instruction_sha256=_digest("instruction:%d" % index),
                strict=index < 359,
                path=Path("/sealed/%04d.mp4" % index),
                media_sha256=_digest("media:%d" % index),
            )
            for index in range(644)
        ]
        saved = []

        class FakeFrozen:
            def __init__(self, model_root: Path, device: str):
                self.processor_call_count = 0
                self.forward_call_count = 0
                self.device_uuid = "fake-mi210-uuid"

            def final_closure(self):
                return {"fake_frozen_closure": True}

        def fake_extract_one(item, frozen):
            frozen.processor_call_count += 1
            frozen.forward_call_count += 5
            return {"ordinal": item.ordinal, "iid": item.iid, "view_sequences": {}}

        def fake_save(path, payload):
            saved.append(payload)
            path.write_bytes(b"sealed-placeholder")
            return {
                "path": str(path), "sha256": _digest("shard:%d" % payload["shard_index"]),
                "size_bytes": path.stat().st_size, "mode": 0o444, "nlink": 1,
                "semantic_sha256": _digest("semantic:%d" % payload["shard_index"]),
                "fresh_torch_load_readback_exact": True,
            }

        partitions = []
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runtime, "load_anchor_manifest", return_value=(anchors, {})
        ), mock.patch.object(runtime, "FrozenVJepa2", FakeFrozen), mock.patch.object(
            runtime, "extract_one", side_effect=fake_extract_one
        ), mock.patch.object(runtime, "_save_torch_create_only", side_effect=fake_save), mock.patch.object(
            runtime, "implementation_binding", return_value={"sha256": "frozen"}
        ), mock.patch.object(runtime.torch.cuda, "reset_peak_memory_stats"), mock.patch.object(
            runtime.torch.cuda, "max_memory_allocated", return_value=123
        ), mock.patch.object(runtime.torch.cuda, "device_count", return_value=1), mock.patch.object(
            runtime.torch.cuda, "get_device_name", return_value="AMD Instinct MI210"
        ), mock.patch("builtins.print"):
            root = Path(temporary).resolve()
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            for shard_index in range(6):
                result = runtime.extract_shard(argparse.Namespace(
                    num_shards=6,
                    shard_index=shard_index,
                    manifest=str(root / "manifest.json"),
                    expected_manifest_sha256=runtime.FEATURE_MANIFEST_SHA256,
                    model_root=str(root / "model"),
                    device="cuda:0",
                    output=str(root / ("shard-%d.pt" % shard_index)),
                ))
                payload = saved[-1]
                self.assertEqual(result["record_count"], len(payload["records"]))
                self.assertEqual(result["processor_calls"], len(payload["records"]))
                self.assertEqual(result["backbone_forwards"], 5 * len(payload["records"]))
                partitions.append(payload["global_anchor_ordinals"])

        self.assertEqual([len(rows) for rows in partitions], [108, 108, 107, 107, 107, 107])
        flattened = [ordinal for rows in partitions for ordinal in rows]
        self.assertEqual(len(flattened), 644)
        self.assertEqual(len(set(flattened)), 644)
        self.assertEqual(sorted(flattened), list(range(644)))
        for index, rows in enumerate(partitions):
            self.assertTrue(all(ordinal % 6 == index for ordinal in rows))

    def test_full_shard_payload_weights_only_roundtrip_has_plain_runtime_strings(self) -> None:
        shard_index = 5
        anchors = [
            runtime.AnchorItem(
                ordinal=index,
                iid="iid-%04d" % index,
                family="family-%02d" % (index % 28),
                group_id=_digest("group:%d" % index),
                instruction_sha256=_digest("instruction:%d" % index),
                strict=index < 359,
                path=Path("/sealed/%04d.mp4" % index),
                media_sha256=_digest("media:%d" % index),
            )
            for index in range(644) if index % 6 == shard_index
        ]
        shared = torch.arange(4, dtype=torch.float32)

        class FakeFrozen:
            def __init__(self, model_root: Path, device: str):
                self.processor_call_count = 0
                self.forward_call_count = 0
                self.device_uuid = "fake-mi210-uuid"

            def final_closure(self):
                return {
                    "model_files_before_and_after_exact": True,
                    "transformers_modules_before_and_after_exact": True,
                }

        def fake_extract_one(item, frozen):
            frozen.processor_call_count += 1
            frozen.forward_call_count += 5
            return {
                "ordinal": item.ordinal,
                "iid": item.iid,
                "view_sequences": {name: shared for name in runtime.VIEW_NAMES},
            }

        implementation = {"path": "/sealed/extractor.py", "sha256": _digest("extractor")}
        torch_version = _VersionStringSubclass("2.7.1+rocm6.3")
        hip_version = _VersionStringSubclass("6.3.42131")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            output = root / "shard-5.pt"
            with mock.patch.object(
                runtime, "load_anchor_manifest", return_value=(anchors, {})
            ), mock.patch.object(runtime, "FrozenVJepa2", FakeFrozen), mock.patch.object(
                runtime, "extract_one", side_effect=fake_extract_one
            ), mock.patch.object(
                runtime, "implementation_binding", return_value=implementation
            ), mock.patch.object(
                runtime.torch, "__version__", torch_version
            ), mock.patch.object(
                runtime.torch.version, "hip", hip_version
            ), mock.patch.object(
                runtime.torch.cuda, "reset_peak_memory_stats"
            ), mock.patch.object(
                runtime.torch.cuda, "max_memory_allocated", return_value=123
            ), mock.patch.object(
                runtime.torch.cuda, "device_count", return_value=1
            ), mock.patch.object(
                runtime.torch.cuda, "get_device_name", return_value="AMD Instinct MI210"
            ), mock.patch("builtins.print"):
                result = runtime.extract_shard(argparse.Namespace(
                    num_shards=6,
                    shard_index=shard_index,
                    manifest=str(manifest),
                    expected_manifest_sha256=runtime.FEATURE_MANIFEST_SHA256,
                    model_root=str(root / "model"),
                    device="cuda:0",
                    output=str(output),
                ))

            with output.open("rb") as handle:
                reloaded = torch.load(handle, map_location="cpu", weights_only=True)
            self.assertEqual(result["record_count"], 107)
            self.assertTrue(result["output_binding"]["fresh_torch_load_readback_exact"])
            self.assertEqual(len(reloaded["records"]), 107)
            self.assertIs(type(reloaded["runtime"]["torch"]), str)
            self.assertIs(type(reloaded["runtime"]["torch_hip"]), str)
            self.assertEqual(reloaded["runtime"]["torch"], "2.7.1+rocm6.3")
            self.assertEqual(reloaded["runtime"]["torch_hip"], "6.3.42131")
            self.assertEqual(
                runtime._shard_semantic_sha256(reloaded),
                result["output_binding"]["semantic_sha256"],
            )

    def test_extract_shard_rejects_any_non_exact6_layout(self) -> None:
        base = dict(
            shard_index=0,
            num_shards=6,
            manifest="/manifest.json",
            expected_manifest_sha256=runtime.FEATURE_MANIFEST_SHA256,
            model_root="/model",
            device="cuda:0",
            output="/shard.pt",
        )
        for update in ({"num_shards": 5}, {"num_shards": 7}, {"shard_index": -1}, {"shard_index": 6}):
            values = dict(base)
            values.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                runtime.extract_shard(argparse.Namespace(**values))


class VJepa2ExtractorV4CPostflightTests(unittest.TestCase):
    def _anchors(self):
        return [
            runtime.AnchorItem(
                ordinal=index,
                iid="iid-%04d" % index,
                family="family-%02d" % (index % 28),
                group_id=_digest("group:%d" % index),
                instruction_sha256=_digest("instruction:%d" % index),
                strict=index < 359,
                path=Path("/sealed/%04d.mp4" % index),
                media_sha256=_digest("media:%d" % index),
            )
            for index in range(644)
        ]

    def _binding_and_closure(self):
        binding = {"path": "/sealed/extractor.py", "sha256": _digest("extractor")}
        model = {"root": "/sealed/model", "closure_sha256": _digest("model")}
        transformers = {"closure_sha256": _digest("transformers")}
        closure = {
            "model_files_before_and_after_exact": True,
            "transformers_modules_before_and_after_exact": True,
            "model": model,
            "transformers": transformers,
        }
        return binding, closure

    def _payloads(self):
        binding, closure = self._binding_and_closure()
        shared = torch.zeros((1,), dtype=torch.float32)
        payloads = []
        for index in range(6):
            ordinals = [ordinal for ordinal in range(644) if ordinal % 6 == index]
            records = [
                {
                    "ordinal": ordinal,
                    "iid": "iid-%04d" % ordinal,
                    "view_sequences": {name: shared for name in runtime.VIEW_NAMES},
                }
                for ordinal in ordinals
            ]
            payloads.append({
                "schema_version": runtime.FEATURE_SCHEMA,
                "status": "V4C_VJEPA2_ORDERED_CONTEXTUAL_SHARD_COMPLETE_BURNED_DEVELOPMENT",
                "authority": "feature_mechanics_diagnostic_only",
                "formal_training_authorized": False,
                "paired_ground_truth_claimed": False,
                "implementation": binding,
                "manifest_sha256": runtime.FEATURE_MANIFEST_SHA256,
                "manifest_digest": runtime.FEATURE_MANIFEST_DIGEST,
                "source_manifest_sha256": runtime.SOURCE_MANIFEST_FILE_SHA256,
                "source_manifest_digest": runtime.SOURCE_MANIFEST_DIGEST,
                "shard_index": index,
                "num_shards": 6,
                "global_anchor_ordinals": ordinals,
                "record_count": len(records),
                "processor_call_count": len(records),
                "frozen_backbone_forward_count": 5 * len(records),
                "one_processor_then_exact5_separate_forwards_per_anchor": True,
                "model_forward_batching_across_views": False,
                "model_repo": runtime.MODEL_REPO,
                "model_revision": runtime.MODEL_REVISION,
                "model_dtype": "torch.float16",
                "skip_predictor": True,
                "model_and_source_closure": closure,
                "sampling_and_transform_abi": {"synthetic_test_authority": True},
                "records": records,
            })
        return payloads

    def _args(self, root=None):
        base = Path("/sealed") if root is None else Path(root)
        return argparse.Namespace(
            manifest=str(base / "manifest.json"),
            expected_manifest_sha256=runtime.FEATURE_MANIFEST_SHA256,
            shard=[str(base / ("shard-%d.pt" % index)) for index in range(6)],
            expected_shard_sha256=[_digest("shard:%d" % index) for index in range(6)],
            output=str(base / "receipt.json"),
        )

    def _run_aggregate(self, payloads, writer=None, validator=None):
        anchors = self._anchors()
        binding, closure = self._binding_and_closure()
        loaded = [
            (payload, {
                "path": "/sealed/shard-%d.pt" % index,
                "sha256": _digest("shard:%d" % index),
                "size_bytes": 1000 + index,
                "mode": 0o444,
                "nlink": 1,
                "semantic_sha256": _digest("semantic:%d" % index),
                "single_fd_pre_post_sha256_exact": True,
            })
            for index, payload in enumerate(payloads)
        ]
        captured = []

        def fake_writer(path, receipt):
            captured.append(receipt)
            if writer is not None:
                return writer(path, receipt)
            return {
                "path": str(path), "sha256": _digest("receipt"),
                "size_bytes": 999, "mode": 0o444, "nlink": 1,
            }

        if validator is None:
            validator = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(runtime, "implementation_binding", return_value=binding), mock.patch.object(
                runtime, "load_anchor_manifest", return_value=(anchors, {})
            ), mock.patch.object(runtime, "_load_sealed_shard", side_effect=loaded), mock.patch.object(
                runtime, "sealed_model_closure", return_value=closure["model"]
            ), mock.patch.object(runtime, "transformers_module_closure", return_value=closure["transformers"]), mock.patch.object(
                runtime, "_validate_record", validator
            ), mock.patch.object(runtime, "tensor_sha256", return_value="a" * 64), mock.patch.object(
                runtime, "_write_json_create_only", side_effect=fake_writer
            ):
                result = runtime.aggregate_shards(self._args(root))
        return result, captured, validator

    def test_postflight_exact6_union_and_receipt_are_closed(self) -> None:
        result, captured, validator = self._run_aggregate(self._payloads())
        self.assertEqual(result["record_count"], 644)
        self.assertEqual(result["shard_count"], 6)
        self.assertEqual(result["receipt_sha256"], _digest("receipt"))
        self.assertEqual(validator.call_count, 644)
        receipt = captured[0]
        self.assertEqual(receipt["schema_version"], runtime.RECEIPT_SCHEMA)
        self.assertEqual(receipt["status"], "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED")
        self.assertEqual(receipt["population"], {
            "unique_base_clips": 644,
            "action_anchor_records": 644,
            "source_records": 0,
            "total_feature_records": 644,
            "view_evaluations_per_anchor": 5,
            "derived_views_are_independent_samples": False,
            "family_count": 28,
            "strict_true": 359,
            "strict_false": 285,
        })
        self.assertEqual([row["index"] for row in receipt["shards"]], list(range(6)))
        self.assertTrue(receipt["exact6_shards"])
        self.assertFalse(receipt["action_representation_qualified"])
        self.assertFalse(receipt["renderer_authorized"])
        self.assertFalse(receipt["inference_authorized"])
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(declared, runtime.object_sha256(unsigned))

    def test_postflight_rejects_wrong_count_duplicate_index_and_wrong_partition(self) -> None:
        args = self._args()
        args.shard.pop()
        with self.assertRaises(ValueError):
            runtime.aggregate_shards(args)

        duplicate = self._payloads()
        duplicate[-1]["shard_index"] = 0
        with self.assertRaises(ValueError):
            self._run_aggregate(duplicate)

        wrong_partition = self._payloads()
        wrong_partition[2]["global_anchor_ordinals"] = list(
            reversed(wrong_partition[2]["global_anchor_ordinals"])
        )
        with self.assertRaises(ValueError):
            self._run_aggregate(wrong_partition)

    def test_postflight_rejects_authority_and_model_source_closure_drift(self) -> None:
        wrong_implementation = self._payloads()
        wrong_implementation[4]["implementation"] = {"sha256": _digest("changed")}
        with self.assertRaises(ValueError):
            self._run_aggregate(wrong_implementation)

        wrong_closure = self._payloads()
        wrong_closure[3]["model_and_source_closure"] = dict(
            wrong_closure[3]["model_and_source_closure"], changed=True
        )
        with self.assertRaises(ValueError):
            self._run_aggregate(wrong_closure)

    def _valid_record(self, expected):
        sequences = {
            name: torch.full((32, 1024), float(ordinal), dtype=torch.float32)
            for ordinal, name in enumerate(runtime.VIEW_NAMES, start=1)
        }
        receipts = {
            name: {
                "forward_ordinal_within_anchor": ordinal,
                "model_input_pixel_values_videos_sha256": _digest("pixels:%s" % name),
                "model_input_shape": [1, 64, 3, 256, 256],
                "model_input_dtype": "torch.float32",
                "last_hidden_state_shape": [1, 8192, 1024],
                "token_output_permuted_or_reindexed": False,
                "time_major_reshape": [1, 32, 256, 1024],
                "spatial_mean_output_shape": [32, 1024],
                "ordered_contextual_sequence_sha256": runtime.tensor_sha256(sequences[name]),
            }
            for ordinal, name in enumerate(runtime.VIEW_NAMES, start=1)
        }
        return {
            "ordinal": expected.ordinal,
            "iid": expected.iid,
            "family": expected.family,
            "group_id": expected.group_id,
            "instruction_sha256": expected.instruction_sha256,
            "strict_selection_gates_all_true": expected.strict,
            "role": "action_anchor",
            "media_sha256": expected.media_sha256,
            "media": {
                "manifest_logical_path": str(expected.path),
                "resolved_path": str(expected.path),
                "logical_equals_resolved_path": True,
                "decoder": "PyAV",
                "pyav_version": "13.1.0",
                "decoded_display_order_is_iteration_order": True,
                "average_rate": "25/1",
                "stream_frames_metadata": 81,
                "decoded_frame_count": 81,
                "all_pts_integral": True,
                "pts_strictly_increasing": True,
                "every_pts_delta_is_exactly_one_over_25_seconds": True,
                "pts": list(range(81)),
                "pts_sha256": runtime.object_sha256(list(range(81))),
                "single_time_base": "1/25",
                "decoded_exact81_rgb24_sha256": _digest("decoded"),
            },
            "exact81_to_base64_indices": runtime.base64_indices().tolist(),
            "exact81_to_base64_indices_sha256": runtime.BASE64_INDICES_SHA256,
            "selected_base64_rgb24_sha256": _digest("selected"),
            "block64_indices": runtime.block64_indices(expected.iid).tolist(),
            "block64_indices_sha256": runtime.tensor_sha256(runtime.block64_indices(expected.iid)),
            "phase64_indices": runtime.phase64_indices().tolist(),
            "phase64_indices_sha256": runtime.tensor_sha256(runtime.phase64_indices()),
            "canonical_processor_call_count": 1,
            "independent_frozen_backbone_forward_count": 5,
            "model_forward_batching_across_views": False,
            "post_backbone_token_permutation_used": False,
            "view_order": list(runtime.VIEW_NAMES),
            "view_sequences": sequences,
            "view_receipts": receipts,
        }

    def test_record_postflight_accepts_exact_views_and_rejects_hostile_values(self) -> None:
        expected = self._anchors()[17]
        valid = self._valid_record(expected)
        runtime._validate_record(valid, expected)

        hostile = self._valid_record(expected)
        hostile["view_sequences"]["reverse"] = hostile["view_sequences"]["reverse"].half()
        with self.assertRaises(ValueError):
            runtime._validate_record(hostile, expected)

        hostile = self._valid_record(expected)
        hostile["view_sequences"].pop("phase_swap")
        with self.assertRaises(ValueError):
            runtime._validate_record(hostile, expected)

        hostile = self._valid_record(expected)
        hostile["media"]["pts_strictly_increasing"] = False
        with self.assertRaises(ValueError):
            runtime._validate_record(hostile, expected)

        for key, value in (
            ("pts", list(range(80)) + [79]),
            ("pts_sha256", "0" * 64),
            ("single_time_base", "1/24"),
            ("pyav_version", "13.0.0"),
            ("decoded_display_order_is_iteration_order", False),
        ):
            hostile = self._valid_record(expected)
            hostile["media"][key] = value
            with self.subTest(media_key=key), self.assertRaises(ValueError):
                runtime._validate_record(hostile, expected)

        hostile = self._valid_record(expected)
        hostile["view_sequences"]["original"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            runtime._validate_record(hostile, expected)

    def test_load_sealed_shard_binds_sha_envelope_and_single_fd_readback(self) -> None:
        payload = {
            "schema_version": runtime.FEATURE_SCHEMA,
            "record_count": 1,
            "records": [{
                "iid": "sealed",
                "view_sequences": {
                    name: torch.ones((2, 3), dtype=torch.float32)
                    for name in runtime.VIEW_NAMES
                },
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "shard.pt"
            output = runtime._save_torch_create_only(path, payload)
            reloaded, binding = runtime._load_sealed_shard(path, output["sha256"])
            self.assertEqual(runtime._shard_semantic_sha256(reloaded), output["semantic_sha256"])
            self.assertEqual(binding["sha256"], output["sha256"])
            self.assertTrue(binding["single_fd_pre_post_sha256_exact"])
            with self.assertRaises(ValueError):
                runtime._load_sealed_shard(path, "0" * 64)
            os.chmod(path, 0o644)
            with self.assertRaises(ValueError):
                runtime._load_sealed_shard(path, output["sha256"])

    def test_load_sealed_shard_rejects_pickle_globals_before_schema_access(self) -> None:
        class HostileGlobal:
            def __reduce__(self):
                return eval, ("{'records': []}",)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "hostile.pt"
            torch.save({"records": [HostileGlobal()]}, path)
            os.chmod(path, 0o444)
            with self.assertRaises(Exception):
                runtime._load_sealed_shard(path, runtime.file_sha256(path))

    def test_json_receipt_writer_is_create_only_and_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "receipt.json"
            binding = runtime._write_json_create_only(output, {"finite": 1.25})
            info = output.stat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(info.st_nlink, 1)
            self.assertEqual(binding["sha256"], runtime.file_sha256(output))
            self.assertEqual(json.loads(output.read_text(encoding="ascii")), {"finite": 1.25})
            with self.assertRaises(ValueError):
                runtime._write_json_create_only(output, {"other": True})
            with self.assertRaises(ValueError):
                runtime._write_json_create_only(Path("relative.json"), {"other": True})


class VJepa2ExtractorV4CCLITests(unittest.TestCase):
    def test_parser_accepts_only_the_two_preregistered_commands(self) -> None:
        parser = runtime.build_parser()
        shard = parser.parse_args([
            "extract-shard", "--manifest", "/m", "--expected-manifest-sha256", "0" * 64,
            "--model-root", "/model", "--shard-index", "3", "--num-shards", "6",
            "--output", "/out.pt",
        ])
        self.assertEqual(shard.command, "extract-shard")
        self.assertEqual(shard.device, "cuda:0")
        self.assertEqual(shard.shard_index, 3)
        aggregate = parser.parse_args([
            "aggregate-shards", "--manifest", "/m", "--expected-manifest-sha256", "0" * 64,
            "--shard", "/s0", "--expected-shard-sha256", "1" * 64,
            "--output", "/receipt.json",
        ])
        self.assertEqual(aggregate.command, "aggregate-shards")
        self.assertEqual(aggregate.shard, ["/s0"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["unknown-command"])

    def test_main_dispatches_extract_and_aggregate_without_real_model(self) -> None:
        extract_result = {"record_count": 108, "shard": "/out.pt"}
        with mock.patch.object(runtime, "extract_shard", return_value=extract_result) as handler:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = runtime.main([
                    "extract-shard", "--manifest", "/m", "--expected-manifest-sha256", "0" * 64,
                    "--model-root", "/model", "--shard-index", "0", "--num-shards", "6",
                    "--output", "/out.pt",
                ])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue()), extract_result)
            self.assertEqual(handler.call_args.args[0].num_shards, 6)

        aggregate_result = {"record_count": 644, "shard_count": 6}
        aggregate_argv = [
            "aggregate-shards", "--manifest", "/m", "--expected-manifest-sha256", "0" * 64,
            "--output", "/receipt.json",
        ]
        for index in range(6):
            aggregate_argv.extend((
                "--shard", "/s%d" % index,
                "--expected-shard-sha256", ("%x" % index) * 64,
            ))
        with mock.patch.object(runtime, "aggregate_shards", return_value=aggregate_result) as handler:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = runtime.main(aggregate_argv)
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue()), aggregate_result)
            self.assertEqual(len(handler.call_args.args[0].shard), 6)
            self.assertEqual(len(handler.call_args.args[0].expected_shard_sha256), 6)


if __name__ == "__main__":
    unittest.main()
