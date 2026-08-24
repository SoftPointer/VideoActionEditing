from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.bernini_action_editing import (
    infer_pair_v5_reserve_display_anchor_overlay_v1 as overlay,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
SELECTION = METHOD_ROOT / "assets/pair_v5_t2v_calibration_reserve4_selection_v1.json"
REGISTRY = METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"


def _rank_row(rank: int, *, job0: str = "135412", job1: str = "135407") -> dict:
    node_rank = rank // 2
    job = job0 if node_rank == 0 else job1
    node = overlay.HOLDER_NODE[job]
    row = {
        "schema_version": "bernini-pair-v5-reserve-display-anchor-rank-v1",
        "world_size": 4,
        "local_world_size": 2,
        "rank": rank,
        "local_rank": rank % 2,
        "node_rank": node_rank,
        "slurm_job_id": job,
        "slurm_step_id": "31" if node_rank == 0 else "47",
        "hostname": node,
        "torch_cuda_device_count": 2,
        "torch_hip": "6.3.42131",
        "rocr_visible_devices": "0,1",
        "hip_visible_devices": None,
        "cuda_visible_devices": None,
    }
    row["row_digest"] = overlay.object_digest(row)
    return row


def _write_receipt(path: Path, unsigned: dict) -> dict:
    overlay.write_exclusive_json(path, unsigned)
    return json.loads(path.read_text(encoding="ascii"))


def _make_branch(root: Path, branch: str, topology_dir: Path) -> Path:
    branch_root = root / "outputs" / branch
    branch_root.mkdir(parents=True)
    artifacts = {}
    for name, payload in (
        ("t2v.mp4", b"exact81-display-mp4"),
        ("t2v.normalized-clean-latent.safetensors", b"clean-latent"),
        ("t2v.official-initial-gaussian.safetensors", b"gaussian"),
    ):
        path = branch_root / name
        path.write_bytes(payload)
        artifacts[name] = (str(path), hashlib.sha256(payload).hexdigest())
    topology_path = topology_dir / "physical2x2-receipt.json"
    topology = json.loads(topology_path.read_text(encoding="ascii"))
    native_path = branch_root / "receipt.json"
    native = _write_receipt(
        native_path,
        {
            "method_source_archive_sha256": overlay.SOURCE_ARCHIVE_SHA256,
            "method_source_revision": overlay.SOURCE_REVISION,
            "arms": ["t2v"],
            "scientific_claim_authorized": False,
            "production_claim_forbidden": True,
            "source_condition_artifact": None,
            "initial_noise_artifacts": {
                "t2v": {
                    "raw_value_sha256": "1" * 64,
                    "content_sha256": "2" * 64,
                    "tensor_value_sha256": "1" * 64,
                    "shape": list(overlay.EXPECTED_LATENT_SHAPE),
                    "dtype": "torch.float32",
                    "generator_initial_seed": overlay.SEED,
                    "path": artifacts["t2v.official-initial-gaussian.safetensors"][0],
                    "sha256": artifacts["t2v.official-initial-gaussian.safetensors"][1],
                    "captured_from_native_sampler": True,
                    "external_initial_noise_injection": False,
                    "source_or_target_derived": False,
                    "observer_changed_return_value": False,
                    "official_randn_tensor_call_count": 1,
                },
            },
        },
    )
    unsigned = {
        "schema_version": overlay.SCHEMA,
        "complete": True,
        "iid": overlay.IID,
        "semantic_branch": branch,
        "candidate_id": f"pair5-t2v-reserve4-v1-{overlay.IID}-{branch}",
        "analysis_split": "fit",
        "authoring_authority": {
            "selection_sha256": overlay.SELECTION_SHA256,
            "registry_sha256": overlay.REGISTRY_SHA256,
            "row_digest": overlay.ROW_DIGEST,
            "seed": overlay.SEED,
            "prompt_sha256": overlay.PROMPT_SHA256[branch],
            "source_video_sha256": overlay.SOURCE_VIDEO_SHA256,
            "source_video_role": "exact81_bucket_geometry_probe_only",
        },
        "method_authority": {
            "source_archive_sha256": overlay.SOURCE_ARCHIVE_SHA256,
            "source_revision": overlay.SOURCE_REVISION,
            "native_generator_sha256": overlay.METHOD_FILE_SHA256["infer_native_identity_generation_canary.py"],
            "bank_wrapper_sha256": overlay.METHOD_FILE_SHA256["infer_pair_v5_t2v_calibration_bank.py"],
            "old_bank_wrapper_invoked": False,
        },
        "physical_topology": {
            "path": str(topology_path),
            "sha256": overlay.file_sha256(topology_path),
            "receipt_digest": topology["receipt_digest"],
            "world_size": 4,
            "physical_nodes": 2,
            "local_ranks_per_node": 2,
        },
        "native_receipt": {
            "path": str(native_path),
            "sha256": overlay.file_sha256(native_path),
            "receipt_digest": native["receipt_digest"],
        },
        "output": {
            "mp4_path": artifacts["t2v.mp4"][0],
            "mp4_sha256": artifacts["t2v.mp4"][1],
            "frame_count": 81,
            "fps": 25,
            "height": 592,
            "width": 400,
            "latent_shape": [1, 16, 21, 74, 50],
            "clean_latent_path": artifacts["t2v.normalized-clean-latent.safetensors"][0],
            "clean_latent_sha256": artifacts["t2v.normalized-clean-latent.safetensors"][1],
            "gaussian_path": artifacts["t2v.official-initial-gaussian.safetensors"][0],
            "gaussian_sha256": artifacts["t2v.official-initial-gaussian.safetensors"][1],
        },
        "use_contract": {
            "display_only": True,
            "stage_b_condition": False,
            "passed_to_stage_b_runtime": False,
            "used_as_model_condition": False,
            "anchor_pixels_or_latent_transplanted": False,
            "source_pixels_entered_t2v_transformer": False,
            "old40_bank_audit_claimed": False,
            "old40_bank_receipt_created": False,
            "action_success_claimed": False,
            "scientific_claim_authorized": False,
            "training_or_parameter_update_performed": False,
        },
    }
    receipt_path = branch_root / "display-anchor-receipt.json"
    _write_receipt(receipt_path, unsigned)
    return receipt_path


class PairV5ReserveDisplayAnchorOverlayTests(unittest.TestCase):
    def test_json_publication_is_complete_before_atomic_create_only_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "receipt.json"
            real_link = os.link
            observed = []

            def checked_link(source, target, **kwargs):
                self.assertFalse(Path(target).exists())
                raw = Path(source).read_bytes()
                self.assertTrue(raw.endswith(b"\n"))
                value = json.loads(raw)
                overlay.validate_embedded_digest(value, label="atomic test")
                observed.append(True)
                return real_link(source, target, **kwargs)

            with mock.patch.object(overlay.os, "link", side_effect=checked_link):
                overlay.write_exclusive_json(output, {"complete": True})
            self.assertEqual(observed, [True])
            self.assertTrue(output.is_file())
            self.assertEqual(list(output.parent.glob(".receipt.json.tmp.*")), [])

    def test_exact_reserve_authoring_builds_registered_four_prompts(self) -> None:
        row = overlay.load_authoring(SELECTION.resolve(), REGISTRY.resolve())
        self.assertEqual(overlay.object_digest(row), overlay.ROW_DIGEST)
        rows = [overlay.candidate_from_authoring(row, branch) for branch in overlay.BRANCH_ORDER]
        self.assertEqual([row["semantic_branch"] for row in rows], list(overlay.BRANCH_ORDER))
        self.assertEqual({row["seed"] for row in rows}, {2026080821})
        self.assertEqual(
            {row["semantic_branch"]: row["full_t2v_caption_utf8_sha256"] for row in rows},
            overlay.PROMPT_SHA256,
        )
        self.assertTrue(all(row["geometry_source_video"] == overlay.SOURCE_VIDEO_PATH for row in rows))

    def test_synchronized_selection_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / REGISTRY.name
            registry.write_bytes(REGISTRY.read_bytes())
            value = json.loads(SELECTION.read_text(encoding="utf-8"))
            value["first_gpu_job_default"] = True
            selection = root / SELECTION.name
            selection.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(overlay, "SELECTION_SHA256", overlay.file_sha256(selection)):
                with self.assertRaises(overlay.DisplayAnchorError):
                    overlay.load_authoring(selection.resolve(), registry.resolve())

    def test_physical_two_by_two_topology_accepts_exact_and_rejects_single_host(self) -> None:
        rows = [_rank_row(rank) for rank in range(4)]
        receipt = overlay.validate_topology_rows(
            rows,
            job0="135412", node0="auh7-1b-gpu-293",
            job1="135407", node1="auh7-1b-gpu-260",
        )
        self.assertEqual(receipt["world_size"], 4)
        self.assertEqual(receipt["physical_nodes"], 2)
        self.assertEqual(receipt["local_ranks_per_node"], 2)
        hostile = [dict(row) for row in rows]
        hostile[2]["hostname"] = "auh7-1b-gpu-293"
        unsigned = dict(hostile[2]); unsigned.pop("row_digest")
        hostile[2]["row_digest"] = overlay.object_digest(unsigned)
        with self.assertRaises(overlay.DisplayAnchorError):
            overlay.validate_topology_rows(
                hostile,
                job0="135412", node0="auh7-1b-gpu-293",
                job1="135407", node1="auh7-1b-gpu-260",
            )

    def test_action_only_set_is_display_only_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            topology_dir = root / "topology" / "action"
            topology_dir.mkdir(parents=True)
            topology_unsigned = overlay.validate_topology_rows(
                [_rank_row(rank) for rank in range(4)],
                job0="135412", node0="auh7-1b-gpu-293",
                job1="135407", node1="auh7-1b-gpu-260",
            )
            _write_receipt(topology_dir / "physical2x2-receipt.json", topology_unsigned)
            _make_branch(root, "action", topology_dir)
            args = type("Args", (), {"root": str(root), "profile": "action-only"})()
            self.assertEqual(overlay.verify_set(args), 0)
            receipt = json.loads((root / "display-anchor-set-receipt.json").read_text())
            self.assertEqual(receipt["generation_order"], ["action"])
            self.assertTrue(
                receipt["same_official_initial_gaussian_tensor_value_across_requested_branches"]
            )
            self.assertTrue(receipt["use_contract"]["display_only"])
            self.assertFalse(receipt["use_contract"]["stage_b_condition"])
            self.assertFalse(receipt["use_contract"]["old40_bank_audit_claimed"])
            with self.assertRaises(overlay.DisplayAnchorError):
                overlay.verify_set(args)

    def test_family4_requires_exact_registered_order_and_one_holder_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for branch in overlay.BRANCH_ORDER:
                topology_dir = root / "topology" / branch
                topology_dir.mkdir(parents=True)
                topology_unsigned = overlay.validate_topology_rows(
                    [_rank_row(rank) for rank in range(4)],
                    job0="135412", node0="auh7-1b-gpu-293",
                    job1="135407", node1="auh7-1b-gpu-260",
                )
                _write_receipt(topology_dir / "physical2x2-receipt.json", topology_unsigned)
                _make_branch(root, branch, topology_dir)
            args = type("Args", (), {"root": str(root), "profile": "family4"})()
            self.assertEqual(overlay.verify_set(args), 0)
            receipt = json.loads((root / "display-anchor-set-receipt.json").read_text())
            self.assertEqual(receipt["generation_order"], list(overlay.BRANCH_ORDER))
            self.assertTrue(receipt["action_canary_first"])
            self.assertTrue(
                receipt["same_official_initial_gaussian_tensor_value_across_requested_branches"]
            )

    def test_synchronized_external_topology_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            topology_dir = root / "topology" / "action"
            topology_dir.mkdir(parents=True)
            topology_unsigned = overlay.validate_topology_rows(
                [_rank_row(rank) for rank in range(4)],
                job0="135412", node0="auh7-1b-gpu-293",
                job1="135407", node1="auh7-1b-gpu-260",
            )
            canonical = topology_dir / "physical2x2-receipt.json"
            _write_receipt(canonical, topology_unsigned)
            receipt_path = _make_branch(root, "action", topology_dir)
            external_dir = root / "external"
            external_dir.mkdir()
            external = external_dir / "physical2x2-receipt.json"
            external.write_bytes(canonical.read_bytes())
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            receipt.pop("receipt_digest")
            receipt["physical_topology"]["path"] = str(external)
            receipt["physical_topology"]["sha256"] = overlay.file_sha256(external)
            receipt_path.unlink()
            _write_receipt(receipt_path, receipt)
            args = type("Args", (), {"root": str(root), "profile": "action-only"})()
            with self.assertRaisesRegex(
                overlay.DisplayAnchorError, "topology receipt path is non-canonical"
            ):
                overlay.verify_set(args)


if __name__ == "__main__":
    unittest.main()
