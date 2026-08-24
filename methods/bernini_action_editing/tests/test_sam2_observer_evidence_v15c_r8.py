#!/usr/bin/env python3
"""Permanent r8 regressions for byte-derived, local-only SAM2 evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = METHOD_ROOT / "sam2_observer_evidence_v15c_r8.py"


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVIDENCE = load_path("sam2_observer_evidence_v15c_r8_test", MODULE_PATH)


def standard_safetensors(path: Path, arrays: dict, metadata: dict) -> None:
    """Small standard-format writer; no safetensors wheel is required."""

    import numpy as np

    dtype_names = {
        np.dtype("float32"): "F32",
        np.dtype("float64"): "F64",
        np.dtype("uint8"): "U8",
        np.dtype("int8"): "I8",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("bool"): "BOOL",
    }
    header = {"__metadata__": dict(metadata)}
    payload = []
    offset = 0
    for key, raw_value in arrays.items():
        value = np.ascontiguousarray(raw_value)
        raw = value.tobytes(order="C")
        header[key] = {
            "dtype": dtype_names[value.dtype],
            "shape": [int(item) for item in value.shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    encoded += b" " * ((-len(encoded)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


def artifact(path: Path, relative: str, arrays: dict, schema: str) -> dict:
    return {
        "schema_version": schema,
        "relative_path": relative,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensor_order": list(arrays),
        "tensor_array_sha256": {
            key: EVIDENCE.array_sha256(value) for key, value in arrays.items()
        },
    }


def model_manifest(kind: str, salt: int) -> dict:
    entry = {
        "name": "encoder.weight" if kind == "parameters" else "pixel_mean",
        "dtype": "torch.float32",
        "shape": [1],
        "numel": 1,
        "array_sha256": hashlib.sha256(f"tensor-{kind}-{salt}".encode()).hexdigest(),
    }
    value = {
        "schema_version": EVIDENCE.MODEL_MANIFEST_SCHEMA,
        "tensor_kind": kind,
        "tensor_count": 1,
        "element_count": 1,
        "entries": [entry],
        "stream_sha256": hashlib.sha256(
            f"stream-{kind}-{salt}".encode()
        ).hexdigest(),
    }
    value["manifest_sha256"] = EVIDENCE.object_sha256(value)
    return value


def freeze_transcript(binding: dict, run_ordinal: int, model_kind: str) -> dict:
    state = {
        "eval_mode": True,
        "requires_grad_true_count": 0,
        "non_none_grad_count": 0,
        "parameters": model_manifest("parameters", 1),
        "buffers": model_manifest("buffers", 2),
    }
    state["state_sha256"] = EVIDENCE.object_sha256(
        {
            "parameters_manifest_sha256": state["parameters"]["manifest_sha256"],
            "buffers_manifest_sha256": state["buffers"]["manifest_sha256"],
        }
    )
    value = {
        "schema_version": EVIDENCE.FREEZE_SCHEMA,
        "run_ordinal": run_ordinal,
        "model_kind": model_kind,
        "evidence_mode": "sealed_deterministic_worker_tensor_manifest",
        "construction_binding": binding,
        "before": state,
        "after": copy.deepcopy(state),
        "all_freeze_gates_pass": True,
    }
    value["transcript_sha256"] = EVIDENCE.object_sha256(value)
    return value


@unittest.skipUnless(
    importlib.util.find_spec("numpy") is not None,
    "numpy is required for strict safetensors replay",
)
class LocalEvidenceReplayTests(unittest.TestCase):
    height = 4
    width = 5
    frame_count = 3

    def setUp(self) -> None:
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.binding = {
            key: hashlib.sha256(key.encode("ascii")).hexdigest()
            for key in EVIDENCE.LOCAL_BINDING_KEYS
        }
        self.admission = {
            "minimum_area_pixels": 1,
            "maximum_area_fraction": 0.8,
            "near_duplicate_iou": 0.9,
            "maximum_distinct_proposals": 4,
        }
        self.generator = {
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.8,
        }

    def tearDown(self) -> None:
        self.context.cleanup()

    def _model_binding(self, model_kind: str) -> dict:
        return {
            "source_video_sha256": self.binding["source_video_sha256"],
            "source_frame0_array_sha256": self.binding[
                "source_frame0_array_sha256"
            ],
            "checkpoint_sha256": self.binding["checkpoint_sha256"],
            "config_sha256": self.binding["config_sha256"],
            "sam2_tree_sha256": self.binding["sam2_tree_sha256"],
            "key_module_sha256": self.binding["key_module_sha256"],
            "resolved_config_sha256": self.binding[
                "image_resolved_config_sha256"
                if model_kind == "image_model"
                else "video_resolved_config_sha256"
            ],
            "worker_code_sha256": self.binding["worker_code_sha256"],
            "model_class": (
                "sam2.modeling.sam2_base.SAM2Base"
                if model_kind == "image_model"
                else "sam2.sam2_video_predictor.SAM2VideoPredictor"
            ),
        }

    def _run(self, run_ordinal: int, *, score: float = 0.9) -> dict:
        import numpy as np

        run_root = self.root / "observer_evidence" / f"run_{run_ordinal}"
        mask = np.zeros((1, self.height, self.width), dtype=np.uint8)
        mask[0, 1:3, 2:4] = 1
        amg_arrays = {
            "area": np.asarray([4], dtype=np.int64),
            "bbox_xywh": np.asarray([[2.0, 1.0, 2.0, 2.0]], dtype=np.float32),
            "predicted_iou": np.asarray([score], dtype=np.float32),
            "stability_score": np.asarray([score], dtype=np.float32),
            "masks": mask,
        }
        amg_relative = f"observer_evidence/run_{run_ordinal}/amg.safetensors"
        amg_path = self.root / amg_relative
        standard_safetensors(
            amg_path,
            amg_arrays,
            {
                "schema_version": EVIDENCE.AMG_SCHEMA,
                "run_ordinal": str(run_ordinal),
                "source_frame_index": "0",
            },
        )
        batch_root = run_root / "batch_000"
        prompt = np.zeros((1, 1, self.height, self.width), dtype=np.float32)
        prompt[0, 0, 1:3, 2:4] = 0.25
        prompt_relative = (
            f"observer_evidence/run_{run_ordinal}/batch_000/"
            "prompt_call_000.safetensors"
        )
        prompt_path = self.root / prompt_relative
        prompt_arrays = {"logits": prompt}
        standard_safetensors(
            prompt_path,
            prompt_arrays,
            {
                "schema_version": EVIDENCE.LOGIT_FILE_SCHEMA,
                "kind": "prompt",
                "run_ordinal": str(run_ordinal),
                "batch_index": "0",
                "call_index": "0",
                "inserted_object_id": "0",
                "out_ids": "0",
            },
        )
        propagation = []
        for frame_index in range(self.frame_count):
            logits = np.full(
                (1, 1, self.height, self.width), -0.5, dtype=np.float32
            )
            logits[0, 0, 1:3, 2:4] = 0.5 + frame_index
            relative = (
                f"observer_evidence/run_{run_ordinal}/batch_000/"
                f"propagation_frame_{frame_index:05d}.safetensors"
            )
            path = self.root / relative
            arrays = {"logits": logits}
            standard_safetensors(
                path,
                arrays,
                {
                    "schema_version": EVIDENCE.LOGIT_FILE_SCHEMA,
                    "kind": "propagation",
                    "run_ordinal": str(run_ordinal),
                    "batch_index": "0",
                    "frame_index": str(frame_index),
                    "out_ids": "0",
                },
            )
            propagation.append(
                artifact(path, relative, arrays, EVIDENCE.LOGIT_FILE_SCHEMA)
            )
        return {
            "schema_version": EVIDENCE.RUN_SCHEMA,
            "run_ordinal": run_ordinal,
            "sam2_execution": {
                "sam2_python_module_loaded": True,
                "automatic_generator_class": "sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator",
                "video_predictor_class": "sam2.sam2_video_predictor.SAM2VideoPredictor",
                "automatic_generate_call_count": 1,
                "add_new_mask_call_count": 1,
                "propagate_in_video_call_count": 1,
                "sam2__C_imported": False,
            },
            "amg_artifact": artifact(
                amg_path, amg_relative, amg_arrays, EVIDENCE.AMG_SCHEMA
            ),
            "tracking_batches": [
                {
                    "schema_version": EVIDENCE.RUN_SCHEMA,
                    "batch_index": 0,
                    "batch_start": 0,
                    "batch_stop": 1,
                    "prompt_artifacts": [
                        artifact(
                            prompt_path,
                            prompt_relative,
                            prompt_arrays,
                            EVIDENCE.LOGIT_FILE_SCHEMA,
                        )
                    ],
                    "propagation_artifacts": propagation,
                }
            ],
            "freeze_transcripts": {
                kind: freeze_transcript(
                    self._model_binding(kind), run_ordinal, kind
                )
                for kind in EVIDENCE.MODEL_KINDS
            },
        }

    def _receipt(self) -> dict:
        runs = [self._run(1), self._run(2)]
        replayed = [
            EVIDENCE.replay_worker_run(
                root=self.root,
                run=run,
                run_ordinal=index + 1,
                proposal_count=1,
                expected_binding=self.binding,
                admission=self.admission,
                automatic_generator=self.generator,
                tracking_batch_size=1,
                frame_count=self.frame_count,
                height=self.height,
                width=self.width,
            )
            for index, run in enumerate(runs)
        ]
        semantic = [
            EVIDENCE.object_sha256(EVIDENCE.semantic_run_payload(run))
            for run in replayed
        ]
        self.assertEqual(semantic[0], semantic[1])
        receipt = {
            "schema_version": EVIDENCE.LOCAL_SCHEMA,
            "status": "LOCAL_SCHEMA_ARTIFACTS_PUBLISHED_REMOTE_UNVERIFIED",
            "binding": self.binding,
            "proposal_count": 1,
            "runs": runs,
            "repeat_semantic_sha256": semantic[0],
            "local_schema_replay_only": True,
            "remote_worker_execution_verified": False,
            "observer_execution_authorized": False,
            "localization_semantically_certified": False,
            "scientific_claim_authorized": False,
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
        }
        receipt["receipt_sha256"] = EVIDENCE.object_sha256(receipt)
        return receipt

    def _replay(self, receipt: dict) -> dict:
        return EVIDENCE.replay_local_evidence(
            root=self.root,
            receipt=receipt,
            expected_binding=self.binding,
            admission=self.admission,
            automatic_generator=self.generator,
            tracking_batch_size=1,
            frame_count=self.frame_count,
            height=self.height,
            width=self.width,
        )

    def test_complete_standard_safetensors_bundle_is_local_replay_only(self):
        replay = self._replay(self._receipt())
        self.assertEqual(
            replay["status"], "LOCAL_SCHEMA_REPLAY_PASS_REMOTE_OBSERVER_UNVERIFIED"
        )
        self.assertIs(replay["remote_worker_execution_verified"], False)
        self.assertIs(replay["observer_execution_authorized"], False)
        self.assertIs(replay["route_authorized"], False)
        self.assertIs(replay["decode_authorized"], False)
        self.assertIs(replay["training_authorized"], False)

    def test_r7_no_sam2_hash_only_bundle_is_permanently_rejected(self):
        old_fake = {
            "schema_version": "bernini-source-sam2-proposal-tracks-v15c-r3",
            "proposal_count": 1,
            "proposals": [{"area": 999999999, "predicted_iou": 12345.0}],
            "tracking_batches": [{"logits_sha256": "3" * 64}],
            "freeze_receipts": {"parameter_sha256_before": "1" * 64},
        }
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            self._replay(old_fake)

    def test_missing_actual_prompt_or_propagation_tensor_is_rejected(self):
        receipt = self._receipt()
        missing = (
            self.root
            / receipt["runs"][0]["tracking_batches"][0]["prompt_artifacts"][0][
                "relative_path"
            ]
        )
        missing.unlink()
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            self._replay(receipt)

    def test_symlinked_tensor_is_rejected_even_when_target_bytes_match(self):
        receipt = self._receipt()
        relative = receipt["runs"][0]["tracking_batches"][0][
            "prompt_artifacts"
        ][0]["relative_path"]
        path = self.root / relative
        target = self.root / "same-bytes.safetensors"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            self._replay(receipt)

    def test_resigned_out_of_range_amg_scores_are_rejected(self):
        # A fully coherent artifact/descriptor/receipt is still rejected by the
        # [0,1] score gate before it can become a proposal transcript.
        self._run(1, score=2.0)
        run = self._run(1, score=2.0)
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            EVIDENCE.replay_worker_run(
                root=self.root,
                run=run,
                run_ordinal=1,
                proposal_count=1,
                expected_binding=self.binding,
                admission=self.admission,
                automatic_generator=self.generator,
                tracking_batch_size=1,
                frame_count=self.frame_count,
                height=self.height,
                width=self.width,
            )

    def test_amg_area_bbox_are_recomputed_from_mask(self):
        import numpy as np

        run = self._run(1)
        descriptor = run["amg_artifact"]
        path = self.root / descriptor["relative_path"]
        mask = np.zeros((1, self.height, self.width), dtype=np.uint8)
        mask[0, 1:3, 2:4] = 1
        arrays = {
            "area": np.asarray([999], dtype=np.int64),
            "bbox_xywh": np.asarray([[0.0, 0.0, 5.0, 4.0]], dtype=np.float32),
            "predicted_iou": np.asarray([0.9], dtype=np.float32),
            "stability_score": np.asarray([0.9], dtype=np.float32),
            "masks": mask,
        }
        standard_safetensors(
            path,
            arrays,
            {
                "schema_version": EVIDENCE.AMG_SCHEMA,
                "run_ordinal": "1",
                "source_frame_index": "0",
            },
        )
        run["amg_artifact"] = artifact(
            path, descriptor["relative_path"], arrays, EVIDENCE.AMG_SCHEMA
        )
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            EVIDENCE.replay_worker_run(
                root=self.root,
                run=run,
                run_ordinal=1,
                proposal_count=1,
                expected_binding=self.binding,
                admission=self.admission,
                automatic_generator=self.generator,
                tracking_batch_size=1,
                frame_count=self.frame_count,
                height=self.height,
                width=self.width,
            )

    def test_freeze_manifest_is_ordered_bound_and_byte_stable(self):
        receipt = self._receipt()
        freeze = receipt["runs"][0]["freeze_transcripts"]["image_model"]
        freeze["before"]["parameters"]["entries"][0]["array_sha256"] = "a" * 64
        # Even a re-signed outer receipt cannot repair the now-invalid inner
        # manifest/state/freeze transcript chain.
        receipt["receipt_sha256"] = EVIDENCE.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            self._replay(receipt)

    def test_header_order_gap_nonfinite_and_trailing_bytes_fail_closed(self):
        import numpy as np

        path = self.root / "one.safetensors"
        arrays = {"logits": np.zeros((1, 1, 2, 2), dtype=np.float32)}
        metadata = {
            "schema_version": EVIDENCE.LOGIT_FILE_SCHEMA,
            "kind": "test",
        }
        standard_safetensors(path, arrays, metadata)
        descriptor = artifact(
            path, "one.safetensors", arrays, EVIDENCE.LOGIT_FILE_SCHEMA
        )
        parsed = EVIDENCE.strict_safetensors(
            path,
            expected_order=("logits",),
            expected_contract={"logits": ("F32", (1, 1, 2, 2))},
            expected_file_sha256=descriptor["file_sha256"],
            expected_array_sha256=descriptor["tensor_array_sha256"],
            expected_metadata=metadata,
        )
        self.assertEqual(parsed["tensor_order"], ["logits"])
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaises(EVIDENCE.SAM2ObserverEvidenceV15CR8Error):
            EVIDENCE.strict_safetensors(
                path,
                expected_order=("logits",),
                expected_contract={"logits": ("F32", (1, 1, 2, 2))},
                expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_array_sha256=descriptor["tensor_array_sha256"],
                expected_metadata=metadata,
            )


class StaticClaimBoundaryTests(unittest.TestCase):
    def test_no_true_observer_route_decode_training_or_science_literal(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            '"remote_worker_execution_verified": True',
            '"observer_execution_authorized": True',
            '"localization_semantically_certified": True',
            '"scientific_claim_authorized": True',
            '"route_authorized": True',
            '"decode_authorized": True',
            '"training_authorized": True',
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("assert ", source)


if __name__ == "__main__":
    unittest.main()
