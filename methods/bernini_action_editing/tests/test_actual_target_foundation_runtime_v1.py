#!/usr/bin/env python3
"""End-to-end CPU integration tests with deterministic fake frozen models."""

from __future__ import annotations

import importlib.util
import inspect
import json
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("actual_target_foundation_runtime_v1", ROOT / "actual_target_foundation_runtime_v1.py")
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = runtime; SPEC.loader.exec_module(runtime)
graph = runtime.graph_v2
POST_SPEC=importlib.util.spec_from_file_location("actual_target_foundation_postflight_v2",ROOT/"actual_target_foundation_postflight_v2.py")
assert POST_SPEC is not None and POST_SPEC.loader is not None
postflight=importlib.util.module_from_spec(POST_SPEC); sys.modules[POST_SPEC.name]=postflight; POST_SPEC.loader.exec_module(postflight)


def basis(slot: int, size: int = 32) -> tuple[float, ...]:
    return tuple(1.0 if index == slot else 0.0 for index in range(size))


class FakeFrozenBackend:
    model_names = ("fake_sam2", "fake_cotracker", "fake_dinov2", "fake_vjepa2")

    def __init__(self): self.scrubs = 0
    def decode(self, path, expected_sha256): return tuple(range(64))
    def node(self, frames, view):
        if view == "source_noop": signature = tuple(value for phase in range(8) for value in basis((phase + 4) % 8, 8))
        else: signature = tuple(value for phase in range(8) for value in basis(phase, 8))
        return runtime.NodeSketch(signature, (0, 1, 3, 2, 5, 1, 0, 4), 6, True, unbalanced_phase_pair_count=7, dustbin_unmatched_count=2, dustbin_transport_mass=0.1)
    def motion(self, frames, view, nodes):
        slots = {"target_forward_reference": 0, "target_forward_eval": 0, "target_reverse": 3, "target_deterministic_shuffle": 5, "source_noop": 7}
        slot = slots[view]
        return runtime.MotionSketch(basis(slot), basis(slot + 8), 0.9, True, 1, 32, 4, 2, 8)
    def phase(self, frames, view):
        slots = {"target_forward_reference": 0, "target_forward_eval": 0, "target_reverse": 3, "target_deterministic_shuffle": 5, "source_noop": 7}
        return runtime.PhaseSketch(basis(slots[view]))
    def frozen_receipt(self):
        return {"all_models_eval_frozen": True, "source_and_weight_closure_unchanged": True, "parameter_updates": 0, "generator_forward_calls": 0}
    def scrub_case(self): self.scrubs += 1


class ActualTargetFoundationRuntimeV1Test(unittest.TestCase):
    @staticmethod
    def node(mask, descriptor):
        ys,xs=np.asarray(mask).nonzero()
        return graph.AnonymousNodeV2(np.asarray(mask,dtype=bool),tuple(descriptor),float(np.mean(mask)),(float(xs.mean()/max(mask.shape[1]-1,1)),float(ys.mean()/max(mask.shape[0]-1,1))))

    def test_fake_models_run_all_four_cases_end_to_end_with_exact_counts(self):
        backend = FakeFrozenBackend(); result = runtime.run_canary(backend)
        self.assertEqual(result["logical_forward_counts"], {"media_decode": 8, "sam2": 96, "dinov2": 96, "cotracker": 20, "vjepa2": 20})
        self.assertEqual(len(result["cases"]), 4)
        self.assertIn("scalar_metrics",result["cases"][0])
        self.assertEqual(result["cases"][0]["scalar_metrics"]["node"]["unbalanced_phase_pair_count"],7)
        self.assertTrue(result["aggregate"]["diagnostic_canary_pass"])
        self.assertFalse(result["aggregate"]["representation_admitted"])
        self.assertEqual(backend.scrubs, 4)

    def test_receipt_is_zero_train_zero_generator_and_contains_no_raw_payload(self):
        result = runtime.run_canary(FakeFrozenBackend())
        self.assertFalse(result["training_performed"]); self.assertFalse(result["optimizer_created"])
        self.assertEqual(result["parameter_updates"], 0); self.assertFalse(result["generator_loaded"])
        self.assertEqual(result["generator_forward_calls"], 0); self.assertFalse(result["raw_teacher_payload_persisted"])
        text = json.dumps(result, sort_keys=True)
        for key in runtime.FORBIDDEN_RECEIPT_KEYS: self.assertNotIn(f'"{key}"', text)

    def test_output_and_case_cache_are_absent_create_only_scalar_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); cache = root / "cache"; cache.mkdir(); output = root / "result.json"
            result = runtime.run_canary(FakeFrozenBackend(), output=output, cache_dir=cache)
            self.assertEqual(json.loads(output.read_text())["digest"], result["digest"])
            self.assertEqual(len(list(cache.glob("*.json"))), 4)
            with self.assertRaises(runtime.RuntimeErrorV1): runtime.run_canary(FakeFrozenBackend(), output=output)
            for path in cache.glob("*.json"):
                row=json.loads(path.read_text()); self.assertFalse(row["raw_teacher_payload_persisted"]); self.assertNotIn("cases", row)

    def test_counterfactuals_and_input_controls_are_branchwise_and(self):
        result = runtime.run_canary(FakeFrozenBackend())
        for case in result["cases"]:
            self.assertTrue(all(case["branch_pass"].values()))
            self.assertEqual(set(case["branch_pass"]), set(runtime.authority.BRANCHES))

    def test_reference_eval_sampling_is_disjoint_and_controls_are_exact_permutations(self):
        views=runtime._views(tuple(range(100,180)),tuple(range(80)))
        self.assertEqual(len(views["target_forward_reference"]),8); self.assertEqual(len(views["target_forward_eval"]),8)
        self.assertFalse(set(views["target_forward_reference"]) & set(views["target_forward_eval"]))
        self.assertEqual(views["target_reverse"],tuple(reversed(views["target_forward_eval"])))
        self.assertEqual(views["target_deterministic_shuffle"],tuple(views["target_forward_eval"][index] for index in runtime.SHUFFLE))
        phase=runtime._phase_views(tuple(range(100,180)),tuple(range(80)))
        self.assertEqual(len(phase["target_forward_reference"]),16); self.assertEqual(len(phase["target_forward_eval"]),16)
        self.assertFalse(set(phase["target_forward_reference"]) & set(phase["target_forward_eval"]))

    def test_launch_contract_is_implemented_but_default_false(self):
        contract = runtime.launch_contract()
        self.assertEqual(contract["implementation_status"], "V2_CLOSURE_IMPLEMENTED_UNEXECUTED_PRE_FLIP_NO")
        self.assertFalse(contract["real_gpu_launch_authorized"])
        self.assertTrue(contract["independent_audit_required_before_gpu"])
        self.assertFalse(contract["raw_teacher_payload_persisted"])
        self.assertEqual(contract["source_closure"]["file_count"], 10)

    def test_real_cli_fails_before_model_import_or_gpu_use(self):
        result=subprocess.run([sys.executable,str(ROOT/"actual_target_foundation_runtime_v1.py"),"--run-real","--output","/tmp/never-created-foundation-canary.json"],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        self.assertNotEqual(result.returncode,0); self.assertIn("blocked pending a new independent audit",result.stderr)

    @unittest.skipIf(np is None,"numeric geometry dependency is unavailable locally")
    def test_unbalanced_sinkhorn_has_explicit_dustbin_and_real_matches(self):
        a=np.zeros((8,8),bool); a[1:3,1:3]=1; b=np.zeros((8,8),bool); b[2:4,2:4]=1
        left=[self.node(a,(1,0)),self.node(np.rot90(a),(0,1))]; right=[self.node(b,(1,0))]
        plan=graph.unbalanced_sinkhorn_dustbin(left,right)
        self.assertEqual(plan.shape,(3,2)); self.assertTrue(np.isfinite(plan).all())
        self.assertEqual(graph.hard_matches_with_dustbin(plan),((0,0),))
        self.assertGreater(plan[1,-1],plan[1,0])

    @unittest.skipIf(np is None,"numeric geometry dependency is unavailable locally")
    def test_slot_relabel_is_exactly_invariant_but_binding_break_is_negative(self):
        a=np.zeros((8,8),bool); a[1:3,1:3]=1; b=np.zeros((8,8),bool); b[5:7,5:7]=1
        phases=tuple((self.node(a,(1,0)),self.node(b,(0,1))) for _ in range(8))
        positive=graph.canonical_node_signature(phases)
        self.assertEqual(positive,graph.canonical_node_signature(graph.relabel_slots(phases)))
        assigned=graph.assign_anonymous_tracks(phases); relabeled=graph.assign_anonymous_tracks(graph.relabel_slots(phases))
        ids=lambda rows: tuple(tuple((node.descriptor,node.track_id) for node in sorted(phase,key=lambda node:node.descriptor)) for phase in rows)
        self.assertEqual(ids(assigned),ids(relabeled))
        negative=graph.canonical_node_signature(graph.break_mask_descriptor_binding(phases))
        self.assertNotEqual(positive,negative)
        singleton=tuple((self.node(a,(1,0)),) for _ in range(8))
        self.assertTrue(any(len(phase)==0 for phase in graph.break_mask_descriptor_binding(singleton)))

    @unittest.skipIf(np is None,"numeric geometry dependency is unavailable locally")
    def test_edge_contains_boundary_overlap_relative_velocity_and_lifecycle(self):
        a=np.zeros((8,8),bool); a[1:4,1:4]=1; b=np.zeros((8,8),bool); b[2:5,2:5]=1
        phases=[]
        for phase in range(8):
            nodes=[self.node(a,(1,0))]; nodes[0]=graph.replace(nodes[0],track_id=0)
            if phase<6:
                other=graph.replace(self.node(b,(0,1)),track_id=1); nodes.append(other)
            phases.append(tuple(nodes))
        velocities={(phase,0):(1.0,0.0) for phase in range(8)}; velocities.update({(phase,1):(0.0,1.0) for phase in range(8)})
        signature,lifecycle=graph.pairwise_edge_signature(phases,velocities)
        self.assertEqual(len(signature),33); self.assertGreater(signature[1],0.0); self.assertGreater(signature[2],0.0); self.assertGreater(lifecycle,0)
        self.assertNotEqual(signature,tuple(0.0 for _ in signature))

    @unittest.skipIf(np is None,"numeric geometry dependency is unavailable locally")
    def test_dino_area_pool_and_zero_support_abstention(self):
        tokens=np.asarray([[1.,0.],[0.,2.],[3.,3.],[5.,1.]])
        self.assertEqual(graph.patch_area_pool(np.asarray([0.5,0.5,0,0]),tokens),(0.5,1.0))
        with self.assertRaises(graph.GraphV2Error): graph.patch_area_pool(np.zeros(4),tokens)

    @unittest.skipIf(np is None,"numeric geometry dependency is unavailable locally")
    def test_vjepa_is_exactly_eight_real_tubelet2_blocks(self):
        hidden=np.arange(8*4*3,dtype=np.float32).reshape(32,3)
        blocks=graph.tubelet2_eight_blocks(hidden,4); self.assertEqual(blocks.shape,(8,3))
        with self.assertRaises(graph.GraphV2Error): graph.tubelet2_eight_blocks(hidden[:-1],4)

    @unittest.skipIf(np is None,"numeric geometry dependency is unavailable locally")
    def test_success_and_exception_raw_ownership_zeroize(self):
        class Cuda:
            @staticmethod
            def is_available(): return False
        class Torch: cuda=Cuda()
        backend=runtime.RealFrozenBackend.__new__(runtime.RealFrozenBackend); backend.torch=Torch(); backend._owned_raw=[]; backend._raw_registered=backend._raw_zeroized=0
        first=np.ones((2,2),dtype=np.uint8); backend._own(first); backend.scrub_case(); self.assertFalse(first.any())
        second=np.ones((2,2),dtype=np.uint8)
        try:
            backend.begin_case(); backend._own(second); raise ValueError("synthetic failure")
        except ValueError: pass
        finally: backend.scrub_case()
        self.assertFalse(second.any()); self.assertTrue(backend.raw_ownership_receipt()["verified"])

    def test_decode_receipt_and_runtime_class_authority_are_sealed(self):
        decode=runtime.authority.load_decode_receipt(); self.assertEqual(len(decode["rows"]),8)
        self.assertEqual({row["frame_count"] for row in decode["rows"]},{169,80,76,91,138,204,141,126})
        availability=runtime.authority.load_availability(); self.assertEqual(len(availability["runtime_class_authority"]),5)

    def test_auh_dynamic_class_source_and_config_binding_without_model_load(self):
        availability=runtime.authority.load_availability(); cot_root=Path(availability["foundations"]["cotracker"]["repository_root"])
        if not cot_root.is_dir(): self.skipTest("AUH foundation sources are unavailable locally")
        from sam2.modeling.sam2_base import SAM2Base
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        if str(cot_root) not in sys.path: sys.path.insert(0,str(cot_root))
        from cotracker.predictor import CoTrackerPredictor
        from transformers import AutoConfig
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
        from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Model
        expected={(row["module"],row["class"]):row["source_sha256"] for row in availability["runtime_class_authority"]}
        for cls in (SAM2Base,SAM2AutomaticMaskGenerator,CoTrackerPredictor,Dinov2Model,VJEPA2Model):
            path=Path(inspect.getsourcefile(cls)); self.assertEqual(runtime.authority.file_sha256(path),expected[(cls.__module__,cls.__name__)])
        dino=AutoConfig.from_pretrained(availability["foundations"]["dinov2"]["model_root"],local_files_only=True)
        vjepa=AutoConfig.from_pretrained(availability["foundations"]["vjepa2"]["model_root"],local_files_only=True)
        self.assertEqual(dino.model_type,"dinov2"); self.assertEqual(vjepa.model_type,"vjepa2"); self.assertEqual(vjepa.tubelet_size,2)

    def test_external_postflight_alone_creates_completion_seal(self):
        candidate=dict(runtime.run_canary(FakeFrozenBackend()))
        candidate["asset_closure"]={"verified":True,"digest":"a"*64}
        decode=runtime.authority.load_decode_receipt(); rows=[{"r1b_ordinal":row["r1b_ordinal"],"role":row["role"],"compressed_sha256":row["compressed_sha256"],"frame_count":row["frame_count"],"shape_hwc":[720,1280,3],"dtype":"uint8","decoded_rgb_sha256":row["decoded_rgb_sha256"]} for row in decode["rows"]]
        media={"verified":True,"rows":rows,"decode_receipt_file_sha256":runtime.authority.file_sha256(runtime.authority.DECODE_RECEIPT_PATH),"decode_receipt_self_sha256":decode["decode_receipt_self_sha256"]}; candidate["decoded_media_closure"]={**media,"digest":runtime.authority.object_sha256(media)}
        model={"verified":True,"device":{"visible_device_count":1}}; candidate["model_device_closure"]={**model,"digest":runtime.authority.object_sha256(model)}
        raw={"registered":2,"zeroized":2,"verified":True}; candidate["raw_ownership"]={**raw,"digest":runtime.authority.object_sha256(raw)}
        candidate.pop("digest"); candidate["digest"]=runtime.authority.object_sha256(candidate)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); candidate_path=root/"candidate.json"; seal_path=root/"completion.json"
            candidate_path.write_text(json.dumps(candidate,sort_keys=True),encoding="ascii")
            contract=runtime.launch_contract(); verifier=lambda:{"digest":"a"*64}; seal=postflight.seal_candidate(candidate_path,seal_path,contract["digest"],0,asset_verifier=verifier)
            self.assertTrue(seal["external_postflight_pass"]); self.assertTrue(seal_path.is_file())
            with self.assertRaises(runtime.RuntimeErrorV1): postflight.seal_candidate(candidate_path,seal_path,contract["digest"],0,asset_verifier=verifier)
            with self.assertRaises(postflight.PostflightV2Error): postflight.seal_candidate(candidate_path,root/"bad.json",contract["digest"],1,asset_verifier=verifier)


if __name__ == "__main__": unittest.main()
