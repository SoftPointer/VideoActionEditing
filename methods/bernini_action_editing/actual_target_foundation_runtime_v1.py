#!/usr/bin/env python3
"""Frozen one-device runtime for the seen actual-target foundation canary.

The real backend is lazy: importing and CPU contract testing never imports
torch, SAM2, CoTracker or transformers.  The only persistent artifacts are
scalar/digest receipts; masks, trajectories and embeddings remain in memory
and are scrubbed at each case boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Protocol, Sequence

import actual_target_foundation_canary_v1 as authority
import actual_target_foundation_graph_v2 as graph_v2


SCHEMA = "actual-target-foundation-runtime-v1"
REAL_GPU_LAUNCH_AUTHORIZED = False
PHASES = 8
SHUFFLE = (0, 2, 4, 6, 7, 5, 3, 1)
VIEWS = (
    "target_forward_reference",
    "target_forward_eval",
    "target_reverse",
    "target_deterministic_shuffle",
    "source_noop",
)
FORBIDDEN_RECEIPT_KEYS = {
    "masks",
    "mask_payload",
    "embeddings",
    "trajectories",
    "track_coordinates",
    "teacher_payload",
}


class RuntimeErrorV1(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise RuntimeErrorV1(message)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    ln = math.sqrt(sum(float(a) ** 2 for a in left))
    rn = math.sqrt(sum(float(b) ** 2 for b in right))
    if ln <= 1e-12 and rn <= 1e-12:
        return float("nan")
    if ln <= 1e-12 or rn <= 1e-12:
        return 0.0
    return dot / (ln * rn)


def _margin(reference: Sequence[float], positive: Sequence[float], control: Sequence[float]) -> float:
    return _cosine(reference, positive) - _cosine(reference, control)


def _permute_blocks(vector: Sequence[float], order: Sequence[int]) -> tuple[float, ...]:
    if len(vector) % PHASES:
        _fail("feature vector is not divisible into eight phase blocks")
    width = len(vector) // PHASES
    return tuple(float(vector[p * width + j]) for p in order for j in range(width))


def _rotate(vector: Sequence[float], offset: int = 1) -> tuple[float, ...]:
    values = tuple(float(v) for v in vector)
    if not values:
        return values
    offset %= len(values)
    return values[offset:] + values[:offset]


def _mask_descriptor_negative(nodes: "NodeSketch") -> tuple[float, ...]:
    if nodes.private_payload is None:
        return _rotate(nodes.signature)
    broken=graph_v2.break_mask_descriptor_binding(nodes.private_payload)
    if any(len(phase)<2 for phase in nodes.private_payload):
        return tuple()
    return graph_v2.canonical_node_signature(broken)


def _track_permutation(motion: "MotionSketch") -> tuple[float, ...]:
    width = motion.track_block_width
    if width <= 0 or motion.assigned_track_count<2 or len(motion.track_signature) != width * motion.track_count:
        return tuple()
    blocks=[list(motion.track_signature[index*width:(index+1)*width]) for index in range(motion.track_count)]
    descriptors=[block[:8] for block in blocks[:motion.assigned_track_count]]
    for index,block in enumerate(blocks[:motion.assigned_track_count]): block[:8]=descriptors[(index+1)%len(descriptors)]
    return tuple(value for block in blocks for value in block)


@dataclass(frozen=True)
class NodeSketch:
    signature: tuple[float, ...]
    cardinalities: tuple[int, ...]
    mechanically_valid_phases: int
    dustbin_used: bool
    slot_width: int = 1
    slots_per_phase: int = 1
    private_payload: Any = None
    unbalanced_phase_pair_count: int = 0
    dustbin_unmatched_count: int = 0
    dustbin_transport_mass: float = 0.0


@dataclass(frozen=True)
class MotionSketch:
    track_signature: tuple[float, ...]
    edge_signature: tuple[float, ...]
    visible_fraction: float
    dynamic_lifecycle_observed: bool
    track_block_width: int = 1
    track_count: int = 0
    assigned_track_count: int = 0
    pairwise_lifecycle_count: int = 0
    evaluated_pairwise_edge_count: int = 0


@dataclass(frozen=True)
class PhaseSketch:
    signature: tuple[float, ...]


class FrozenBackend(Protocol):
    model_names: Sequence[str]

    def decode(self, path: str, expected_sha256: str) -> Sequence[Any]: ...
    def node(self, frames: Sequence[Any], view: str) -> NodeSketch: ...
    def motion(self, frames: Sequence[Any], view: str, nodes: NodeSketch) -> MotionSketch: ...
    def phase(self, frames: Sequence[Any], view: str) -> PhaseSketch: ...
    def frozen_receipt(self) -> Mapping[str, Any]: ...
    def begin_case(self) -> None: ...
    def scrub_case(self) -> None: ...


class CountedBackend:
    """Counts logical foundation calls independently of backend batching."""

    def __init__(self, backend: FrozenBackend):
        self.backend = backend
        self.counts = {"media_decode": 0, "sam2": 0, "dinov2": 0, "cotracker": 0, "vjepa2": 0}

    def decode(self, path: str, digest: str) -> Sequence[Any]:
        self.counts["media_decode"] += 1
        return self.backend.decode(path, digest)

    def node(self, frames: Sequence[Any], view: str) -> NodeSketch:
        result = self.backend.node(frames, view)
        self.counts["sam2"] += PHASES
        self.counts["dinov2"] += PHASES
        return result

    def motion(self, frames: Sequence[Any], view: str, nodes: NodeSketch) -> MotionSketch:
        self.counts["cotracker"] += 1
        return self.backend.motion(frames, view, nodes)

    def phase(self, frames: Sequence[Any], view: str) -> PhaseSketch:
        self.counts["vjepa2"] += 1
        return self.backend.phase(frames, view)


def _sample(frames: Sequence[Any], count: int = PHASES) -> tuple[Any, ...]:
    if len(frames) < count:
        _fail(f"video has fewer than {count} decoded frames")
    indices = tuple(round(index * (len(frames) - 1) / (count - 1)) for index in range(count))
    if len(set(indices)) != count:
        _fail("fixed phase sampling produced duplicate indices")
    return tuple(frames[index] for index in indices)


def _views(source: Sequence[Any], target: Sequence[Any]) -> Mapping[str, tuple[Any, ...]]:
    source8, target16 = _sample(source), _sample(target,16)
    reference=target16[0::2]; evaluation=target16[1::2]
    return {
        "target_forward_reference": reference,
        "target_forward_eval": evaluation,
        "target_reverse": tuple(reversed(evaluation)),
        "target_deterministic_shuffle": tuple(evaluation[index] for index in SHUFFLE),
        "source_noop": source8,
    }


def _phase_views(source: Sequence[Any], target: Sequence[Any]) -> Mapping[str, tuple[Any, ...]]:
    source16,target32=_sample(source,16),_sample(target,32)
    reference=target32[0::2]; evaluation=target32[1::2]
    blocks=tuple(evaluation[index:index+2] for index in range(0,16,2))
    return {
        "target_forward_reference":reference,
        "target_forward_eval":evaluation,
        "target_reverse":tuple(reversed(evaluation)),
        "target_deterministic_shuffle":tuple(frame for index in SHUFFLE for frame in blocks[index]),
        "source_noop":source16,
    }


def _case_evidence(pair: Mapping[str, Any], backend: CountedBackend) -> authority.CaseEvidenceV1:
    source = backend.decode(pair["source_video_path"], pair["source_video_sha256"])
    target = backend.decode(pair["target_video_path"], pair["target_video_sha256"])
    views = _views(source, target)
    phase_views = _phase_views(source, target)
    # SAM2/DINO execute only on the three preregistered base sequences.  Reverse
    # and shuffle are exact temporal permutations of reference proposals.
    nodes = {
        name: backend.node(views[name], name)
        for name in ("target_forward_reference", "target_forward_eval", "source_noop")
    }
    reference_node = nodes["target_forward_eval"]
    nodes["target_reverse"] = NodeSketch(
        _permute_blocks(reference_node.signature, tuple(reversed(range(PHASES)))),
        tuple(reversed(reference_node.cardinalities)),
        reference_node.mechanically_valid_phases,
        reference_node.dustbin_used,
        reference_node.slot_width,
        reference_node.slots_per_phase,
        tuple(reversed(reference_node.private_payload)) if reference_node.private_payload is not None else None,
    )
    nodes["target_deterministic_shuffle"] = NodeSketch(
        _permute_blocks(reference_node.signature, SHUFFLE),
        tuple(reference_node.cardinalities[index] for index in SHUFFLE),
        reference_node.mechanically_valid_phases,
        reference_node.dustbin_used,
        reference_node.slot_width,
        reference_node.slots_per_phase,
        tuple(reference_node.private_payload[index] for index in SHUFFLE) if reference_node.private_payload is not None else None,
    )
    motions = {name: backend.motion(views[name], name, nodes[name]) for name in VIEWS}
    phases = {name: backend.phase(phase_views[name], name) for name in VIEWS}
    ref, pos = "target_forward_reference", "target_forward_eval"
    controls = {
        "target_reverse": "target_reverse",
        "target_deterministic_shuffle": "target_deterministic_shuffle",
        "source_noop": "source_noop",
    }
    node_margins = {key: _margin(nodes[ref].signature, nodes[pos].signature, nodes[name].signature) for key, name in controls.items()}
    track_margins = {key: _margin(motions[ref].track_signature, motions[pos].track_signature, motions[name].track_signature) for key, name in controls.items()}
    edge_margins = {key: _margin(motions[ref].edge_signature, motions[pos].edge_signature, motions[name].edge_signature) for key, name in controls.items()}
    phase_margins = {key: _margin(phases[ref].signature, phases[pos].signature, phases[name].signature) for key, name in controls.items()}
    frozen = backend.backend.frozen_receipt()
    evidence = authority.CaseEvidenceV1(
        family=pair["family"], pair_id=pair["pair_id"], branches={
            "frozen_base": {
                "all_models_eval_frozen": frozen.get("all_models_eval_frozen") is True,
                "source_and_weight_closure_unchanged": frozen.get("source_and_weight_closure_unchanged") is True,
                "parameter_updates": frozen.get("parameter_updates"),
                "generator_forward_calls": frozen.get("generator_forward_calls"),
            },
            "node": {
                "dustbin_used": nodes[pos].dustbin_used,
                "unbalanced_phase_pair_count": nodes[pos].unbalanced_phase_pair_count,
                "dustbin_unmatched_count": nodes[pos].dustbin_unmatched_count,
                "dustbin_transport_mass": nodes[pos].dustbin_transport_mass,
                "forced_nonempty_slot_used": False,
                "phase_cardinalities": list(nodes[pos].cardinalities),
                "mechanically_valid_phases": nodes[pos].mechanically_valid_phases,
                "positive_similarity": _cosine(nodes[ref].signature, nodes[pos].signature),
                "input_margins": node_margins,
                "mask_descriptor_binding_break_margin": _margin(nodes[ref].signature, nodes[pos].signature, _mask_descriptor_negative(nodes[pos])),
                "anonymous_slot_relabel_invariant": nodes[pos].private_payload is None or graph_v2.canonical_node_signature(graph_v2.relabel_slots(nodes[pos].private_payload)) == nodes[pos].signature,
            },
            "track": {
                "assigned_track_count": motions[pos].assigned_track_count,
                "visible_fraction": motions[pos].visible_fraction,
                "positive_similarity": _cosine(motions[ref].track_signature, motions[pos].track_signature),
                "input_margins": track_margins,
                "cross_phase_track_identity_break_margin": _margin(motions[ref].track_signature, motions[pos].track_signature, _track_permutation(motions[pos])),
            },
            "edge": {
                "dynamic_lifecycle_observed": motions[pos].dynamic_lifecycle_observed,
                "pairwise_lifecycle_count": motions[pos].pairwise_lifecycle_count,
                "evaluated_pairwise_edge_count": motions[pos].evaluated_pairwise_edge_count,
                "positive_similarity": _cosine(motions[ref].edge_signature, motions[pos].edge_signature),
                "input_margins": edge_margins,
                "drop_edge_margin": _margin(motions[ref].edge_signature, motions[pos].edge_signature, tuple(0.0 for _ in motions[pos].edge_signature)),
            },
            "ordered_phase": {"input_margins": phase_margins},
        },
    )
    nodes.clear()
    motions.clear()
    phases.clear()
    return evidence


def _safe_receipt(value: Mapping[str, Any]) -> None:
    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if FORBIDDEN_RECEIPT_KEYS.intersection(node):
                _fail("receipt attempts to persist raw teacher payload")
            for child in node.values(): walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node: walk(child)
    walk(value)
    authority.canonical_json_bytes(value)


def _create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        _fail("output/cache receipt must be an absolute absent path")
    _safe_receipt(value)
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    except BaseException:
        try: path.unlink()
        except OSError: pass
        raise


def run_canary(backend: FrozenBackend, *, output: Optional[Path] = None, cache_dir: Optional[Path] = None) -> Mapping[str, Any]:
    prereg = authority.load_preregistration()
    counted = CountedBackend(backend)
    rows = []
    for pair in prereg["pairs"]:
        begin=getattr(backend,"begin_case",None)
        if begin is not None: begin()
        try:
            evaluated = authority.evaluate_case(_case_evidence(pair, counted), prereg)
        finally:
            backend.scrub_case()
        rows.append(evaluated)
        if cache_dir is not None:
            if not cache_dir.is_absolute() or cache_dir.is_symlink() or not cache_dir.is_dir():
                _fail("cache directory must be absolute existing non-symlink directory")
            cache_value = {"schema_version": SCHEMA, "kind": "derived_scalar_case_receipt", "pair_id": pair["pair_id"], "evaluation_digest": evaluated["digest"], "raw_teacher_payload_persisted": False}
            _create_only_json(cache_dir / f'{pair["pair_id"]}.json', {**cache_value, "digest": authority.object_sha256(cache_value)})
    expected = {"media_decode": 8, "sam2": 96, "dinov2": 96, "cotracker": 20, "vjepa2": 20}
    if counted.counts != expected:
        _fail(f"logical forward counts differ: {counted.counts}")
    aggregate = authority.aggregate_canary(rows)
    value = {
        "schema_version": SCHEMA,
        "experiment_id": authority.EXPERIMENT_ID,
        "scope": "seen_development_only_not_locked_validation",
        "cases": rows,
        "aggregate": aggregate,
        "logical_forward_counts": counted.counts,
        "training_performed": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "generator_loaded": False,
        "generator_forward_calls": 0,
        "raw_teacher_payload_persisted": False,
        "representation_admission_hard_false": True,
        "raw_ownership": dict(getattr(backend,"raw_ownership_receipt",lambda:{"registered":0,"zeroized":0,"verified":True})()),
        "runtime_source_closure": source_closure(),
        "asset_closure": dict(getattr(backend,"asset_closure_receipt",lambda:{"mode":"fake_cpu_contract","verified":True})()),
        "decoded_media_closure": dict(getattr(backend,"decoded_media_receipt",lambda:{"mode":"fake_cpu_contract","verified":True})()),
        "model_device_closure": dict(getattr(backend,"model_device_receipt",lambda:{"mode":"fake_cpu_contract","verified":True})()),
        "completion_authority": {"probe_output_kind":"candidate_only","candidate_file_presence_is_completion_authority":False,"external_completion_seal_required":True,"external_completion_seal_written_by_probe":False},
        "launch_contract_digest": launch_contract()["digest"],
    }
    receipt = {**value, "digest": authority.object_sha256(value)}
    _safe_receipt(receipt)
    if output is not None: _create_only_json(output, receipt)
    return receipt


class RealFrozenBackend:
    """Lazy AUH backend; construction is the only place foundation code loads."""

    model_names = ("sam2", "cotracker", "dinov2", "vjepa2")

    def __init__(self, device: str = "cuda:0"):
        if device != "cuda:0": _fail("real canary is sealed to exactly one GPU cuda:0")
        self._asset_closure_digest = authority.verify_remote_assets()["digest"]
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoVideoProcessor
        availability = authority.load_availability()["foundations"]
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        cot_root = availability["cotracker"]["repository_root"]
        if cot_root not in sys.path: sys.path.insert(0, cot_root)
        from cotracker.predictor import CoTrackerPredictor
        self.torch, self.device = torch, device
        sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", availability["sam2"]["checkpoint_path"], device=device, mode="eval")
        self.sam = SAM2AutomaticMaskGenerator(sam_model, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.90, output_mode="binary_mask")
        self.cotracker = CoTrackerPredictor(checkpoint=availability["cotracker"]["checkpoint_path"], offline=True).to(device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(availability["dinov2"]["model_root"], local_files_only=True)
        self.dino = AutoModel.from_pretrained(availability["dinov2"]["model_root"], local_files_only=True).to(device).eval()
        self.vjepa_processor = AutoVideoProcessor.from_pretrained(availability["vjepa2"]["model_root"], local_files_only=True)
        self.vjepa = AutoModel.from_pretrained(availability["vjepa2"]["model_root"], local_files_only=True).to(device).eval()
        self.models = (sam_model, self.cotracker, self.dino, self.vjepa)
        for model in self.models:
            model.eval()
            for parameter in model.parameters(): parameter.requires_grad_(False)
        self._versions = tuple(tuple(parameter._version for parameter in model.parameters()) for model in self.models)
        self._owned_raw=[]; self._raw_registered=0; self._raw_zeroized=0
        self._decoded_rows=[]
        self._model_device_binding=self._binding_receipt()

    def _binding_receipt(self) -> Mapping[str, Any]:
        expected={(row["module"],row["class"]):row["source_sha256"] for row in authority.load_availability()["runtime_class_authority"]}
        classes=(self.models[0].__class__,self.sam.__class__,self.cotracker.__class__,self.dino.__class__,self.vjepa.__class__)
        rows=[]
        for cls in classes:
            key=(cls.__module__,cls.__name__); path=Path(inspect.getsourcefile(cls) or "")
            if key not in expected or not path.is_absolute() or path.is_symlink() or path.resolve(strict=True)!=path or authority.file_sha256(path)!=expected[key]: _fail(f"foundation class/source binding differs: {key}")
            rows.append({"module":key[0],"class":key[1],"source_path":str(path),"source_sha256":expected[key]})
        if self.dino.config.model_type!="dinov2" or self.vjepa.config.model_type!="vjepa2" or int(self.vjepa.config.tubelet_size)!=2: _fail("foundation model config binding differs")
        if self.torch.cuda.device_count()!=1 or self.torch.device(self.device).index!=0: _fail("exact one-GPU device binding differs")
        device={"type":"cuda","index":0,"name":self.torch.cuda.get_device_name(0),"visible_device_count":1}
        value={"verified":True,"classes":rows,"configs":{"dinov2_model_type":"dinov2","vjepa2_model_type":"vjepa2","vjepa2_tubelet_size":2},"device":device}
        return {**value,"digest":authority.object_sha256(value)}

    def begin_case(self) -> None:
        if self._owned_raw: _fail("raw ownership from prior case was not scrubbed")

    def _own(self, value: Any) -> Any:
        self._owned_raw.append(value); self._raw_registered+=1; return value

    def _release_owned(self, value: Any) -> None:
        index=next((i for i,item in enumerate(self._owned_raw) if item is value),None)
        if index is None: _fail("raw ownership release is not single-owner")
        try:
            if hasattr(value,"zero_"): value.zero_(); clean=not bool(value.any())
            else: value[...] = 0; clean=not bool(value.any())
        except BaseException as error: raise RuntimeErrorV1("raw ownership immediate zeroization failed") from error
        if not clean: _fail("raw ownership immediate zeroization did not clear payload")
        self._owned_raw.pop(index); self._raw_zeroized+=1

    def decode(self, path: str, expected_sha256: str) -> Sequence[Any]:
        import cv2, numpy as np
        candidate = Path(path)
        if not candidate.is_absolute() or candidate.is_symlink() or candidate.resolve(strict=True)!=candidate or not candidate.is_file(): _fail("media path is not canonical absolute plain file")
        before=candidate.stat()
        if authority.file_sha256(candidate) != expected_sha256: _fail("media SHA differs")
        decode_row=next((row for row in authority.load_decode_receipt()["rows"] if row["compressed_sha256"]==expected_sha256),None)
        if decode_row is None: _fail("media is absent from decoded RGB authority")
        digest=hashlib.sha256(authority.canonical_json_bytes({"dtype":"uint8","shape":[decode_row["frame_count"],720,1280,3]}))
        capture = cv2.VideoCapture(path); frames = []
        while True:
            ok, frame = capture.read()
            if not ok: break
            rgb=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if rgb.dtype!=np.uint8 or rgb.shape!=(720,1280,3): _fail("decoded RGB dtype/shape differs")
            digest.update(rgb.tobytes(order="C")); frames.append(rgb)
        capture.release()
        after=candidate.stat(); identity=lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
        if identity(before)!=identity(after): _fail("media changed during decode")
        if len(frames)!=decode_row["frame_count"] or digest.hexdigest()!=decode_row["decoded_rgb_sha256"]: _fail("decoded RGB closure differs")
        self._decoded_rows.append({"r1b_ordinal":decode_row["r1b_ordinal"],"role":decode_row["role"],"compressed_sha256":expected_sha256,"frame_count":len(frames),"shape_hwc":[720,1280,3],"dtype":"uint8","decoded_rgb_sha256":digest.hexdigest()})
        return frames

    def node(self, frames: Sequence[Any], view: str) -> NodeSketch:
        torch = self.torch; import torch.nn.functional as F
        cards=[]; valid=0; payload=[]
        with torch.inference_mode():
            for frame in frames:
                anns=self.sam.generate(frame)
                anns=[a for a in anns if 0.001 <= a["area"]/(frame.shape[0]*frame.shape[1]) <= 0.45]
                anns=sorted(anns,key=lambda a:(-a["predicted_iou"],-a["stability_score"],-a["area"],a["bbox"]))[:12]
                inputs=self.dino_processor(images=frame,return_tensors="pt"); inputs={k:self._own(v.to(self.device)) for k,v in inputs.items()}
                try: tokens=self._own(self.dino(**inputs).last_hidden_state[:,1:,:])
                finally:
                    for value in inputs.values(): self._release_owned(value)
                side=int(math.isqrt(tokens.shape[1]))
                pooled=[]; valid_anns=[]
                try:
                    if side*side != tokens.shape[1]: _fail("DINO patch geometry is not square")
                    for ann in anns:
                        self._own(ann["segmentation"])
                        mask=torch.as_tensor(ann["segmentation"],device=self.device,dtype=torch.float32)[None,None]
                        height, width = mask.shape[-2:]
                        if height <= width:
                            resized = (256, round(width * 256 / height))
                        else:
                            resized = (round(height * 256 / width), 256)
                        mask = F.interpolate(mask, size=resized, mode="nearest")
                        top = (resized[0] - 224) // 2
                        left = (resized[1] - 224) // 2
                        mask = mask[:, :, top : top + 224, left : left + 224]
                        mask=self._own(F.interpolate(mask,size=(side,side),mode="area").flatten())
                        support=mask.sum()
                        if not bool(torch.isfinite(support)) or float(support)<=1e-6:
                            self._release_owned(mask); continue
                        pooled.append(self._own((tokens[0]*mask[:,None]).sum(0)/support))
                        valid_anns.append(ann); self._release_owned(mask)
                    cards.append(len(pooled)); valid += bool(pooled)
                    descriptors = [item[:8].float().cpu().tolist() for item in pooled]
                    for item in pooled: self._release_owned(item)
                finally:
                    if any(item is tokens for item in self._owned_raw): self._release_owned(tokens)
                phase_nodes=[]
                for ann,descriptor in zip(valid_anns,descriptors):
                    raw=ann["segmentation"]; ys,xs=raw.nonzero()
                    if not len(xs): continue
                    phase_nodes.append(graph_v2.AnonymousNodeV2(raw,tuple(descriptor),float(raw.mean()),(float(xs.mean()/max(raw.shape[1]-1,1)),float(ys.mean()/max(raw.shape[0]-1,1)))))
                payload.append(tuple(phase_nodes))
        tracked=graph_v2.assign_anonymous_tracks(payload)
        signature=graph_v2.canonical_node_signature(tracked)
        diagnostics=graph_v2.unbalanced_matching_diagnostics(payload)
        return NodeSketch(signature,tuple(cards),int(valid),True,11,12,tracked,int(diagnostics["phase_pair_count"]),int(diagnostics["unmatched_count"]),float(diagnostics["dustbin_transport_mass"]))

    def motion(self, frames: Sequence[Any], view: str, nodes: NodeSketch) -> MotionSketch:
        torch=self.torch
        if nodes.private_payload is None or len(nodes.private_payload) != PHASES:
            _fail("CoTracker requires in-memory automatic mask/DINO payload")
        video=self._own(torch.stack([torch.as_tensor(frame).permute(2,0,1) for frame in frames]).float()[None].to(self.device))
        try:
            with torch.inference_mode(): tracks,visible=self.cotracker(video,grid_size=12,grid_query_frame=0,backward_tracking=True)
            tracks=self._own(tracks); visible=self._own(visible)
            xy=self._own(tracks[0].float().cpu()); vis=self._own(visible[0].bool().cpu())
            self._release_owned(tracks); self._release_owned(visible)
        finally:
            if any(item is video for item in self._owned_raw): self._release_owned(video)
        groups={}
        for point in range(xy.shape[1]):
            identity=None
            for phase in range(PHASES):
                if not bool(vis[phase,point]): continue
                x=int(round(float(xy[phase,point,0]))); y=int(round(float(xy[phase,point,1])))
                for node in nodes.private_payload[phase]:
                    mask=node.mask
                    y0=max(0,min(y,mask.shape[0]-1)); x0=max(0,min(x,mask.shape[1]-1))
                    if bool(mask[y0,x0]): identity=(node.track_id,node.descriptor); break
                if identity is not None: break
            if identity is not None: groups.setdefault(identity[0],{"descriptor":identity[1],"points":[]})["points"].append(point)
        ordered=sorted(groups.items(),key=lambda item:hashlib.sha256(repr(tuple(round(float(x),8) for x in item[1]["descriptor"])).encode("ascii")).digest())[:12]
        track_signature=[]; centers=[]
        velocities={}
        for track_id,group in ordered:
            indices=group["points"]; group_xy=xy[:,indices].mean(1); group_vis=vis[:,indices].float().mean(1)
            velocity=(group_xy[1:]-group_xy[:-1]).mean(0)
            block=list(group["descriptor"])+velocity.tolist()+[float(group_vis.mean()),float((group_vis>0.5).sum()/PHASES)]
            track_signature.extend(block); centers.append((track_id,group_xy,group_vis))
            for phase in range(PHASES):
                if phase:
                    delta=group_xy[phase]-group_xy[phase-1]
                    velocities[(phase,track_id)]=(float(delta[0]),float(delta[1]))
                else: velocities[(phase,track_id)]=(0.0,0.0)
        track_signature.extend([0.0]*((12-len(ordered))*12))
        assigned_ids={track_id for track_id,_ in ordered}
        assigned_phases=tuple(tuple(node for node in phase if node.track_id in assigned_ids) for phase in nodes.private_payload)
        edge,lifecycle_count=graph_v2.pairwise_edge_signature(assigned_phases,velocities)
        assigned_points=sum(len(group["points"]) for _,group in ordered)
        assigned_visible=sum(float(vis[:,group["points"]].float().sum()) for _,group in ordered)
        denominator=PHASES*assigned_points
        assigned_visibility=assigned_visible/denominator if denominator else float("nan")
        pair_count=int(sum(edge[index] for index in range(3,PHASES*4,4)))
        return MotionSketch(tuple(track_signature),edge,float(assigned_visibility),lifecycle_count>0,12,12,len(ordered),lifecycle_count,pair_count)

    def phase(self, frames: Sequence[Any], view: str) -> PhaseSketch:
        if len(frames)!=16 or int(self.vjepa.config.tubelet_size)!=2:
            _fail("V-JEPA branch requires sixteen frames and tubelet_size=2")
        with self.torch.inference_mode():
            inputs=self.vjepa_processor(videos=[list(frames)],return_tensors="pt"); inputs={k:self._own(v.to(self.device)) for k,v in inputs.items()}
            try: hidden=self._own(self.vjepa(**inputs).last_hidden_state[0].float())
            finally:
                for value in inputs.values(): self._release_owned(value)
            spatial=(int(self.vjepa.config.image_size)//int(self.vjepa.config.patch_size))**2
            if hidden.shape[0] != PHASES*spatial:
                _fail("V-JEPA output does not contain exactly eight real temporal tubelet blocks")
            blocks=hidden.reshape(PHASES,spatial,hidden.shape[-1])
            signature=blocks.mean(1)[:,:16].flatten().cpu().tolist()
            self._release_owned(hidden)
        return PhaseSketch(tuple(signature))

    def frozen_receipt(self) -> Mapping[str, Any]:
        current=tuple(tuple(parameter._version for parameter in model.parameters()) for model in self.models)
        closure_unchanged = (
            authority.verify_remote_assets()["digest"]
            == self._asset_closure_digest
            and self._binding_receipt()["digest"]==self._model_device_binding["digest"]
        )
        return {"all_models_eval_frozen":all(not model.training and all(not p.requires_grad for p in model.parameters()) for model in self.models),"source_and_weight_closure_unchanged":current==self._versions and closure_unchanged,"parameter_updates":0,"generator_forward_calls":0}

    def scrub_case(self) -> None:
        failures=[]
        for value in self._owned_raw:
            try:
                if hasattr(value,"zero_"): value.zero_(); clean=not bool(value.any())
                else: value[...] = 0; clean=not bool(value.any())
                if not clean: failures.append(type(value).__name__)
                else: self._raw_zeroized+=1
            except BaseException: failures.append(type(value).__name__)
        self._owned_raw.clear()
        if self.torch.cuda.is_available(): self.torch.cuda.empty_cache()
        if failures: _fail(f"raw ownership zeroization failed: {failures}")

    def raw_ownership_receipt(self) -> Mapping[str, Any]:
        value={"registered":self._raw_registered,"zeroized":self._raw_zeroized,"verified":self._raw_registered==self._raw_zeroized and not self._owned_raw}
        return {**value,"digest":authority.object_sha256(value)}

    def model_device_receipt(self) -> Mapping[str, Any]:
        return self._model_device_binding

    def asset_closure_receipt(self) -> Mapping[str, Any]:
        return {"digest":self._asset_closure_digest,"verified":True,"verified_file_count":22}

    def decoded_media_receipt(self) -> Mapping[str, Any]:
        value={"verified":len(self._decoded_rows)==8,"decode_receipt_file_sha256":authority.file_sha256(authority.DECODE_RECEIPT_PATH),"decode_receipt_self_sha256":authority.load_decode_receipt()["decode_receipt_self_sha256"],"rows":list(self._decoded_rows)}
        return {**value,"digest":authority.object_sha256(value)}


def source_closure() -> Mapping[str, Any]:
    paths=(Path(__file__).resolve(),Path(authority.__file__).resolve(),Path(graph_v2.__file__).resolve(),Path(__file__).resolve().parent/"actual_target_foundation_postflight_v2.py",authority.PREREG_PATH,authority.AVAILABILITY_PATH,authority.DECODE_RECEIPT_PATH,Path(__file__).resolve().parent/"scripts"/"auh_actual_target_foundation_canary_rank_wrapper_v1.sh",Path(__file__).resolve().parent/"tests"/"test_actual_target_foundation_canary_v1.py",Path(__file__).resolve().parent/"tests"/"test_actual_target_foundation_runtime_v1.py")
    rows=[]
    for path in paths:
        if not path.is_absolute() or not path.is_file() or path.is_symlink() or path.resolve(strict=True)!=path: _fail(f"source closure differs: {path}")
        rows.append({"path":str(path),"sha256":authority.file_sha256(path)})
    value={"file_count":len(rows),"files":rows}
    return {**value,"digest":authority.object_sha256(value)}


def launch_contract() -> Mapping[str, Any]:
    value={"schema_version":"actual-target-foundation-runtime-v2-closure","implementation_status":"V2_CLOSURE_IMPLEMENTED_UNEXECUTED_PRE_FLIP_NO","real_gpu_launch_authorized":REAL_GPU_LAUNCH_AUTHORIZED,"independent_audit_required_before_gpu":True,"device":"exactly cuda:0 / one externally isolated MI210","source_closure":source_closure(),"output":"absolute absent create-only scalar/digest candidate JSON","cache":"optional absolute directory; one absent create-only scalar/digest JSON per case","raw_teacher_payload_persisted":False,"training_performed":False,"generator_loaded":False,"command":"auh_actual_target_foundation_canary_rank_wrapper_v1.sh ABSENT_OUTPUT_JSON OPTIONAL_CACHE_DIR","completion":"candidate is non-authoritative; after rank-wrapper/Slurm exit zero an external CPU postflight must verify all closures and create an absent create-only completion seal"}
    return {**value,"digest":authority.object_sha256(value)}


def main(argv: Optional[Sequence[str]]=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print-contract",action="store_true"); group.add_argument("--run-real",action="store_true")
    parser.add_argument("--output",type=Path); parser.add_argument("--cache-dir",type=Path)
    args=parser.parse_args(argv)
    if args.print_contract:
        print(json.dumps(launch_contract(),indent=2,sort_keys=True)); return 0
    if not REAL_GPU_LAUNCH_AUTHORIZED: _fail("real GPU launch blocked pending a new independent audit")
    if args.output is None: _fail("real run requires --output")
    run_canary(RealFrozenBackend(),output=args.output,cache_dir=args.cache_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
