#!/usr/bin/env python3
"""CPU-only V8 anonymous tube/graph reducer and LOAO diagnostic.

The module consumes already captured projected intermediates.  It has no model,
GPU, generator, renderer, decoder, optimizer, training, or routing entry point.
All returned objects contain scalar values and digests only.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import math
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

import self_generated_loao_action_registry_v8 as registry


SCHEMA_VERSION = "bernini-self-generated-loao-action-tube-reducer-v8"
BRANCHES = ("A_to_B", "B_to_A")
PROPOSAL_BLOCKS = {"A_to_B": (6,18), "B_to_A": (12,24)}
EVALUATION_BLOCKS = {"A_to_B": (12,24), "B_to_A": (6,18)}
PROPOSAL_PHASES = {
    "A_to_B": (0, 2, 4, 6, 8, 10, 12, 14, 16, 18),
    "B_to_A": (1, 3, 5, 7, 9, 11, 13, 15, 17, 19),
}
EVALUATION_PHASES = {
    "A_to_B": (1, 3, 5, 7, 9, 11, 13, 15, 17, 19),
    "B_to_A": (2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
}
PROPOSAL_SHUFFLES = {
    "A_to_B": (0, 4, 8, 12, 16, 18, 14, 10, 6, 2),
    "B_to_A": (1, 5, 9, 13, 17, 19, 15, 11, 7, 3),
}
SEED_DIRECTIONS = {
    "D_to_H": (registry.SEED_IDS[0], registry.SEED_IDS[1]),
    "H_to_D": (registry.SEED_IDS[1], registry.SEED_IDS[0]),
}
TUBE_SHAPE = (registry.PHASES, registry.PATCH_HEIGHT, registry.PATCH_WIDTH)

REPRESENTATION_ADMISSION_HARD_FALSE = True
SCIENTIFIC_CLAIM_AUTHORIZED = False
TRAINING_AUTHORIZED = False
DECODER_AUTHORIZED = False
ROUTE_AUTHORIZED = False
GPU_AUTHORIZED = False


class ReducerV8Error(RuntimeError):
    pass


def tensor_sha256_v8(value: np.ndarray) -> str:
    array = np.asarray(value)
    header = registry.canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest = hashlib.sha256(header)
    digest.update(np.ascontiguousarray(array).view(np.uint8).tobytes())
    return digest.hexdigest()


def proposal_only_phase_shuffle_v8(value: np.ndarray, branch: str) -> np.ndarray:
    """Shuffle only a branch's ten proposal-source phases; eval is bit-exact."""

    array = np.asarray(value)
    if array.shape[:3] != TUBE_SHAPE or branch not in BRANCHES:
        raise ReducerV8Error("proposal-only shuffle tensor/branch differs")
    proposal = PROPOSAL_PHASES[branch]
    shuffled = PROPOSAL_SHUFFLES[branch]
    evaluation = EVALUATION_PHASES[branch]
    if set(proposal) != set(shuffled) or set(proposal) & set(evaluation):
        raise ReducerV8Error("proposal-only phase bijection differs")
    output = array.copy()
    for destination, source in zip(proposal, shuffled):
        output[destination] = array[source]
    if not np.array_equal(output[list(evaluation)], array[list(evaluation)], equal_nan=True):
        raise ReducerV8Error("phase shuffle leaked into evaluation phases")
    return output


def _is_owned_leaf(value: Any) -> bool:
    return isinstance(value, np.ndarray) or callable(getattr(value, "zero_", None))


@dataclass(frozen=True)
class _OwnershipDiscoveryV8:
    leaves: tuple[tuple[str, Any], ...]
    repeated_containers: tuple[tuple[str, str], ...]


def _discover_ownership_v8(root: Any) -> _OwnershipDiscoveryV8:
    """Discover leaves without hiding shared/cyclic container ownership."""

    output: list[tuple[str, Any]] = []
    first_container_path: dict[int, str] = {}
    repeated_containers: list[tuple[str, str]] = []

    def walk(value: Any, path: str) -> None:
        # Leaves are recorded on every path so repeated objects and storage
        # aliases remain visible to reject_aliased_ownership_v8.
        if _is_owned_leaf(value):
            output.append((path, value))
            return
        if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
            return
        identity = id(value)
        if identity in first_container_path:
            repeated_containers.append((first_container_path[identity], path))
            return
        first_container_path[identity] = path
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif is_dataclass(value):
            for field in fields(value):
                walk(getattr(value, field.name), f"{path}.{field.name}")
        else:
            if hasattr(value, "__dict__"):
                for name, child in vars(value).items():
                    walk(child, f"{path}.{name}")
            if hasattr(value, "__slots__"):
                slots = value.__slots__
                names=(slots,) if isinstance(slots,str) else tuple(slots)
                for name in names:
                    if hasattr(value, name):
                        walk(getattr(value, name), f"{path}.{name}")

    walk(root, "root")
    return _OwnershipDiscoveryV8(tuple(output), tuple(repeated_containers))


def discover_owned_values_v8(root: Any) -> tuple[tuple[str, Any], ...]:
    """Recursively discover raw arrays/tensors through mappings and odd nesting."""
    return _discover_ownership_v8(root).leaves


def _shares_storage(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.shares_memory(left, right))
    for candidate in ("untyped_storage", "storage"):
        lmethod, rmethod = getattr(left, candidate, None), getattr(right, candidate, None)
        if callable(lmethod) and callable(rmethod):
            try:
                return int(lmethod().data_ptr()) == int(rmethod().data_ptr())
            except Exception:
                pass
    return False


def reject_aliased_ownership_v8(root: Any) -> tuple[tuple[str, Any], ...]:
    discovery = _discover_ownership_v8(root)
    if discovery.repeated_containers:
        first, repeated = discovery.repeated_containers[0]
        raise ReducerV8Error(
            f"raw capture ownership repeats container: {first} and {repeated}"
        )
    leaves = discovery.leaves
    for index, (left_path, left) in enumerate(leaves):
        for right_path, right in leaves[index + 1 :]:
            if _shares_storage(left, right):
                raise ReducerV8Error(
                    f"raw capture ownership aliases: {left_path} and {right_path}"
                )
    return leaves


def best_effort_scrub_v8(root: Any) -> Mapping[str, Any]:
    """Attempt every zeroization even when an earlier owner raises."""

    discovery = _discover_ownership_v8(root)
    # Shared paths are invalid authority, but scrubbing still attempts every
    # unique leaf exactly once after any alias rejection or zeroizer failure.
    leaves = []
    seen_leaf_objects: set[int] = set()
    for path, value in discovery.leaves:
        if id(value) not in seen_leaf_objects:
            seen_leaf_objects.add(id(value)); leaves.append((path, value))
    failures: list[str] = []
    zeroized = 0
    for path, value in leaves:
        try:
            if isinstance(value, np.ndarray):
                value[...] = 0
                clean = not bool(np.any(value))
            else:
                value.zero_()
                probe = getattr(value, "any", None)
                if not callable(probe):
                    failures.append(f"{path}:unverifiable_zero")
                    continue
                clean = not bool(probe())
            if not clean:
                failures.append(f"{path}:not_zero")
            else:
                zeroized += 1
        except BaseException as error:
            failures.append(f"{path}:{type(error).__name__}")
    value = {
        "discovered": len(leaves),
        "repeated_container_count": len(discovery.repeated_containers),
        "zeroized": zeroized,
        "failures": tuple(failures),
        "best_effort_completed": True,
        "verified": not failures and zeroized == len(leaves),
    }
    return {**value, "digest": registry.object_sha256(value)}


@dataclass(frozen=True)
class ComponentV8:
    proposal_support_id: int
    phase: int
    mask: np.ndarray
    centroid_xy: tuple[float, float]
    soft_mass: float
    descriptor: tuple[float, ...]


@dataclass(frozen=True)
class TrackedNodeV8:
    component: ComponentV8
    track_id: int


@dataclass(frozen=True)
class GraphSummaryV8:
    signature: tuple[float, ...]
    support: tuple[tuple[int, ...], ...]
    component_count: int
    qualified_track_count: int
    track_coverage: float
    dynamic_edge_lifecycle_count: int
    raw_dustbin_assignment_count: int
    qualified_dustbin_assignment_count: int
    raw_proposal_support_count: int
    lifecycle_counts: Mapping[str, int]
    endpoint_direction_xy: tuple[float, float]
    vertical_velocity_early: float
    vertical_velocity_late: float
    signed_winding_area: float
    path_acceleration: float
    neutral_correspondence_valid: bool
    neutral_visual_cosine_margin: float
    neutral_top_vs_median_margin: float
    neutral_top10_mass_fraction: float
    neutral_nonpositional_std: float

    def scalar_receipt(self) -> Mapping[str, Any]:
        value = {
            "signature_sha256": registry.object_sha256(list(self.signature)),
            "signature_width": len(self.signature),
            "component_count": self.component_count,
            "qualified_track_count": self.qualified_track_count,
            "track_coverage": self.track_coverage,
            "dynamic_edge_lifecycle_count": self.dynamic_edge_lifecycle_count,
            "raw_dustbin_assignment_count": self.raw_dustbin_assignment_count,
            "qualified_dustbin_assignment_count": self.qualified_dustbin_assignment_count,
            "raw_proposal_support_count": self.raw_proposal_support_count,
            "lifecycle_counts": dict(self.lifecycle_counts),
            "endpoint_direction_xy": list(self.endpoint_direction_xy),
            "vertical_velocity_early": self.vertical_velocity_early,
            "vertical_velocity_late": self.vertical_velocity_late,
            "signed_winding_area": self.signed_winding_area,
            "path_acceleration": self.path_acceleration,
            "neutral_correspondence_valid": self.neutral_correspondence_valid,
            "neutral_visual_cosine_margin": self.neutral_visual_cosine_margin,
            "neutral_top_vs_median_margin": self.neutral_top_vs_median_margin,
            "neutral_top10_mass_fraction": self.neutral_top10_mass_fraction,
            "neutral_nonpositional_std": self.neutral_nonpositional_std,
        }
        return {**value, "digest": registry.object_sha256(value)}


def _four_connected(mask: np.ndarray) -> list[np.ndarray]:
    pending = set(map(tuple, np.argwhere(mask)))
    groups: list[np.ndarray] = []
    while pending:
        seed = pending.pop(); stack = [seed]; group = [seed]
        while stack:
            y, x = stack.pop()
            for item in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if item in pending:
                    pending.remove(item); stack.append(item); group.append(item)
        result = np.zeros(mask.shape, dtype=bool)
        ys, xs = zip(*group); result[np.asarray(ys), np.asarray(xs)] = True
        groups.append(result)
    return groups


def _joint_support_components(
    proposal: np.ndarray, neutral: np.ndarray, branch: str
) -> tuple[tuple[tuple[ComponentV8, ...], ...], int, tuple[tuple[float, float, float], ...]]:
    """Propose joint supports, then evaluate each slice with neutral tokens.

    A proposal-support id is deliberately never copied into a track id.  It
    only records which connected action-residual support proposed a slice;
    neutral visual correspondence and dustbin Sinkhorn establish tracking.
    """

    sources=PROPOSAL_PHASES[branch]; evaluations=EVALUATION_PHASES[branch]
    width=max(int(proposal.shape[-1]),1)
    energy=np.stack([np.linalg.norm(proposal[phase].astype(np.float64),axis=-1)/math.sqrt(width) for phase in sources])
    active=np.zeros(energy.shape,dtype=bool); soft=np.zeros(energy.shape,dtype=np.float64)
    for index,frame in enumerate(energy):
        if float(frame.max())<1e-4:
            continue
        top_count=max(1,int(math.ceil(frame.size*0.1)))
        concentration=float(np.partition(frame.reshape(-1),-top_count)[-top_count:].sum())/max(float(frame.sum()),1e-12)
        if concentration<0.2:
            continue
        median=float(np.median(frame)); mad=float(np.median(np.abs(frame-median))); scale=max(1.4826*mad,1e-8)
        z=(frame-median)/scale; active[index]=(z>=2.5)&(frame>=1e-4)
        soft[index]=1.0/(1.0+np.exp(-np.clip((z-2.5)/0.5,-30.0,30.0)))
    pending=set(map(tuple,np.argwhere(active))); tubes=[]
    while pending:
        seed=pending.pop(); stack=[seed]; group=[seed]
        while stack:
            t,y,x=stack.pop(); neighbors=[]
            neighbors.extend(((t,y-1,x),(t,y+1,x),(t,y,x-1),(t,y,x+1)))
            for dt in (-1,1):
                for dy in (-1,0,1):
                    for dx in (-1,0,1): neighbors.append((t+dt,y+dy,x+dx))
            for item in neighbors:
                if item in pending: pending.remove(item); stack.append(item); group.append(item)
        total=float(sum(soft[item] for item in group))
        if total>=1.5 and len(group)>=2: tubes.append((total,group))
    tubes.sort(key=lambda row:-row[0]); raw_support_count=len(tubes); tubes=tubes[:64]
    phases=[[] for _ in evaluations]
    diagnostics: list[tuple[float, float, float]]=[]
    nonpositional_std=_neutral_nonpositional_std_v8(neutral)
    for support_id,(_,group) in enumerate(tubes):
        for time_index,evaluation_phase in enumerate(evaluations):
            coords=[(y,x) for t,y,x in group if t==time_index]
            if len(coords)<2: continue
            mask=np.zeros((registry.PATCH_HEIGHT,registry.PATCH_WIDTH),dtype=bool)
            ys=np.asarray([item[0] for item in coords]); xs=np.asarray([item[1] for item in coords]); mask[ys,xs]=True
            mass=float(sum(soft[time_index,y,x] for y,x in coords))
            if mass<1.5: continue
            if nonpositional_std<1e-6:
                continue
            evaluated=_evaluate_neutral_component_v8(
                mask,
                soft[time_index],
                neutral[sources[time_index]],
                neutral[evaluation_phase],
            )
            if evaluated is None:
                continue
            evaluated_mask,centroid,descriptor,evaluated_mass,visual_margin,spatial_margin,top10=evaluated
            diagnostics.append((visual_margin,spatial_margin,top10))
            phases[time_index].append(
                ComponentV8(
                    support_id,
                    evaluation_phase,
                    evaluated_mask,
                    centroid,
                    evaluated_mass,
                    descriptor,
                )
            )
    return (
        tuple(tuple(sorted(rows,key=lambda row:(-row.soft_mass,row.centroid_xy))[:8]) for rows in phases),
        raw_support_count,
        tuple(diagnostics),
    )


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        return 1.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - float(a @ b / denominator) if denominator > 1e-12 else 1.0


def _normalized_grid_v8() -> np.ndarray:
    y=np.linspace(-1.0,1.0,registry.PATCH_HEIGHT,dtype=np.float64)
    x=np.linspace(-1.0,1.0,registry.PATCH_WIDTH,dtype=np.float64)
    yy,xx=np.meshgrid(y,x,indexing="ij")
    return np.stack((xx,yy),axis=-1).reshape(-1,2)


def _neutral_nonpositional_std_v8(neutral: np.ndarray) -> float:
    """Remove fixed token position and frame-global offsets before a visual gate."""

    value=np.asarray(neutral,dtype=np.float64)
    temporal=value-value.mean(axis=0,keepdims=True)
    temporal=temporal-temporal.mean(axis=(1,2),keepdims=True)
    return float(np.std(temporal))


def _evaluate_neutral_component_v8(
    source_mask: np.ndarray,
    source_soft: np.ndarray,
    neutral_source: np.ndarray,
    neutral_evaluation: np.ndarray,
) -> Optional[tuple[np.ndarray,tuple[float,float],tuple[float,...],float,float,float,float]]:
    """Numpy transcription of the frozen V6 visual-then-spatial gate."""

    source=np.asarray(neutral_source,dtype=np.float64).reshape(-1,neutral_source.shape[-1])
    target=np.asarray(neutral_evaluation,dtype=np.float64).reshape(-1,neutral_evaluation.shape[-1])
    target=target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-12)
    flat_mask=np.asarray(source_mask,dtype=bool).reshape(-1)
    weights=np.asarray(source_soft,dtype=np.float64).reshape(-1)[flat_mask]
    if flat_mask.sum()<2 or float(weights.sum())<=1e-12:
        return None
    weights=weights/weights.sum()
    descriptor=(source[flat_mask]*weights[:,None]).sum(axis=0)
    descriptor=descriptor/max(float(np.linalg.norm(descriptor)),1e-12)
    cosine=target@descriptor
    visual_margin=float(cosine.max()-np.median(cosine))
    if visual_margin<0.03:
        return None
    grid=_normalized_grid_v8()
    source_centroid=(grid[flat_mask]*weights[:,None]).sum(axis=0)
    distance2=np.sum((grid-source_centroid)**2,axis=1)
    logits=cosine-distance2/(2.0*0.35*0.35)
    spatial_margin=float(logits.max()-np.median(logits))
    shifted=(logits-float(logits.max()))/0.1
    probability=np.exp(np.clip(shifted,-700.0,0.0)); probability/=max(float(probability.sum()),1e-300)
    top_count=max(1,int(math.ceil(probability.size*0.1)))
    top10=float(np.partition(probability,-top_count)[-top_count:].sum())
    if spatial_margin<0.03 or top10<0.15:
        return None
    centroid=(grid*probability[:,None]).sum(axis=0)
    evaluated_descriptor=(target*probability[:,None]).sum(axis=0)
    evaluated_descriptor/=max(float(np.linalg.norm(evaluated_descriptor)),1e-12)
    support_count=max(2,int(flat_mask.sum()))
    selected=np.argpartition(probability,-support_count)[-support_count:]
    evaluated_mask=np.zeros(probability.size,dtype=bool); evaluated_mask[selected]=True
    return (
        evaluated_mask.reshape(registry.PATCH_HEIGHT,registry.PATCH_WIDTH),
        (float(centroid[0]),float(centroid[1])),
        tuple(float(item) for item in evaluated_descriptor),
        float(probability.max()),
        visual_margin,
        spatial_margin,
        top10,
    )


def _uot_plan(left: Sequence[TrackedNodeV8], right: Sequence[ComponentV8]) -> tuple[np.ndarray,np.ndarray]:
    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), 0.65, dtype=np.float64); cost[-1, -1] = 0.0
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            spatial = math.dist(a.component.centroid_xy, b.centroid_xy) / math.sqrt(8.0)
            embedding = _cosine_distance(a.component.descriptor, b.descriptor)
            mass = abs(math.log(max(a.component.soft_mass, 1e-12) / max(b.soft_mass, 1e-12)))
            cost[i, j] = 0.45 * spatial + 0.45 * embedding + 0.1 * min(mass, 1.0)
    kernel = np.exp(-cost / 0.08).clip(1e-30, None)
    left_mass=np.asarray([max(row.component.soft_mass,1e-12) for row in left],dtype=np.float64)
    right_mass=np.asarray([max(row.soft_mass,1e-12) for row in right],dtype=np.float64)
    supply=np.concatenate((left_mass,np.asarray([float(right_mass.sum())])))
    demand=np.concatenate((right_mass,np.asarray([float(left_mass.sum())])))
    total=max(float(supply.sum()),1e-12); supply/=total; demand/=total
    u=np.ones(n+1); v=np.ones(m+1)
    for _ in range(64):
        u=supply/(kernel@v).clip(1e-300,None)
        v=demand/(kernel.T@u).clip(1e-300,None)
    return (u[:,None]*kernel)*v[None,:],cost


def _hard_matches(plan: np.ndarray, cost: np.ndarray) -> tuple[tuple[int, int], ...]:
    n, m = plan.shape[0] - 1, plan.shape[1] - 1; output = []
    for i in range(n):
        if not m:
            continue
        j = int(np.argmax(plan[i, :m]))
        if int(np.argmax(plan[:n, j])) != i:
            continue
        if plan[i, j] <= plan[i, m] or plan[i, j] <= plan[n, j]:
            continue
        if cost[i,j] > 0.75 or plan[i,:m].sum()/max(plan[i,:].sum(),1e-12) < 0.2 or plan[:n,j].sum()/max(plan[:,j].sum(),1e-12) < 0.2:
            continue
        output.append((i, j))
    return tuple(output)


def _track(phases: Sequence[Sequence[ComponentV8]]) -> tuple[tuple[TrackedNodeV8, ...], Mapping[str, int], int]:
    output: list[tuple[TrackedNodeV8, ...]] = []; next_id = 0; dustbin = 0
    lifecycle = {name: 0 for name in ("birth", "occlusion", "reentry", "death")}
    missing: dict[int, int] = {}; last: dict[int, TrackedNodeV8] = {}
    for phase in phases:
        candidates = list(phase); active = [last[key] for key in sorted(last) if missing.get(key, 0) <= 2]
        plan,cost=_uot_plan(active,candidates); matches=_hard_matches(plan,cost); used = set(); current = []
        for left_index, right_index in matches:
            prior, component = active[left_index], candidates[right_index]
            node = TrackedNodeV8(component, prior.track_id); current.append(node); used.add(right_index)
            if missing.get(prior.track_id, 0): lifecycle["reentry"] += 1
            missing[prior.track_id] = 0; last[prior.track_id] = node
        for index, component in enumerate(candidates):
            if index in used:
                continue
            node = TrackedNodeV8(component, next_id); next_id += 1; current.append(node)
            last[node.track_id] = node; missing[node.track_id] = 0; lifecycle["birth"] += 1; dustbin += 1
        current_ids = {node.track_id for node in current}
        for track_id in list(last):
            if track_id not in current_ids:
                missing[track_id] = missing.get(track_id, 0) + 1; dustbin += 1
                if missing[track_id] <= 2: lifecycle["occlusion"] += 1
                else: lifecycle["death"] += 1; last.pop(track_id); missing.pop(track_id, None)
        output.append(tuple(sorted(current, key=lambda row: row.track_id)))
    lifecycle["right_censored"] = len(last)
    return tuple(output), lifecycle, dustbin


def track_neutral_components_v8(
    phases: Sequence[Sequence[ComponentV8]],
) -> Mapping[str,Any]:
    """Scalar diagnostic exposing neutral/UOT identity and gap reentry."""

    tracked,lifecycle,dustbin=_track(phases)
    value={
        "track_ids_by_phase":[[node.track_id for node in phase] for phase in tracked],
        "proposal_support_ids_by_phase":[[node.component.proposal_support_id for node in phase] for phase in tracked],
        "lifecycle_counts":dict(lifecycle),
        "dustbin_assignment_count":dustbin,
        "proposal_support_id_used_as_track_id":False,
        "cross_proposal_support_matching_permitted":True,
    }
    return {**value,"digest":registry.object_sha256(value)}


def _contact(left: np.ndarray, right: np.ndarray) -> float:
    dilated = left.copy()
    dilated[1:] |= left[:-1]; dilated[:-1] |= left[1:]; dilated[:, 1:] |= left[:, :-1]; dilated[:, :-1] |= left[:, 1:]
    union = np.logical_or(dilated, right).sum()
    return float(np.logical_and(dilated, right).sum() / union) if union else 0.0


def path_program_metrics_v8(
    phase_positions: Sequence[tuple[int,Sequence[float]]],
) -> Mapping[str,float]:
    """Signed metrics from one common carrier coordinate frame."""

    if len(phase_positions)<2:
        return {"vertical_velocity_early":0.0,"vertical_velocity_late":0.0,"signed_winding_area":0.0}
    phases=np.asarray([row[0] for row in phase_positions],dtype=np.float64)
    points=np.asarray([row[1] for row in phase_positions],dtype=np.float64)
    if points.shape!=(len(phase_positions),2) or not np.isfinite(points).all() or not np.isfinite(phases).all() or np.any(np.diff(phases)<=0):
        raise ReducerV8Error("path program metric geometry differs")
    velocity=np.diff(points,axis=0)/np.diff(phases)[:,None]
    split=max(1,len(velocity)//2)
    early=float(np.mean(velocity[:split,1]))
    late=float(np.mean(velocity[split:,1])) if split<len(velocity) else 0.0
    closed=np.concatenate((points,points[:1]),axis=0)
    winding=0.5*float(np.sum(closed[:-1,0]*closed[1:,1]-closed[1:,0]*closed[:-1,1]))
    return {"vertical_velocity_early":early,"vertical_velocity_late":late,"signed_winding_area":winding}


def graph_signature_v8(
    residual: np.ndarray,
    neutral_descriptor: np.ndarray,
    branch: str,
    *,
    proposal_block_ids: Sequence[int],
    evaluation_block_ids: Sequence[int],
    proposal_override: Optional[np.ndarray] = None,
) -> GraphSummaryV8:
    value = np.asarray(residual); neutral = np.asarray(neutral_descriptor)
    if value.shape[:3] != TUBE_SHAPE or neutral.shape != value.shape or value.ndim != 4 or not np.isfinite(value).all() or not np.isfinite(neutral).all():
        raise ReducerV8Error("full 21x37x25 tube input differs")
    if tuple(proposal_block_ids)!=PROPOSAL_BLOCKS.get(branch) or tuple(evaluation_block_ids)!=EVALUATION_BLOCKS.get(branch) or set(proposal_block_ids)&set(evaluation_block_ids):
        raise ReducerV8Error("layer cross-fit closure differs")
    proposal = value if proposal_override is None else np.asarray(proposal_override)
    if proposal.shape != value.shape:
        raise ReducerV8Error("proposal override shape differs")
    # Action residual is proposal-only.  The held/evaluation side contributes
    # only prompt-neutral visual descriptors for correspondence.
    phase_components,raw_proposal_support_count,correspondence = _joint_support_components(proposal,neutral,branch)
    nonpositional_std=_neutral_nonpositional_std_v8(neutral)
    visual_margin=min((row[0] for row in correspondence),default=float("nan"))
    spatial_margin=min((row[1] for row in correspondence),default=float("nan"))
    top10=min((row[2] for row in correspondence),default=float("nan"))
    neutral_valid=(
        nonpositional_std>=1e-6
        and math.isfinite(visual_margin) and visual_margin>=0.03
        and math.isfinite(spatial_margin) and spatial_margin>=0.03
        and math.isfinite(top10) and top10>=0.15
    )
    tracked, lifecycle, dustbin = _track(phase_components)
    observations: dict[int, list[TrackedNodeV8]] = {}
    for phase in tracked:
        for node in phase: observations.setdefault(node.track_id, []).append(node)
    qualified = {key: rows for key, rows in observations.items() if len({row.component.phase for row in rows}) >= 3}
    qualified_ids=set(qualified)
    qualified_phases=tuple(tuple(node for node in phase if node.track_id in qualified_ids) for phase in tracked)
    evaluated = len(EVALUATION_PHASES[branch]); coverage = float(sum(len(rows) for rows in qualified.values()) / max(evaluated * max(len(qualified), 1), 1))
    all_points=np.asarray([node.component.centroid_xy for rows in qualified.values() for node in rows],dtype=np.float64)
    if len(all_points):
        minimum=all_points.min(0); maximum=all_points.max(0); carrier_center=(minimum+maximum)/2.0; carrier_scale=max(float(np.linalg.norm(maximum-minimum)),1e-6)
    else: carrier_center=np.zeros(2); carrier_scale=1.0
    displacements=[]; velocities=[]; accelerations=[]; endpoint=[]; graph_position={}
    qualified_lifecycle={"birth":len(qualified),"occlusion":0,"reentry":0,"death":0,"right_censored":0}
    qualified_missing=0
    for track_id, rows in qualified.items():
        rows=sorted(rows,key=lambda row:row.component.phase); points=np.asarray([row.component.centroid_xy for row in rows],dtype=np.float64); phase_ids=np.asarray([row.component.phase for row in rows],dtype=np.float64)
        raw_delta=np.diff(points,axis=0); gaps=np.diff(phase_ids)[:,None] if len(rows)>1 else np.empty((0,1)); delta=raw_delta/np.maximum(gaps,1.0)
        normalized=(points-carrier_center)/carrier_scale
        for row,position in zip(rows,normalized): graph_position[(track_id,row.component.phase)]=position
        if len(delta):
            displacements.extend((np.linalg.norm(raw_delta,axis=1)/carrier_scale).tolist()); velocities.extend((delta/carrier_scale).tolist()); endpoint.append(((points[-1]-points[0])/carrier_scale).tolist())
        if len(delta)>1: accelerations.extend((np.linalg.norm(np.diff(delta,axis=0),axis=1)/carrier_scale).tolist())
        gaps_in_eval=[int(round(item/2.0))-1 for item in np.diff(phase_ids) if item>2]
        qualified_missing+=sum(gaps_in_eval); qualified_lifecycle["occlusion"]+=sum(gaps_in_eval); qualified_lifecycle["reentry"]+=len(gaps_in_eval)
        remaining=max((EVALUATION_PHASES[branch][-1]-rows[-1].component.phase)//2,0)
        if remaining>2:
            qualified_missing+=3; qualified_lifecycle["occlusion"]+=2; qualified_lifecycle["death"]+=1
        else:
            qualified_missing+=remaining; qualified_lifecycle["occlusion"]+=remaining; qualified_lifecycle["right_censored"]+=1
    endpoint_mean=np.mean(np.asarray(endpoint),axis=0) if endpoint else np.zeros(2)
    centroid_path=[]
    for phase in qualified_phases:
        if phase:
            positions=np.asarray(
                [graph_position[(row.track_id,row.component.phase)] for row in phase],
                dtype=np.float64,
            )
            centroid_path.append((phase[0].component.phase,positions.mean(axis=0)))
    program_path=path_program_metrics_v8(centroid_path)
    vertical_early=program_path["vertical_velocity_early"]
    vertical_late=program_path["vertical_velocity_late"]
    winding=program_path["signed_winding_area"]
    edge_temporal=[]; prior_edges=set(); edge_lifecycle=0
    prior_position: dict[int, tuple[float,float]]={}
    prior_phase: dict[int,int]={}
    for phase in qualified_phases:
        active=[]; current_edges=set()
        for index,left in enumerate(phase):
            for right in phase[index+1:]:
                left_position=graph_position[(left.track_id,left.component.phase)]; right_position=graph_position[(right.track_id,right.component.phase)]
                distance=float(np.linalg.norm(left_position-right_position))
                contact=_contact(left.component.mask,right.component.mask)
                affinity=math.exp(-distance/0.25)*(0.5+0.5*contact)
                if affinity<0.35: continue
                key=tuple(sorted((left.track_id,right.track_id))); current_edges.add(key)
                lprev=np.asarray(prior_position.get(left.track_id,tuple(left_position))); rprev=np.asarray(prior_position.get(right.track_id,tuple(right_position)))
                rel_now=left_position-right_position; rel_prev=lprev-rprev
                phase_gap=max(left.component.phase-max(prior_phase.get(left.track_id,left.component.phase),prior_phase.get(right.track_id,right.component.phase)),1)
                rel_velocity=float(np.linalg.norm(rel_now-rel_prev)/phase_gap)
                uncertainty=-(affinity*math.log(max(affinity,1e-12))+(1-affinity)*math.log(max(1-affinity,1e-12)))
                active.append((abs(rel_now[0]),abs(rel_now[1]),rel_velocity,contact,uncertainty))
        activate=len(current_edges-prior_edges); persist=len(current_edges&prior_edges); deactivate=len(prior_edges-current_edges)
        edge_lifecycle += activate+deactivate
        means=np.mean(np.asarray(active),axis=0).tolist() if active else [0.0]*5
        edge_temporal.extend([float(len(active))/28.0,*means,float(activate),float(persist),float(deactivate)])
        prior_edges=current_edges
        for node in phase:
            prior_position[node.track_id]=tuple(graph_position[(node.track_id,node.component.phase)]); prior_phase[node.track_id]=node.component.phase
    masses=[node.component.soft_mass for phase in qualified_phases for node in phase]
    node_counts=[len(phase) for phase in qualified_phases]
    qualified_dustbin=qualified_missing+qualified_lifecycle["birth"]+qualified_lifecycle["death"]
    worldline=[
        len(qualified)/64.0, coverage, qualified_dustbin/max(sum(node_counts)+qualified_dustbin,1),
        qualified_lifecycle["birth"]/64.0,qualified_lifecycle["occlusion"]/64.0,qualified_lifecycle["reentry"]/64.0,qualified_lifecycle["death"]/64.0,
        float(np.mean(displacements)) if displacements else 0.0,float(np.std(displacements)) if displacements else 0.0,
        float(np.mean(np.linalg.norm(np.asarray(velocities),axis=1))) if velocities else 0.0,
        float(np.mean(accelerations)) if accelerations else 0.0,float(np.std(accelerations)) if accelerations else 0.0,
        float(np.mean(masses)) if masses else 0.0,float(np.std(masses)) if masses else 0.0,
        float(np.mean(node_counts))/8.0 if node_counts else 0.0,float(np.std(node_counts))/8.0 if node_counts else 0.0,
    ]
    temporal=[]
    for phase in qualified_phases:
        if phase:
            centroids=np.asarray([graph_position[(row.track_id,row.component.phase)] for row in phase]); temporal.extend([float(centroids[:,0].mean()),float(centroids[:,1].mean()),len(phase)/8.0])
        else: temporal.extend([0.0,0.0,0.0])
    signature=tuple(float(item) for item in (*worldline,*edge_temporal,*temporal))
    support=tuple(tuple(int(item) for item in np.flatnonzero(np.logical_or.reduce([node.component.mask for node in phase]) if phase else np.zeros((37,25),bool))) for phase in qualified_phases)
    return GraphSummaryV8(
        signature,support,sum(node_counts),len(qualified),coverage,edge_lifecycle,
        dustbin,qualified_dustbin,raw_proposal_support_count,qualified_lifecycle,
        (float(endpoint_mean[0]),float(endpoint_mean[1])),vertical_early,
        vertical_late,winding,
        float(np.mean(accelerations)) if accelerations else 0.0,
        neutral_valid,visual_margin,spatial_margin,top10,nonpositional_std,
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a,b=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=1: return float("nan")
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/denominator) if denominator>1e-12 else float("nan")


def _normalized_graph_signature(value: Sequence[float]) -> tuple[float,...]:
    array=np.asarray(value,dtype=np.float64)
    if array.shape!=(136,) or not np.isfinite(array).all():
        raise ReducerV8Error("graph signature schema/nonfinite closure differs")
    output=[]
    for section in (array[:16],array[16:106],array[106:136]):
        scale=float(np.linalg.norm(section)); output.extend((section/scale).tolist() if scale>1e-12 else [0.0]*len(section))
    return tuple(float(item) for item in output)


def _support_iou(left: GraphSummaryV8,right: GraphSummaryV8) -> float:
    intersection=union=0
    for a,b in zip(left.support,right.support):
        sa,sb=set(a),set(b); intersection+=len(sa&sb); union+=len(sa|sb)
    return intersection/union if union else 0.0


def _magnitude_ratio(left: float, right: float) -> float:
    denominator=max(abs(left),abs(right))
    return min(abs(left),abs(right))/denominator if denominator>1e-12 else float("nan")


def action_specific_program_gates_v8(
    action_id: str,
    primary: GraphSummaryV8,
    paraphrase: GraphSummaryV8,
    reverse: GraphSummaryV8,
) -> Mapping[str,Any]:
    """Frozen signed program metrics; closed-loop actions never use endpoints."""

    if action_id not in registry.ACTION_IDS:
        raise ReducerV8Error("action-specific control action differs")
    metrics: dict[str,Any]={"action_id":action_id}
    if action_id=="transfer":
        p=primary.endpoint_direction_xy[0]; q=paraphrase.endpoint_direction_xy[0]; r=reverse.endpoint_direction_xy[0]
        qratio=_magnitude_ratio(p,q); rratio=_magnitude_ratio(p,r)
        primary_valid=math.isfinite(p) and abs(p)>=0.02
        paraphrase_pass=(primary_valid and math.isfinite(q) and abs(q)>=0.02 and p*q>0 and math.isfinite(qratio) and qratio>=0.5)
        reverse_pass=(primary_valid and math.isfinite(r) and abs(r)>=0.02 and p*r<0 and math.isfinite(rratio) and rratio>=0.5)
        metrics.update({"primary_signed_horizontal_endpoint":p,"paraphrase_signed_horizontal_endpoint":q,"reverse_signed_horizontal_endpoint":r,"paraphrase_magnitude_ratio":qratio,"reverse_magnitude_ratio":rratio})
    elif action_id=="lift_pause_return":
        p=np.asarray((primary.vertical_velocity_early,primary.vertical_velocity_late),dtype=np.float64)
        q=np.asarray((paraphrase.vertical_velocity_early,paraphrase.vertical_velocity_late),dtype=np.float64)
        r=np.asarray((reverse.vertical_velocity_early,reverse.vertical_velocity_late),dtype=np.float64)
        pscore=float(p[1]-p[0]); qscore=float(q[1]-q[0]); rscore=float(r[1]-r[0])
        qcos=_cosine(p,q); rcos=_cosine(p,r)
        primary_valid=bool(np.isfinite(p).all() and np.all(np.abs(p)>=0.005) and abs(pscore)>=0.01)
        paraphrase_pass=bool(primary_valid and np.isfinite(q).all() and np.all(np.abs(q)>=0.005) and pscore*qscore>0 and p[0]*q[0]>0 and p[1]*q[1]>0 and math.isfinite(qcos) and qcos>=0.5)
        reverse_pass=bool(primary_valid and np.isfinite(r).all() and np.all(np.abs(r)>=0.005) and pscore*rscore<0 and p[0]*r[0]<0 and p[1]*r[1]<0 and math.isfinite(rcos) and rcos<=-0.1)
        metrics.update({"primary_early_vertical_velocity":float(p[0]),"primary_late_vertical_velocity":float(p[1]),"primary_vertical_order_score":pscore,"paraphrase_vertical_order_score":qscore,"reverse_vertical_order_score":rscore,"paraphrase_velocity_order_cosine":qcos,"reverse_velocity_order_cosine":rcos})
    else:
        p=primary.signed_winding_area; q=paraphrase.signed_winding_area; r=reverse.signed_winding_area
        qratio=_magnitude_ratio(p,q); rratio=_magnitude_ratio(p,r)
        primary_valid=math.isfinite(p) and abs(p)>=0.01
        paraphrase_pass=(primary_valid and math.isfinite(q) and abs(q)>=0.01 and p*q>0 and math.isfinite(qratio) and qratio>=0.5)
        reverse_pass=(primary_valid and math.isfinite(r) and abs(r)>=0.01 and p*r<0 and math.isfinite(rratio) and rratio>=0.5)
        metrics.update({"primary_signed_winding_area":p,"paraphrase_signed_winding_area":q,"reverse_signed_winding_area":r,"paraphrase_magnitude_ratio":qratio,"reverse_magnitude_ratio":rratio})
    scalar_metrics={key:(float(value) if isinstance(value,(float,np.floating)) and math.isfinite(float(value)) else value if not isinstance(value,(float,np.floating)) else None) for key,value in metrics.items()}
    value={"primary_program_metric_valid":bool(primary_valid),"paraphrase_program_pass":bool(paraphrase_pass),"reverse_program_pass":bool(reverse_pass),"metrics":scalar_metrics}
    return {**value,"digest":registry.object_sha256(value)}


def local_control_gates_v8(
    action_id: str,
    summaries: Mapping[str,GraphSummaryV8],
    shuffle_summary: GraphSummaryV8,
    null_energy_ratio: float,
) -> Mapping[str,Any]:
    """Pure, non-caller-boolean recomputation shared by runtime and fixtures."""

    if set(summaries)!={"primary","paraphrase","reverse","lexical_placebo","noop"}:
        raise ReducerV8Error("local graph summary arm matrix differs")
    primary=summaries["primary"]
    program=action_specific_program_gates_v8(action_id,primary,summaries["paraphrase"],summaries["reverse"])
    primary_valid=(primary.neutral_correspondence_valid and primary.qualified_track_count>=1 and primary.track_coverage>=0.35 and primary.dynamic_edge_lifecycle_count>=1 and program["primary_program_metric_valid"])
    paraphrase_iou=_support_iou(primary,summaries["paraphrase"])
    primary_raw=primary.raw_proposal_support_count
    lexical_ratio=summaries["lexical_placebo"].raw_proposal_support_count/primary_raw if primary_raw>0 else float("nan")
    if primary.path_acceleration<=1e-8:
        shuffle_pass=shuffle_summary.path_acceleration>=0.02
        shuffle_ratio=None
    else:
        shuffle_ratio=shuffle_summary.path_acceleration/primary.path_acceleration
        shuffle_pass=shuffle_ratio>=1.2 and shuffle_summary.path_acceleration>=0.02
    gates={
        "primary_graph_valid":bool(primary_valid),
        "noop_pass":summaries["noop"].raw_proposal_support_count<=0,
        "reverse_pass":bool(program["reverse_program_pass"]),
        "phase_shuffle_pass":bool(shuffle_pass),
        "paraphrase_pass":bool(paraphrase_iou>=0.35 and program["paraphrase_program_pass"]),
        "lexical_placebo_pass":bool(math.isfinite(lexical_ratio) and lexical_ratio<=0.5),
        "independent_null_pass":bool(math.isfinite(null_energy_ratio) and null_energy_ratio<=0.5),
        "neutral_correspondence_pass":bool(primary.neutral_correspondence_valid),
    }
    metrics={
        "action_program":program,
        "paraphrase_support_iou":paraphrase_iou,
        "lexical_component_ratio":lexical_ratio if math.isfinite(lexical_ratio) else None,
        "phase_shuffle_acceleration_ratio":shuffle_ratio,
        "phase_shuffle_acceleration":shuffle_summary.path_acceleration,
        "null_energy_ratio":null_energy_ratio if math.isfinite(null_energy_ratio) else None,
        "neutral_visual_cosine_margin":primary.neutral_visual_cosine_margin if math.isfinite(primary.neutral_visual_cosine_margin) else None,
        "neutral_top_vs_median_margin":primary.neutral_top_vs_median_margin if math.isfinite(primary.neutral_top_vs_median_margin) else None,
        "neutral_top10_mass_fraction":primary.neutral_top10_mass_fraction if math.isfinite(primary.neutral_top10_mass_fraction) else None,
        "neutral_nonpositional_std":primary.neutral_nonpositional_std,
    }
    value={"gates":gates,"metrics":metrics,"all_gates_pass":all(gates.values())}
    return {**value,"digest":registry.object_sha256(value)}


def evaluate_local_controls_v8(
    arms_by_block: Mapping[int,Mapping[str,np.ndarray]], branch: str, *, action_id: str
) -> Mapping[str,Any]:
    """Internally recompute every executed V8 local control from all four layers."""

    if set(arms_by_block)!=set(registry.BLOCKS) or branch not in BRANCHES:
        raise ReducerV8Error("bound four-block local matrix differs")
    factorized={block:factorize_carrier_arms_v8(arms_by_block[block]) for block in registry.BLOCKS}
    proposer={arm:np.mean([factorized[block][arm] for block in PROPOSAL_BLOCKS[branch]],axis=0) for arm in ("primary","paraphrase","reverse","lexical_placebo","noop","null_sanity")}
    neutral=np.mean([factorized[block]["neutral_descriptor"] for block in EVALUATION_BLOCKS[branch]],axis=0)
    summaries={arm:graph_signature_v8(proposer[arm],neutral,branch,proposal_block_ids=PROPOSAL_BLOCKS[branch],evaluation_block_ids=EVALUATION_BLOCKS[branch]) for arm in ("primary","paraphrase","reverse","lexical_placebo","noop")}
    shuffled=proposal_only_phase_shuffle_v8(proposer["primary"],branch)
    shuffle_summary=graph_signature_v8(proposer["primary"],neutral,branch,proposal_block_ids=PROPOSAL_BLOCKS[branch],evaluation_block_ids=EVALUATION_BLOCKS[branch],proposal_override=shuffled)
    primary=summaries["primary"]
    action_energy=float(np.mean(proposer["primary"]**2)); null_energy=float(np.mean(proposer["null_sanity"]**2))
    null_ratio=null_energy/action_energy if action_energy>1e-12 else float("nan")
    decision=local_control_gates_v8(action_id,summaries,shuffle_summary,null_ratio)
    body={"branch":branch,"action_id":action_id,"gates":decision["gates"],"branch_pass":decision["all_gates_pass"],"graph_signature":list(_normalized_graph_signature(primary.signature)),"graph_signature_sections":{"node_worldlines":[0,16],"edge_event_sequence":[16,106],"temporal_ordering":[106,136]},"metrics":decision["metrics"],"local_gate_decision_digest":decision["digest"],"primary":primary.scalar_receipt(),"proposal_blocks":list(PROPOSAL_BLOCKS[branch]),"evaluation_blocks":list(EVALUATION_BLOCKS[branch]),"phase_shuffle_proposal_only":True,"all_controls_recomputed":True}
    return {**body,"digest":registry.object_sha256(body)}


def loao_reduce_signatures_v8(
    local_evidence: Mapping[tuple[int,str,str,str,str], Mapping[str,Any]],
) -> Mapping[str, Any]:
    """Pure CPU LOAO reducer; official authority must call reduce_bound_matrix_v8."""

    expected={
        (seed,sigma,state,action,branch)
        for seed in registry.SEED_IDS for sigma in registry.SIGMA_CELL_INDICES
        for state in registry.APPEARANCE_IDS for action in registry.ACTION_IDS for branch in BRANCHES
    }
    if set(local_evidence)!=expected:
        raise ReducerV8Error("LOAO local evidence matrix differs")
    signatures={}
    exact_gates={"primary_graph_valid","noop_pass","reverse_pass","phase_shuffle_pass","paraphrase_pass","lexical_placebo_pass","independent_null_pass","neutral_correspondence_pass"}
    for key,row in local_evidence.items():
        if not isinstance(row,Mapping): raise ReducerV8Error("LOAO local evidence is not an object")
        body=dict(row); claim=body.pop("digest",None)
        if claim!=registry.object_sha256(body): raise ReducerV8Error("LOAO local evidence digest differs")
        gates=row.get("gates")
        if not isinstance(gates,Mapping) or set(gates)!=exact_gates or any(type(value) is not bool for value in gates.values()): raise ReducerV8Error("LOAO local control matrix differs")
        if row.get("branch")!=key[-1] or row.get("action_id")!=key[3] or row.get("all_controls_recomputed") is not True or row.get("branch_pass") is not all(gates.values()): raise ReducerV8Error("LOAO local branch/action authority differs")
        signatures[key]=_normalized_graph_signature(row.get("graph_signature",()))
    rows=[]
    for direction,(reference_seed,held_seed) in SEED_DIRECTIONS.items():
        for sigma in registry.SIGMA_CELL_INDICES:
            for held_state in registry.APPEARANCE_IDS:
                reference_states=tuple(state for state in registry.APPEARANCE_IDS if state!=held_state)
                for action in registry.ACTION_IDS:
                    branch_rows={}
                    for branch in BRANCHES:
                        prototype=np.mean([np.asarray(signatures[(reference_seed,sigma,state,action,branch)],dtype=np.float64) for state in reference_states],axis=0)
                        held=np.asarray(signatures[(held_seed,sigma,held_state,action,branch)],dtype=np.float64)
                        same=_cosine(held,prototype); wrong=[]
                        for other in registry.ACTION_IDS:
                            if other==action: continue
                            wrong_proto=np.mean([np.asarray(signatures[(reference_seed,sigma,state,other,branch)],dtype=np.float64) for state in reference_states],axis=0)
                            wrong.append(_cosine(held,wrong_proto))
                        margin=same-max(wrong) if all(math.isfinite(item) for item in (same,*wrong)) else float("nan")
                        local_ok=all(local_evidence[(seed,sigma,state,action,branch)]["branch_pass"] for seed,state in ((reference_seed,reference_states[0]),(reference_seed,reference_states[1]),(held_seed,held_state)))
                        branch_pass=local_ok and math.isfinite(same) and same>=0.5 and math.isfinite(margin) and margin>=0.03
                        body={"same_action_cosine":same if math.isfinite(same) else None,"wrong_action_max_cosine":max(wrong) if all(math.isfinite(item) for item in wrong) else None,"separation_margin":margin if math.isfinite(margin) else None,"reference_seed":reference_seed,"held_seed":held_seed,"reference_appearances":list(reference_states),"held_appearance":held_state,"held_excluded_before_aggregation":held_state not in reference_states,"local_graph_controls_pass":local_ok,"branch_pass":branch_pass,"prototype_digest":tensor_sha256_v8(prototype)}
                        branch_rows[branch]={**body,"digest":registry.object_sha256(body)}
                    cell_pass=all(row["branch_pass"] for row in branch_rows.values())
                    value={"direction":direction,"sigma":sigma,"held_appearance":held_state,"action_id":action,"branches":branch_rows,"cell_pass":cell_pass,"representation_admitted":False,"scientific_claim":False}
                    rows.append({**value,"digest":registry.object_sha256(value)})
    passed=sum(row["cell_pass"] for row in rows)
    value={"schema_version":SCHEMA_VERSION,"held_cell_count":len(rows),"expected_held_cell_count":54,"passed_cell_count":passed,"diagnostic_pass":passed==54,"rows":rows,"official_bound_capture_authority":False,"representation_admission_hard_false":True,"stable_transferable_action_representation_established":False,"scientific_claim_authorized":False,"training_performed":False,"decoder_called":False,"route_or_injection_called":False,"gpu_used":False}
    return {**value,"digest":registry.object_sha256(value)}


def factorize_carrier_arms_v8(arms: Mapping[str,np.ndarray]) -> Mapping[str,np.ndarray]:
    if set(arms)!=set(registry.ARMS): raise ReducerV8Error("carrier arm matrix differs")
    shapes={np.asarray(value).shape for value in arms.values()}
    if len(shapes)!=1 or next(iter(shapes))[:3]!=TUBE_SHAPE: raise ReducerV8Error("carrier arm tensor shape differs")
    if any(not np.issubdtype(np.asarray(value).dtype,np.floating) or not np.isfinite(value).all() for value in arms.values()): raise ReducerV8Error("carrier arm dtype/nonfinite differs")
    null_center=(np.asarray(arms["null_a"],dtype=np.float64)+np.asarray(arms["null_b"],dtype=np.float64))/2.0
    result={arm:np.asarray(arms[arm],dtype=np.float64)-null_center for arm in ("primary","paraphrase","reverse","lexical_placebo","noop")}
    result["null_sanity"]=np.asarray(arms["null_a"],dtype=np.float64)-np.asarray(arms["null_b"],dtype=np.float64)
    result["neutral_descriptor"]=np.asarray(arms["neutral"],dtype=np.float64)
    return result


@dataclass(frozen=True)
class BoundForwardV8:
    bindings: Mapping[int,registry.CaptureBindingV8]
    projected_by_block: Mapping[int,np.ndarray]


@dataclass(frozen=True)
class BoundMiddleStateV8:
    seed_id: int
    sigma_name: str
    state_id: str
    action_id: str
    tensor: np.ndarray


def reduce_bound_cell_v8(
    forwards: Mapping[str,BoundForwardV8],
    middle: BoundMiddleStateV8,
    runtime: registry.RuntimeIdentityV8,
    embedding_recomputer: registry.FrozenEmbeddingRecomputerV8,
) -> Mapping[str,Any]:
    """Single ownership-consuming authority path for one 8-arm carrier cell."""

    processing_error=None; result=None
    try:
        reject_aliased_ownership_v8({"forwards":forwards,"middle":middle})
        if set(forwards)!=set(registry.ARMS): raise ReducerV8Error("bound cell arm matrix differs")
        if embedding_recomputer.prompt_encoder_sha256!=runtime.prompt_encoder_sha256 or embedding_recomputer.nontext_encoder_sha256!=runtime.nontext_encoder_sha256: raise ReducerV8Error("bound cell encoder runtime differs")
        cell=(middle.seed_id,middle.sigma_name,middle.state_id,middle.action_id)
        middle_sha=tensor_sha256_v8(middle.tensor)
        middle_embedding=embedding_recomputer.middle_embedding_sha256(middle.tensor)
        arms_by_block={block:{} for block in registry.BLOCKS}; binding_digests=[]
        for arm,forward in forwards.items():
            if set(forward.bindings)!=set(registry.BLOCKS) or set(forward.projected_by_block)!=set(registry.BLOCKS): raise ReducerV8Error("bound forward four-block closure differs")
            for block in registry.BLOCKS:
                binding=forward.bindings[block]; tensor=np.asarray(forward.projected_by_block[block])
                if binding.key!=(middle.seed_id,middle.sigma_name,middle.state_id,middle.action_id,arm,block): raise ReducerV8Error("bound forward identity key differs")
                if binding.runtime_identity_digest!=runtime.digest or binding.projection_sha256!=runtime.projection_sha256: raise ReducerV8Error("bound runtime/projection differs")
                if binding.carrier_state_sha256!=middle_sha or binding.middle_nontext_embedding_sha256!=middle_embedding: raise ReducerV8Error("actual middle tensor binding differs")
                if binding.projected_tensor_sha256!=tensor_sha256_v8(tensor): raise ReducerV8Error("actual projected tensor binding differs")
                prompt=registry.PROMPT_BY_KEY[(binding.state_id,binding.action_id,binding.arm)]
                action=registry.ACTION_BY_ID[binding.action_id]; state=registry.APPEARANCE_BY_ID[binding.state_id]
                if binding.prompt_embedding_sha256!=embedding_recomputer.prompt_embedding_sha256(prompt.caption) or binding.action_embedding_sha256!=embedding_recomputer.action_embedding_sha256(action.dynamics) or binding.state_embedding_sha256!=embedding_recomputer.state_embedding_sha256(state.body()): raise ReducerV8Error("bound prompt/action/state live embedding differs")
                if tensor.shape[:3]!=TUBE_SHAPE or tensor.ndim!=4 or not np.isfinite(tensor).all(): raise ReducerV8Error("bound projected tensor geometry/nonfinite differs")
                arms_by_block[block][arm]=tensor
                binding_digests.append(registry.object_sha256(dict(binding.__dict__)))
        # A single captured transformer invocation must expose all four blocks;
        # caller-provided block labels cannot substitute for this event binding.
        forward_events=set(); invocation_ids=set()
        for forward in forwards.values():
            forward_events.add(next(iter(forward.bindings.values())).forward_event_sha256)
            invocation_ids.add(next(iter(forward.bindings.values())).four_block_invocation_sha256)
            if len({row.forward_event_sha256 for row in forward.bindings.values()})!=1 or len({row.four_block_invocation_sha256 for row in forward.bindings.values()})!=1:
                raise ReducerV8Error("bound four blocks do not share one forward invocation")
        if len(forward_events)!=len(registry.ARMS) or len(invocation_ids)!=len(registry.ARMS):
            raise ReducerV8Error("bound arm forward invocation is aliased")
        branches={branch:evaluate_local_controls_v8(arms_by_block,branch,action_id=middle.action_id) for branch in BRANCHES}
        body={"seed_id":middle.seed_id,"sigma":middle.sigma_name,"state_id":middle.state_id,"action_id":middle.action_id,"carrier_state_sha256":middle_sha,"middle_embedding_sha256":middle_embedding,"binding_digest":registry.object_sha256(binding_digests),"branches":branches,"all_raw_consumed":True,"representation_admitted":False,"scientific_claim":False}
        result={**body,"digest":registry.object_sha256(body)}
    except BaseException as error:
        processing_error=error
    scrub=best_effort_scrub_v8({"forwards":forwards,"middle":middle})
    if processing_error is not None:
        if not scrub["verified"]: raise ReducerV8Error(f"bound cell failed and scrub also failed: {type(processing_error).__name__}; {scrub['failures']}") from processing_error
        raise processing_error
    if not scrub["verified"]: raise ReducerV8Error(f"bound cell scrub failed: {scrub['failures']}")
    assert result is not None
    final_body=dict(result); final_body.pop("digest",None); final_body["ownership_scrub_digest"]=scrub["digest"]
    return {**final_body,"digest":registry.object_sha256(final_body)}


def reduce_bound_matrix_v8(
    cells: Mapping[tuple[int,str,str,str],Mapping[str,BoundForwardV8]],
    middle_states: Mapping[tuple[int,str,str,str],BoundMiddleStateV8],
    b0_rows: Sequence[registry.B0BindingV8],
    runtime: registry.RuntimeIdentityV8,
    embedding_recomputer: registry.FrozenEmbeddingRecomputerV8,
) -> Mapping[str,Any]:
    """Official CPU authority: validate, consume, scrub, then aggregate exact 54 cells."""
    expected={(seed,sigma,state,action) for seed in registry.SEED_IDS for sigma in registry.SIGMA_CELL_INDICES for state in registry.APPEARANCE_IDS for action in registry.ACTION_IDS}
    processing_error=None; final=None
    try:
        reject_aliased_ownership_v8({"cells":cells,"middle_states":middle_states})
        if set(cells)!=expected or set(middle_states)!=expected: raise ReducerV8Error("official bound 54-cell matrix differs")
        bindings=[binding for forwards in cells.values() for forward in forwards.values() for binding in forward.bindings.values()]
        capture_receipt=registry.validate_capture_bindings_v8(bindings,runtime,embedding_recomputer)
        b0_receipt=registry.validate_b0_bindings_v8(b0_rows,bindings,runtime)
        local={}; cell_receipts=[]
        for key in sorted(expected):
            receipt=reduce_bound_cell_v8(cells[key],middle_states[key],runtime,embedding_recomputer); cell_receipts.append(receipt["digest"])
            seed,sigma,state,action=key
            for branch in BRANCHES: local[(seed,sigma,state,action,branch)]=receipt["branches"][branch]
        aggregate=loao_reduce_signatures_v8(local); aggregate_body=dict(aggregate); aggregate_body.pop("digest",None); aggregate_body["official_bound_capture_authority"]=True; aggregate={**aggregate_body,"digest":registry.object_sha256(aggregate_body)}
        body={"schema_version":SCHEMA_VERSION,"capture_binding_receipt":capture_receipt,"B0_schema_receipt":b0_receipt,"bound_cell_digest":cell_receipts,"aggregate":aggregate,"B0_execution_proven":False,"runner_and_external_completion_pending":True,"representation_admission_hard_false":True,"scientific_claim_authorized":False,"training_performed":False,"decoder_called":False,"route_or_injection_called":False,"gpu_used":False}
        final={**body,"digest":registry.object_sha256(body)}
    except BaseException as error: processing_error=error
    scrub=best_effort_scrub_v8({"cells":cells,"middle_states":middle_states})
    if processing_error is not None:
        if not scrub["verified"]: raise ReducerV8Error(f"official matrix failed and scrub also failed: {type(processing_error).__name__}; {scrub['failures']}") from processing_error
        raise processing_error
    if not scrub["verified"]: raise ReducerV8Error(f"official matrix scrub failed: {scrub['failures']}")
    assert final is not None
    body=dict(final); body.pop("digest",None); body["ownership_scrub_digest"]=scrub["digest"]
    return {**body,"digest":registry.object_sha256(body)}


def hard_false_receipt_v8() -> Mapping[str,Any]:
    value={"representation_admission_hard_false":True,"stable_transferable_action_representation_established":False,"scientific_claim_authorized":False,"local_same_carrier_internal_plan_compatibility_only":True,"full_trajectory_action_representation_claimed":False,"per_arm_native_40_step_trajectory_required_for_future_v9":True,"training_authorized":False,"decoder_authorized":False,"route_authorized":False,"gpu_authorized":False,"runner_implemented":False,"completion_authority_created":False}
    return {**value,"digest":registry.object_sha256(value)}


__all__=["BRANCHES","BoundForwardV8","BoundMiddleStateV8","ComponentV8","EVALUATION_BLOCKS","EVALUATION_PHASES","GraphSummaryV8","PROPOSAL_BLOCKS","PROPOSAL_PHASES","PROPOSAL_SHUFFLES","ReducerV8Error","action_specific_program_gates_v8","best_effort_scrub_v8","discover_owned_values_v8","evaluate_local_controls_v8","factorize_carrier_arms_v8","graph_signature_v8","hard_false_receipt_v8","loao_reduce_signatures_v8","local_control_gates_v8","path_program_metrics_v8","proposal_only_phase_shuffle_v8","reduce_bound_cell_v8","reduce_bound_matrix_v8","reject_aliased_ownership_v8","tensor_sha256_v8","track_neutral_components_v8"]
