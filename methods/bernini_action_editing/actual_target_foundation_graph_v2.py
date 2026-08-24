#!/usr/bin/env python3
"""Pure-CPU anonymous geometry and unbalanced matching primitives for v2."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # local contract-only Python may omit numeric deps
    np = None  # type: ignore[assignment]


class GraphV2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class AnonymousNodeV2:
    mask: np.ndarray
    descriptor: tuple[float, ...]
    area_fraction: float
    centroid_xy: tuple[float, float]
    track_id: int = -1


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a=np.asarray(left,dtype=np.float64); b=np.asarray(right,dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1: raise GraphV2Error("descriptor shapes differ")
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return 1.0-float(a@b/denominator) if denominator>1e-12 else 1.0


def unbalanced_sinkhorn_dustbin(
    left: Sequence[AnonymousNodeV2], right: Sequence[AnonymousNodeV2], *,
    epsilon: float=0.08, rho: float=0.35, dustbin_cost: float=0.42,
    iterations: int=80,
) -> np.ndarray:
    """KL-relaxed unbalanced OT with an explicit last-row/last-column dustbin."""
    n,m=len(left),len(right); cost=np.full((n+1,m+1),dustbin_cost,dtype=np.float64)
    cost[-1,-1]=0.0
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            geometry=math.dist(a.centroid_xy,b.centroid_xy)
            cost[i,j]=0.8*cosine_distance(a.descriptor,b.descriptor)+0.2*min(geometry,1.0)
    kernel=np.exp(-cost/epsilon).clip(1e-30,None)
    scale=max(n,m,1); a=np.full(n+1,1.0/scale); b=np.full(m+1,1.0/scale)
    a[-1]=max(m-n,0)/scale+1.0/scale; b[-1]=max(n-m,0)/scale+1.0/scale
    tau=rho/(rho+epsilon); u=np.ones(n+1); v=np.ones(m+1)
    for _ in range(iterations):
        u=np.power(a/(kernel@v).clip(1e-30,None),tau)
        v=np.power(b/(kernel.T@u).clip(1e-30,None),tau)
    plan=(u[:,None]*kernel)*v[None,:]
    if not np.isfinite(plan).all(): raise GraphV2Error("nonfinite unbalanced transport")
    return plan


def hard_matches_with_dustbin(plan: np.ndarray) -> tuple[tuple[int,int], ...]:
    n,m=plan.shape[0]-1,plan.shape[1]-1; matches=[]
    for i in range(n):
        if m==0: continue
        j=int(np.argmax(plan[i,:m]))
        if int(np.argmax(plan[:n,j])) != i: continue
        if plan[i,j] <= plan[i,m] or plan[i,j] <= plan[n,j]: continue
        matches.append((i,j))
    return tuple(matches)


def assign_anonymous_tracks(phases: Sequence[Sequence[AnonymousNodeV2]]) -> tuple[tuple[AnonymousNodeV2,...], ...]:
    output=[]; next_id=0
    for phase_index,phase in enumerate(phases):
        current=[replace(node,track_id=-1) for node in sorted(phase,key=_node_key)]
        if phase_index:
            previous=output[-1]
            for i,j in hard_matches_with_dustbin(unbalanced_sinkhorn_dustbin(previous,current)):
                current[j]=replace(current[j],track_id=previous[i].track_id)
        for index,node in enumerate(current):
            if node.track_id<0: current[index]=replace(node,track_id=next_id); next_id+=1
        output.append(tuple(current))
    return tuple(output)


def unbalanced_matching_diagnostics(phases: Sequence[Sequence[AnonymousNodeV2]]) -> dict:
    unmatched=0; dustbin_mass=0.0; pairs=0
    for left,right in zip(phases,phases[1:]):
        plan=unbalanced_sinkhorn_dustbin(left,right); matches=hard_matches_with_dustbin(plan); pairs+=1
        unmatched += len(left)+len(right)-2*len(matches)
        dustbin_mass += float(plan[-1,:-1].sum()+plan[:-1,-1].sum())
    return {"phase_pair_count":pairs,"explicit_dustbin":True,"unmatched_count":unmatched,"dustbin_transport_mass":dustbin_mass}


def _node_key(node: AnonymousNodeV2) -> bytes:
    values=tuple(round(float(x),8) for x in (*node.descriptor,node.area_fraction,*node.centroid_xy))
    digest=hashlib.sha256(repr(values).encode("ascii")); digest.update(np.ascontiguousarray(node.mask,dtype=np.uint8).tobytes())
    return digest.digest()


def canonical_node_signature(phases: Sequence[Sequence[AnonymousNodeV2]], *, slots: int=12, descriptor_width: int=8) -> tuple[float,...]:
    signature=[]
    for phase in phases:
        ordered=sorted(phase,key=_node_key)[:slots]
        for node in ordered:
            descriptor=tuple(node.descriptor[:descriptor_width]); descriptor += (0.0,)*(descriptor_width-len(descriptor))
            signature.extend((*descriptor,node.area_fraction,*node.centroid_xy))
        signature.extend((0.0,)*((slots-len(ordered))*(descriptor_width+3)))
    return tuple(signature)


def break_mask_descriptor_binding(phases: Sequence[Sequence[AnonymousNodeV2]]) -> tuple[tuple[AnonymousNodeV2,...], ...]:
    broken=[]
    for phase in phases:
        if len(phase)<2: broken.append(tuple()); continue
        descriptors=[node.descriptor for node in phase]
        broken.append(tuple(replace(node,descriptor=descriptors[(index+1)%len(phase)]) for index,node in enumerate(phase)))
    return tuple(broken)


def relabel_slots(phases: Sequence[Sequence[AnonymousNodeV2]]) -> tuple[tuple[AnonymousNodeV2,...], ...]:
    return tuple(tuple(reversed(phase)) for phase in phases)


def boundary_gap_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float,float]:
    a=np.asarray(left,dtype=bool); b=np.asarray(right,dtype=bool)
    if a.shape!=b.shape or a.ndim!=2: raise GraphV2Error("mask geometry differs")
    union=np.logical_or(a,b).sum(); overlap=float(np.logical_and(a,b).sum()/union) if union else 0.0
    def boundary(mask):
        eroded=mask.copy(); eroded[1:,:]&=mask[:-1,:]; eroded[:-1,:]&=mask[1:,:]; eroded[:,1:]&=mask[:,:-1]; eroded[:,:-1]&=mask[:,1:]
        return np.argwhere(mask & ~eroded)
    pa,pb=boundary(a),boundary(b)
    if not len(pa) or not len(pb): return 1.0,overlap
    minimum=min(float(((pa[start:start+512,None,:]-pb[None,:,:])**2).sum(-1).min()) for start in range(0,len(pa),512))
    return math.sqrt(minimum)/math.hypot(*a.shape),overlap


def patch_area_pool(mask_weights: np.ndarray, patch_tokens: np.ndarray) -> tuple[float,...]:
    weights=np.asarray(mask_weights,dtype=np.float64).reshape(-1); tokens=np.asarray(patch_tokens,dtype=np.float64)
    if tokens.ndim!=2 or tokens.shape[0]!=weights.size: raise GraphV2Error("patch/token geometry differs")
    support=float(weights.sum())
    if not np.isfinite(weights).all() or support<=1e-6: raise GraphV2Error("zero/nonfinite DINO patch support abstains")
    return tuple((tokens*weights[:,None]).sum(0)/support)


def tubelet2_eight_blocks(hidden: np.ndarray, spatial_tokens: int) -> np.ndarray:
    value=np.asarray(hidden)
    if value.ndim!=2 or spatial_tokens<=0 or value.shape[0]!=8*spatial_tokens: raise GraphV2Error("V-JEPA output is not exactly eight tubelet2 temporal blocks")
    return value.reshape(8,spatial_tokens,value.shape[-1]).mean(1)


def pairwise_edge_signature(
    phases: Sequence[Sequence[AnonymousNodeV2]], velocities: dict[tuple[int,int],tuple[float,float]],
) -> tuple[tuple[float,...], int]:
    values=[]; previous_edges=set(); lifecycle=0
    for phase_index,phase in enumerate(phases):
        ordered=sorted(phase,key=lambda n:n.track_id)
        phase_values=[]
        for i,left in enumerate(ordered):
            for right in ordered[i+1:]:
                gap,overlap=boundary_gap_overlap(left.mask,right.mask)
                lv=velocities.get((phase_index,left.track_id),(0.0,0.0)); rv=velocities.get((phase_index,right.track_id),(0.0,0.0))
                relative=math.dist(lv,rv)
                phase_values.append((left.track_id,right.track_id,gap,overlap,relative))
        current_edges={(x[0],x[1]) for x in phase_values}
        lifecycle += len(current_edges.symmetric_difference(previous_edges)) if phase_index else len(current_edges)
        # Symmetric scalar moments make slot relabel irrelevant while retaining all edge channels.
        if phase_values:
            values.extend((sum(x[2] for x in phase_values)/len(phase_values),sum(x[3] for x in phase_values)/len(phase_values),sum(x[4] for x in phase_values)/len(phase_values),float(len(phase_values))))
        else: values.extend((1.0,0.0,0.0,0.0))
        previous_edges=current_edges
    lifecycle += len(previous_edges)
    values.append(float(lifecycle))
    return tuple(values),lifecycle


__all__=["AnonymousNodeV2","GraphV2Error","assign_anonymous_tracks","boundary_gap_overlap","break_mask_descriptor_binding","canonical_node_signature","hard_matches_with_dustbin","pairwise_edge_signature","patch_area_pool","relabel_slots","tubelet2_eight_blocks","unbalanced_matching_diagnostics","unbalanced_sinkhorn_dustbin"]
