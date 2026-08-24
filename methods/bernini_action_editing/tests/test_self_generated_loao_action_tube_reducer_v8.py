#!/usr/bin/env python3

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
import self_generated_loao_action_registry_v8 as registry
import self_generated_loao_action_tube_reducer_v8 as reducer


def h(value): return hashlib.sha256(str(value).encode()).hexdigest()


class Recomputer:
    prompt_encoder_sha256=h("prompt-encoder"); nontext_encoder_sha256=h("nontext-encoder")
    def prompt_embedding_sha256(self,caption): return h(("prompt",caption))
    def action_embedding_sha256(self,dynamics): return h(("action",dynamics))
    def state_embedding_sha256(self,state): return h(("state",registry.object_sha256(dict(state))))
    def middle_embedding_sha256(self,tensor): return h(("middle",reducer.tensor_sha256_v8(np.asarray(tensor))))


def runtime(): return registry.RuntimeIdentityV8(h("runtime"),Recomputer.prompt_encoder_sha256,Recomputer.nontext_encoder_sha256,h("projection"),h("frozen"),h("frozen"))


def moving_tensor(scale=8.0,reverse=False):
    value=np.zeros((21,37,25,4),dtype=np.float32)
    for phase in range(21):
        step=20-phase if reverse else phase; y=8+step//6; x=3+step//2
        value[phase,y:y+3,x:x+3,0]=scale
        value[phase,24:27,18-step//4:21-step//4,1]=scale*0.8
    return value


def neutral_tensor():
    yy,xx=np.meshgrid(np.linspace(-1,1,37),np.linspace(-1,1,25),indexing="ij")
    base=(0.03*np.stack((xx,yy,np.sin(xx*3),np.cos(yy*4)),axis=-1)).astype(np.float32)
    output=[]
    for phase in range(21):
        row=base.copy(); step=phase; y=8+step//6; x=3+step//2
        row[y:y+3,x:x+3]+=np.asarray((3.0,0.2,0.1,0.0),dtype=np.float32)
        row[24:27,18-step//4:21-step//4]+=np.asarray((0.1,3.0,0.2,0.0),dtype=np.float32)
        output.append(row)
    return np.stack(output)


def evidence(signature,branch,action,passed=True):
    gates={name:passed for name in ("primary_graph_valid","noop_pass","reverse_pass","phase_shuffle_pass","paraphrase_pass","lexical_placebo_pass","independent_null_pass","neutral_correspondence_pass")}
    body={"branch":branch,"action_id":action,"gates":gates,"branch_pass":all(gates.values()),"graph_signature":list(signature),"all_controls_recomputed":True}
    return {**body,"digest":registry.object_sha256(body)}


class Throwing:
    def __init__(self): self.called=False
    def zero_(self): self.called=True; raise RuntimeError("expected")


class NoopZero:
    def __init__(self): self.called=False
    def zero_(self): self.called=True
    def any(self): return True


class UnverifiableZero:
    def __init__(self): self.called=False
    def zero_(self): self.called=True


@dataclass
class Weird:
    nested: object


class ReducerV8Tests(unittest.TestCase):
    def graph_summary(self,**overrides):
        value=dict(
            signature=tuple([1.0]+[0.0]*15+[1.0]+[0.0]*89+[1.0]+[0.0]*29),
            support=tuple((1,2,3) for _ in range(10)),component_count=10,
            qualified_track_count=1,track_coverage=1.0,dynamic_edge_lifecycle_count=1,
            raw_dustbin_assignment_count=1,qualified_dustbin_assignment_count=1,
            raw_proposal_support_count=2,
            lifecycle_counts={"birth":1,"occlusion":0,"reentry":0,"death":0,"right_censored":1},
            endpoint_direction_xy=(0.2,0.0),vertical_velocity_early=-0.02,
            vertical_velocity_late=0.02,signed_winding_area=0.1,path_acceleration=0.03,
            neutral_correspondence_valid=True,neutral_visual_cosine_margin=0.2,
            neutral_top_vs_median_margin=0.3,neutral_top10_mass_fraction=0.8,
            neutral_nonpositional_std=0.1,
        ); value.update(overrides); return reducer.GraphSummaryV8(**value)

    def arm_matrix(self):
        neutral=neutral_tensor(); primary=moving_tensor(); reverse=moving_tensor(reverse=True); tiny=neutral*0.0001
        output={}
        for block in registry.BLOCKS:
            output[block]={"primary":primary.copy(),"paraphrase":primary.copy(),"reverse":reverse.copy(),"lexical_placebo":np.zeros_like(primary),"null_a":tiny.copy(),"null_b":(-tiny).copy(),"noop":np.zeros_like(primary),"neutral":neutral.copy()}
        return output

    def test_phase_shuffle_is_exact_proposal_bijection_without_eval_leakage(self):
        value=np.arange(21*37*25,dtype=np.float32).reshape(21,37,25,1)
        for branch in reducer.BRANCHES:
            output=reducer.proposal_only_phase_shuffle_v8(value,branch)
            self.assertTrue(np.array_equal(output[list(reducer.EVALUATION_PHASES[branch])],value[list(reducer.EVALUATION_PHASES[branch])]))
            self.assertEqual(set(reducer.PROPOSAL_SHUFFLES[branch]),set(reducer.PROPOSAL_PHASES[branch]))

    def test_factorization_removes_additive_carrier_effect_and_consumes_independent_null(self):
        carrier=np.full((21,37,25,2),7.0); arms={arm:np.full_like(carrier,index+1.0) for index,arm in enumerate(registry.ARMS)}
        first=reducer.factorize_carrier_arms_v8({key:value+carrier for key,value in arms.items()})
        second=reducer.factorize_carrier_arms_v8(arms)
        for arm in ("primary","paraphrase","reverse","lexical_placebo","noop","null_sanity"):
            self.assertTrue(np.array_equal(first[arm],second[arm]))
        changed=dict(arms); changed["null_b"]=changed["null_b"]+3
        self.assertFalse(np.array_equal(reducer.factorize_carrier_arms_v8(changed)["null_sanity"],second["null_sanity"]))

    def test_joint_full_geometry_graph_and_layer_fold_binding(self):
        primary=moving_tensor(); neutral=neutral_tensor()
        summary=reducer.graph_signature_v8(primary,neutral,"A_to_B",proposal_block_ids=(6,18),evaluation_block_ids=(12,24))
        self.assertEqual(len(summary.signature),136); self.assertGreaterEqual(summary.raw_proposal_support_count,1)
        self.assertIn("right_censored",summary.lifecycle_counts)
        with self.assertRaises(reducer.ReducerV8Error): reducer.graph_signature_v8(primary,neutral,"A_to_B",proposal_block_ids=(6,12),evaluation_block_ids=(18,24))

    def test_all_local_controls_are_internally_recomputed(self):
        row=reducer.evaluate_local_controls_v8(self.arm_matrix(),"A_to_B",action_id="transfer")
        self.assertEqual(set(row["gates"]),{"primary_graph_valid","noop_pass","reverse_pass","phase_shuffle_pass","paraphrase_pass","lexical_placebo_pass","independent_null_pass","neutral_correspondence_pass"})
        self.assertTrue(row["all_controls_recomputed"]); self.assertEqual(row["proposal_blocks"],[6,18]); self.assertEqual(row["evaluation_blocks"],[12,24])

    def make_bound_cell(self,wrong_middle=False):
        rt=runtime(); rec=Recomputer(); middle_tensor=np.arange(64,dtype=np.float32).reshape(4,4,4); middle_sha=reducer.tensor_sha256_v8(middle_tensor)
        arms=self.arm_matrix(); forwards={}; state=registry.APPEARANCE_BY_ID["appearance_0"]; action=registry.ACTION_BY_ID["transfer"]
        neutral_embedding=rec.prompt_embedding_sha256(registry.NEUTRAL_CAPTION)
        for arm in registry.ARMS:
            prompt=registry.PROMPT_BY_KEY[(state.appearance_id,action.action_id,arm)]; bindings={}; tensors={}
            for block in registry.BLOCKS:
                tensor=arms[block][arm]; tensors[block]=tensor
                bindings[block]=registry.CaptureBindingV8(state.appearance_id,action.action_id,arm,registry.SEED_IDS[0],"high",18,block,prompt.prompt_id,prompt.prompt_sha256,rec.prompt_embedding_sha256(prompt.caption),action.action_sha256,rec.action_embedding_sha256(action.dynamics),state.state_sha256,rec.state_embedding_sha256(state.body()),rec.middle_embedding_sha256(middle_tensor),neutral_embedding,h("carrier-cell"),h("wrong-middle") if wrong_middle else middle_sha,h(("timestep","high")),h("rotary-21x37x25"),rt.projection_sha256,reducer.tensor_sha256_v8(tensor),h(("forward-event",arm)),h(("four-block-invocation",arm)),rt.digest)
            forwards[arm]=reducer.BoundForwardV8(bindings,tensors)
        return forwards,reducer.BoundMiddleStateV8(registry.SEED_IDS[0],"high",state.appearance_id,action.action_id,middle_tensor),rt,rec

    def test_bound_cell_consumes_actual_middle_and_projected_tensors_then_scrubs(self):
        forwards,middle,rt,rec=self.make_bound_cell(); raw=[tensor for forward in forwards.values() for tensor in forward.projected_by_block.values()]
        receipt=reducer.reduce_bound_cell_v8(forwards,middle,rt,rec)
        self.assertFalse(receipt["representation_admitted"]); self.assertTrue(all(not np.any(tensor) for tensor in raw)); self.assertFalse(np.any(middle.tensor))
        forwards,middle,rt,rec=self.make_bound_cell(wrong_middle=True)
        with self.assertRaises(reducer.ReducerV8Error): reducer.reduce_bound_cell_v8(forwards,middle,rt,rec)
        self.assertFalse(np.any(middle.tensor))

    def test_recursive_alias_and_best_effort_exception_scrub(self):
        base=np.ones((4,4)); alias=base[:,1:]
        with self.assertRaises(reducer.ReducerV8Error): reducer.reject_aliased_ownership_v8(Weird({"a":[base],"b":({"deep":alias},)}))
        shared=[np.ones((2,))]
        with self.assertRaises(reducer.ReducerV8Error): reducer.reject_aliased_ownership_v8({"first":shared,"second":shared})
        good=np.ones((3,)); bad=Throwing(); noop=NoopZero(); unverifiable=UnverifiableZero()
        receipt=reducer.best_effort_scrub_v8(Weird({"bad":bad,"noop":noop,"unverifiable":unverifiable,"deep":[{"good":good}]}))
        self.assertTrue(bad.called); self.assertTrue(np.array_equal(good,np.zeros_like(good))); self.assertTrue(receipt["best_effort_completed"]); self.assertFalse(receipt["verified"])
        self.assertTrue(noop.called); self.assertTrue(unverifiable.called)
        self.assertTrue(any("not_zero" in row for row in receipt["failures"]))
        self.assertTrue(any("unverifiable_zero" in row for row in receipt["failures"]))

    def test_neutral_uot_assigns_cross_support_identity_and_true_gap_reentry(self):
        mask=np.zeros((37,25),dtype=bool); mask[8:10,8:10]=True
        descriptor=tuple([1.0]+[0.0]*15)
        first=reducer.ComponentV8(7,1,mask,(0.0,0.0),2.0,descriptor)
        returned=reducer.ComponentV8(99,7,mask,(0.01,0.0),2.0,descriptor)
        receipt=reducer.track_neutral_components_v8(((first,),(),(),(returned,)))
        self.assertEqual(receipt["track_ids_by_phase"][0],receipt["track_ids_by_phase"][3])
        self.assertNotEqual(receipt["proposal_support_ids_by_phase"][0],receipt["proposal_support_ids_by_phase"][3])
        self.assertEqual(receipt["lifecycle_counts"]["reentry"],1)
        self.assertFalse(receipt["proposal_support_id_used_as_track_id"])
        censored=reducer.track_neutral_components_v8(((first,),(),()))
        self.assertEqual(censored["lifecycle_counts"]["death"],0); self.assertEqual(censored["lifecycle_counts"]["right_censored"],1)
        dead=reducer.track_neutral_components_v8(((first,),(),(),()))
        self.assertEqual(dead["lifecycle_counts"]["death"],1); self.assertEqual(dead["lifecycle_counts"]["right_censored"],0)

    def test_action_specific_valid_fixtures_pass_all_local_gates_and_ephemeral_noop_fails(self):
        lexical=self.graph_summary(raw_proposal_support_count=0,qualified_track_count=0,component_count=0)
        noop=self.graph_summary(raw_proposal_support_count=0,qualified_track_count=0,component_count=0)
        shuffle=self.graph_summary(path_acceleration=0.05)
        fixtures={
            "transfer":(
                self.graph_summary(endpoint_direction_xy=(0.20,0.0)),
                self.graph_summary(endpoint_direction_xy=(0.18,0.0)),
                self.graph_summary(endpoint_direction_xy=(-0.20,0.0)),
            ),
            "lift_pause_return":(
                self.graph_summary(endpoint_direction_xy=(0.0,0.0),vertical_velocity_early=-0.02,vertical_velocity_late=0.02),
                self.graph_summary(endpoint_direction_xy=(0.0,0.0),vertical_velocity_early=-0.018,vertical_velocity_late=0.019),
                self.graph_summary(endpoint_direction_xy=(0.0,0.0),vertical_velocity_early=0.02,vertical_velocity_late=-0.02),
            ),
            "clockwise_orbit_return":(
                self.graph_summary(endpoint_direction_xy=(0.0,0.0),signed_winding_area=0.10),
                self.graph_summary(endpoint_direction_xy=(0.0,0.0),signed_winding_area=0.09),
                self.graph_summary(endpoint_direction_xy=(0.0,0.0),signed_winding_area=-0.10),
            ),
        }
        for action,(primary,paraphrase,reverse) in fixtures.items():
            decision=reducer.local_control_gates_v8(action,{"primary":primary,"paraphrase":paraphrase,"reverse":reverse,"lexical_placebo":lexical,"noop":noop},shuffle,0.1)
            self.assertTrue(all(decision["gates"].values()),(action,decision))
        primary,paraphrase,reverse=fixtures["transfer"]
        ephemeral=replace(noop,raw_proposal_support_count=1,qualified_track_count=0,component_count=0)
        rejected=reducer.local_control_gates_v8("transfer",{"primary":primary,"paraphrase":paraphrase,"reverse":reverse,"lexical_placebo":lexical,"noop":ephemeral},shuffle,0.1)
        self.assertFalse(rejected["gates"]["noop_pass"])

    def test_closed_loop_reverse_is_signed_winding_not_endpoint_cosine(self):
        loop=[(0,(1.0,0.0)),(1,(0.0,1.0)),(2,(-1.0,0.0)),(3,(0.0,-1.0)),(4,(1.0,0.0))]
        reversed_loop=[(index,point) for index,(_,point) in enumerate(reversed(loop))]
        forward=reducer.path_program_metrics_v8(loop); backward=reducer.path_program_metrics_v8(reversed_loop)
        self.assertGreater(abs(forward["signed_winding_area"]),0.01)
        self.assertLess(forward["signed_winding_area"]*backward["signed_winding_area"],0.0)
        self.assertEqual(loop[0][1],loop[-1][1]); self.assertEqual(reversed_loop[0][1],reversed_loop[-1][1])

    def test_v6_spatial_concentration_and_nonpositional_neutral_abstention(self):
        diffuse=np.zeros((21,37,25,4),dtype=np.float32)
        for phase in reducer.PROPOSAL_PHASES["A_to_B"]:
            frame=np.ones((37,25),dtype=np.float32); frame.reshape(-1)[:frame.size//2]=2.0
            diffuse[phase,:,:,0]=frame
        summary=reducer.graph_signature_v8(diffuse,neutral_tensor(),"A_to_B",proposal_block_ids=(6,18),evaluation_block_ids=(12,24))
        self.assertEqual(summary.raw_proposal_support_count,0)
        yy,xx=np.meshgrid(np.linspace(-1,1,37),np.linspace(-1,1,25),indexing="ij")
        positional=np.stack((xx,yy,np.sin(xx),np.cos(yy)),axis=-1).astype(np.float32)
        positional=np.stack([positional.copy() for _ in range(21)])
        summary=reducer.graph_signature_v8(moving_tensor(),positional,"A_to_B",proposal_block_ids=(6,18),evaluation_block_ids=(12,24))
        self.assertFalse(summary.neutral_correspondence_valid); self.assertEqual(summary.qualified_track_count,0)

    def test_loao_same_action_transfer_wrong_action_separation_and_held_independence(self):
        local={}
        for seed in registry.SEED_IDS:
          for sigma in registry.SIGMA_CELL_INDICES:
           for state_index,state in enumerate(registry.APPEARANCE_IDS):
            for action_index,action in enumerate(registry.ACTION_IDS):
             for branch in reducer.BRANCHES:
              vector=np.zeros(136); vector[action_index]=1; vector[16+action_index]=1; vector[106+action_index]=1; vector[3+state_index]=0.005; vector[20+(seed-registry.SEED_IDS[0])]=0.002
              local[(seed,sigma,state,action,branch)]=evidence(vector,branch,action)
        first=reducer.loao_reduce_signatures_v8(local); self.assertEqual(first["passed_cell_count"],54)
        chosen=next(row for row in first["rows"] if row["direction"]=="D_to_H" and row["sigma"]=="high" and row["held_appearance"]=="appearance_0" and row["action_id"]=="transfer")
        key=(registry.SEED_IDS[1],"high","appearance_0","transfer","A_to_B"); changed=dict(local); vector=np.asarray(changed[key]["graph_signature"]); vector[40]=9
        changed[key]=evidence(vector,"A_to_B","transfer")
        second=reducer.loao_reduce_signatures_v8(changed); chosen2=next(row for row in second["rows"] if row["direction"]=="D_to_H" and row["sigma"]=="high" and row["held_appearance"]=="appearance_0" and row["action_id"]=="transfer")
        self.assertEqual(chosen["branches"]["A_to_B"]["prototype_digest"],chosen2["branches"]["A_to_B"]["prototype_digest"])

    def test_official_matrix_consumes_and_aggregates_exact_54_cells(self):
        expected={(seed,sigma,state,action) for seed in registry.SEED_IDS for sigma in registry.SIGMA_CELL_INDICES for state in registry.APPEARANCE_IDS for action in registry.ACTION_IDS}
        rt=runtime(); rec=Recomputer(); cells={}; middle={}; raw=[]; b0=[]
        for cell_index,key in enumerate(sorted(expected)):
            seed,sigma,state_id,action_id=key; state=registry.APPEARANCE_BY_ID[state_id]; action=registry.ACTION_BY_ID[action_id]
            middle_tensor=np.asarray([cell_index+1.0],dtype=np.float32); middle[key]=reducer.BoundMiddleStateV8(seed,sigma,state_id,action_id,middle_tensor)
            middle_sha=reducer.tensor_sha256_v8(middle_tensor); forwards={}
            for arm in registry.ARMS:
                prompt=registry.PROMPT_BY_KEY[(state_id,action_id,arm)]; bindings={}; tensors={}
                for block in registry.BLOCKS:
                    tensor=np.full((21,37,25,1),cell_index+1,dtype=np.float16); raw.append(tensor); tensors[block]=tensor
                    bindings[block]=registry.CaptureBindingV8(state_id,action_id,arm,seed,sigma,registry.SIGMA_CELL_INDICES[sigma],block,prompt.prompt_id,prompt.prompt_sha256,rec.prompt_embedding_sha256(prompt.caption),action.action_sha256,rec.action_embedding_sha256(action.dynamics),state.state_sha256,rec.state_embedding_sha256(state.body()),rec.middle_embedding_sha256(middle_tensor),rec.prompt_embedding_sha256(registry.NEUTRAL_CAPTION),h(("carrier",seed,state_id,action_id)),middle_sha,h(("timestep",sigma)),h("rotary-21x37x25"),rt.projection_sha256,reducer.tensor_sha256_v8(tensor),h(("forward-event",seed,sigma,state_id,action_id,arm)),h(("four-block-invocation",seed,sigma,state_id,action_id,arm)),rt.digest)
                forwards[arm]=reducer.BoundForwardV8(bindings,tensors)
            cells[key]=forwards; output=h(("B0",key)); b0.append(registry.B0BindingV8(state_id,action_id,seed,sigma,registry.SIGMA_CELL_INDICES[sigma],middle_sha,output,output,0,rt.digest))
        consumed=[]
        def fake_local(_arms,branch,*,action_id):
            consumed.append((branch,action_id)); index=registry.ACTION_IDS.index(action_id)
            vector=np.zeros(136); vector[index]=1; vector[16+index]=1; vector[106+index]=1
            return evidence(vector,branch,action_id)
        with mock.patch.object(reducer,"evaluate_local_controls_v8",side_effect=fake_local):
            receipt=reducer.reduce_bound_matrix_v8(cells,middle,b0,rt,rec)
        self.assertEqual(len(consumed),108)
        self.assertTrue(receipt["aggregate"]["official_bound_capture_authority"])
        self.assertEqual(receipt["aggregate"]["passed_cell_count"],54)
        self.assertTrue(all(not np.any(row.tensor) for row in middle.values())); self.assertTrue(all(not np.any(row) for row in raw))

    def test_zero_evidence_is_zero_of_nine_per_direction_sigma_and_hard_false(self):
        local={(seed,sigma,state,action,branch):evidence(np.zeros(136),branch,action) for seed in registry.SEED_IDS for sigma in registry.SIGMA_CELL_INDICES for state in registry.APPEARANCE_IDS for action in registry.ACTION_IDS for branch in reducer.BRANCHES}
        result=reducer.loao_reduce_signatures_v8(local); self.assertEqual(result["passed_cell_count"],0)
        subset=[row for row in result["rows"] if row["direction"]=="D_to_H" and row["sigma"]=="high"]
        self.assertEqual(len(subset),9); self.assertEqual(sum(row["cell_pass"] for row in subset),0)
        hard=reducer.hard_false_receipt_v8(); self.assertTrue(hard["representation_admission_hard_false"]); self.assertFalse(hard["scientific_claim_authorized"])


if __name__=="__main__": unittest.main()
