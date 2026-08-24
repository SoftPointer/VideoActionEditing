#!/usr/bin/env python3

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import self_generated_loao_action_registry_v8 as registry


def h(value): return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class FakeRecomputer:
    prompt_encoder_sha256=h("prompt-encoder")
    nontext_encoder_sha256=h("nontext-encoder")
    def prompt_embedding_sha256(self,caption): return h(("prompt",caption))
    def action_embedding_sha256(self,dynamics): return h(("action",dynamics))
    def state_embedding_sha256(self,state): return h(("state",registry.object_sha256(dict(state))))
    def middle_embedding_sha256(self,tensor):
        import numpy as np
        return h(("middle",np.ascontiguousarray(tensor).tobytes()))


def runtime():
    return registry.RuntimeIdentityV8(h("runtime"),FakeRecomputer.prompt_encoder_sha256,FakeRecomputer.nontext_encoder_sha256,h("projection"),h("frozen"),h("frozen"))


def rows():
    rt=runtime(); rec=FakeRecomputer(); neutral=rec.prompt_embedding_sha256(registry.NEUTRAL_CAPTION); output=[]
    for seed in registry.SEED_IDS:
      for sigma,index in registry.SIGMA_CELL_INDICES.items():
       timestep=h(("timestep",sigma)); rotary=h("rotary-21x37x25")
       for prompt in registry.PROMPTS:
        action=registry.ACTION_BY_ID[prompt.action_id]; state=registry.APPEARANCE_BY_ID[prompt.state_id]
        carrier=h(("carrier",seed,prompt.state_id,prompt.action_id)); carrier_state=h(("middle",seed,sigma,prompt.state_id,prompt.action_id)); middle=h(("middle-embedding",carrier_state))
        for block in registry.BLOCKS:
         output.append(registry.CaptureBindingV8(prompt.state_id,prompt.action_id,prompt.arm,seed,sigma,index,block,prompt.prompt_id,prompt.prompt_sha256,rec.prompt_embedding_sha256(prompt.caption),action.action_sha256,rec.action_embedding_sha256(action.dynamics),state.state_sha256,rec.state_embedding_sha256(state.body()),middle,neutral,carrier,carrier_state,timestep,rotary,rt.projection_sha256,h(("projected",seed,sigma,prompt.prompt_id,block)),h(("forward-event",seed,sigma,prompt.prompt_id)),h(("four-block-invocation",seed,sigma,prompt.prompt_id)),rt.digest))
    return output


class RegistryV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rows=rows(); cls.runtime=runtime(); cls.recomputer=FakeRecomputer()

    def test_frozen_matrix_and_state_local_rendering(self):
        spec=registry.load_preregistration(); self.assertEqual(len(registry.PROMPTS),72)
        self.assertEqual(spec["frozen_base"]["expected_total_frozen_forward_invocations"],1440+432+54)
        self.assertEqual(spec["frozen_base"]["expected_projected_block_capture_rows"],432*4)
        self.assertEqual(len({row.state_sha256 for row in registry.APPEARANCES}),3)
        for prompt in registry.PROMPTS:
            if prompt.arm!="neutral":
                own=registry.APPEARANCE_BY_ID[prompt.state_id]
                self.assertTrue(all(noun in prompt.caption for noun in (own.actor,own.object,own.source,own.destination)))
        for state in registry.APPEARANCE_IDS:
            for arm in ("noop","null_a","null_b","neutral"):
                captions={registry.PROMPT_BY_KEY[(state,action,arm)].caption for action in registry.ACTION_IDS}
                self.assertEqual(len(captions),1)

    def test_full_binding_matrix_passes(self):
        receipt=registry.validate_capture_bindings_v8(self.rows,self.runtime,self.recomputer)
        self.assertTrue(receipt["verified"]); self.assertEqual(receipt["row_count"],1728)

    def test_caption_permutation_rejected(self):
        row=self.rows[0]
        with self.assertRaises(registry.RegistryV8Error):
            replace(row,prompt_sha256=registry.PROMPTS[8].prompt_sha256)

    def test_duplicate_embedding_rejected(self):
        bad=list(self.rows); target=next(i for i,row in enumerate(bad) if row.prompt_id!=bad[0].prompt_id and row.prompt_sha256!=bad[0].prompt_sha256)
        bad[target]=replace(bad[target],prompt_embedding_sha256=bad[0].prompt_embedding_sha256)
        with self.assertRaises(registry.RegistryV8Error): registry.validate_capture_bindings_v8(bad,self.runtime,self.recomputer)

    def test_state_embedding_duplicate_rejected(self):
        first=next(row for row in self.rows if row.state_id=="appearance_0")
        bad=[replace(row,state_embedding_sha256=first.state_embedding_sha256) if row.state_id=="appearance_1" else row for row in self.rows]
        with self.assertRaises(registry.RegistryV8Error): registry.validate_capture_bindings_v8(bad,self.runtime,self.recomputer)

    def test_forward_event_and_four_block_invocation_are_one_per_arm_forward(self):
        bad=list(self.rows); target=bad[4].forward_event_sha256
        selected=[index for index,row in enumerate(bad) if row.key[:5]==bad[0].key[:5]]
        for index in selected: bad[index]=replace(bad[index],forward_event_sha256=target)
        with self.assertRaises(registry.RegistryV8Error): registry.validate_capture_bindings_v8(bad,self.runtime,self.recomputer)
        bad=list(self.rows); bad[0]=replace(bad[0],four_block_invocation_sha256=h("wrong invocation"))
        with self.assertRaises(registry.RegistryV8Error): registry.validate_capture_bindings_v8(bad,self.runtime,self.recomputer)

    def test_b0_exact_54_schema_rows_and_pending_execution(self):
        b0=[]
        primary={(r.seed_id,r.sigma_name,r.state_id,r.action_id):r for r in self.rows if r.arm=="primary" and r.block==6}
        for key,row in primary.items():
            output=h(("b0",key)); b0.append(registry.B0BindingV8(row.state_id,row.action_id,row.seed_id,row.sigma_name,row.sigma_cell_index,row.carrier_state_sha256,output,output,0,self.runtime.digest))
        receipt=registry.validate_b0_bindings_v8(b0,self.rows,self.runtime)
        self.assertEqual(receipt["row_count"],54); self.assertFalse(receipt["execution_proven"])
        with self.assertRaises(registry.RegistryV8Error): replace(b0[0],matching_primary_output_sha256=h("different"))


if __name__=="__main__": unittest.main()
