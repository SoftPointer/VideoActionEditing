import unittest
from dataclasses import replace
from typing import Optional

import torch

from methods.bernini_action_editing import source_role_graph_preservation_v15b as core


SHA_A = "a" * 64
SHA_B = "b" * 64
SOURCE_LATENT_SHA = "e" * 64
EXTRACTOR_CODE_SHA = "c" * 64
EXTRACTOR_CONFIG_SHA = "d" * 64
ANCHOR_ASSETS = {
    "v0": "0" * 64,
    "v1": "1" * 64,
    "v2": "2" * 64,
    "v3": "3" * 64,
}
TRACK_RECEIPT_SHAS = ("4" * 64, "5" * 64, "6" * 64, "7" * 64)
HEIGHT = 9
WIDTH = 9
HEADS = 1
HEAD_DIM = 8
HIDDEN_WIDTH = 4


def position_reference(base_key):
    if tuple(base_key.shape[2:]) != (
            core.POSITION_CALIBRATION_HEADS,
            core.POSITION_CALIBRATION_HEAD_DIM):
        raise ValueError("test K geometry differs from the sealed calibration ABI")
    return core.build_position_counterfactual_reference_v15b()


def role_trace(slot: str, asset_sha256: Optional[str] = None) -> core.RoleContactTraceV15B:
    phase = torch.arange(core.LATENT_PHASES, dtype=torch.float32)
    energy = torch.stack(
        (
            phase,
            phase.square() / 20,
            torch.sin(phase / 7).abs(),
            torch.cos(phase / 9).abs(),
        ),
        dim=1,
    )
    return core.RoleContactTraceV15B.create(
        anchor_slot=slot,
        asset_sha256=asset_sha256 or ("9" * 64),
        extractor_code_sha256=EXTRACTOR_CODE_SHA,
        extractor_config_sha256=EXTRACTOR_CONFIG_SHA,
        energy=energy,
    )


def manual_warp(source, canonical, path):
    resample = torch.zeros(core.LATENT_PHASES, core.LATENT_PHASES)
    for canonical_phase in range(core.LATENT_PHASES):
        indices = [i for i, j in path if j == canonical_phase]
        resample[canonical_phase, indices] = 1 / len(indices)
    resample[0].zero_(); resample[0, 0] = 1
    resample[-1].zero_(); resample[-1, -1] = 1
    gate = core._warp_gate_metrics(path)
    payload = {
        "schema_version": core.WARP_SCHEMA,
        "source_slot": source.anchor_slot,
        "canonical_slot": canonical.anchor_slot,
        "source_trace_digest": source.digest,
        "canonical_trace_digest": canonical.digest,
        "path": [list(item) for item in path],
        "resample_sha256": core.tensor_sha256(resample),
        "max_phase_displacement": gate[0],
        "max_source_only_run": gate[1],
        "max_canonical_only_run": gate[2],
        "path_length": gate[3],
    }
    return core.MonotonicEventWarpV15B(
        core.WARP_SCHEMA,
        source.anchor_slot,
        canonical.anchor_slot,
        source.digest,
        canonical.digest,
        tuple(path),
        resample,
        gate[0],
        gate[1],
        gate[2],
        gate[3],
        core.object_sha256(payload),
    )


def identity_warp(trace, canonical=None):
    canonical = canonical or trace
    return manual_warp(trace, canonical, [(i, i) for i in range(core.LATENT_PHASES)])


def shifted_path():
    path = [(i, i) for i in range(8)]
    path += [(8, 7), (9, 7), (10, 8), (11, 9), (12, 10)]
    path += [(i, i - 2) for i in range(13, 21)]
    path += [(20, 19), (20, 20)]
    return path


def _put_centered_edge(graph, roles, edge, query_phases, key_phases, scale=1.0):
    query_role, key_role = edge
    qi = roles.index(query_role); ki = roles.index(key_role)
    for query_phase, key_phase in zip(query_phases, key_phases):
        graph[0, query_phase, qi, key_phase, ki] = scale
        graph[0, query_phase, qi, key_phase + 1, ki] = -scale


def anchor_tensor(*, shifted=False, wrong=False):
    graph = torch.zeros(1, 21, 3, 21, 3)
    query_phases = (10, 11, 12) if shifted else (8, 9, 10)
    key_phases = (8, 9, 10) if shifted else (6, 7, 8)
    _put_centered_edge(
        graph, core.GENERIC_ROLES, ("human_agent", "moving_object"),
        query_phases, key_phases, 0.75,
    )
    recipient_key_phases = tuple(phase + 3 for phase in key_phases) if wrong else key_phases
    _put_centered_edge(
        graph, core.GENERIC_ROLES, ("moving_object", "recipient"),
        query_phases, recipient_key_phases, 1.0,
    )
    if wrong:
        _put_centered_edge(
            graph, core.GENERIC_ROLES, ("recipient", "moving_object"),
            query_phases, key_phases, 1.0,
        )
    return graph


def source_tensor():
    graph = torch.zeros(1, 21, 4, 21, 4)
    for edge, scale in (
        (("human_agent", "old_actor"), 0.8),
        (("old_actor", "moving_object"), 1.0),
    ):
        _put_centered_edge(
            graph, core.SIGNED_ROLES, edge, (8, 9, 10), (6, 7, 8), scale,
        )
    return graph


def bank(slot, trace, *, shifted=False, wrong=False):
    graph = core.AnchorRelationGraphV15B.create(
        action_id="pour", graph=anchor_tensor(shifted=shifted, wrong=wrong), confidence=1.0
    )
    return core.AnchorGraphBankV15B.create(
        action_id="pour", graph_set_id=f"pour_{slot}", anchor_slot=slot,
        timing_trace_digest=trace.digest, relation_graph=graph,
    )


def binding():
    return core.SourceActionRoleBindingV15B.create(
        action_id="pour", source_iid="event00",
        human_agent_source_role="woman",
        old_actor_source_role="white_ceramic_vessel_1",
        moving_object_source_role="clear_glass_pitcher_2",
        recipient_source_role="small_white_cup_3",
    )


def source_role_masks(bound):
    shape = (1, core.LATENT_PHASES, HEIGHT, WIDTH)
    masks = {role: torch.zeros(shape, dtype=torch.bool) for role in bound.source_roles}
    for phase in range(core.LATENT_PHASES):
        masks[bound.human_agent_source_role][0, phase, 4, 0:2] = True
        masks[bound.old_actor_source_role][0, phase, 1, 1:3] = True
        moving_x = 4 if phase < 11 else 5
        masks[bound.moving_object_source_role][0, phase, 4:6, moving_x] = True
        masks[bound.recipient_source_role][0, phase, 7, 7:9] = True
    return {role: mask.reshape(1, -1) for role, mask in masks.items()}


def mask_set(bound):
    masks = source_role_masks(bound)
    contact = torch.zeros(1, core.LATENT_PHASES, HEIGHT, WIDTH, dtype=torch.bool)
    contact[:, 7:15, 4, 2:5] = True
    receipts = dict(zip(bound.source_roles, TRACK_RECEIPT_SHAS))
    return core.SourceRoleMaskSetV15B.create(
        source_video_sha256=SHA_A, binding=bound, role_masks=masks,
        contact_mask=contact.reshape(1, -1), height=HEIGHT, width=WIDTH,
        role_track_receipt_sha256=receipts,
    )


def graph_fixture():
    traces = {
        slot: role_trace(slot, ANCHOR_ASSETS[slot])
        for slot in core.DIAGNOSTIC_ANCHOR_SLOTS
    }
    canonical = traces["v0"]
    banks = {
        "v0": bank("v0", traces["v0"]),
        "v1": bank("v1", traces["v1"]),
        "v2": bank("v2", traces["v2"]),
        "v3": bank("v3", traces["v3"]),
    }
    warps = {
        "v0": identity_warp(traces["v0"], canonical),
        "v1": identity_warp(traces["v1"], canonical),
        "v2": identity_warp(traces["v2"], canonical),
        "v3": identity_warp(traces["v3"], canonical),
    }
    source_trace = role_trace("v9", "8" * 64)
    source_graph = core.SourceRelationGraphV15B.create(
        action_id="pour", timing_trace_digest=source_trace.digest,
        graph=source_tensor(),
    )
    source_warp = identity_warp(source_trace, canonical)
    signed_a = core.build_signed_edit_graph_v15b(
        source_graph=source_graph, anchor_bank=banks["v0"],
        source_warp=source_warp, anchor_warp=warps["v0"],
    )
    signed_b = core.build_signed_edit_graph_v15b(
        source_graph=source_graph, anchor_bank=banks["v1"],
        source_warp=source_warp, anchor_warp=warps["v1"],
    )
    report = core.compare_anchor_graphs_v15b(
        banks["v0"], banks["v1"], warps["v0"], warps["v1"]
    )
    consensus = core.diagnose_four_anchor_consensus_v15b(
        list(banks.values()), list(warps.values()), list(traces.values()), canonical
    )
    return {
        "traces": traces, "canonical": canonical, "banks": banks, "warps": warps,
        "source_trace": source_trace, "source_graph": source_graph,
        "source_warp": source_warp, "signed_a": signed_a, "signed_b": signed_b,
        "report": report, "consensus": consensus,
    }


def carrier_and_memory(bound, masks, *, step=3, block=4, branch="conditional"):
    tokens = core.LATENT_PHASES * HEIGHT * WIDTH
    source_hidden = torch.full((1, tokens, HIDDEN_WIDTH), 7.0)
    source_key = torch.full((1, tokens, HEADS, HEAD_DIM), 8.0)
    source_value = torch.full((1, tokens, HEADS, HEAD_DIM), 9.0)
    post_phase0_corridor = masks.editable_corridor_mask.clone()
    post_phase0_corridor[:, :HEIGHT * WIDTH] = False
    source_hidden[post_phase0_corridor] = 0
    source_key[post_phase0_corridor] = 0
    source_value[post_phase0_corridor] = 0
    raw_key = torch.zeros(1, tokens, HEADS, HEAD_DIM)
    raw_key[..., 4] = 1.0  # phase-0 background/null content basis
    raw_key[..., 7] = torch.arange(tokens, dtype=torch.float32).reshape(1, -1, 1) % 7
    raw_value = torch.zeros_like(raw_key)
    for role_index, role in enumerate(sorted(bound.source_roles)):
        phase0_mask = masks.role_masks[role].clone()
        phase0_mask[:, HEIGHT * WIDTH:] = False
        selected = torch.zeros(int(phase0_mask.sum()), HEADS, HEAD_DIM)
        selected[..., role_index] = 1.0
        raw_key[phase0_mask] = selected
        raw_value[phase0_mask] = float(role_index + 1)
    # Full phase 0 is one raw authority.  Later background/support remains the
    # explicitly caller-supplied carrier material exercised by this fixture.
    raw_hidden = source_hidden.clone()
    source_key[:, :HEIGHT * WIDTH] = raw_key[:, :HEIGHT * WIDTH]
    source_value[:, :HEIGHT * WIDTH] = raw_value[:, :HEIGHT * WIDTH]
    reference = position_reference(raw_key)
    memory = core.build_source_role_content_memory_v15b(
        source_video_sha256=SHA_A, source_latent_sha256=SOURCE_LATENT_SHA,
        binding=bound, masks=masks,
        step_index=step, block_index=block, branch=branch,
        source_hidden=raw_hidden,
        source_pre_rope_key=raw_key, source_value=raw_value,
        position_reference=reference,
    )
    carrier = core.SourceBackgroundCarrierV15B.create(
        source_video_sha256=SHA_A, binding=bound, masks=masks,
        step_index=step, block_index=block, branch=branch,
        hidden=source_hidden, key=source_key, value=source_value,
        raw_source_material=memory.raw_source_material,
    )
    return carrier, memory, raw_key, raw_value, reference


def native_motion_from_tracks(bound, masks, tracks, memory, *, step=3, block=4,
                              branch="conditional"):
    reference = core.build_target_native_motion_reference_v15b(
        role_physical_candidate_masks={
            role: tracks[role] for role in memory.role_ids
        },
        height=HEIGHT, width=WIDTH,
    )
    transport = core.build_target_native_transport_v15b(
        source_video_sha256=SHA_A, binding=bound, masks=masks,
        step_index=step, block_index=block, branch=branch,
        motion_reference=reference,
    )
    return transport


def persistent_target_fixture(bound, masks, memory, *, move_phase=6):
    spatial = HEIGHT * WIDTH
    tracks = {
        role: torch.zeros(core.LATENT_PHASES, HEIGHT, WIDTH, dtype=torch.bool)
        for role in memory.role_ids
    }
    for role in memory.role_ids:
        tracks[role][0] = masks.role_masks[role].reshape(
            1, core.LATENT_PHASES, HEIGHT, WIDTH
        )[0, 0]
        for phase in range(1, core.LATENT_PHASES):
            tracks[role][phase] = tracks[role][phase - 1]
    moving = bound.moving_object_source_role
    for phase in range(move_phase, core.LATENT_PHASES):
        tracks[moving][phase].zero_()
        tracks[moving][phase, 4:6, 5] = True
    transport = native_motion_from_tracks(
        bound, masks, tracks, memory, step=memory.step_index,
        block=memory.block_index, branch=memory.branch,
    )
    key = torch.zeros(1, core.LATENT_PHASES * spatial, HEADS, HEAD_DIM)
    key[..., 4] = 1.0
    key[..., 7] = torch.arange(
        core.LATENT_PHASES * spatial, dtype=torch.float32
    ).reshape(1, -1, 1) % 5
    for role_index, role in enumerate(memory.role_ids):
        packed = tracks[role].reshape(1, -1).clone()
        packed[:, :spatial] = False
        selected = torch.zeros(int(packed.sum()), HEADS, HEAD_DIM)
        selected[..., role_index] = 1.0
        selected[..., 7] = 3.0
        key[packed] = selected
    return key, transport, tracks


class RelationGraphAndTraceTest(unittest.TestCase):
    def test_anchor_requires_confident_add_edges_with_phase_coverage(self):
        valid = core.AnchorRelationGraphV15B.create(
            action_id="pour", graph=anchor_tensor(), confidence=0.95
        )
        self.assertEqual(valid.graph.shape, (1, 21, 3, 21, 3))
        with self.assertRaises(core.V15BContractError):
            core.AnchorRelationGraphV15B.create(
                action_id="pour", graph=anchor_tensor(), confidence=0.949
            )
        missing = anchor_tensor()
        human = core.GENERIC_ROLES.index("human_agent")
        moving = core.GENERIC_ROLES.index("moving_object")
        missing[:, :, human, :, moving] = 0
        with self.assertRaises(core.V15BContractError):
            core.AnchorRelationGraphV15B.create(action_id="pour", graph=missing)
        one_phase = anchor_tensor()
        one_phase[:, 10] = 0
        with self.assertRaises(core.V15BContractError):
            core.AnchorRelationGraphV15B.create(action_id="pour", graph=one_phase)

    def test_source_requires_both_old_relation_remove_edges(self):
        graph = source_tensor()
        human = core.SIGNED_ROLES.index("human_agent")
        old = core.SIGNED_ROLES.index("old_actor")
        graph[:, :, human, :, old] = 0
        trace = role_trace("v9")
        with self.assertRaises(core.V15BContractError):
            core.SourceRelationGraphV15B.create(
                action_id="pour", timing_trace_digest=trace.digest, graph=graph
            )

    def test_signed_graph_has_exact_required_add_remove_and_phase0_zero(self):
        fixture = graph_fixture(); signed = fixture["signed_a"]
        for edge in core.ACTION_REQUIRED_ADD_EDGES["pour"]:
            norm, coverage = core._edge_norm_and_query_phase_coverage(
                signed.add_component, core.SIGNED_ROLES, edge
            )
            self.assertGreaterEqual(norm, core.REQUIRED_EDGE_MIN_NORM)
            self.assertGreaterEqual(coverage, core.REQUIRED_EDGE_MIN_QUERY_PHASES)
            self.assertGreaterEqual(
                core._edge_key_phase_coverage(signed.add_component, core.SIGNED_ROLES, edge),
                core.REQUIRED_EDGE_MIN_KEY_PHASES,
            )
        for edge in core.ACTION_REQUIRED_REMOVE_EDGES["pour"]:
            norm, coverage = core._edge_norm_and_query_phase_coverage(
                signed.remove_component, core.SIGNED_ROLES, edge
            )
            self.assertGreaterEqual(norm, core.REQUIRED_EDGE_MIN_NORM)
            self.assertGreaterEqual(coverage, core.REQUIRED_EDGE_MIN_QUERY_PHASES)
            self.assertGreaterEqual(
                core._edge_key_phase_coverage(
                    signed.remove_component, core.SIGNED_ROLES, edge
                ),
                core.REQUIRED_EDGE_MIN_KEY_PHASES,
            )
        self.assertEqual(signed.disallowed_add_edge_max_abs, 0)
        self.assertEqual(signed.disallowed_remove_edge_max_abs, 0)
        self.assertEqual(int(torch.count_nonzero(signed.graph[:, 0])), 0)

    def test_create_rejects_dtype_bool_and_nonfinite_laundering(self):
        with self.assertRaises(core.V15BContractError):
            core.AnchorRelationGraphV15B.create(
                action_id="pour", graph=anchor_tensor().double(), confidence=1.0
            )
        with self.assertRaises(core.V15BContractError):
            core.AnchorRelationGraphV15B.create(
                action_id="pour", graph=anchor_tensor(), confidence=True
            )
        poisoned = anchor_tensor(); poisoned[0, 8, 0, 6, 1] = float("nan")
        with self.assertRaises(core.V15BContractError):
            core.AnchorRelationGraphV15B.create(
                action_id="pour", graph=poisoned, confidence=1.0
            )
        too_weak = anchor_tensor() * 0.01
        with self.assertRaises(core.V15BContractError):
            core.AnchorRelationGraphV15B.create(
                action_id="pour", graph=too_weak, confidence=1.0
            )

    def test_timing_alignment_and_unique_v0_v3_authority(self):
        fixture = graph_fixture()
        self.assertTrue(fixture["report"].passed)
        consensus = fixture["consensus"]
        self.assertTrue(consensus.robust_passed)
        self.assertEqual(
            tuple((pair.slot_a, pair.slot_b) for pair in consensus.pairs),
            tuple(__import__("itertools").combinations(core.DIAGNOSTIC_ANCHOR_SLOTS, 2)),
        )
        self.assertEqual(consensus.canonical_trace_digest, fixture["canonical"].digest)
        duplicate_asset = core.RoleContactTraceV15B.create(
            anchor_slot="v3", asset_sha256=fixture["traces"]["v2"].asset_sha256,
            extractor_code_sha256=EXTRACTOR_CODE_SHA,
            extractor_config_sha256=EXTRACTOR_CONFIG_SHA,
            energy=fixture["traces"]["v3"].energy,
        )
        duplicate_bank = bank("v3", duplicate_asset)
        duplicate_warp = identity_warp(duplicate_asset, fixture["canonical"])
        with self.assertRaises(core.V15BContractError):
            core.diagnose_four_anchor_consensus_v15b(
                [fixture["banks"]["v0"], fixture["banks"]["v1"],
                 fixture["banks"]["v2"], duplicate_bank],
                [fixture["warps"]["v0"], fixture["warps"]["v1"],
                 fixture["warps"]["v2"], duplicate_warp],
                [fixture["traces"]["v0"], fixture["traces"]["v1"],
                 fixture["traces"]["v2"], duplicate_asset], fixture["canonical"],
            )

    def test_alignment_does_not_hide_role_relation_change(self):
        fixture = graph_fixture()
        wrong = bank("v1", fixture["traces"]["v1"], wrong=True)
        report = core.compare_anchor_graphs_v15b(
            fixture["banks"]["v0"], wrong,
            fixture["warps"]["v0"], fixture["warps"]["v1"],
        )
        self.assertFalse(report.passed)


class SourceCorridorTest(unittest.TestCase):
    def setUp(self):
        self.bound = binding(); self.masks = mask_set(self.bound)

    def test_corridor_is_source_only_deterministic_and_phase0_empty(self):
        duplicate = mask_set(self.bound)
        self.assertEqual(self.masks.digest, duplicate.digest)
        self.assertTrue(torch.equal(
            self.masks.editable_corridor_mask, duplicate.editable_corridor_mask
        ))
        spatial = HEIGHT * WIDTH
        self.assertFalse(self.masks.editable_corridor_mask[:, :spatial].any())
        self.assertTrue(self.masks.background_support_mask[:, :spatial].all())
        self.assertTrue(torch.all(
            self.masks.editable_corridor_mask[self.masks.transition_path_mask]
        ))
        self.assertFalse(torch.any(
            self.masks.background_support_mask & self.masks.transition_path_mask
        ))
        background = self.masks.background_support_mask.reshape(1, 21, HEIGHT, WIDTH)
        self.assertTrue(background.any(dim=(2, 3)).all())

    def test_vessel_overlap_or_missing_phase_fails_closed(self):
        masks = source_role_masks(self.bound)
        old = self.bound.old_actor_source_role
        moving = self.bound.moving_object_source_role
        masks[moving][:] = masks[old]
        receipts = dict(zip(self.bound.source_roles, TRACK_RECEIPT_SHAS))
        contact = torch.zeros(1, 21 * HEIGHT * WIDTH, dtype=torch.bool)
        with self.assertRaises(core.V15BContractError):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound, role_masks=masks,
                contact_mask=contact, height=HEIGHT, width=WIDTH,
                role_track_receipt_sha256=receipts,
            )

    def test_human_vessel_overlap_is_rejected_not_treated_as_contact(self):
        moving = self.bound.moving_object_source_role
        masks = source_role_masks(self.bound)
        masks[self.bound.human_agent_source_role][:] = masks[
            self.bound.moving_object_source_role
        ]
        receipts = dict(zip(self.bound.source_roles, TRACK_RECEIPT_SHAS))
        contact = torch.zeros(1, 21 * HEIGHT * WIDTH, dtype=torch.bool)
        with self.assertRaisesRegex(core.V15BContractError, "human/#1/#2/#3"):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound, role_masks=masks,
                contact_mask=contact, height=HEIGHT, width=WIDTH,
                role_track_receipt_sha256=receipts,
            )
        masks = source_role_masks(self.bound)
        masks[moving].reshape(1, 21, HEIGHT, WIDTH)[:, 10].zero_()
        with self.assertRaises(core.V15BContractError):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound, role_masks=masks,
                contact_mask=contact, height=HEIGHT, width=WIDTH,
                role_track_receipt_sha256=receipts,
            )

    def test_track_receipts_are_distinct_and_geometry_is_explicit(self):
        masks = source_role_masks(self.bound)
        contact = torch.zeros(1, 21 * HEIGHT * WIDTH, dtype=torch.bool)
        duplicate = {role: "4" * 64 for role in self.bound.source_roles}
        with self.assertRaises(core.V15BContractError):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound, role_masks=masks,
                contact_mask=contact, height=HEIGHT, width=WIDTH,
                role_track_receipt_sha256=duplicate,
            )
        receipts = dict(zip(self.bound.source_roles, TRACK_RECEIPT_SHAS))
        with self.assertRaises(core.V15BContractError):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound, role_masks=masks,
                contact_mask=contact, height=HEIGHT - 1, width=WIDTH,
                role_track_receipt_sha256=receipts,
            )

    def test_mutated_corridor_is_rejected(self):
        changed = self.masks.editable_corridor_mask.clone()
        changed[:, HEIGHT * WIDTH + 8] ^= True
        with self.assertRaises(core.V15BContractError):
            replace(self.masks, editable_corridor_mask=changed)

    def test_oversized_corridor_disconnected_track_and_bool_laundering_fail(self):
        role_masks = source_role_masks(self.bound)
        receipts = dict(zip(self.bound.source_roles, TRACK_RECEIPT_SHAS))
        oversized_contact = torch.ones(
            1, core.LATENT_PHASES * HEIGHT * WIDTH, dtype=torch.bool
        )
        with self.assertRaises(core.V15BContractError):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound,
                role_masks=role_masks, contact_mask=oversized_contact,
                height=HEIGHT, width=WIDTH,
                role_track_receipt_sha256=receipts,
            )
        disconnected = source_role_masks(self.bound)
        moving = disconnected[self.bound.moving_object_source_role].reshape(
            1, core.LATENT_PHASES, HEIGHT, WIDTH
        )
        moving[:, 5].zero_(); moving[:, 5, 0, 0] = True; moving[:, 5, 8, 8] = True
        empty_contact = torch.zeros_like(oversized_contact)
        with self.assertRaises(core.V15BContractError):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound,
                role_masks=disconnected, contact_mask=empty_contact,
                height=HEIGHT, width=WIDTH,
                role_track_receipt_sha256=receipts,
            )
        uint8_masks = dict(role_masks)
        uint8_masks[self.bound.human_agent_source_role] = uint8_masks[
            self.bound.human_agent_source_role
        ].to(torch.uint8)
        with self.assertRaises(core.V15BContractError):
            core.SourceRoleMaskSetV15B.create(
                source_video_sha256=SHA_A, binding=self.bound,
                role_masks=uint8_masks, contact_mask=empty_contact,
                height=HEIGHT, width=WIDTH,
                role_track_receipt_sha256=receipts,
            )


class SourcePropertyAndPersistentSlotTest(unittest.TestCase):
    def setUp(self):
        self.bound = binding(); self.masks = mask_set(self.bound)
        self.fixture = graph_fixture(); self.signed = self.fixture["signed_a"]
        (self.carrier, self.memory, self.raw_key, self.raw_value,
         self.position_reference) = carrier_and_memory(self.bound, self.masks)
        self.native_key, self.transport, self.tracks = persistent_target_fixture(
            self.bound, self.masks, self.memory
        )
        self.tokens = core.LATENT_PHASES * HEIGHT * WIDTH

    def target(self):
        return (
            torch.zeros(1, self.tokens, HIDDEN_WIDTH),
            torch.zeros(1, self.tokens, HEADS, HEAD_DIM),
            self.native_key.clone(),
            torch.zeros(1, self.tokens, HEADS, HEAD_DIM),
        )

    def token(self, phase, y, x):
        return phase * HEIGHT * WIDTH + y * WIDTH + x

    def state(self, key=None, transport=None, memory=None):
        return core.build_target_role_state_v15b(
            native_target_pre_rope_key=self.native_key if key is None else key,
            memory=self.memory if memory is None else memory,
            masks=self.masks, binding=self.bound,
            target_native_transport=self.transport if transport is None else transport,
        )

    def apply(self, *, key=None, query=None, value=None, memory=True, route=True,
              transport=None, restore=False):
        hidden, native_query, native_key, native_value = self.target()
        return core.apply_pre_block_v15b(
            target_hidden=hidden,
            target_query=native_query if query is None else query,
            target_key=native_key if key is None else key,
            target_value=native_value if value is None else value,
            carrier=self.carrier, binding=self.bound,
            signed_graph=self.signed if route else None,
            content_memory=self.memory if memory else None,
            target_native_transport=(
                (self.transport if transport is None else transport) if memory else None
            ),
            route_strength=1.0 if route else 0.0,
            memory_strength=1.0 if memory else 0.0,
            restore_background=restore,
        )

    def test_persistent_track_moves_editable_support_and_releases_old_position(self):
        state = self.state(); phase = 8
        moved = self.token(phase, 4, 5); old = self.token(phase, 4, 4)
        role_index = self.memory.role_ids.index(self.bound.moving_object_source_role)
        self.assertEqual(int(state.confident_role_index[0, moved]), role_index)
        self.assertEqual(int(state.confident_role_index[0, old]), -1)
        self.assertEqual(state.cross_role_rename_count, 0)
        self.assertEqual(state.corridor_escape_count, 0)
        self.assertEqual(state.dual_position_component_count, 0)
        self.assertEqual(
            state.assignment_kind,
            "previous_owner_synthetic_transport_reference_then_k_retain_or_unassign",
        )
        self.assertFalse(state.route_authorized)
        self.assertFalse(state.position_removed_claimed)
        self.assertFalse(state.native_flow_claimed)
        self.assertFalse(state.scientific_claim_authorized)
        self.assertTrue(all(
            len(phases) == core.LATENT_PHASES and
            all(holes == 0 and euler == 1 and occlusion is False
                for holes, euler, _boundary, occlusion in phases)
            for _role, phases in state.role_topology_by_phase
        ))

    def test_whitening_can_only_unassign_never_self_reinforce_as_old_actor(self):
        key = self.native_key.clone(); phase = 8
        moving = self.tracks[self.bound.moving_object_source_role][phase].reshape(-1)
        tokens = torch.nonzero(moving).flatten() + phase * HEIGHT * WIDTH
        old_index = self.memory.role_ids.index(self.bound.old_actor_source_role)
        key[:, tokens].zero_(); key[:, tokens, :, old_index] = 1.0
        with self.assertRaises(core.V15BContractError):
            self.state(key=key)

        one = self.native_key.clone(); token = int(tokens[0])
        one[:, token].zero_(); one[:, token, :, old_index] = 1.0
        with self.assertRaisesRegex(core.V15BContractError, "role-specific"):
            self.state(key=one)

    def test_global_feature_permutation_and_number1_number3_swap_do_not_rename(self):
        key = self.native_key.clone()
        old = self.bound.old_actor_source_role
        recipient = self.bound.recipient_source_role
        old_index = self.memory.role_ids.index(old)
        recipient_index = self.memory.role_ids.index(recipient)
        for phase in range(1, core.LATENT_PHASES):
            for source_role, wrong_index in (
                (old, recipient_index), (recipient, old_index),
            ):
                tokens = torch.nonzero(
                    self.tracks[source_role][phase].reshape(-1)
                ).flatten() + phase * HEIGHT * WIDTH
                key[:, tokens].zero_(); key[:, tokens, :, wrong_index] = 1.0
        with self.assertRaises(core.V15BContractError):
            self.state(key=key)

    def test_partial_number1_number3_exchange_inside_other_physical_mask_is_ghost(self):
        key = self.native_key.clone(); phase = 8
        old_index = self.memory.role_ids.index(self.bound.old_actor_source_role)
        recipient_tokens = torch.nonzero(
            self.tracks[self.bound.recipient_source_role][phase].reshape(-1)
        ).flatten() + phase * HEIGHT * WIDTH
        self.assertEqual(int(recipient_tokens.numel()), 2)
        token = int(recipient_tokens[0])
        key[:, token].zero_(); key[:, token, :, old_index] = 1.0
        with self.assertRaisesRegex(core.V15BContractError, "role-specific"):
            self.state(key=key)

    def test_global_spatial_permutation_is_equivariant_when_all_authorities_move(self):
        flip_packed = lambda tensor: tensor.reshape(
            1, core.LATENT_PHASES, HEIGHT, WIDTH, *tensor.shape[2:]
        ).flip(3).reshape_as(tensor)
        flipped_role_masks = {
            role: flip_packed(mask) for role, mask in self.masks.role_masks.items()
        }
        flipped_contact = flip_packed(self.masks.contact_mask)
        receipts = dict(zip(self.bound.source_roles, TRACK_RECEIPT_SHAS))
        flipped_masks = core.SourceRoleMaskSetV15B.create(
            source_video_sha256=SHA_A, binding=self.bound,
            role_masks=flipped_role_masks, contact_mask=flipped_contact,
            height=HEIGHT, width=WIDTH,
            role_track_receipt_sha256=receipts,
        )
        flipped_source_key = flip_packed(self.raw_key)
        flipped_source_value = flip_packed(self.raw_value)
        flipped_memory = core.build_source_role_content_memory_v15b(
            source_video_sha256=SHA_A, source_latent_sha256=SOURCE_LATENT_SHA,
            binding=self.bound, masks=flipped_masks,
            step_index=3, block_index=4, branch="conditional",
            source_hidden=flip_packed(self.carrier.hidden),
            source_pre_rope_key=flipped_source_key,
            source_value=flipped_source_value,
            position_reference=position_reference(flipped_source_key),
        )
        spatial = HEIGHT * WIDTH
        flip_index = torch.tensor([
            y * WIDTH + (WIDTH - 1 - x)
            for y in range(HEIGHT) for x in range(WIDTH)
        ], dtype=torch.int64)
        flipped_tracks = {
            role: self.tracks[role].flip(2) for role in self.memory.role_ids
        }
        flipped_transport = native_motion_from_tracks(
            self.bound, flipped_masks, flipped_tracks, flipped_memory,
        )
        flipped_key = flip_packed(self.native_key)
        original = self.state()
        flipped = core.build_target_role_state_v15b(
            native_target_pre_rope_key=flipped_key, memory=flipped_memory,
            masks=flipped_masks, binding=self.bound,
            target_native_transport=flipped_transport,
        )
        expected = flip_packed(original.confident_role_index)
        self.assertTrue(torch.equal(flipped.confident_role_index, expected))
        self.assertEqual(flipped.cross_role_rename_count, 0)

    def test_source_v_reads_only_fixed_slot_and_follows_moved_track(self):
        _, _, key, value = self.target(); phase = 8
        source_old = self.token(phase, 4, 4)
        moved_target = self.token(phase, 4, 5)
        value[:, source_old] = 37.0; value[:, moved_target] = -11.0
        state = self.apply(key=key, value=value, route=False)
        role_index = self.memory.role_ids.index(self.bound.moving_object_source_role)
        self.assertEqual(
            int(state.target_role_state.confident_role_index[0, moved_target]), role_index
        )
        self.assertEqual(
            int(state.target_role_state.confident_role_index[0, source_old]), -1
        )
        self.assertTrue(torch.equal(state.value[:, source_old], value[:, source_old]))
        self.assertFalse(torch.equal(state.value[:, moved_target], value[:, moved_target]))
        self.assertEqual(float(state.appearance_residual[:, source_old].abs().sum()), 0.0)
        self.assertGreater(float(state.appearance_residual[:, moved_target].abs().sum()), 0.0)

    def test_old_and_new_position_components_fail_as_ghost(self):
        phase = 6
        tracks = {role: value.clone() for role, value in self.tracks.items()}
        moving = tracks[self.bound.moving_object_source_role]
        for current_phase in range(phase, core.LATENT_PHASES):
            moving[current_phase].zero_()
            moving[current_phase, 4, 4] = True
            moving[current_phase, 5, 5] = True
        with self.assertRaisesRegex(
                core.V15BContractError, "integer translation|shape/topology"):
            native_motion_from_tracks(
                self.bound, self.masks, tracks, self.memory
            )

    def test_untransported_high_affinity_new_number2_component_is_rejected(self):
        key = self.native_key.clone(); phase = 8
        corridor = self.masks.editable_corridor_mask.reshape(
            1, core.LATENT_PHASES, HEIGHT, WIDTH
        )[0, phase]
        physical = self.transport.motion_reference.physical_candidate_mask.reshape(
            1, core.LATENT_PHASES, HEIGHT, WIDTH
        )[0, phase]
        candidates = torch.nonzero(corridor & ~physical, as_tuple=False)
        self.assertGreater(int(candidates.shape[0]), 0)
        y, x = (int(value) for value in candidates[0])
        token = self.token(phase, y, x)
        moving_index = self.memory.role_ids.index(
            self.bound.moving_object_source_role
        )
        key[:, token].zero_(); key[:, token, :, moving_index] = 1.0
        with self.assertRaisesRegex(core.V15BContractError, "role-specific"):
            self.state(key=key)

    def test_transport_outside_fixed_corridor_fails_not_silently_clips(self):
        phase = 6
        tracks = {role: value.clone() for role, value in self.tracks.items()}
        moving = tracks[self.bound.moving_object_source_role]
        # (1,4)/(2,4) are connected and within 3.5 of (4,4)/(5,4), but outside
        # this source-derived maximum envelope at phase 6.
        for current_phase in range(phase, core.LATENT_PHASES):
            moving[current_phase].zero_()
            moving[current_phase, 1:3, 4] = True
        with self.assertRaisesRegex(core.V15BContractError, "escape"):
            native_motion_from_tracks(
                self.bound, self.masks, tracks, self.memory
            )

    def test_position_encoding_mutation_is_scrubbed_before_identity_or_v_read(self):
        source = self.raw_key.clone()
        source[..., 7] += torch.randn_like(source[..., 7]) * 100
        memory = core.build_source_role_content_memory_v15b(
            source_video_sha256=SHA_A, source_latent_sha256=SOURCE_LATENT_SHA,
            binding=self.bound, masks=self.masks,
            step_index=3, block_index=4, branch="conditional",
            source_hidden=self.carrier.hidden,
            source_pre_rope_key=source, source_value=self.raw_value,
            position_reference=position_reference(source),
        )
        self.assertTrue(torch.equal(memory.key_content, self.memory.key_content))
        key = self.native_key.clone()
        key[..., 7] += torch.randn_like(key[..., 7]) * 100
        original = self.state(); mutated = self.state(key=key)
        self.assertTrue(torch.equal(original.confident_role_index,
                                    mutated.confident_role_index))
        self.assertTrue(torch.equal(
            core._role_content_read(self.native_key, self.memory, self.bound, original),
            core._role_content_read(key, self.memory, self.bound, mutated),
        ))

    def test_float_centroid_gate_and_vacancy_are_recomputed(self):
        previous = torch.zeros(9, 9, dtype=torch.bool); previous[4:6, 4] = True
        exact = torch.zeros_like(previous); exact[7:9, 4] = True
        area, centroid, vacancy, released = core._track_gate(
            previous=previous, candidate=exact, final=exact,
            corridor=torch.ones_like(previous), role="clear_glass_pitcher_2", phase=1,
        )
        self.assertEqual(core._centroid(previous), (4.5, 4.0))
        self.assertEqual(centroid, (7.5, 4.0))
        self.assertEqual((area, vacancy, released), (2, 0, 2))
        too_far = torch.zeros_like(previous); too_far[0:2, 4] = True
        with self.assertRaisesRegex(core.V15BContractError, "3.5"):
            core._track_gate(
                previous=previous, candidate=too_far, final=too_far,
                corridor=torch.ones_like(previous),
                role="clear_glass_pitcher_2", phase=1,
            )

    def test_shape_collapse_and_completely_stale_native_motion_fail_closed(self):
        phase0 = torch.zeros(9, 9, dtype=torch.bool); phase0[3:5, 3:5] = True
        one_by_four = torch.zeros_like(phase0); one_by_four[4, 2:6] = True
        with self.assertRaisesRegex(core.V15BContractError, "aspect"):
            core._track_gate(
                phase0=phase0, previous=phase0, candidate=one_by_four,
                final=one_by_four, corridor=torch.ones_like(phase0),
                role="clear_glass_pitcher_2", phase=1,
            )
        two = torch.zeros_like(phase0); two[4:6, 4] = True
        one = torch.zeros_like(phase0); one[4, 4] = True
        with self.assertRaisesRegex(core.V15BContractError, "phase0-relative"):
            core._track_gate(
                phase0=phase0, previous=two, candidate=two, final=one,
                corridor=torch.ones_like(phase0),
                role="clear_glass_pitcher_2", phase=2,
            )
        solid = torch.zeros_like(phase0); solid[3:6, 3:6] = True
        ring = solid.clone(); ring[4, 4] = False
        with self.assertRaisesRegex(core.V15BContractError, "topology"):
            core._track_gate(
                phase0=solid, previous=solid, candidate=solid, final=ring,
                corridor=torch.ones_like(solid),
                role="clear_glass_pitcher_2", phase=3,
            )
        stale_key, stale_transport, _ = persistent_target_fixture(
            self.bound, self.masks, self.memory, move_phase=core.LATENT_PHASES + 1
        )
        with self.assertRaisesRegex(core.V15BContractError, "completely stale"):
            self.state(key=stale_key, transport=stale_transport)

    def test_relation_is_query_independent_and_position_scrubbed_k_driven(self):
        _, query, key, _ = self.target()
        baseline = self.apply(query=query, key=key)
        changed_query = self.apply(query=torch.randn_like(query), key=key)
        self.assertTrue(torch.equal(baseline.route_delta, changed_query.route_delta))
        recipient = self.bound.recipient_source_role; phase = 7
        tokens = torch.nonzero(self.tracks[recipient][phase].reshape(-1)).flatten()
        tokens += phase * HEIGHT * WIDTH
        changed_key = key.clone(); changed_key[:, tokens] *= 3.0
        changed = self.apply(query=query, key=changed_key)
        self.assertFalse(torch.equal(baseline.route_delta, changed.route_delta))
        self.assertEqual(
            baseline.audit.relation_operator,
            "position_scrubbed_target_key_persistent_role_pool_query_scatter",
        )

    def test_memory_is_phase0_unordered_position_scrubbed_and_convex_v_only(self):
        self.assertTrue(self.memory.builder_receipt.permutation_invariant)
        self.assertEqual(
            self.memory.builder_receipt.position_scrub_kind,
            "fixed_synthetic_fixture_projection_reference_only",
        )
        self.assertFalse(self.memory.coordinate_free)
        self.assertFalse(self.memory.position_removed_claimed)
        self.assertFalse(self.memory.scientific_claim_authorized)
        perm_key = self.raw_key.clone(); perm_value = self.raw_value.clone()
        for role in self.bound.source_roles:
            mask = self.masks.role_masks[role].clone(); mask[:, HEIGHT * WIDTH:] = False
            perm_key[mask] = perm_key[mask].flip(0)
            perm_value[mask] = perm_value[mask].flip(0)
        permuted = core.build_source_role_content_memory_v15b(
            source_video_sha256=SHA_A, source_latent_sha256=SOURCE_LATENT_SHA,
            binding=self.bound, masks=self.masks,
            step_index=3, block_index=4, branch="conditional",
            source_hidden=self.carrier.hidden,
            source_pre_rope_key=perm_key, source_value=perm_value,
            position_reference=position_reference(perm_key),
        )
        self.assertTrue(torch.equal(self.memory.key_content, permuted.key_content))
        self.assertTrue(torch.equal(self.memory.value_content, permuted.value_content))
        hidden, query, key, value = self.target(); value.normal_()
        no_memory = core.apply_pre_block_v15b(
            target_hidden=hidden, target_query=query, target_key=key,
            target_value=value, carrier=self.carrier, binding=self.bound,
            signed_graph=None, content_memory=None, target_native_transport=None,
            route_strength=0, memory_strength=0, restore_background=False,
        )
        full = self.apply(key=key, query=query, value=value, route=False)
        support = full.target_role_state.assigned_support_mask.clone()
        support[:, :HEIGHT * WIDTH] = False
        source_read = core._role_content_read(
            key, self.memory, self.bound, full.target_role_state
        )
        self.assertTrue(torch.equal(full.hidden, no_memory.hidden))
        self.assertTrue(torch.equal(full.key, no_memory.key))
        self.assertTrue(torch.allclose(full.value[support], source_read[support]))
        self.assertTrue(torch.equal(full.value[~support], no_memory.value[~support]))

    def test_raw_source_bytes_reopen_fresh_and_synchronized_number2_number3_reseal_fails(self):
        raw = self.memory.raw_source_material
        first_hidden, first_key, first_value, first_masks = raw.reopen()
        second_hidden, second_key, second_value, second_masks = raw.reopen()
        self.assertTrue(raw.immutable_byte_material)
        self.assertTrue(raw.material_reopenable)
        self.assertFalse(raw.externally_authenticated)
        self.assertFalse(raw.scientific_claim_authorized)
        self.assertFalse(raw.route_authorized)
        self.assertTrue(torch.equal(first_hidden, second_hidden))
        self.assertTrue(torch.equal(first_key, second_key))
        self.assertTrue(torch.equal(first_value, second_value))
        self.assertNotEqual(first_hidden.data_ptr(), second_hidden.data_ptr())
        self.assertNotEqual(first_key.data_ptr(), second_key.data_ptr())
        self.assertNotEqual(first_value.data_ptr(), second_value.data_ptr())
        for role in raw.role_ids:
            self.assertTrue(torch.equal(first_masks[role], second_masks[role]))
            self.assertNotEqual(
                first_masks[role].data_ptr(), second_masks[role].data_ptr()
            )
        audited = self.apply(route=False)
        self.assertEqual(audited.audit.cross_role_memory_write_max_abs, 0.0)
        self.assertTrue(audited.audit.slot_uuid_mask_provenance_verified)
        self.assertTrue(audited.audit.target_write_ownership_verified)
        self.assertIsNotNone(audited.audit.cross_role_zero_proof_sha256)

        # Strong attack: exchange only derived #2/#3 V rows, then synchronously
        # reseal every mutable downstream receipt/digest (including UUID rows).
        # Immutable raw K/V/masks remain the independent authority, so replay
        # must still reject the forged internally-consistent hash closure.
        moving = self.memory.role_ids.index(
            self.bound.moving_object_source_role
        )
        recipient = self.memory.role_ids.index(
            self.bound.recipient_source_role
        )
        forged_value = self.memory.value_content.clone()
        forged_value[[moving, recipient]] = forged_value[[recipient, moving]]
        forged_provenance = list(self.memory.slot_provenance_by_role)
        moving_entries = forged_provenance[moving][1]
        recipient_entries = forged_provenance[recipient][1]
        forged_provenance[moving] = (
            forged_provenance[moving][0], recipient_entries
        )
        forged_provenance[recipient] = (
            forged_provenance[recipient][0], moving_entries
        )
        forged_provenance = tuple(forged_provenance)
        receipt = self.memory.builder_receipt
        object.__setattr__(self.memory, "value_content", forged_value)
        object.__setattr__(self.memory, "slot_provenance_by_role", forged_provenance)
        object.__setattr__(receipt, "output_value_sha256", core.tensor_sha256(forged_value))
        object.__setattr__(receipt, "per_role_output_value_sha256", tuple(
            (role, core.tensor_sha256(
                forged_value[index, self.memory.slot_valid_mask[index]]
            )) for index, role in enumerate(self.memory.role_ids)
        ))
        object.__setattr__(receipt, "per_role_slot_uuid_sha256", tuple(
            (role, core.object_sha256(tuple(entry[0] for entry in entries)))
            for role, entries in forged_provenance
        ))
        object.__setattr__(receipt, "slot_provenance_digest",
                           core.object_sha256(forged_provenance))
        object.__setattr__(receipt, "permutation_probe_sha256", core.object_sha256({
            "synchronized_reseal": True,
            "forged_value_sha256": core.tensor_sha256(forged_value),
            "forged_provenance_digest": core.object_sha256(forged_provenance),
        }))
        object.__setattr__(receipt, "digest", core.object_sha256(receipt._payload()))
        object.__setattr__(self.memory, "digest",
                           core.object_sha256(self.memory._payload()))
        with self.assertRaisesRegex(
            core.V15BContractError,
            "raw re-extraction|provenance replay|extraction receipt replay",
        ):
            self.memory.__post_init__()

    def test_builder_receipt_source_and_replay_fields_cannot_be_resealed(self):
        receipt = self.memory.builder_receipt
        residual = receipt.position_scrub_projection_residual_max_abs
        alternative_residual = (
            core.POSITION_PROJECTOR_TOLERANCE / 2.0
            if residual != core.POSITION_PROJECTOR_TOLERANCE / 2.0 else 0.0
        )
        mutations = {
            "source_shape": (
                receipt.source_shape[0], receipt.source_shape[1] + 1,
                receipt.source_shape[2], receipt.source_shape[3],
            ),
            "source_pre_rope_key_sha256": "0" * 64,
            "source_scrubbed_pre_rope_key_sha256": "0" * 64,
            "source_value_sha256": "0" * 64,
            "permutation_probe_sha256": "0" * 64,
            "position_scrub_projection_residual_max_abs": alternative_residual,
        }
        for field, value in mutations.items():
            receipt_payload = receipt._payload()
            receipt_payload[field] = value
            forged_receipt = core.SourceContentBuilderReceiptV15B(
                **receipt_payload, digest=core.object_sha256(receipt_payload)
            )
            memory_payload = self.memory._payload()
            memory_payload["builder_receipt_digest"] = forged_receipt.digest
            memory_kwargs = {
                name: getattr(self.memory, name)
                for name in self.memory.__dataclass_fields__
                if name != "digest"
            }
            memory_kwargs["builder_receipt"] = forged_receipt
            with self.subTest(field=field), self.assertRaisesRegex(
                core.V15BContractError, "receipt|replay|provenance"
            ):
                core.SourceRoleContentMemoryV15B(
                    **memory_kwargs, digest=core.object_sha256(memory_payload)
                )

    def test_phase0_hkv_exact_and_background_restore_remain(self):
        hidden, query, key, value = self.target()
        hidden.normal_(); query.normal_(); key.normal_(); value.normal_()
        # A random K is allowed only for the inactive-memory K0 control.
        state = core.apply_pre_block_v15b(
            target_hidden=hidden, target_query=query, target_key=key,
            target_value=value, carrier=self.carrier, binding=self.bound,
            signed_graph=None, content_memory=None, target_native_transport=None,
            route_strength=0, memory_strength=0, restore_background=True,
        )
        spatial = HEIGHT * WIDTH
        raw_hidden, raw_key, raw_value, _ = (
            self.carrier.raw_source_material.reopen()
        )
        self.assertTrue(self.carrier.phase0_raw_hkv_exact)
        self.assertTrue(self.carrier.post_phase0_background_caller_supplied)
        self.assertFalse(self.carrier.externally_authenticated)
        self.assertFalse(self.carrier.scientific_claim_authorized)
        self.assertFalse(self.carrier.route_authorized)
        self.assertTrue(torch.equal(state.hidden[:, :spatial], raw_hidden))
        self.assertTrue(torch.equal(state.key[:, :spatial], raw_key))
        self.assertTrue(torch.equal(state.value[:, :spatial], raw_value))
        self.assertTrue(torch.equal(state.hidden[:, :spatial],
                                    self.carrier.hidden[:, :spatial]))
        self.assertTrue(torch.equal(state.key[:, :spatial], self.carrier.key[:, :spatial]))
        self.assertTrue(torch.equal(state.value[:, :spatial],
                                    self.carrier.value[:, :spatial]))
        self.assertTrue(torch.equal(
            state.hidden[self.masks.background_support_mask],
            self.carrier.hidden[self.masks.background_support_mask],
        ))
        self.assertTrue(torch.equal(
            state.hidden[self.masks.transition_path_mask],
            hidden[self.masks.transition_path_mask],
        ))

    def test_resealed_fake_carrier_phase0_hkv_is_rejected_by_raw_reopen(self):
        spatial = HEIGHT * WIDTH
        forged_hidden = self.carrier.hidden.clone()
        forged_key = self.carrier.key.clone()
        forged_value = self.carrier.value.clone()
        forged_hidden[:, :spatial] += 31.0
        forged_key[:, :spatial] += 123.0
        forged_value[:, :spatial] -= 77.0
        object.__setattr__(self.carrier, "hidden", forged_hidden)
        object.__setattr__(self.carrier, "key", forged_key)
        object.__setattr__(self.carrier, "value", forged_value)
        object.__setattr__(
            self.carrier, "digest", core.object_sha256(self.carrier._payload())
        )
        with self.assertRaisesRegex(core.V15BContractError, "phase0 H/K/V"):
            self.carrier.__post_init__()

    def test_execution_cell_transport_and_dtype_mismatch_fail_closed(self):
        wrong_memory = core.build_source_role_content_memory_v15b(
            source_video_sha256=SHA_A, source_latent_sha256=SOURCE_LATENT_SHA,
            binding=self.bound, masks=self.masks,
            step_index=2, block_index=4, branch="conditional",
            source_hidden=self.carrier.hidden,
            source_pre_rope_key=self.raw_key, source_value=self.raw_value,
            position_reference=self.position_reference,
        )
        with self.assertRaises(core.V15BContractError):
            self.state(memory=wrong_memory)
        with self.assertRaises(core.V15BContractError):
            core.build_target_role_state_v15b(
                native_target_pre_rope_key=self.native_key.double(),
                memory=self.memory, masks=self.masks, binding=self.bound,
                target_native_transport=self.transport,
            )
        hidden, query, key, value = self.target()
        with self.assertRaises(core.V15BContractError):
            core.apply_pre_block_v15b(
                target_hidden=hidden, target_query=query, target_key=key,
                target_value=value, carrier=self.carrier, binding=self.bound,
                signed_graph=self.signed, content_memory=None,
                target_native_transport=None, route_strength=1,
                memory_strength=0, restore_background=False,
            )

    def test_transport_injectivity_and_position_projector_cannot_be_laundered(self):
        with self.assertRaisesRegex(core.V15BContractError, "arbitrary"):
            core.TargetNativeTransportV15B.create(
                previous_token_index=self.transport.previous_token_index.float(),
            )
        reference = self.transport.motion_reference
        current = int(torch.nonzero(
            reference.backward_token_index[0, 0] >= 0
        ).flatten()[0])
        reference.backward_displacement_yx[0, 0, current, 1] += 1
        with self.assertRaisesRegex(core.V15BContractError, "estimator replay"):
            self.state()

        # Fresh material for the remaining independent mutation checks.
        self.native_key, self.transport, self.tracks = persistent_target_fixture(
            self.bound, self.masks, self.memory
        )
        reference = self.transport.motion_reference
        current = int(torch.nonzero(
            reference.backward_token_index[0, 0] >= 0
        ).flatten()[0])
        self.transport.motion_reference.backward_token_index[0, 0, current] = -1
        with self.assertRaisesRegex(core.V15BContractError, "replay"):
            self.state()

        base = self.raw_key.clone()
        reference2 = core.build_position_counterfactual_reference_v15b()
        self.assertEqual(reference2.rank_by_head, (2,))
        self.assertEqual(reference2.fit_y_rank_by_head, (1,))
        self.assertEqual(reference2.fit_x_rank_by_head, (1,))
        self.assertEqual(reference2.heldout_translation_count, 6)
        self.assertFalse(reference2.position_removed_claimed)
        reference2.projector[:, 6].zero_(); reference2.projector[:, :, 6].zero_()
        with self.assertRaisesRegex(core.V15BContractError, "fixed-fixture replay"):
            core.build_source_role_content_memory_v15b(
                source_video_sha256=SHA_A,
                source_latent_sha256=SOURCE_LATENT_SHA,
                binding=self.bound, masks=self.masks,
                step_index=3, block_index=4, branch="conditional",
                source_hidden=self.carrier.hidden,
                source_pre_rope_key=base, source_value=self.raw_value,
                position_reference=reference2,
            )
        fixture = core.build_position_calibration_fixture_v15b()
        fixture.fit_translated_input_latents[0][0, 0, 0, 0] = 99.0
        with self.assertRaisesRegex(core.V15BContractError, "fixed position calibration"):
            core.build_position_counterfactual_reference_v15b(
                calibration_fixture=fixture
            )
        with self.assertRaises(TypeError):
            core.build_position_counterfactual_reference_v15b(
                calibration_base_pre_rope_key=torch.zeros(1, 4, HEADS, HEAD_DIM)
            )

    def test_calibration_and_transport_material_provenance_remain_no_authority(self):
        fixture = core.build_position_calibration_fixture_v15b()
        self.assertTrue(set(fixture.fit_translations_yx).isdisjoint(
            fixture.heldout_translations_yx
        ))
        self.assertFalse(fixture.source_or_video_phase_key_accepted)
        self.assertFalse(fixture.externally_authenticated)
        self.assertFalse(fixture.position_removed_claimed)
        self.assertFalse(fixture.scientific_claim_authorized)
        self.assertFalse(fixture.route_authorized)
        self.assertTrue(fixture.material_bytes_reopenable)
        self.assertTrue(fixture.translation_correspondence_recomputed)
        self.assertIsInstance(fixture.frozen_model_code_material, str)
        self.assertIsInstance(fixture.frozen_model_checkpoint_material, torch.Tensor)
        with self.assertRaisesRegex(core.V15BContractError, "code_material"):
            replace(fixture, frozen_model_code_material="{}")
        heldout = fixture.heldout_counterfactual_pre_rope_keys[0].clone()
        heldout[0, 0, 0, 0] += 1
        poisoned_heldout = (heldout,) + fixture.heldout_counterfactual_pre_rope_keys[1:]
        with self.assertRaisesRegex(core.V15BContractError, "heldout_counterfactual"):
            replace(fixture, heldout_counterfactual_pre_rope_keys=poisoned_heldout)

        flow = self.transport.motion_reference
        self.assertEqual(
            flow.evidence_kind,
            "self_contained_role_label_translation_reference",
        )
        self.assertTrue(flow.target_input_reopenable)
        self.assertTrue(flow.estimator_material_reopenable)
        self.assertTrue(flow.estimator_output_recomputed)
        self.assertFalse(flow.externally_authenticated)
        self.assertFalse(flow.native_flow_claimed)
        self.assertFalse(flow.scientific_claim_authorized)
        self.assertFalse(flow.route_authorized)
        with self.assertRaisesRegex(core.V15BContractError, "checkpoint material"):
            replace(flow, estimator_checkpoint_material="forged")
        poisoned_input = flow.target_input_role_label_tensor.clone()
        poisoned_input[0, 1, 0] = 3
        with self.assertRaisesRegex(core.V15BContractError, "target input"):
            replace(flow, target_input_role_label_tensor=poisoned_input)
        with self.assertRaisesRegex(core.V15BContractError, "transcript"):
            replace(flow, estimator_output_transcript_sha256="0" * 64)

    def test_nested_memory_and_mask_tensor_mutations_are_rejected_by_consumers(self):
        self.memory.value_content[0, 0, 0, 0] += 1
        with self.assertRaisesRegex(core.V15BContractError, "builder output|digest"):
            self.state()
        # A fresh fixture proves the nested carrier->mask path independently.
        bound = binding(); masks = mask_set(bound)
        carrier, memory, _, _, _ = carrier_and_memory(bound, masks)
        key, transport, _ = persistent_target_fixture(bound, masks, memory)
        role = bound.moving_object_source_role
        masks.role_masks[role][0, 0] ^= True
        hidden = torch.zeros(1, self.tokens, HIDDEN_WIDTH)
        query = torch.zeros(1, self.tokens, HEADS, HEAD_DIM)
        value = torch.zeros_like(key)
        with self.assertRaisesRegex(core.V15BContractError, "track|mask|connected"):
            core.apply_pre_block_v15b(
                target_hidden=hidden, target_query=query, target_key=key,
                target_value=value, carrier=carrier, binding=bound,
                signed_graph=None, content_memory=memory,
                target_native_transport=transport, route_strength=0,
                memory_strength=1, restore_background=False,
            )

    def test_post_block_still_has_phase0_identity_exception(self):
        hidden, _, key, value = self.target()
        hidden.normal_(); key.normal_(); value.normal_()
        state = core.apply_post_block_v15b(
            target_hidden=hidden, target_key=key, target_value=value,
            carrier=self.carrier, binding=self.bound,
            signed_graph_digest=self.signed.digest, restore_background=False,
        )
        spatial = HEIGHT * WIDTH
        self.assertTrue(torch.equal(state.hidden[:, :spatial],
                                    self.carrier.hidden[:, :spatial]))
        self.assertTrue(torch.equal(state.key[:, :spatial], self.carrier.key[:, :spatial]))
        self.assertTrue(torch.equal(state.value[:, :spatial],
                                    self.carrier.value[:, :spatial]))


if __name__ == "__main__":
    unittest.main()
