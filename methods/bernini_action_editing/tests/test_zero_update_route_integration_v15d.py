import ast
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from methods.bernini_action_editing import zero_update_route_integration_v15d as core


ASSET = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "e00_zero_update_route_integration_v15d.json"
)
SPATIAL = core.GRID[1] * core.GRID[2]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def token(phase: int, spatial_index: int) -> int:
    return phase * SPATIAL + spatial_index


def resign_digest(payload):
    unsigned = dict(payload)
    unsigned.pop("digest")
    payload["digest"] = core.object_sha256(unsigned)
    return payload


def set_like_list_mutations(values):
    reversed_values = list(reversed(copy.deepcopy(values)))
    nonascending = copy.deepcopy(values)
    nonascending[0], nonascending[1] = nonascending[1], nonascending[0]
    duplicate = copy.deepcopy(values)
    duplicate.insert(1, copy.deepcopy(duplicate[0]))
    return {
        "reversed": reversed_values,
        "nonascending": nonascending,
        "duplicate": duplicate,
    }


def source_initial():
    return {
        core.HUMAN_ROLE: {0},
        core.OLD_ACTOR_ROLE: {10},
        core.NEW_ACTOR_ROLE: {100},
        core.RECIPIENT_ROLE: {200},
    }


def target_role_indices(*, move_new_actor: bool = True):
    result = {role: set() for role in core.FOREGROUND_ROLES}
    for phase in range(core.GRID[0]):
        result[core.HUMAN_ROLE].add(token(phase, 0))
        result[core.OLD_ACTOR_ROLE].add(token(phase, 10))
        new_spatial = 100 if phase == 0 or not move_new_actor else 101
        result[core.NEW_ACTOR_ROLE].add(token(phase, new_spatial))
        result[core.RECIPIENT_ROLE].add(token(phase, 200))
    return result


def motion(*, roles=None, corridor=None, contact=None, pour_phases=(10, 11, 12)):
    return core.TargetNativeMotionPlanV15D.create(
        role_indices=roles if roles is not None else target_role_indices(),
        motion_corridor_indices=(
            corridor if corridor is not None else {token(1, 102)}
        ),
        contact_indices=(contact if contact is not None else {token(10, 500)}),
        pour_phases=pour_phases,
    )


def property_memory(
    *, source=SHA_A, runtime=SHA_B, static_reforward=SHA_C
):
    return core.SourcePropertyMemoryBoundaryV15D.create(
        source_instance_sha256=source,
        source_runtime_receipt_sha256=runtime,
        source_static_reforward_receipt_sha256=static_reforward,
        role_slot_sha256={
            role: str(index + 1) * 64
            for index, role in enumerate(core.FOREGROUND_ROLES)
        },
        liquid_property_sha256="5" * 64,
    )


def background_carrier(
    *,
    source=SHA_A,
    runtime=SHA_B,
    static_reforward=SHA_C,
    background=(300,),
    support=(400,),
):
    return core.SourceStaticBackgroundCarrierV15D.create(
        source_instance_sha256=source,
        source_runtime_receipt_sha256=runtime,
        source_static_reforward_receipt_sha256=static_reforward,
        strict_background_indices=background,
        strict_support_indices=support,
    )


def abi(arm=core.ARM_A):
    return core.build_no_anchor_target_abi_v15d(case_id="e00", arm_id=arm)


def ledger(
    *,
    arm=core.ARM_A,
    target_motion=None,
    initial=None,
    properties=None,
    background=None,
):
    return core.InstanceLedgerV15D(
        target_abi=abi(arm),
        target_motion=target_motion if target_motion is not None else motion(),
        source_initial_spatial_indices=(
            initial if initial is not None else source_initial()
        ),
        property_memory=(properties if properties is not None else property_memory()),
        background_carrier=(
            background if background is not None else background_carrier()
        ),
    )


def fill_all_cells(value: core.InstanceLedgerV15D) -> None:
    for step in range(core.DENOISE_STEPS):
        for branch in core.CFG_BRANCHES:
            for physical_id in core.PHYSICAL_BLOCK_IDS:
                value.plan_cell(
                    step_index=step,
                    cfg_branch=branch,
                    physical_block=core.PhysicalBlockAddressV15D.for_id(
                        physical_id
                    ),
                    requested_write_scales=core.ZERO_WRITE_SCALES,
                )


class ContractAndAddressTests(unittest.TestCase):
    def test_asset_is_exact_preregistration(self):
        loaded = core.load_contract_v15d(ASSET)
        self.assertEqual(loaded, core.expected_contract_v15d())
        self.assertEqual(loaded["grid"], [21, 37, 25])
        self.assertEqual(loaded["physical_block_ids"], list(range(30)))
        self.assertEqual(
            loaded["route_physical_block_ids"],
            [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18,
             19, 21, 22, 23, 25, 26, 27, 29],
        )
        self.assertEqual(loaded["route_physical_block_count"], 22)

    def test_contract_mutations_are_rejected(self):
        base = core.expected_contract_v15d()
        mutations = []
        changed_grid = copy.deepcopy(base)
        changed_grid["grid"] = [21, 36, 26]
        mutations.append(changed_grid)
        float_grid = copy.deepcopy(base)
        float_grid["grid"] = [21.0, 37.0, 25.0]
        mutations.append(float_grid)
        bool_as_int_action = copy.deepcopy(base)
        bool_as_int_action["action_contract"][
            "appearance_values_in_prompt_or_graph_forbidden"
        ] = 1
        mutations.append(bool_as_int_action)
        ordinal_allowlist = copy.deepcopy(base)
        ordinal_allowlist["route_physical_block_ids"] = list(range(22))
        mutations.append(ordinal_allowlist)
        enabled_route = copy.deepcopy(base)
        enabled_route["authorizations"]["route_execution_authorized"] = True
        mutations.append(enabled_route)
        enabled_write = copy.deepcopy(base)
        enabled_write["default_write_scales"]["graph_route_v0"] = 1.0
        mutations.append(enabled_write)
        for mutation in mutations:
            with self.subTest(mutation=core.object_sha256(mutation)):
                with self.assertRaises(core.V15DContractError):
                    core.validate_contract_v15d(mutation)

    def test_four_arms_are_exactly_preregistered_and_zero_write(self):
        contract = core.expected_contract_v15d()
        self.assertEqual(
            tuple(item["arm_id"] for item in contract["arms"]), core.ARM_IDS
        )
        for item in contract["arms"]:
            self.assertEqual(
                item["current_write_policy"],
                "synthetic_plan_exact_zero_not_runtime_observation",
            )
        for arm_id in core.ARM_IDS:
            target = abi(arm_id)
            self.assertEqual(target.arm_id, arm_id)
            self.assertEqual(target.requested_write_scales, core.ZERO_WRITE_SCALES)
            self.assertEqual(
                target.authorizations, core.hard_false_authorizations_v15d()
            )
        mutable_copy = core.hard_false_authorizations_v15d()
        mutable_copy["route_execution_authorized"] = True
        self.assertIs(
            core.hard_false_authorizations_v15d()["route_execution_authorized"],
            False,
        )
        payload = abi().payload()
        payload["authorizations"]["route_execution_authorized"] = 0
        with self.assertRaisesRegex(core.V15DContractError, "bool False"):
            core.NoAnchorTargetABIV15D.from_mapping(payload)

    def test_physical_addresses_are_typed_not_ordinals(self):
        address = core.PhysicalBlockAddressV15D.for_id(21)
        self.assertEqual(address.physical_id, 21)
        for bad in (
            {
                "address_kind": "route_ordinal",
                "namespace": core.BLOCK_NAMESPACE,
                "physical_id": 21,
            },
            {
                "address_kind": "physical_transformer_block",
                "namespace": core.BLOCK_NAMESPACE,
                "physical_id": 21,
                "ordinal": 15,
            },
            {
                "address_kind": "physical_transformer_block",
                "namespace": "hooks.route",
                "physical_id": 21,
            },
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(core.V15DContractError):
                    core.PhysicalBlockAddressV15D.from_mapping(bad)
        baseline = abi().payload()
        self.assertEqual(
            core.NoAnchorTargetABIV15D.from_mapping(baseline).payload(),
            baseline,
        )
        for field in ("physical_block_inventory", "route_physical_allowlist"):
            for mutation, replacement in set_like_list_mutations(
                baseline[field]
            ).items():
                changed = copy.deepcopy(baseline)
                changed[field] = replacement
                with self.subTest(field=field, mutation=mutation):
                    with self.assertRaisesRegex(
                        core.V15DContractError, "ascending|duplicate|unique"
                    ):
                        core.NoAnchorTargetABIV15D.from_mapping(changed)

    def test_module_has_no_runner_controller_trainer_or_tensor_dependency(self):
        path = Path(core.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue(
            imported_roots.isdisjoint(
                {"torch", "numpy", "subprocess", "train_lora", "anchor_qk_transport"}
            )
        )


class NoAnchorActionABITests(unittest.TestCase):
    def test_media_latent_gaussian_path_fd_and_bytes_are_rejected(self):
        cases = {
            "anchor_video": "clip.mp4",
            "anchor_latent": [0.0],
            "anchor_gaussian": [0.0],
            "anchor_path": "/tmp/anchor.mp4",
            "anchor_fd": 9,
            "donor_rgb": [1, 2, 3],
            "reference_media": "x",
            "mystery": Path("clip.mp4"),
            "payload": b"video-bytes",
        }
        for field, material in cases.items():
            payload = abi().payload()
            payload[field] = material
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    core.V15DContractError, "forbidden|media/path/FD"
                ):
                    core.NoAnchorTargetABIV15D.from_mapping(payload)
        malformed = abi().payload()
        malformed["grid"] = 7
        with self.assertRaises(core.V15DContractError):
            core.NoAnchorTargetABIV15D.from_mapping(malformed)
        cyclic = abi().payload()
        cyclic["nested"] = cyclic
        with self.assertRaisesRegex(core.V15DContractError, "cyclic"):
            core.NoAnchorTargetABIV15D.from_mapping(cyclic)
        deep: list[object] = []
        cursor = deep
        for _ in range(core.MAX_INPUT_NESTING_DEPTH + 2):
            child: list[object] = []
            cursor.append(child)
            cursor = child
        nested = abi().payload()
        nested["nested"] = deep
        with self.assertRaisesRegex(core.V15DContractError, "nesting depth"):
            core.NoAnchorTargetABIV15D.from_mapping(nested)

    def test_bare_route_ordinals_cannot_replace_physical_addresses(self):
        payload = abi().payload()
        payload["route_physical_allowlist"] = list(range(22))
        with self.assertRaises(core.V15DContractError):
            core.NoAnchorTargetABIV15D.from_mapping(payload)

    def test_source_appearance_words_are_not_action_targets(self):
        examples = (
            "original glass pitcher must pour into original small white cup",
            "the amber liquid pours from #2 into #3",
            "#2 keeps its transparent material and round shape while pouring",
        )
        for text in examples:
            payload = abi().payload()
            payload["target_action"]["canonical_target_action_text"] = text
            with self.subTest(text=text):
                with self.assertRaisesRegex(
                    core.V15DContractError, "source-owned appearance"
                ):
                    core.NoAnchorTargetABIV15D.from_mapping(payload)

    def test_identity_lock_cannot_be_changed_into_appearance_supervision(self):
        payload = abi().payload()
        payload["target_action"]["identity_locks"][core.NEW_ACTOR_ROLE] = (
            "learn transparent glass appearance"
        )
        with self.assertRaisesRegex(core.V15DContractError, "action-only"):
            core.NoAnchorTargetABIV15D.from_mapping(payload)

    def test_graph_cannot_add_color_material_shape_or_liquid_appearance(self):
        for field in ("color", "material", "shape", "liquid_appearance"):
            payload = abi().payload()
            payload["target_action"][field] = "source value"
            with self.subTest(field=field):
                with self.assertRaises(core.V15DContractError):
                    core.NoAnchorTargetABIV15D.from_mapping(payload)

    def test_external_authority_and_nonzero_writes_cannot_be_self_claimed(self):
        authority = abi().payload()
        authority["source_authority_state"] = "PASSED"
        authority["authorizations"]["external_source_authority_passed"] = True
        with self.assertRaises(core.V15DContractError):
            core.NoAnchorTargetABIV15D.from_mapping(authority)
        for field, replacement in (
            ("grid", [21.0, 37.0, 25.0]),
            ("denoise_steps", 40.0),
        ):
            payload = abi().payload()
            payload[field] = replacement
            with self.subTest(exact_int_field=field):
                with self.assertRaises(core.V15DContractError):
                    core.NoAnchorTargetABIV15D.from_mapping(payload)
        for component in core.WRITE_COMPONENTS:
            payload = abi().payload()
            payload["requested_write_scales"][component] = 0.01
            with self.subTest(component=component):
                with self.assertRaisesRegex(core.V15DContractError, "exact zero"):
                    core.NoAnchorTargetABIV15D.from_mapping(payload)


class MotionAndOwnershipTests(unittest.TestCase):
    def test_target_motion_has_only_target_native_provenance_and_explicit_timing(self):
        value = motion()
        self.assertEqual(value.source_future_track_input_count, 0)
        self.assertFalse(value.externally_authenticated)
        self.assertEqual(value.contact_phases, (10,))
        self.assertEqual(value.pour_phases, (10, 11, 12))
        self.assertIn("no_source_future_input", value.provenance_kind)

    def test_source_future_track_inlet_and_claimed_authentication_are_rejected(self):
        value = motion()
        with self.assertRaisesRegex(core.V15DContractError, "source-future"):
            replace(value, source_future_track_input_count=1)
        with self.assertRaisesRegex(core.V15DContractError, "external"):
            replace(value, externally_authenticated=True)
        payload = abi().payload()
        payload["source_future_tracks"] = [1, 2, 3]
        with self.assertRaisesRegex(core.V15DContractError, "forbidden"):
            core.NoAnchorTargetABIV15D.from_mapping(payload)

    def test_motion_masks_must_cover_each_role_each_phase_without_overlap(self):
        missing_phase = target_role_indices()
        missing_phase[core.HUMAN_ROLE].remove(token(5, 0))
        with self.assertRaisesRegex(core.V15DContractError, "every temporal phase"):
            motion(roles=missing_phase)
        overlap = target_role_indices()
        overlap[core.OLD_ACTOR_ROLE].add(token(0, 100))
        with self.assertRaisesRegex(core.V15DContractError, "overlap"):
            motion(roles=overlap)
        value = motion()
        role_rows = list(value.role_indices)
        role, indices = role_rows[0]
        role_rows[0] = (role, indices.union({core.TOKEN_CELLS}))
        with self.assertRaises(core.V15DContractError):
            replace(value, role_indices=tuple(role_rows))
        with self.assertRaises(core.V15DContractError):
            replace(value, grid=(21.0, 37.0, 25.0))
        serialized = value.payload()
        serialized["role_indices"][0][1].append(
            serialized["role_indices"][0][1][0]
        )
        unsigned = dict(serialized)
        unsigned.pop("digest")
        serialized["digest"] = core.object_sha256(unsigned)
        with self.assertRaisesRegex(core.V15DContractError, "duplicate"):
            core.TargetNativeMotionPlanV15D.from_mapping(serialized)
        canonical = motion(
            corridor={token(1, 102), token(2, 102), token(3, 102)},
            contact={token(10, 500), token(10, 501), token(10, 502)},
        ).payload()
        self.assertEqual(
            core.TargetNativeMotionPlanV15D.from_mapping(canonical).payload(),
            canonical,
        )
        set_like_paths = (
            ("role_indices", canonical["role_indices"][0][1]),
            ("motion_corridor_indices", canonical["motion_corridor_indices"]),
            ("contact_indices", canonical["contact_indices"]),
        )
        for field, values in set_like_paths:
            for mutation, replacement in set_like_list_mutations(values).items():
                changed = copy.deepcopy(canonical)
                if field == "role_indices":
                    changed[field][0][1] = replacement
                else:
                    changed[field] = replacement
                resign_digest(changed)
                with self.subTest(field=field, mutation=mutation):
                    with self.assertRaisesRegex(
                        core.V15DContractError, "ascending|duplicate|unique"
                    ):
                        core.TargetNativeMotionPlanV15D.from_mapping(changed)

    def test_contact_and_pour_timing_are_nonempty_and_causal(self):
        with self.assertRaises(core.V15DContractError):
            motion(contact=set())
        with self.assertRaisesRegex(core.V15DContractError, "cannot precede"):
            motion(pour_phases=(9, 10))
        with self.assertRaises(core.V15DContractError):
            motion(pour_phases=(10, 10))

    def test_ledger_has_six_roles_and_vacated_new_actor_site_is_hole(self):
        value = ledger()
        self.assertEqual(value.roles, core.LEDGER_ROLES)
        self.assertEqual(len(value.released_hole_indices), core.GRID[0] - 1)
        for phase in range(1, core.GRID[0]):
            self.assertEqual(value.owner_at(token(phase, 100)), core.HOLE_ROLE)
            self.assertEqual(value.owner_at(token(phase, 101)), core.NEW_ACTOR_ROLE)
        self.assertEqual(value.owner_at(token(0, 100)), core.NEW_ACTOR_ROLE)
        self.assertEqual(value.owner_at(token(0, 300)), core.BACKGROUND_ROLE)

    def test_static_new_actor_or_phase0_identity_mismatch_is_rejected(self):
        with self.assertRaisesRegex(core.V15DContractError, "release"):
            ledger(target_motion=motion(roles=target_role_indices(move_new_actor=False)))
        bad_initial = source_initial()
        bad_initial[core.NEW_ACTOR_ROLE] = {101}
        with self.assertRaisesRegex(core.V15DContractError, "phase 0"):
            ledger(initial=bad_initial)

    def test_vacated_new_actor_site_cannot_be_reassigned_to_foreground(self):
        roles = target_role_indices()
        roles[core.HUMAN_ROLE].remove(token(1, 0))
        roles[core.HUMAN_ROLE].add(token(1, 100))
        value = motion(roles=roles)
        with self.assertRaisesRegex(core.V15DContractError, "must be HOLE"):
            ledger(target_motion=value)


class SourcePropertyAndBackgroundTests(unittest.TestCase):
    def test_property_and_background_bind_same_source_static_reforward_runtime(self):
        variants = (
            background_carrier(source="d" * 64),
            background_carrier(runtime="d" * 64),
            background_carrier(static_reforward="d" * 64),
        )
        for variant in variants:
            with self.subTest(digest=variant.digest):
                with self.assertRaisesRegex(core.V15DContractError, "same source"):
                    ledger(background=variant)

    def test_background_scope_rejects_object_corridor_contact_and_hole(self):
        forbidden = {
            "object": token(0, 0),
            "corridor": token(1, 102),
            "contact": token(10, 500),
            "hole": token(1, 100),
        }
        for label, index in forbidden.items():
            carrier = background_carrier(background=(index,), support=(401,))
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    core.V15DContractError, "strict background|object/corridor/contact/HOLE"
                ):
                    ledger(background=carrier)

    def test_background_scope_must_be_nonempty_and_disjoint(self):
        with self.assertRaisesRegex(core.V15DContractError, "nonempty"):
            background_carrier(background=(), support=())
        with self.assertRaisesRegex(core.V15DContractError, "disjoint"):
            background_carrier(background=(300,), support=(300,))
        canonical = background_carrier(
            background=(300, 301, 302), support=(400, 401, 402)
        ).payload()
        self.assertEqual(
            core.SourceStaticBackgroundCarrierV15D.from_mapping(
                canonical
            ).payload(),
            canonical,
        )
        for field in ("strict_background_indices", "strict_support_indices"):
            for mutation, replacement in set_like_list_mutations(
                canonical[field]
            ).items():
                changed = copy.deepcopy(canonical)
                changed[field] = replacement
                resign_digest(changed)
                with self.subTest(field=field, mutation=mutation):
                    with self.assertRaisesRegex(
                        core.V15DContractError, "ascending|duplicate|unique"
                    ):
                        core.SourceStaticBackgroundCarrierV15D.from_mapping(
                            changed
                        )

    def test_local_property_and_background_cannot_claim_authority(self):
        with self.assertRaisesRegex(core.V15DContractError, "grant authority"):
            replace(property_memory(), externally_authenticated=True)
        with self.assertRaisesRegex(core.V15DContractError, "grant authority"):
            replace(background_carrier(), route_authorized=True)

    def test_property_memory_requires_exact_four_source_roles(self):
        slots = {
            role: str(index + 1) * 64
            for index, role in enumerate(core.FOREGROUND_ROLES)
        }
        del slots[core.RECIPIENT_ROLE]
        with self.assertRaisesRegex(core.V15DContractError, "role slots"):
            core.SourcePropertyMemoryBoundaryV15D.create(
                source_instance_sha256=SHA_A,
                source_runtime_receipt_sha256=SHA_B,
                source_static_reforward_receipt_sha256=SHA_C,
                role_slot_sha256=slots,
                liquid_property_sha256="5" * 64,
            )

    def test_liquid_appearance_has_an_explicit_source_property_binding(self):
        value = property_memory()
        self.assertEqual(value.liquid_property_sha256, "5" * 64)
        with self.assertRaisesRegex(core.V15DContractError, "liquid property"):
            core.SourcePropertyMemoryBoundaryV15D.create(
                source_instance_sha256=SHA_A,
                source_runtime_receipt_sha256=SHA_B,
                source_static_reforward_receipt_sha256=SHA_C,
                role_slot_sha256={
                    role: str(index + 1) * 64
                    for index, role in enumerate(core.FOREGROUND_ROLES)
                },
                liquid_property_sha256="not-a-source-digest",
            )


class PerCellAndRunReceiptTests(unittest.TestCase):
    def test_cell_requires_typed_physical_address_and_rejects_duplicate(self):
        value = ledger()
        kwargs = {
            "step_index": 0,
            "cfg_branch": core.CFG_BRANCHES[0],
            "requested_write_scales": core.ZERO_WRITE_SCALES,
        }
        with self.assertRaisesRegex(core.V15DContractError, "never an ordinal"):
            value.plan_cell(physical_block=0, **kwargs)
        address = core.PhysicalBlockAddressV15D.for_id(0)
        value.plan_cell(physical_block=address, **kwargs)
        with self.assertRaisesRegex(core.V15DContractError, "duplicate"):
            value.plan_cell(physical_block=address, **kwargs)

    def test_every_write_component_is_hard_zero_per_cell(self):
        for component in core.WRITE_COMPONENTS:
            value = ledger()
            writes = dict(core.ZERO_WRITE_SCALES)
            writes[component] = 1e-12
            with self.subTest(component=component):
                with self.assertRaisesRegex(core.V15DContractError, "exact zero"):
                    value.plan_cell(
                        step_index=0,
                        cfg_branch="negative",
                        physical_block=core.PhysicalBlockAddressV15D.for_id(1),
                        requested_write_scales=writes,
                    )

    def test_per_cell_mask_hole_background_and_property_mutations_are_rejected(self):
        cases = (
            {"outside_allowed_mask_delta_max_abs": 1e-9},
            {"hole_source_new_actor_2_restore_count": 1},
            {"background_forbidden_scope_write_count": 1},
            {"source_object_direct_or_graph_appearance_write_count": 1},
            {"object_feature_route": "graph_or_anchor_appearance"},
            {"background_feature_route": "caller_feature"},
            {"target_motion_route": "source_future_tracks"},
        )
        for changes in cases:
            value = ledger()
            with self.subTest(changes=changes):
                with self.assertRaises(core.V15DContractError):
                    value.plan_cell(
                        step_index=0,
                        cfg_branch="negative",
                        physical_block=core.PhysicalBlockAddressV15D.for_id(1),
                        requested_write_scales=core.ZERO_WRITE_SCALES,
                        **changes,
                    )

    def test_incomplete_all30_dual_cfg_schedule_cannot_seal(self):
        value = ledger()
        value.plan_cell(
            step_index=0,
            cfg_branch="negative",
            physical_block=core.PhysicalBlockAddressV15D.for_id(0),
            requested_write_scales=core.ZERO_WRITE_SCALES,
        )
        with self.assertRaisesRegex(core.V15DContractError, "incomplete"):
            value.seal()

    def test_complete_run_uses_one_ledger_across_30_blocks_and_dual_cfg(self):
        value = ledger(arm=core.ARM_C)
        fill_all_cells(value)
        receipt = value.seal()
        self.assertEqual(receipt.cell_count, 40 * 30 * 2)
        self.assertEqual(receipt.selected_route_cell_count, 40 * 22 * 2)
        self.assertTrue(receipt.one_ledger_digest_across_all_cells)
        self.assertTrue(receipt.all_planned_cell_write_scales_exact_zero)
        self.assertTrue(receipt.all_planned_outside_allowed_mask_deltas_exact_zero)
        self.assertFalse(receipt.runtime_execution_observed)
        self.assertTrue(receipt.cell_audits_are_synthetic_plan)
        self.assertTrue(receipt.fresh_bundle_validation_required)
        self.assertFalse(receipt.standalone_receipt_authority)
        self.assertTrue(receipt.hole_never_restored_as_source_new_actor_2)
        self.assertTrue(receipt.source_objects_use_property_memory_only)
        self.assertTrue(receipt.background_scope_strict)
        self.assertFalse(receipt.source_future_tracks_used_for_target_motion)
        self.assertFalse(receipt.external_source_authority_passed)
        self.assertFalse(receipt.route_execution_authorized)
        self.assertFalse(receipt.training_authorized)
        self.assertFalse(receipt.scientific_claim_authorized)
        self.assertEqual(receipt.target_motion_digest, value._target_motion.digest)
        self.assertEqual(receipt.property_memory_digest, value._property_memory.digest)
        self.assertEqual(
            receipt.background_carrier_digest, value._background_carrier.digest
        )
        self.assertEqual(receipt.source_instance_sha256, SHA_A)
        self.assertEqual(receipt.source_runtime_receipt_sha256, SHA_B)
        self.assertEqual(receipt.source_static_reforward_receipt_sha256, SHA_C)
        self.assertFalse(hasattr(receipt, "all_cell_writes_exact_zero"))
        self.assertEqual(receipt.unresolved_dependencies, core.UNRESOLVED_DEPENDENCIES)
        with self.assertRaisesRegex(core.V15DContractError, "already sealed"):
            value.seal()

    def test_resealed_cell_with_wrong_allowlist_or_ledger_is_rejected(self):
        for mutation in ("allowlist", "ledger"):
            value = ledger()
            fill_all_cells(value)
            key = (0, "negative", 1)
            original = value._cells[key]
            if mutation == "allowlist":
                changed = replace(original, route_allowlisted=False, digest="")
            else:
                changed = replace(original, ledger_digest="f" * 64, digest="")
            changed = replace(
                changed,
                digest=core.object_sha256(changed.payload_without_digest()),
            )
            value._cells[key] = changed
            with self.subTest(mutation=mutation):
                with self.assertRaises(core.V15DContractError):
                    value.seal()

    def test_mutated_internal_target_abi_is_revalidated_before_seal(self):
        for mutation in ("action", "cached_digest", "cached_arm"):
            value = ledger()
            if mutation == "action":
                value._target_abi.target_action["canonical_target_action_text"] = (
                    "amber liquid from a glass pitcher"
                )
            elif mutation == "cached_digest":
                value._target_abi_digest = "f" * 64
            else:
                value._target_arm_id = core.ARM_B
            with self.subTest(mutation=mutation):
                with self.assertRaises(core.V15DContractError):
                    value.seal()

    def test_run_receipt_cannot_be_changed_into_route_training_or_science_go(self):
        value = ledger(arm=core.ARM_D)
        fill_all_cells(value)
        receipt = value.seal()
        for field in (
            "external_source_authority_passed",
            "route_execution_authorized",
            "training_authorized",
            "scientific_claim_authorized",
            "standalone_receipt_authority",
        ):
            with self.subTest(field=field):
                with self.assertRaises(core.V15DContractError):
                    replace(receipt, **{field: True})

    def test_status_receipt_is_honest_about_nonexecution(self):
        status = core.preregistration_receipt_v15d()
        self.assertEqual(status["decision"], core.DECISION)
        self.assertTrue(status["unexecuted_local_cpu_synthetic_plan_only"])
        self.assertTrue(status["default_all_planned_write_scales_exact_zero"])
        self.assertFalse(status["runtime_execution_observed"])
        self.assertTrue(status["cell_audits_are_synthetic_plan"])
        self.assertTrue(status["fresh_bundle_validation_required"])
        self.assertFalse(status["standalone_receipt_authority"])
        for key in (
            "external_source_authority_passed",
            "gpu_execution_performed",
            "route_execution_performed",
            "decode_performed",
            "training_performed",
            "route_execution_authorized",
            "training_authorized",
            "scientific_claim_authorized",
        ):
            self.assertFalse(status[key], key)
        self.assertIn("source property", status["source_visible_amber_classification"])


class FreshReplayR2MutationTests(unittest.TestCase):
    def test_seal_rejects_resigned_motion_property_and_background_after_creation(self):
        cases = ("motion", "property_and_background", "background_scope")
        for case in cases:
            value = ledger()
            if case == "motion":
                current = value._target_motion
                object.__setattr__(current, "pour_phases", (10, 11, 13))
                object.__setattr__(
                    current,
                    "digest",
                    core.object_sha256(current._payload_without_digest()),
                )
            elif case == "property_and_background":
                for current in (value._property_memory, value._background_carrier):
                    object.__setattr__(current, "source_instance_sha256", "d" * 64)
                    object.__setattr__(
                        current,
                        "digest",
                        core.object_sha256(current._payload_without_digest()),
                    )
            else:
                current = value._background_carrier
                object.__setattr__(current, "strict_background_indices", frozenset({301}))
                object.__setattr__(
                    current,
                    "digest",
                    core.object_sha256(current._payload_without_digest()),
                )
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    core.V15DContractError, "changed after ledger construction"
                ):
                    value.seal()

    def test_seal_freshly_recomputes_owner_assignment_not_only_counts(self):
        value = ledger()
        owners = list(value._owners)
        index, _ = owners[300]
        owners[300] = (index, core.OLD_ACTOR_ROLE)
        value._owners = tuple(owners)
        with self.assertRaisesRegex(core.V15DContractError, "recomputed ownership"):
            value.seal()

    def test_resigned_cloned_cell_fails_coordinate_nonce(self):
        value = ledger()
        fill_all_cells(value)
        source = value._cells[(0, "negative", 1)]
        destination_key = (0, "negative", 2)
        cloned = replace(
            source,
            physical_block=core.PhysicalBlockAddressV15D.for_id(2),
            digest="",
        )
        cloned = replace(
            cloned,
            digest=core.object_sha256(cloned.payload_without_digest()),
        )
        value._cells[destination_key] = cloned
        with self.assertRaisesRegex(core.V15DContractError, "coordinate nonce"):
            value.seal()

    def test_cell_binds_motion_property_and_background_digests(self):
        for field in (
            "target_motion_digest",
            "property_memory_digest",
            "background_carrier_digest",
        ):
            value = ledger()
            fill_all_cells(value)
            key = (0, "negative", 1)
            changed = replace(value._cells[key], **{field: "f" * 64}, digest="")
            changed = replace(
                changed,
                digest=core.object_sha256(changed.payload_without_digest()),
            )
            value._cells[key] = changed
            with self.subTest(field=field):
                with self.assertRaisesRegex(core.V15DContractError, "binding differs"):
                    value.seal()

    def test_canonical_bundle_freshly_deserializes_lists_and_breaks_aliases(self):
        initial = source_initial()
        initial[core.HUMAN_ROLE] = {0, 1, 2}
        initial[core.OLD_ACTOR_ROLE] = {10, 11, 12}
        initial[core.RECIPIENT_ROLE] = {200, 201, 202}
        value = ledger(arm=core.ARM_D, initial=initial)
        fill_all_cells(value)
        receipt = value.seal()
        serialized = core.serialize_synthetic_plan_bundle_v15d(value, receipt)
        decoded = json.loads(serialized)
        self.assertIsInstance(decoded["synthetic_cells"], list)
        self.assertIsInstance(decoded["target_motion"]["role_indices"], list)
        object.__setattr__(value._property_memory, "digest", "f" * 64)
        replayed = core.fresh_deserialize_validate_synthetic_plan_v15d(serialized)
        self.assertEqual(replayed.payload(), receipt.payload())
        reordered = json.loads(serialized)
        reordered["synthetic_cells"][0], reordered["synthetic_cells"][1] = (
            reordered["synthetic_cells"][1],
            reordered["synthetic_cells"][0],
        )
        with self.assertRaisesRegex(core.V15DContractError, "canonical coordinate order"):
            core.fresh_deserialize_validate_synthetic_plan_v15d(
                core.canonical_json_bytes(reordered)
            )
        set_like_paths = (
            (
                "source_initial_spatial_indices",
                decoded["source_initial_spatial_indices"][0][1],
            ),
            (
                "released_hole_indices",
                decoded["ledger_creation_payload"]["released_hole_indices"],
            ),
        )
        for field, values in set_like_paths:
            for mutation, replacement in set_like_list_mutations(values).items():
                changed = copy.deepcopy(decoded)
                if field == "source_initial_spatial_indices":
                    changed[field][0][1] = replacement
                else:
                    changed["ledger_creation_payload"][field] = replacement
                with self.subTest(field=field, mutation=mutation):
                    with self.assertRaisesRegex(
                        core.V15DContractError, "ascending|duplicate|unique"
                    ):
                        core.fresh_deserialize_validate_synthetic_plan_v15d(
                            core.canonical_json_bytes(changed)
                        )

    def test_fresh_consumer_rejects_noncanonical_or_mutable_serialization(self):
        value = ledger()
        fill_all_cells(value)
        receipt = value.seal()
        serialized = core.serialize_synthetic_plan_bundle_v15d(value, receipt)
        with self.assertRaisesRegex(core.V15DContractError, "immutable canonical bytes"):
            core.fresh_deserialize_validate_synthetic_plan_v15d(bytearray(serialized))
        pretty = json.dumps(json.loads(serialized), indent=2).encode("utf-8")
        with self.assertRaisesRegex(core.V15DContractError, "not exact canonical"):
            core.fresh_deserialize_validate_synthetic_plan_v15d(pretty)

    def test_resigned_receipt_identity_cannot_escape_fresh_cross_binding(self):
        value = ledger()
        fill_all_cells(value)
        receipt = value.seal()
        serialized = core.serialize_synthetic_plan_bundle_v15d(value, receipt)
        bundle = json.loads(serialized)
        submitted = bundle["receipt"]
        submitted["source_instance_sha256"] = "d" * 64
        unsigned = dict(submitted)
        unsigned.pop("digest")
        submitted["digest"] = core.object_sha256(unsigned)
        mutated = core.canonical_json_bytes(bundle)
        with self.assertRaisesRegex(core.V15DContractError, "fresh replay"):
            core.fresh_deserialize_validate_synthetic_plan_v15d(mutated)

    def test_runtime_execution_claim_is_rejected_even_when_receipt_is_resigned(self):
        value = ledger()
        fill_all_cells(value)
        receipt = value.seal()
        payload = receipt.payload()
        payload["runtime_execution_observed"] = True
        unsigned = dict(payload)
        unsigned.pop("digest")
        payload["digest"] = core.object_sha256(unsigned)
        with self.assertRaisesRegex(core.V15DContractError, "hard false"):
            core.ZeroUpdateRunReceiptV15D.from_mapping(payload)
        for field, replacement in (
            ("cell_count", float(receipt.cell_count)),
            ("expected_cell_count", float(receipt.expected_cell_count)),
            ("selected_route_cell_count", float(receipt.selected_route_cell_count)),
            ("all_physical_blocks_seen_per_step_cfg", 1),
            ("runtime_execution_observed", 0),
        ):
            changed = receipt.payload()
            changed[field] = replacement
            unsigned = dict(changed)
            unsigned.pop("digest")
            changed["digest"] = core.object_sha256(unsigned)
            with self.subTest(exact_scalar=field):
                with self.assertRaises(core.V15DContractError):
                    core.ZeroUpdateRunReceiptV15D.from_mapping(changed)


if __name__ == "__main__":
    unittest.main()
