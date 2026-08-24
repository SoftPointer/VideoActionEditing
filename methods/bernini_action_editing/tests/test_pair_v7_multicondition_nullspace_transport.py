#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if torch is not None:
    import pair_v7_dual_coordinate_nullspace_transport as core
    import pair_v7_multicondition_nullspace_transport as multi
else:
    core = None
    multi = None


B_NAME = "blocks.0.attn2.to_q.action_lora_b.weight"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def world_input_receipt(actions, identities, *, authority):
    action_rows = []
    for row in actions:
        layout = core.GradientLayout.from_named_gradients(row.gradient_by_parameter)
        action_rows.append(
            {
                "condition_id": row.condition_id,
                "pair_id": row.pair_id,
                "source_sample_id": row.source_sample_id,
                "schedule_index": row.schedule_index,
                "candidate_id": row.candidate_id,
                "action_family": row.action_family,
                "event_digest": row.event_digest,
                "gradient_computation_receipt_digest": row.gradient_computation_receipt_digest,
                "gradient_sha256": multi._gradient_digest(
                    layout, row.gradient_by_parameter, label=row.condition_id
                ),
            }
        )
    identity_rows = []
    grouped = {}
    for row in identities:
        layout = core.GradientLayout.from_named_gradients(
            row.probe.gradient_by_parameter
        )
        identity_rows.append(
            {
                "probe_id": row.probe.probe_id,
                "pair_id": row.pair_id,
                "source_sample_id": row.source_sample_id,
                "schedule_index": row.schedule_index,
                "family": row.probe.family,
                "sketch_index": row.sketch_index,
                "feature_sketch_sha256": row.probe.feature_sketch_sha256,
                "source_coordinate_receipt_digest": row.probe.source_coordinate_receipt_digest,
                "gradient_computation_receipt_digest": row.probe.gradient_computation_receipt_digest,
                "gradient_sha256": multi._gradient_digest(
                    layout,
                    row.probe.gradient_by_parameter,
                    label=row.probe.probe_id,
                ),
            }
        )
        grouped.setdefault(
            (row.pair_id, row.source_sample_id, row.schedule_index), []
        ).append(row)
    coordinate_cells = []
    for (pair, source, schedule), rows in sorted(grouped.items()):
        coordinate_cells.append(
            {
                "pair_id": pair,
                "source_sample_id": source,
                "schedule_index": schedule,
                "identity_family_count": 2,
                "identity_probe_count": len(rows),
                "source_coordinate_receipt_digest": sorted(
                    {row.probe.source_coordinate_receipt_digest for row in rows}
                )[0],
                "cross_family_coordinate_consensus": True,
            }
        )
    unsigned = {
        "schema_version": multi.WORLD_INPUT_SCHEMA,
        "manifest_digest": authority,
        "checkpoint_content_receipt_digest": actions[0].checkpoint_content_receipt_digest,
        "parameter_state_sha256": actions[0].parameter_state_sha256,
        "action_condition_count": len(actions),
        "identity_probe_count": len(identities),
        "action_rows": sorted(action_rows, key=lambda row: row["condition_id"]),
        "identity_rows": sorted(identity_rows, key=lambda row: row["probe_id"]),
        "identity_cross_family_coordinate_cells": coordinate_cells,
        "raw_gradient_values_persisted": False,
    }
    return {**unsigned, "input_digest": core.object_sha256(unsigned)}


@unittest.skipIf(torch is None, "torch is required for geometry tests")
class MulticonditionCommonDirectionTests(unittest.TestCase):
    def solve(self, actions, identities, *, authority=sha("authority"), receipt=None):
        return multi.solve_multicondition_common_direction(
            action_conditions=actions,
            identity_conditions=identities,
            multicondition_authority_digest=authority,
            validated_measurement_input_receipt=(
                world_input_receipt(actions, identities, authority=authority)
                if receipt is None
                else receipt
            ),
        )

    def build_rows(self):
        pairs = {
            "fit": (
                ("fit-dog", "dog-sit-facing-camera"),
                ("fit-human", "human-rise-to-stand"),
            ),
            "confirmation": (
                ("confirmation-dog", "dog-sit-facing-camera"),
                ("confirmation-human", "human-rise-to-stand"),
            ),
        }
        checkpoint = sha("checkpoint")
        state = sha("state")
        actions = []
        identities = []
        group_ordinal = 0
        for pair, sources in pairs.items():
            for source, family in sources:
                for schedule in multi.PRIMARY_SCHEDULE_INDICES:
                    action = torch.zeros(7, 7, dtype=torch.float32)
                    action.reshape(-1)[0] = 1.0 + 0.01 * len(actions)
                    condition_id = f"{pair}.{source}.s{schedule}"
                    actions.append(
                        multi.ActionConditionGradient(
                            condition_id=condition_id,
                            pair_id=pair,
                            source_sample_id=source,
                            schedule_index=schedule,
                            candidate_id=f"candidate-{source}",
                            action_family=family,
                            event_digest=sha(f"event-{source}"),
                            gradient_computation_receipt_digest=sha(
                                f"action-receipt-{condition_id}"
                            ),
                            gradient_by_parameter={B_NAME: action},
                            checkpoint_content_receipt_digest=checkpoint,
                            parameter_state_sha256=state,
                        )
                    )
                    coordinate = sha(f"coordinate-{source}-s{schedule}")
                    for identity_family in core.REQUIRED_IDENTITY_FAMILIES:
                        start = 1 + 3 * group_ordinal
                        basis = []
                        for offset in range(3):
                            vector = torch.zeros(49, dtype=torch.float32)
                            vector[start + offset] = 1.0
                            basis.append(vector)
                        basis.append(basis[0] + basis[1] + basis[2])
                        for sketch, vector in enumerate(basis):
                            probe_id = (
                                f"{pair}.{source}.s{schedule}."
                                f"{identity_family}.k{sketch}"
                            )
                            probe = core.IdentityGradientProbe(
                                probe_id=probe_id,
                                family=identity_family,
                                gradient_by_parameter={
                                    B_NAME: vector.reshape(7, 7).contiguous()
                                },
                                feature_sketch_sha256=sha(
                                    f"feature-{probe_id}"
                                ),
                                source_coordinate_receipt_digest=coordinate,
                                gradient_computation_receipt_digest=sha(
                                    f"identity-receipt-{probe_id}"
                                ),
                                checkpoint_content_receipt_digest=checkpoint,
                                parameter_state_sha256=state,
                            )
                            identities.append(
                                multi.IdentityConditionProbe(
                                    pair_id=pair,
                                    source_sample_id=source,
                                    schedule_index=schedule,
                                    sketch_index=sketch,
                                    probe=probe,
                                )
                            )
                        group_ordinal += 1
        return actions, identities

    def test_fixed_8x64_panel_has_one_common_go_direction(self) -> None:
        actions, identities = self.build_rows()
        result = self.solve(actions, identities)
        self.assertTrue(result.primary_replication_go)
        self.assertEqual(result.receipt["failure_codes"], [])
        self.assertEqual(result.receipt["action_condition_count"], 8)
        self.assertEqual(result.receipt["identity_probe_count"], 64)
        self.assertFalse(
            result.receipt["pilot_schedule_index_33_included_in_primary_gate"]
        )
        self.assertEqual(len(result.receipt["local_cell_receipts"]), 4)
        self.assertTrue(
            all(
                row["geometry_audit_passed"]
                for row in result.receipt["local_cell_receipts"]
            )
        )
        routing = result.receipt["typed_routing_diagnostic"]
        self.assertEqual(
            routing["schema_version"], multi.ROUTING_DIAGNOSTIC_SCHEMA
        )
        self.assertFalse(routing["optimizer_authorized"])
        self.assertFalse(routing["parameter_update_authorized"])
        self.assertFalse(routing["action_success_claimed"])
        self.assertFalse(routing["parameter_mutation_performed"])
        self.assertEqual(
            [row["route_mode"] for row in routing["routing_modes"]],
            [
                "sigma",
                "action_family",
                "sigma_x_action_family",
                "pair_x_sigma",
            ],
        )
        self.assertTrue(
            all(
                row["all_partition_geometry_passed"]
                for row in routing["routing_modes"]
            )
        )
        self.assertFalse(
            next(
                row
                for row in routing["routing_modes"]
                if row["route_mode"] == "pair_x_sigma"
            )["deployment_observable_route"]
        )
        subspace = result.receipt["event_subspace_diagnostic"]
        self.assertEqual(
            subspace["schema_version"], multi.SUBSPACE_DIAGNOSTIC_SCHEMA
        )
        self.assertFalse(subspace["raw_gradient_basis_or_direction_persisted"])
        self.assertFalse(subspace["parameter_mutation_performed"])
        self.assertEqual(
            [row["route_mode"] for row in subspace["routing_modes"]],
            ["global", "sigma", "action_family", "sigma_x_action_family"],
        )
        self.assertTrue(
            all(row["all_group_geometry_passed"] for row in subspace["routing_modes"])
        )
        self.assertGreaterEqual(
            result.global_transport.receipt["identity_effective_rank"], 16
        )

    def test_one_opposed_condition_makes_global_direction_no_go(self) -> None:
        actions, identities = self.build_rows()
        victim = actions[-1]
        opposed = {
            name: -value for name, value in victim.gradient_by_parameter.items()
        }
        actions[-1] = multi.ActionConditionGradient(
            **{
                **victim.__dict__,
                "gradient_by_parameter": opposed,
            }
        )
        result = self.solve(actions, identities)
        self.assertFalse(result.primary_replication_go)
        global_rows = result.receipt["global_common_direction_receipt"][
            "per_condition_action_descent"
        ]
        self.assertFalse(next(row for row in global_rows if row["condition_id"] == victim.condition_id)["passed"])
        # A fit-learned signed subspace may still support the held-out live
        # action even when one immutable shared direction cannot support all
        # conditions at once.
        subspace_global = result.receipt["event_subspace_diagnostic"][
            "routing_modes"
        ][0]
        self.assertTrue(subspace_global["all_group_geometry_passed"])

    def test_confirmation_identity_tangent_is_not_event_subspace_support(self) -> None:
        actions, identities = self.build_rows()
        victim = actions[-1]
        tangent = next(
            row.probe.gradient_by_parameter
            for row in identities
            if (
                row.pair_id == victim.pair_id
                and row.source_sample_id == victim.source_sample_id
                and row.schedule_index == victim.schedule_index
            )
        )
        actions[-1] = multi.ActionConditionGradient(
            **{**victim.__dict__, "gradient_by_parameter": tangent}
        )
        result = self.solve(actions, identities)
        self.assertFalse(result.primary_replication_go)
        route = next(
            row
            for row in result.receipt["event_subspace_diagnostic"][
                "routing_modes"
            ]
            if row["route_mode"] == "sigma_x_action_family"
        )
        failed = next(
            row
            for row in route["group_receipts"]
            if row["group_id"]
            == f"s{victim.schedule_index}:{victim.action_family}"
        )
        confirmation = failed["confirmation_receipts"][0]
        self.assertFalse(confirmation["passed"])
        self.assertEqual(confirmation["safe_subspace_coverage_cosine"], 0.0)

    def test_degenerate_k4_group_is_reported_not_threshold_relaxed(self) -> None:
        actions, identities = self.build_rows()
        target = identities[:4]
        repeated = target[0].probe.gradient_by_parameter
        for index in range(1, 4):
            row = identities[index]
            identities[index] = multi.IdentityConditionProbe(
                pair_id=row.pair_id,
                source_sample_id=row.source_sample_id,
                schedule_index=row.schedule_index,
                sketch_index=row.sketch_index,
                probe=core.IdentityGradientProbe(
                    **{
                        **row.probe.__dict__,
                        "gradient_by_parameter": repeated,
                    }
                ),
            )
        result = self.solve(actions, identities)
        self.assertFalse(result.primary_replication_go)
        self.assertTrue(
            any(code.startswith("IDENTITY_GROUP_RANK_BELOW_3") for code in result.receipt["failure_codes"])
        )

    def test_unregistered_schedule_is_rejected(self) -> None:
        actions, identities = self.build_rows()
        row = actions[0]
        actions[0] = multi.ActionConditionGradient(
            **{**row.__dict__, "schedule_index": 33}
        )
        with self.assertRaisesRegex(
            multi.PairV7MulticonditionError, "outside preregistration"
        ):
            self.solve(actions, identities)

    def test_missing_identity_row_is_rejected(self) -> None:
        actions, identities = self.build_rows()
        with self.assertRaisesRegex(
            multi.PairV7MulticonditionError, "row count"
        ):
            self.solve(actions, identities[:-1])

    def test_arbitrary_authority_or_live_row_substitution_is_rejected(self) -> None:
        actions, identities = self.build_rows()
        receipt = world_input_receipt(
            actions, identities, authority=sha("authority")
        )
        with self.assertRaisesRegex(
            multi.PairV7MulticonditionError, "seal/binding"
        ):
            self.solve(
                actions,
                identities,
                authority=sha("fictional-authority"),
                receipt=receipt,
            )

        victim = actions[0]
        substituted = {
            name: value.clone() for name, value in victim.gradient_by_parameter.items()
        }
        substituted[B_NAME].reshape(-1)[0] += 0.25
        actions[0] = multi.ActionConditionGradient(
            **{**victim.__dict__, "gradient_by_parameter": substituted}
        )
        with self.assertRaisesRegex(
            multi.PairV7MulticonditionError, "live 8x64"
        ):
            self.solve(actions, identities, receipt=receipt)

    def test_cross_family_coordinate_substitution_is_rejected(self) -> None:
        actions, identities = self.build_rows()
        row = identities[4]
        identities[4] = multi.IdentityConditionProbe(
            pair_id=row.pair_id,
            source_sample_id=row.source_sample_id,
            schedule_index=row.schedule_index,
            sketch_index=row.sketch_index,
            probe=core.IdentityGradientProbe(
                **{
                    **row.probe.__dict__,
                    "source_coordinate_receipt_digest": sha("other-coordinate"),
                }
            ),
        )
        receipt = world_input_receipt(
            actions, identities, authority=sha("authority")
        )
        with self.assertRaisesRegex(
            multi.PairV7MulticonditionError, "share one sealed"
        ):
            self.solve(actions, identities, receipt=receipt)


if __name__ == "__main__":
    unittest.main()
