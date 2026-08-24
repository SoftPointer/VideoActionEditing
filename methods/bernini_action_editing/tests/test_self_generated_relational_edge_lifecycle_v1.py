#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest
from typing import Optional, Sequence

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_relational_action_graph_observer_v1 as observer  # noqa: E402


INTRODUCED_PHASE = 7


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _roles(*, latent_support: bool = False) -> tuple[observer.RoleSpec, ...]:
    return (
        observer.RoleSpec(
            "human_agent",
            "source_owned",
            semantic_role="human_agent",
            source_node_id="source_agent",
        ),
        observer.RoleSpec(
            "new_object",
            "instruction_introduced",
            semantic_role="moving_object",
            first_reliable_phase=INTRODUCED_PHASE,
        ),
        observer.RoleSpec(
            "support_surface",
            "source_owned",
            semantic_role="support_surface",
            evidence_mode=(
                "latent_unobserved" if latent_support else "observed_internal"
            ),
            source_node_id=None if latent_support else "source_support",
            critical=False,
        ),
    )


def _required_edge() -> observer.EdgeSpec:
    return observer.EdgeSpec(
        source_role="human_agent",
        target_role="new_object",
        relation_type="relative_motion",
        first_applicable_phase=INTRODUCED_PHASE,
        last_applicable_phase=observer.PHASES - 1,
        applicability="required",
    )


def _not_applicable_support_edge() -> observer.EdgeSpec:
    return observer.EdgeSpec(
        source_role="new_object",
        target_role="support_surface",
        relation_type="supports",
        first_applicable_phase=INTRODUCED_PHASE,
        last_applicable_phase=observer.PHASES - 1,
        applicability="not_applicable",
    )


def _base_and_lifecycle_events() -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    generator = torch.Generator(device="cpu").manual_seed(71)
    base = torch.randn((1, 6, 8), generator=generator).repeat(
        observer.PHASES,
        1,
        1,
    )
    action = torch.zeros((observer.PHASES, 6, 8), dtype=torch.float32)
    window_length = observer.PHASES - INTRODUCED_PHASE
    for phase in range(INTRODUCED_PHASE, observer.PHASES):
        local_phase = phase - INTRODUCED_PHASE
        moving_patch = min(
            5,
            int(round(5 * local_phase / (window_length - 1))),
        )
        action[phase, 0, 0] = 1.4
        action[phase, moving_patch, 1] = 2.2
        action[phase, 5, 2] = 1.6
        action[phase, moving_patch, 4:6] = (
            0.2 + local_phase / window_length
        )
    reverse = action.clone()
    reverse[INTRODUCED_PHASE:] = torch.flip(
        action[INTRODUCED_PHASE:],
        dims=(0,),
    )
    static = torch.zeros_like(action)
    static[INTRODUCED_PHASE:] = action[
        INTRODUCED_PHASE:INTRODUCED_PHASE + 1
    ]
    return base, {
        "action": action,
        "noop": torch.zeros_like(action),
        "reverse": reverse,
        "static": static,
    }


def _arm_tensors(
    arm: str,
    *,
    latent_support: bool,
    broken_reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base, events = _base_and_lifecycle_events()
    event = events["action"] if arm == "reverse" and broken_reverse else events[arm]
    hidden = base + event
    heads = hidden.reshape(observer.PHASES, 6, 2, 4)
    query = heads.mean(dim=2).detach().contiguous()
    key = torch.roll(heads, shifts=1, dims=1).mean(dim=2).detach().contiguous()
    responsibility = torch.softmax(
        torch.stack((hidden[..., 0], hidden[..., 1], hidden[..., 2]), dim=1),
        dim=1,
    ).detach().contiguous()
    # An introduced node is structurally absent, rather than uncertain, before
    # its first reliable phase.  A latent support never publishes a visual slot.
    responsibility[:INTRODUCED_PHASE, 1].zero_()
    if latent_support:
        responsibility[:, 2].zero_()
    return query, key, responsibility


def _populate_and_finalize(
    *,
    roles: Sequence[observer.RoleSpec],
    edge_specs: tuple[observer.EdgeSpec, ...],
    missing_observation: Optional[tuple[str, str, str, int, int]] = None,
    broken_reverse: bool = False,
) -> dict:
    stream = observer.StreamingRelationalObserver(
        roles=roles,
        config=observer.ObserverConfig(edge_specs=edge_specs),
    )
    latent_support = roles[2].evidence_mode == "latent_unobserved"
    role_ids = tuple(item.role_id for item in roles)
    for appearance_index in range(observer.APPEARANCE_COUNT):
        appearance = f"appearance_{appearance_index}"
        for sigma in observer.SIGMA_BANDS:
            state_sha256 = _sha(f"state:{appearance}:{sigma}")
            for arm in observer.ARMS:
                query, key, responsibility = _arm_tensors(
                    arm,
                    latent_support=latent_support,
                    broken_reverse=broken_reverse,
                )
                for block in observer.BLOCKS:
                    cell_query = query.clone().contiguous()
                    cell_key = key.clone().contiguous()
                    cell_responsibility = responsibility.clone().contiguous()
                    identity = (appearance, arm, sigma, block)
                    if (
                        missing_observation is not None
                        and identity == missing_observation[:4]
                    ):
                        phase = missing_observation[4]
                        cell_responsibility[phase, 1].zero_()
                    stream.add(
                        observer.CaptureCell(
                            appearance_id=appearance,
                            arm=arm,
                            sigma_band=sigma,
                            block_index=block,
                            state_sha256=state_sha256,
                            prompt_sha256=_sha(f"prompt:{arm}"),
                            patch_height=2,
                            patch_width=3,
                            roles=role_ids,
                            queries=cell_query,
                            keys=cell_key,
                            responsibilities=cell_responsibility,
                        ),
                        zeroize=True,
                    )
    return dict(stream.finalize())


class TypedEdgeLifecycleTests(unittest.TestCase):
    def test_self_generated_anchor_role_has_distinct_non_source_provenance(self) -> None:
        role = observer.RoleSpec(
            "anchor_object",
            "self_generated_anchor_owned",
            semantic_role="moving_object",
            evidence_mode="observed_internal",
            first_reliable_phase=0,
            source_node_id=None,
        )
        self.assertEqual(role.ownership, "self_generated_anchor_owned")
        self.assertFalse(role.receipt()["mask_identity_claimed"])
        with self.assertRaisesRegex(
            observer.RelationalObserverError, "cannot inherit a source identity"
        ):
            observer.RoleSpec(
                "anchor_object",
                "self_generated_anchor_owned",
                semantic_role="moving_object",
                source_node_id="source_object_0",
            )

    def test_one_role_pair_can_have_multiple_typed_lifecycle_hypotheses(self) -> None:
        pair = ("human_agent", "new_object")
        config = observer.ObserverConfig(
            edge_specs=(
                observer.EdgeSpec(
                    *pair,
                    relation_type="relative_motion",
                    first_applicable_phase=INTRODUCED_PHASE,
                    last_applicable_phase=12,
                ),
                observer.EdgeSpec(
                    *pair,
                    relation_type="releases",
                    first_applicable_phase=13,
                    last_applicable_phase=observer.PHASES - 1,
                ),
            )
        )
        self.assertEqual(
            [item.relation_type for item in config.resolved_edge_specs()],
            ["relative_motion", "releases"],
        )
        with self.assertRaisesRegex(
            observer.RelationalObserverError, "contains duplicates"
        ):
            observer.ObserverConfig(
                edge_specs=(config.edge_specs[0], config.edge_specs[0])
            )

    def test_reducer_materializes_only_explicit_required_role_pairs(self) -> None:
        roles = _roles(latent_support=False)
        config = observer.ObserverConfig(edge_specs=(_required_edge(),))
        query, key, responsibility = _arm_tensors(
            "action", latent_support=False, broken_reverse=False
        )
        cell = observer.CaptureCell(
            appearance_id="appearance_0",
            arm="action",
            sigma_band="high",
            block_index=6,
            state_sha256=_sha("state"),
            prompt_sha256=_sha("prompt"),
            patch_height=2,
            patch_width=3,
            roles=tuple(item.role_id for item in roles),
            queries=query,
            keys=key,
            responsibilities=responsibility,
        )
        reduced = observer._reduce_cell(cell, roles, config)
        self.assertEqual(reduced.edge_ids, ((_required_edge().pair),))
        self.assertEqual(int(reduced.values.shape[0]), 1)

    def test_introduced_role_phase7_required_window_closes(self) -> None:
        result = _populate_and_finalize(
            roles=_roles(),
            edge_specs=(_required_edge(),),
        )

        self.assertEqual(result["status"], "MECHANICALLY_ADMITTED")
        self.assertEqual(result["edge_registry_summary"]["required_edge_count"], 1)
        self.assertEqual(
            result["edge_registry_summary"]["not_applicable_edge_count"],
            0,
        )
        self.assertFalse(
            result["edge_registry_summary"]["default_cartesian_product_used"]
        )
        row = result["appearance_packets"][0]["edges"][0]
        self.assertEqual(row["relation_type"], "relative_motion")
        self.assertTrue(row["relation_type_is_preregistered_hypothesis"])
        self.assertFalse(row["typed_relation_truth_claimed"])
        self.assertFalse(row["physical_relation_truth_claimed"])
        self.assertTrue(row["contributes_to_reward"])
        self.assertTrue(row["controls_passed"])
        self.assertEqual(
            [item["status"] for item in row["phase_rows"][:INTRODUCED_PHASE]],
            ["not_applicable"] * INTRODUCED_PHASE,
        )
        self.assertTrue(
            all(
                item["relative_features"] is None
                for item in row["phase_rows"][:INTRODUCED_PHASE]
            )
        )
        self.assertTrue(
            all(
                item["status"] == "observed"
                for item in row["phase_rows"][INTRODUCED_PHASE:]
            )
        )

    def test_missing_observation_inside_required_window_rejects(self) -> None:
        missing_phase = 10
        result = _populate_and_finalize(
            roles=_roles(),
            edge_specs=(_required_edge(),),
            missing_observation=("appearance_0", "action", "high", 6, missing_phase),
        )

        self.assertEqual(result["status"], "REJECTED")
        self.assertTrue(result["summary"]["any_critical_phase_uncertain"])
        row = result["appearance_packets"][0]["edges"][0]
        self.assertFalse(row["control_gates"]["all_cells_observed"])
        self.assertEqual(row["phase_rows"][missing_phase]["status"], "uncertain")
        self.assertIsNone(row["phase_rows"][missing_phase]["relative_features"])

    def test_not_applicable_support_never_passes_or_contributes_reward(self) -> None:
        edge_specs = (_required_edge(), _not_applicable_support_edge())
        admitted = _populate_and_finalize(
            roles=_roles(latent_support=True),
            edge_specs=edge_specs,
        )
        rejected = _populate_and_finalize(
            roles=_roles(latent_support=True),
            edge_specs=edge_specs,
            broken_reverse=True,
        )

        self.assertEqual(admitted["status"], "MECHANICALLY_ADMITTED")
        self.assertEqual(rejected["status"], "REJECTED")
        for result in (admitted, rejected):
            self.assertEqual(result["edge_registry_summary"]["required_edge_count"], 1)
            self.assertEqual(
                result["edge_registry_summary"]["not_applicable_edge_count"],
                1,
            )
            self.assertFalse(
                result["edge_registry_summary"][
                    "not_applicable_edges_contribute_to_reward"
                ]
            )
            for appearance in result["appearance_packets"]:
                support_row = next(
                    row
                    for row in appearance["edges"]
                    if row["relation_type"] == "supports"
                )
                self.assertEqual(support_row["edge_status"], "not_applicable")
                self.assertFalse(support_row["contributes_to_reward"])
                self.assertIsNone(support_row["control_metrics"])
                self.assertIsNone(support_row["control_gates"])
                self.assertIsNone(support_row["controls_passed"])
                self.assertTrue(
                    all(
                        phase["status"] == "not_applicable"
                        and phase["relative_features"] is None
                        for phase in support_row["phase_rows"]
                    )
                )
            self.assertTrue(
                all(
                    row["relation_type"] != "supports"
                    for row in result["multiappearance_consensus"]
                )
            )

    def test_empty_or_only_not_applicable_registry_fails_closed(self) -> None:
        with self.subTest("default empty registry"):
            with self.assertRaisesRegex(
                observer.RelationalObserverError,
                "explicit typed edge registry",
            ):
                observer.ObserverConfig()

        with self.subTest("not-applicable only cannot masquerade as success"):
            with self.assertRaisesRegex(
                observer.RelationalObserverError,
                "at least one required edge",
            ):
                observer.ObserverConfig(
                    edge_specs=(_not_applicable_support_edge(),),
                )

    def test_edge_schema_and_role_lifecycle_are_fail_closed(self) -> None:
        with self.subTest("relation allowlist"):
            with self.assertRaisesRegex(
                observer.RelationalObserverError,
                "relation differs",
            ):
                observer.EdgeSpec(
                    "human_agent",
                    "new_object",
                    "arbitrary_relation",
                )

        with self.subTest("required edge cannot predate introduced role"):
            premature = observer.EdgeSpec(
                "human_agent",
                "new_object",
                "relative_motion",
                first_applicable_phase=INTRODUCED_PHASE - 1,
            )
            with self.assertRaisesRegex(
                observer.RelationalObserverError,
                "starts before both roles are observable",
            ):
                observer.StreamingRelationalObserver(
                    roles=_roles(),
                    config=observer.ObserverConfig(edge_specs=(premature,)),
                )

        with self.subTest("shared pair types remain hypotheses, not physical truth"):
            typed = observer.ObserverConfig(
                edge_specs=(
                    observer.EdgeSpec(
                        "human_agent",
                        "support_surface",
                        "supports",
                        first_applicable_phase=0,
                        last_applicable_phase=9,
                    ),
                    observer.EdgeSpec(
                        "human_agent",
                        "support_surface",
                        "releases",
                        first_applicable_phase=10,
                        last_applicable_phase=20,
                    ),
                )
            )
            self.assertEqual(len(typed.edge_specs), 2)
            self.assertTrue(
                all(
                    item.receipt()["relation_type_is_preregistered_hypothesis"]
                    and not item.receipt()["physical_relation_truth_claimed"]
                    for item in typed.edge_specs
                )
            )

        with self.subTest("legacy explicit pair remains a compatibility bridge"):
            legacy = observer.ObserverConfig(
                critical_edges=(("human_agent", "support_surface"),),
            )
            self.assertTrue(legacy.legacy_edge_registry)
            self.assertEqual(
                legacy.resolved_edge_specs(),
                (
                    observer.EdgeSpec(
                        "human_agent",
                        "support_surface",
                        "relative_motion",
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
