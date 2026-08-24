from __future__ import annotations

import hashlib
import unittest

import torch

from methods.bernini_action_editing import self_generated_branch_interaction_graph_observer_v3 as graph
from methods.bernini_action_editing import self_generated_relational_action_graph_observer_v1 as legacy
from methods.bernini_action_editing import self_generated_relational_t2v_probe_registry_v3 as registry


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _roles() -> tuple[legacy.RoleSpec, ...]:
    return tuple(
        legacy.RoleSpec(
            role_id,
            "self_generated_anchor_owned",
            semantic_role={
                "agent": "human_agent",
                "moving_object": "moving_object",
                "start_support": "support_surface",
                "end_support": "support_surface",
                "null_context": "distractor",
            }[role_id],
            critical=role_id != "null_context",
        )
        for role_id in registry.ROLE_IDS
    )


def _cell(
    appearance: str,
    arm: str,
    sigma: str,
    block: int,
    *,
    shared_state: bool = False,
    wrong_direction: bool = False,
    shifted_reverse: bool = False,
) -> legacy.CaptureCell:
    height, width = 4, 8
    patches = height * width
    responsibilities = torch.zeros((graph.PHASES, len(registry.ROLE_IDS), patches))
    for phase in range(graph.PHASES):
        progress = phase / float(graph.PHASES - 1)
        if arm == "action":
            object_progress = 1.0 - progress if wrong_direction else progress
        elif arm == "reverse":
            object_progress = progress if wrong_direction else 1.0 - progress
        elif arm == "noop":
            object_progress = 0.0
        else:
            object_progress = 0.5
        if arm == "reverse" and shifted_reverse:
            # Same signed displacement/time profile as a valid reverse, but
            # endpoints lie one patch to the right.  A center-only retime can
            # be fooled by this; semantic endpoint closure must reject it.
            object_x = min(width - 1, max(0, round(2 + object_progress * (width - 3))))
        else:
            object_x = min(width - 2, max(1, round(1 + object_progress * (width - 3))))
        agent_x = object_x
        indices = {
            "agent": 2 * width + agent_x,
            "moving_object": 1 * width + object_x,
            "start_support": 3 * width + 1,
            "end_support": 3 * width + (width - 2),
        }
        owned = set(indices.values())
        for role, patch in indices.items():
            responsibilities[phase, registry.ROLE_IDS.index(role), patch] = 1.0
        for patch in range(patches):
            if patch not in owned:
                responsibilities[phase, registry.ROLE_IDS.index("null_context"), patch] = 1.0
    generator = torch.Generator().manual_seed(block + len(appearance) + len(arm))
    queries = torch.randn((graph.PHASES, patches, 4), generator=generator).contiguous()
    keys = torch.randn((graph.PHASES, patches, 4), generator=generator).contiguous()
    state_arm = "shared" if shared_state else arm
    return legacy.CaptureCell(
        appearance_id=appearance,
        arm=arm,
        sigma_band=sigma,
        block_index=block,
        state_sha256=_sha(f"{appearance}:{state_arm}:{sigma}:state"),
        prompt_sha256=_sha(f"{appearance}:{arm}:prompt"),
        patch_height=height,
        patch_width=width,
        roles=registry.ROLE_IDS,
        queries=queries,
        keys=keys,
        responsibilities=responsibilities.contiguous(),
    )


def _fill(
    stream: graph.StreamingBranchInteractionGraphObserver,
    *,
    shared_state: bool = False,
    wrong_direction: bool = False,
    shifted_reverse: bool = False,
) -> None:
    for appearance in registry.APPEARANCE_IDS:
        for arm in registry.ARMS:
            for sigma in graph.SIGMA_BANDS:
                for block in graph.BLOCKS:
                    stream.add(
                        _cell(
                            appearance,
                            arm,
                            sigma,
                            block,
                            shared_state=shared_state,
                            wrong_direction=wrong_direction,
                            shifted_reverse=shifted_reverse,
                        )
                    )


class BranchInteractionGraphObserverV3Tests(unittest.TestCase):
    def test_perfect_branch_trajectory_is_mechanically_admitted(self) -> None:
        stream = graph.StreamingBranchInteractionGraphObserver(roles=_roles())
        _fill(stream)
        result = stream.finalize()
        self.assertEqual(result["status"], "MECHANICALLY_ADMITTED")
        self.assertTrue(result["summary"]["mechanical_admission_passed"])
        self.assertEqual(result["capture_matrix"]["capture_count"], 144)
        self.assertEqual(result["capture_matrix"]["raw_zeroized_capture_count"], 144)
        self.assertEqual(result["reduced_internal_tensor_zeroized_count"], 144)
        self.assertFalse(result["capture_matrix"]["same_state_within_appearance_sigma"])
        self.assertFalse(result["scientific_claim_authorized"])

    def test_same_state_prompt_overlay_is_rejected_fail_closed(self) -> None:
        stream = graph.StreamingBranchInteractionGraphObserver(roles=_roles())
        _fill(stream, shared_state=True)
        with self.assertRaisesRegex(graph.BranchInteractionGraphError, "did not diverge"):
            stream.finalize()

    def test_wrong_signed_direction_is_rejected(self) -> None:
        stream = graph.StreamingBranchInteractionGraphObserver(roles=_roles())
        _fill(stream, wrong_direction=True)
        result = stream.finalize()
        self.assertEqual(result["status"], "REJECTED")
        self.assertTrue(
            all(
                not row["control_gates"]["forward_progress_nonzero"]
                for row in result["appearance_control_results"]
            )
        )

    def test_retimed_shape_with_wrong_reverse_endpoints_is_rejected(self) -> None:
        stream = graph.StreamingBranchInteractionGraphObserver(roles=_roles())
        _fill(stream, shifted_reverse=True)
        result = stream.finalize()
        self.assertEqual(result["status"], "REJECTED")
        self.assertTrue(
            any(
                not row["control_gates"]["reverse_endpoint_topology_closes"]
                or not row["control_gates"]["reverse_endpoint_each_feature_closes"]
                for row in result["appearance_control_results"]
            )
        )

    def test_add_zeroizes_raw_capture_immediately(self) -> None:
        stream = graph.StreamingBranchInteractionGraphObserver(roles=_roles())
        cell = _cell("appearance_0", "action", "high", 6)
        stream.add(cell)
        self.assertEqual(int(torch.count_nonzero(cell.queries).item()), 0)
        self.assertEqual(int(torch.count_nonzero(cell.keys).item()), 0)
        self.assertEqual(int(torch.count_nonzero(cell.responsibilities).item()), 0)


if __name__ == "__main__":
    unittest.main()
