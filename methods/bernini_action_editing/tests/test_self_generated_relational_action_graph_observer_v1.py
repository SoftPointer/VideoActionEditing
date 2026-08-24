#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
from typing import Any, Iterable, Optional, Sequence

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_relational_action_graph_observer_v1 as observer  # noqa: E402


CRITICAL_EDGE = ("human_agent", "moving_object")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _roles() -> tuple[observer.RoleSpec, ...]:
    return (
        observer.RoleSpec(
            "human_agent",
            "source_owned",
            semantic_role="human_agent",
            source_node_id="source_agent",
            critical=True,
        ),
        observer.RoleSpec(
            "moving_object",
            "source_owned",
            semantic_role="moving_object",
            source_node_id="source_object",
            critical=True,
        ),
        observer.RoleSpec(
            "support_surface",
            "source_owned",
            semantic_role="support_surface",
            source_node_id="source_support",
            critical=False,
        ),
    )


def _config() -> observer.ObserverConfig:
    return observer.ObserverConfig(
        edge_specs=(
            observer.EdgeSpec(
                source_role=CRITICAL_EDGE[0],
                target_role=CRITICAL_EDGE[1],
                relation_type="relative_motion",
            ),
        )
    )


def _base_and_event() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7)
    base = torch.randn((1, 6, 8), generator=generator).repeat(
        observer.PHASES, 1, 1
    )
    event = torch.zeros((observer.PHASES, 6, 8), dtype=torch.float32)
    for phase in range(observer.PHASES):
        moving_patch = min(
            5,
            int(round(5 * phase / (observer.PHASES - 1))),
        )
        event[phase, 0, 0] = 1.4
        event[phase, moving_patch, 1] = 2.2
        event[phase, 5, 2] = 1.6
        event[phase, moving_patch, 4:6] = 0.2 + phase / observer.PHASES
    return base, event


def _arm_tensors(
    arm: str,
    *,
    appearance_scale: float = 1.0,
    broken_reverse: bool = False,
    broken_static: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base, unit_event = _base_and_event()
    event = unit_event * appearance_scale
    if arm == "action":
        hidden = base + event
    elif arm == "noop":
        hidden = base.clone()
    elif arm == "reverse":
        hidden = base + (
            event if broken_reverse else torch.flip(event, dims=(0,))
        )
    elif arm == "static":
        hidden = base + (
            event
            if broken_static
            else event[:1].repeat(observer.PHASES, 1, 1)
        )
    else:  # pragma: no cover - the helper only accepts the sealed ABI arms.
        raise AssertionError(f"unknown test arm: {arm}")

    heads = hidden.reshape(observer.PHASES, 6, 2, 4)
    queries = heads.mean(dim=2).detach().contiguous()
    keys = torch.roll(heads, shifts=1, dims=1).mean(dim=2).detach().contiguous()
    responsibilities = torch.softmax(
        torch.stack((hidden[..., 0], hidden[..., 1], hidden[..., 2]), dim=1),
        dim=1,
    ).detach().contiguous()
    return queries, keys, responsibilities


def _cell(
    *,
    roles: Sequence[observer.RoleSpec],
    appearance: str = "appearance_0",
    arm: str = "action",
    sigma: str = "high",
    block: int = 6,
    state_sha256: Optional[str] = None,
    appearance_scale: float = 1.0,
    broken_reverse: bool = False,
    broken_static: bool = False,
) -> observer.CaptureCell:
    query, key, responsibility = _arm_tensors(
        arm,
        appearance_scale=appearance_scale,
        broken_reverse=broken_reverse,
        broken_static=broken_static,
    )
    return observer.CaptureCell(
        appearance_id=appearance,
        arm=arm,
        sigma_band=sigma,
        block_index=block,
        state_sha256=(
            state_sha256
            if state_sha256 is not None
            else _digest(f"sealed-state:{appearance}:{sigma}")
        ),
        prompt_sha256=_digest(f"sealed-prompt:{arm}"),
        patch_height=2,
        patch_width=3,
        roles=tuple(role.role_id for role in roles),
        queries=query,
        keys=key,
        responsibilities=responsibility,
    )


def _populate(
    stream: observer.StreamingRelationalObserver,
    *,
    roles: Sequence[observer.RoleSpec],
    missing: Optional[tuple[str, str, str, int]] = None,
    mismatched_state: Optional[tuple[str, str, str, int]] = None,
    broken_reverse: bool = False,
    broken_static: bool = False,
    scaled_appearance: Optional[str] = None,
) -> list[observer.CaptureCell]:
    consumed: list[observer.CaptureCell] = []
    for appearance_index in range(observer.APPEARANCE_COUNT):
        appearance = f"appearance_{appearance_index}"
        appearance_scale = 0.30 if appearance == scaled_appearance else 1.0
        for sigma in observer.SIGMA_BANDS:
            for arm in observer.ARMS:
                for block in observer.BLOCKS:
                    key = (appearance, arm, sigma, block)
                    if key == missing:
                        continue
                    state = (
                        _digest("hostile-mismatched-state")
                        if key == mismatched_state
                        else None
                    )
                    cell = _cell(
                        roles=roles,
                        appearance=appearance,
                        arm=arm,
                        sigma=sigma,
                        block=block,
                        state_sha256=state,
                        appearance_scale=appearance_scale,
                        broken_reverse=broken_reverse,
                        broken_static=broken_static,
                    )
                    stream.add(cell, zeroize=True)
                    consumed.append(cell)
    return consumed


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


class StreamingRelationalObserverAdversarialTests(unittest.TestCase):
    def test_exact_144_cells_are_reduced_and_owned_raw_buffers_zeroized(self) -> None:
        roles = _roles()
        stream = observer.StreamingRelationalObserver(roles=roles, config=_config())
        cells = _populate(stream, roles=roles)

        self.assertEqual(len(cells), 144)
        for cell in cells:
            self.assertEqual(int(torch.count_nonzero(cell.queries).item()), 0)
            self.assertEqual(int(torch.count_nonzero(cell.keys).item()), 0)
            self.assertEqual(
                int(torch.count_nonzero(cell.responsibilities).item()),
                0,
            )

        result = stream.finalize()
        self.assertEqual(result["status"], "MECHANICALLY_ADMITTED")
        self.assertEqual(result["capture_matrix"]["capture_count"], 144)
        self.assertEqual(result["capture_matrix"]["zeroized_capture_count"], 144)
        self.assertTrue(result["summary"]["mechanical_admission_passed"])

    def test_missing_and_duplicate_cells_fail_closed(self) -> None:
        roles = _roles()
        with self.subTest("missing cell"):
            stream = observer.StreamingRelationalObserver(
                roles=roles,
                config=_config(),
            )
            cells = _populate(
                stream,
                roles=roles,
                missing=("appearance_2", "static", "mid_low", 24),
            )
            self.assertEqual(len(cells), 143)
            with self.assertRaisesRegex(
                observer.RelationalObserverError,
                "exactly 144",
            ):
                stream.finalize()

        with self.subTest("duplicate cell"):
            stream = observer.StreamingRelationalObserver(
                roles=roles,
                config=_config(),
            )
            first = _cell(roles=roles)
            stream.add(first, zeroize=True)
            duplicate = _cell(roles=roles)
            with self.assertRaisesRegex(
                observer.RelationalObserverError,
                "duplicate capture cell",
            ):
                stream.add(duplicate, zeroize=True)
            for tensor in (
                duplicate.queries,
                duplicate.keys,
                duplicate.responsibilities,
            ):
                self.assertEqual(int(torch.count_nonzero(tensor).item()), 0)

    def test_latent_offscreen_role_cannot_publish_nonzero_responsibility(self) -> None:
        roles = (
            _roles()[0],
            _roles()[1],
            observer.RoleSpec(
                "offscreen_effector",
                "source_owned",
                semantic_role="effector",
                evidence_mode="latent_unobserved",
                critical=False,
            ),
        )
        stream = observer.StreamingRelationalObserver(roles=roles, config=_config())
        cell = _cell(roles=roles)
        self.assertGreater(
            int(torch.count_nonzero(cell.responsibilities[:, 2]).item()),
            0,
        )
        with self.assertRaisesRegex(
            observer.RelationalObserverError,
            "latent/offscreen role cannot carry visual responsibility",
        ):
            stream.add(cell, zeroize=True)
        self.assertEqual(int(torch.count_nonzero(cell.responsibilities).item()), 0)

    def test_instruction_introduced_role_cannot_fabricate_preappearance_evidence(self) -> None:
        roles = (
            _roles()[0],
            observer.RoleSpec(
                "moving_object",
                "instruction_introduced",
                semantic_role="moving_object",
                evidence_mode="observed_internal",
                first_reliable_phase=7,
                critical=True,
            ),
            _roles()[2],
        )
        stream = observer.StreamingRelationalObserver(
            roles=roles,
            config=observer.ObserverConfig(
                edge_specs=(
                    observer.EdgeSpec(
                        "human_agent",
                        "moving_object",
                        "relative_motion",
                        first_applicable_phase=7,
                    ),
                )
            ),
        )
        cell = _cell(roles=roles)
        self.assertGreater(
            int(torch.count_nonzero(cell.responsibilities[:7, 1]).item()),
            0,
        )
        with self.assertRaisesRegex(
            observer.RelationalObserverError,
            "fabricated preappearance evidence",
        ):
            stream.add(cell, zeroize=True)
        self.assertEqual(int(torch.count_nonzero(cell.responsibilities).item()), 0)

    def test_state_hash_mismatch_across_arms_or_blocks_fails_closed(self) -> None:
        roles = _roles()
        stream = observer.StreamingRelationalObserver(roles=roles, config=_config())
        _populate(
            stream,
            roles=roles,
            mismatched_state=("appearance_0", "action", "high", 24),
        )
        with self.assertRaisesRegex(
            observer.RelationalObserverError,
            "do not share one sealed noisy state",
        ):
            stream.finalize()

    def test_public_receipt_contains_only_relative_sequences_not_raw_or_absolute_payloads(self) -> None:
        roles = _roles()
        stream = observer.StreamingRelationalObserver(roles=roles, config=_config())
        _populate(stream, roles=roles)
        result = stream.finalize()

        published = result["published_representation"]
        for flag in (
            "visual_qk_role_slots",
            "raw_q",
            "raw_k",
            "raw_h",
            "raw_v",
            "dense_role_responsibilities",
            "absolute_coordinates",
            "absolute_anchor_geometry",
        ):
            self.assertIs(published[flag], False)
        self.assertIs(published["relative_role_pair_sequences_only"], True)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in _walk(result)))
        encoded = json.dumps(result, sort_keys=True, allow_nan=False)
        for forbidden_payload_name in (
            '"queries"',
            '"keys"',
            '"responsibilities"',
            '"centroids"',
            '"absolute_x"',
            '"absolute_y"',
        ):
            self.assertNotIn(forbidden_payload_name, encoded)
        for appearance in result["appearance_packets"]:
            for edge in appearance["edges"]:
                for phase in edge["phase_rows"]:
                    self.assertEqual(
                        set(phase["relative_features"]),
                        set(observer.FEATURE_NAMES),
                    )
        unsigned = dict(result)
        digest = unsigned.pop("representation_digest")
        self.assertEqual(digest, observer.object_sha256(unsigned))

    def test_broken_reverse_control_is_rejected(self) -> None:
        roles = _roles()
        stream = observer.StreamingRelationalObserver(roles=roles, config=_config())
        _populate(stream, roles=roles, broken_reverse=True)
        result = stream.finalize()

        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["summary"]["all_control_gates_passed"])
        for appearance in result["appearance_packets"]:
            self.assertFalse(
                appearance["edges"][0]["control_gates"]["reverse_retimes_order"]
            )

    def test_broken_static_control_is_rejected(self) -> None:
        roles = _roles()
        stream = observer.StreamingRelationalObserver(roles=roles, config=_config())
        _populate(stream, roles=roles, broken_static=True)
        result = stream.finalize()

        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["summary"]["all_control_gates_passed"])
        for appearance in result["appearance_packets"]:
            self.assertFalse(
                appearance["edges"][0]["control_gates"]["static_lacks_transition"]
            )

    def test_broken_multiappearance_transfer_is_rejected_even_when_controls_pass(self) -> None:
        roles = _roles()
        stream = observer.StreamingRelationalObserver(roles=roles, config=_config())
        _populate(stream, roles=roles, scaled_appearance="appearance_2")
        result = stream.finalize()

        self.assertEqual(result["status"], "REJECTED")
        self.assertTrue(result["summary"]["all_control_gates_passed"])
        self.assertFalse(
            result["summary"]["all_critical_edges_consistent_across_appearances"]
        )
        self.assertTrue(
            any(not row["passed"] for row in result["multiappearance_consensus"])
        )


if __name__ == "__main__":
    unittest.main()
