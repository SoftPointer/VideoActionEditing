#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_intermediate_action_anchor_v1 as subject


class _Block(nn.Module):
    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((width,), index * 1.0e-3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.bias.reshape(1, 1, -1).to(value.dtype)


class _Transformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width, index) for index in range(30)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


def _config(*, steps: tuple[int, ...] = (8,)) -> subject.AnchorConfig:
    return subject.AnchorConfig(
        patch_height=4,
        patch_width=5,
        hidden_size=8,
        object_slots=3,
        capture_steps=steps,
        min_retained_fraction=1.0e-7,
    )


def _teacher_grids(config: subject.AnchorConfig) -> tuple[torch.Tensor, ...]:
    shape = (
        1,
        config.phases,
        config.patch_height,
        config.patch_width,
        config.hidden_size,
    )
    geometry_noop = torch.zeros(shape, dtype=torch.float32)
    semantic_noop = torch.zeros(shape, dtype=torch.float32)
    geometry_action = geometry_noop.clone()
    semantic_action = semantic_noop.clone()
    # Two independently moving regions plus a static appearance term.  Static
    # content must be removed by the representation.
    appearance = torch.linspace(-0.2, 0.2, config.hidden_size).reshape(1, 1, 1, 1, -1)
    geometry_action += appearance
    semantic_action += appearance * 1.5
    for phase in range(config.phases):
        y1, x1 = 1, min(config.patch_width - 1, phase // 5)
        y2, x2 = 3, max(0, config.patch_width - 1 - phase // 6)
        geometry_action[0, phase, y1, x1, 0:3] += 1.0 + phase / 20.0
        geometry_action[0, phase, y2, x2, 3:6] += 0.8 + phase / 30.0
        semantic_action[0, phase, y1, x1, 0:4] += 0.7 + phase / 15.0
        semantic_action[0, phase, y2, x2, 4:8] -= 0.6 + phase / 17.0
    return geometry_action, geometry_noop, semantic_action, semantic_noop


def _packet(
    config: subject.AnchorConfig, *, action_multiplier: float = 1.0
) -> subject.IntermediateActionAnchorPacket:
    geometry_action, geometry_noop, semantic_action, semantic_noop = _teacher_grids(config)
    geometry_action = geometry_noop + action_multiplier * (
        geometry_action - geometry_noop
    )
    semantic_action = semantic_noop + action_multiplier * (
        semantic_action - semantic_noop
    )
    return subject.build_intermediate_action_anchor(
        geometry_action=geometry_action,
        geometry_noop=geometry_noop,
        semantic_action=semantic_action,
        semantic_noop=semantic_noop,
        config=config,
        step_index=8,
        sigma=0.55,
    )


def _admission(
    packet: subject.IntermediateActionAnchorPacket,
) -> tuple[subject.MultiViewControlAdmission, str]:
    alternate = _packet(packet.config, action_multiplier=1.01)
    source_sha = "1" * 64
    admission = subject.admit_multiview_control(
        primary=subject.SourceViewPacketEvidence(
            "view_a", source_sha, (0, 27, 53, 80), packet
        ),
        alternate=subject.SourceViewPacketEvidence(
            "view_b", source_sha, (0, 20, 60, 80), alternate
        ),
        controls=subject.MultiViewControlEvidence(
            noop_vs_noop_delta_rms=0.0,
            action_delta_rms_reference=float(packet.quality["teacher_delta_rms"]),
            teacher_observer_action_output_bit_exact=True,
            frozen_state_before_sha256="2" * 64,
            frozen_state_after_teacher_sha256="2" * 64,
            target_inputs_absent=True,
        ),
    )
    return admission, source_sha


class IntermediateActionAnchorTests(unittest.TestCase):
    def test_tensor_digest_accepts_stride_zero_size_one_view(self) -> None:
        scalar = torch.tensor(859, dtype=torch.int64)
        expanded = scalar.expand(1)
        base = torch.tensor([859], dtype=torch.int64)
        self.assertEqual(expanded.stride(), (0,))
        self.assertEqual(subject.tensor_sha256(expanded), subject.tensor_sha256(base))

    def test_packet_is_compressed_object_graph_and_phase0_is_zero(self) -> None:
        config = _config()
        packet = _packet(config)
        packet.validate()
        self.assertEqual(tuple(packet.slot_values.shape), (1, 21, 3, 8))
        self.assertEqual(tuple(packet.interaction_graph.shape), (1, 21, 3, 3))
        self.assertTrue(packet.quality["admitted"])
        phase = torch.arange(21).repeat_interleave(config.patch_positions)
        patch = torch.arange(config.patch_positions).repeat(21)
        residual = packet.local_residual(phase, patch, device=torch.device("cpu"))
        self.assertTrue(torch.equal(residual[:, : config.patch_positions], torch.zeros_like(residual[:, : config.patch_positions])))
        self.assertFalse(torch.equal(residual[:, config.patch_positions :], torch.zeros_like(residual[:, config.patch_positions :])))
        receipt = packet.receipt()
        self.assertFalse(receipt["representation"]["raw_teacher_hidden_retained"])
        self.assertFalse(receipt["representation"]["teacher_latent_or_rgb_retained"])

    def test_local_layout_sp4_assembles_target_exactly_once(self) -> None:
        config = _config()
        condition_tokens = 25 * config.patch_positions
        layouts = [
            subject.LocalTokenLayout.build(
                condition_tokens=condition_tokens,
                patch_height=config.patch_height,
                patch_width=config.patch_width,
                sp_rank=rank,
                sp_size=4,
            )
            for rank in range(4)
        ]
        full = torch.arange(
            config.phases * config.patch_positions * config.hidden_size,
            dtype=torch.float32,
        ).reshape(1, config.phases * config.patch_positions, config.hidden_size)
        shards = []
        for layout in layouts:
            flat = layout.target_phase_indices * config.patch_positions + layout.target_patch_indices
            shards.append(full.index_select(1, flat))
        assembled = subject.assemble_sp_target_grid(shards, layouts)
        self.assertTrue(torch.equal(assembled.reshape_as(full), full))

    def test_injection_preserves_prefix_padding_and_phase0(self) -> None:
        config = _config()
        geometry_action, geometry_noop, semantic_action, semantic_noop = _teacher_grids(config)
        packet = _packet(config)
        admission, source_sha = _admission(packet)
        condition_tokens = 25 * config.patch_positions
        layout = subject.LocalTokenLayout.build(
            condition_tokens=condition_tokens,
            patch_height=config.patch_height,
            patch_width=config.patch_width,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(91)
        native = torch.randn((1, layout.local_length, config.hidden_size), generator=generator)
        result, audit = subject.inject_packet_into_local_hidden(
            native,
            packet=packet,
            admission=admission,
            source_video_sha256=source_sha,
            layout=layout,
            sigma=0.55,
            scale=0.06,
        )
        target = layout.local_target_indices
        protected = torch.ones(layout.local_length, dtype=torch.bool)
        protected[target] = False
        phase0 = target[layout.target_phase_indices == 0]
        self.assertTrue(subject.bits_equal(result[:, protected], native[:, protected]))
        self.assertTrue(subject.bits_equal(result[:, phase0], native[:, phase0]))
        self.assertFalse(subject.bits_equal(result, native))
        self.assertTrue(audit.protected_rows_bit_exact)
        self.assertLessEqual(
            audit.applied_delta_rms,
            config.max_injection_to_hidden_rms * audit.native_hidden_rms + 1.0e-6,
        )

    def test_scale_zero_is_same_object_hard_bypass(self) -> None:
        config = _config()
        packet = _packet(config)
        admission, source_sha = _admission(packet)
        layout = subject.LocalTokenLayout.build(
            condition_tokens=10,
            patch_height=config.patch_height,
            patch_width=config.patch_width,
        )
        native = torch.randn(1, layout.local_length, config.hidden_size)
        result, audit = subject.inject_packet_into_local_hidden(
            native,
            packet=packet,
            admission=admission,
            source_video_sha256=source_sha,
            layout=layout,
            sigma=0.55,
            scale=0.0,
        )
        self.assertIs(result, native)
        self.assertTrue(audit.hard_bypass)

    def test_hook_controller_captures_and_injects_only_selected_rows(self) -> None:
        config = _config()
        transformer = _Transformer(config.hidden_size).eval().requires_grad_(False)
        layout = subject.LocalTokenLayout.build(
            condition_tokens=7,
            patch_height=config.patch_height,
            patch_width=config.patch_width,
        )
        controller = subject.IntermediateAnchorHookController(transformer, config)
        controller.install()
        noop_input = torch.zeros(1, layout.local_length, config.hidden_size)
        action_input = noop_input.clone()
        for phase, patch in ((4, 6), (9, 12), (15, 18)):
            global_target = phase * config.patch_positions + patch
            local = layout.condition_tokens + global_target
            action_input[:, local, phase % config.hidden_size] += 1.0
        with controller.invoke(
            subject.HookInvocation(
                mode="capture_action",
                step_index=8,
                sigma=0.55,
                layout=layout,
            )
        ):
            transformer(action_input)
        action_capture = controller.pop_captures()
        with controller.invoke(
            subject.HookInvocation(
                mode="capture_noop",
                step_index=8,
                sigma=0.55,
                layout=layout,
            )
        ):
            transformer(noop_input)
        noop_capture = controller.pop_captures()
        packet = subject.build_intermediate_action_anchor(
            geometry_action=subject.assemble_sp_target_grid(
                [action_capture[config.geometry_block]], [layout]
            ),
            geometry_noop=subject.assemble_sp_target_grid(
                [noop_capture[config.geometry_block]], [layout]
            ),
            semantic_action=subject.assemble_sp_target_grid(
                [action_capture[config.semantic_block]], [layout]
            ),
            semantic_noop=subject.assemble_sp_target_grid(
                [noop_capture[config.semantic_block]], [layout]
            ),
            config=config,
            step_index=8,
            sigma=0.55,
        )
        alternate_packet = subject.build_intermediate_action_anchor(
            geometry_action=1.01
            * subject.assemble_sp_target_grid(
                [action_capture[config.geometry_block]
                 - noop_capture[config.geometry_block]],
                [layout],
            ),
            geometry_noop=torch.zeros(
                (1, config.phases, config.patch_height, config.patch_width, config.hidden_size)
            ),
            semantic_action=1.01
            * subject.assemble_sp_target_grid(
                [action_capture[config.semantic_block]
                 - noop_capture[config.semantic_block]],
                [layout],
            ),
            semantic_noop=torch.zeros(
                (1, config.phases, config.patch_height, config.patch_width, config.hidden_size)
            ),
            config=config,
            step_index=8,
            sigma=0.55,
        )
        source_sha = "3" * 64
        admission = subject.admit_multiview_control(
            primary=subject.SourceViewPacketEvidence(
                "hook_view_a", source_sha, (0, 27, 53, 80), packet
            ),
            alternate=subject.SourceViewPacketEvidence(
                "hook_view_b", source_sha, (0, 20, 60, 80), alternate_packet
            ),
            controls=subject.MultiViewControlEvidence(
                noop_vs_noop_delta_rms=0.0,
                action_delta_rms_reference=float(packet.quality["teacher_delta_rms"]),
                teacher_observer_action_output_bit_exact=True,
                frozen_state_before_sha256="4" * 64,
                frozen_state_after_teacher_sha256="4" * 64,
                target_inputs_absent=True,
            ),
        )
        student = torch.randn_like(action_input)
        with controller.invoke(
            subject.HookInvocation(
                mode="inject_student",
                step_index=8,
                sigma=0.55,
                layout=layout,
                scale=0.06,
                packet=packet,
                admission=admission,
                source_video_sha256=source_sha,
            )
        ):
            output = transformer(student)
        audits = controller.pop_audits()
        self.assertEqual(len(audits), 1)
        self.assertTrue(audits[0].protected_rows_bit_exact)
        controller.remove()
        self.assertFalse(transformer.blocks[15]._forward_hooks)
        self.assertFalse(transformer.blocks[22]._forward_hooks)
        self.assertEqual(output.shape, student.shape)

    def test_target_isolation_freeze_and_p0_replay_contracts(self) -> None:
        with self.assertRaises(subject.IntermediateActionAnchorError):
            subject.assert_target_isolation_payload(
                {"source_video": "/source.mp4", "target_video": "/target.mp4"}
            )
        subject.assert_target_isolation_payload(
            {"source_video": "/source.mp4", "action_prompt": "move the cup"}
        )
        transformer = _Transformer(8).eval().requires_grad_(False)
        before = subject.frozen_module_certificate(transformer)
        after = subject.frozen_module_certificate(transformer)
        self.assertEqual(before["digest"], after["digest"])
        value = torch.randn(1, 2, 3)
        replay = subject.assert_p0_exact_replay(value, value.clone())
        self.assertTrue(replay["bit_exact"])
        changed = value.clone()
        changed.reshape(-1)[0] += 1.0e-3
        with self.assertRaises(subject.IntermediateActionAnchorError):
            subject.assert_p0_exact_replay(value, changed)

    def test_canonical_noop_and_sigma_band(self) -> None:
        subject.validate_canonical_noop(
            subject.CANONICAL_NOOP_INSTRUCTION,
            subject.CANONICAL_NOOP_SHA256,
        )
        config = _config()
        self.assertEqual(subject.smooth_bandpass_gate(0.0, config), 0.0)
        self.assertEqual(subject.smooth_bandpass_gate(0.55, config), 1.0)
        self.assertEqual(subject.smooth_bandpass_gate(1.0, config), 0.0)

    def test_multiview_admission_rejects_identical_packet_and_noop_leak(self) -> None:
        packet = _packet(_config())
        source_sha = "5" * 64
        primary = subject.SourceViewPacketEvidence(
            "view_a", source_sha, (0, 27, 53, 80), packet
        )
        identical = subject.SourceViewPacketEvidence(
            "view_b", source_sha, (0, 20, 60, 80), packet
        )
        controls = subject.MultiViewControlEvidence(
            noop_vs_noop_delta_rms=0.0,
            action_delta_rms_reference=float(packet.quality["teacher_delta_rms"]),
            teacher_observer_action_output_bit_exact=True,
            frozen_state_before_sha256="6" * 64,
            frozen_state_after_teacher_sha256="6" * 64,
            target_inputs_absent=True,
        )
        with self.assertRaises(subject.IntermediateActionAnchorError):
            subject.admit_multiview_control(
                primary=primary, alternate=identical, controls=controls
            )
        alternate = subject.SourceViewPacketEvidence(
            "view_b", source_sha, (0, 20, 60, 80), _packet(_config(), action_multiplier=1.01)
        )
        leaky = subject.MultiViewControlEvidence(
            noop_vs_noop_delta_rms=float(packet.quality["teacher_delta_rms"]),
            action_delta_rms_reference=float(packet.quality["teacher_delta_rms"]),
            teacher_observer_action_output_bit_exact=True,
            frozen_state_before_sha256="6" * 64,
            frozen_state_after_teacher_sha256="6" * 64,
            target_inputs_absent=True,
        )
        with self.assertRaises(subject.IntermediateActionAnchorError):
            subject.admit_multiview_control(
                primary=primary, alternate=alternate, controls=leaky
            )


if __name__ == "__main__":
    unittest.main()
