from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_self_native_ref_contrastive_v3 as v3


class FakeNativeTransformer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def patch_vae_latent(
        self, value: torch.Tensor, *, source_id: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = int(value.shape[2] * value.shape[3] * value.shape[4])
        self.calls.append((tokens, float(source_id)))
        # Encode both the source id and original value into the fake patch so
        # the two image axes are observably different in the test.
        payload = torch.full((1, tokens, 4), float(source_id))
        payload[..., 1] = float(value.mean())
        rotary = torch.full((1, 2, tokens, 3), float(source_id))
        return payload, rotary


class FakeWanDiffusion:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def shared_step(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        return kwargs["noisy_latents"] + 7.0  # type: ignore[operator]


def _video(value: float, phases: int) -> torch.Tensor:
    return torch.full((1, 16, phases, 2, 2), value, dtype=torch.float32)


def test_native_pack_reuses_exact_rv2v_vi_plus_i_axes() -> None:
    transformer = FakeNativeTransformer()
    donor = _video(10.0, 21)
    refs = [_video(20.0 + index, 1) for index in range(4)]
    target = _video(30.0, 21)

    pack = v3.build_native_rv2v_pack(
        transformer,
        donor_video=donor,
        image_references=refs,
        noisy_target=target,
    )

    assert [source_id for _, source_id in transformer.calls] == [
        1.0,
        2.0,
        1.0,
        3.0,
        2.0,
        4.0,
        3.0,
        5.0,
        4.0,
        0.0,
    ]
    assert pack.none.source_ids == (0.0,)
    assert pack.video.source_ids == (1.0, 0.0)
    assert pack.image.source_ids == (1.0, 2.0, 3.0, 4.0, 0.0)
    assert pack.video_image.source_ids == (1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
    assert pack.patch_call_roles == v3.PATCH_CALL_ROLES
    assert pack.none.concat_order == v3.BRANCH_CONCAT_ORDER["none"]
    assert pack.video.concat_order == v3.BRANCH_CONCAT_ORDER["V"]
    assert pack.image.concat_order == v3.BRANCH_CONCAT_ORDER["I"]
    assert pack.video_image.concat_order == v3.BRANCH_CONCAT_ORDER["VI"]
    assert pack.video.condition_tokens == 84
    assert pack.image.condition_tokens == 16
    assert pack.video_image.condition_tokens == 100
    assert pack.none.condition_tokens == 0
    assert all(
        bool(branch.target_mask[branch.condition_tokens :].all())
        for branch in (pack.none, pack.video, pack.image, pack.video_image)
    )
    receipt = pack.receipt()
    assert receipt["image_refs_repatched_on_i_axis"] is True
    assert receipt["reference_count"] == 4
    assert receipt["vi_image_source_ids"] == [2.0, 3.0, 4.0, 5.0]
    assert receipt["i_image_source_ids"] == [1.0, 2.0, 3.0, 4.0]
    assert receipt["patch_call_roles"] == list(v3.PATCH_CALL_ROLES)
    assert receipt["latent_concat_dim"] == 1
    assert receipt["rotary_concat_dim"] == 2
    assert receipt["branches"]["VI"]["concat_order"] == [
        "video", "ref0", "ref1", "ref2", "ref3", "target"
    ]
    assert receipt["native_rv2v4_reference_contract_digest"] == (
        v3.native_rv2v4_reference_contract()["digest"]
    )
    contract = v3.native_rv2v4_reference_contract()
    assert contract["digest"] == v3.PINNED_NATIVE_RV2V4_REFERENCE_CONTRACT_DIGEST
    assert contract["patch_call_roles"] == list(v3.PATCH_CALL_ROLES)
    assert contract["branch_concat_order"]["I"] == [
        "ref0", "ref1", "ref2", "ref3", "target"
    ]

    # The first VI image uses source id 2, while that same physical image on
    # the native I axis uses source id 1.  Reusing one RoPE would fail here.
    assert torch.all(pack.video_image.latents[:, 84:88, 0] == 2.0)
    assert torch.all(pack.image.latents[:, :4, 0] == 1.0)


def test_native_pack_rejects_missing_refs_and_manual_shape_drift() -> None:
    transformer = FakeNativeTransformer()
    with pytest.raises(v3.NativeRefContrastiveV3Error, match="exactly four"):
        v3.build_native_rv2v_pack(
            transformer,
            donor_video=_video(1.0, 21),
            image_references=[_video(2.0, 1)],
            noisy_target=_video(3.0, 21),
        )
    bad_ref = _video(2.0, 2)
    with pytest.raises(v3.NativeRefContrastiveV3Error, match="one latent phase"):
        v3.build_native_rv2v_pack(
            transformer,
            donor_video=_video(1.0, 21),
            image_references=[bad_ref, bad_ref, bad_ref, bad_ref],
            noisy_target=_video(3.0, 21),
        )
    with pytest.raises(v3.NativeRefContrastiveV3Error, match="exact81"):
        v3.build_native_rv2v_pack(
            transformer,
            donor_video=_video(1.0, 20),
            image_references=[_video(2.0, 1) for _ in range(4)],
            noisy_target=_video(3.0, 21),
        )


def test_native_target_forward_does_not_repatch_and_selects_suffix() -> None:
    transformer = FakeNativeTransformer()
    pack = v3.build_native_rv2v_pack(
        transformer,
        donor_video=_video(10.0, 21),
        image_references=[_video(20.0 + index, 1) for index in range(4)],
        noisy_target=_video(30.0, 21),
    )
    patch_calls_before = list(transformer.calls)
    diffusion = FakeWanDiffusion()
    text = torch.zeros((1, 5, 8), dtype=torch.float32)
    output = v3.forward_native_target_branch(
        diffusion,
        pack.video_image,
        timestep=torch.tensor([999.0], dtype=torch.float32),
        cond_embeds=text,
    )
    assert transformer.calls == patch_calls_before
    assert output.shape == (1, 84, 4)
    assert torch.all(output == pack.video_image.latents[:, -84:, :] + 7.0)
    assert len(diffusion.calls) == 1
    assert diffusion.calls[0]["batch_vae_seqlen"] == [184]
    assert diffusion.calls[0]["batch_text_seqlen"] == [5]
    assert diffusion.calls[0]["model_id"] == "transformer_1"
    with pytest.raises(v3.NativeRefContrastiveV3Error, match="FP32"):
        v3.forward_native_target_branch(
            diffusion,
            pack.video_image,
            timestep=torch.tensor([999], dtype=torch.int64),
            cond_embeds=text,
        )


def test_exact40_schedule_is_stratified_without_replacement() -> None:
    assert (
        v3.native_unipc40_schedule_receipt()["digest"]
        == v3.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
    )
    first_cycle = [
        index
        for step in range(10)
        for index in v3.schedule_indices_for_step(
            seed=20260808, step=step, samples_per_step=4
        )
    ]
    second_cycle = [
        index
        for step in range(10, 20)
        for index in v3.schedule_indices_for_step(
            seed=20260808, step=step, samples_per_step=4
        )
    ]
    assert sorted(first_cycle) == list(range(40))
    assert sorted(second_cycle) == list(range(40))
    assert first_cycle != second_cycle
    assert v3.schedule_indices_for_step(seed=7, step=3) == v3.schedule_indices_for_step(
        seed=7, step=3
    )
    with pytest.raises(v3.NativeRefContrastiveV3Error, match="divisor"):
        v3.schedule_indices_for_step(seed=7, step=0, samples_per_step=3)


def test_multi_sigma_states_use_exact_native_coordinates_and_flow_equation() -> None:
    clean = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    epsilon = torch.tensor([[5.0, 10.0]], dtype=torch.float32)
    indices = (0, 17, 39)
    states = v3.build_multi_sigma_states(clean, epsilon, indices=indices)
    assert states.noisy.shape == (3, 1, 2)
    assert states.target_velocity.shape == (3, 1, 2)
    assert states.timesteps.tolist() == [999, 871, 117]
    assert states.timesteps.dtype == torch.float32
    expected = torch.stack(
        [
            (1.0 - v3.NATIVE_UNIPC40_SIGMAS[index]) * clean
            + v3.NATIVE_UNIPC40_SIGMAS[index] * epsilon
            for index in indices
        ]
    )
    torch.testing.assert_close(states.noisy, expected)
    torch.testing.assert_close(
        states.target_velocity, (epsilon - clean).unsqueeze(0).expand_as(expected)
    )
    assert states.weights.dtype == torch.float64
    assert float(states.weights.sum()) == pytest.approx(1.0)
    assert states.receipt()["schedule_digest"] == v3.native_unipc40_schedule_receipt()[
        "digest"
    ]


def test_causal_objective_backpropagates_through_all_four_cells() -> None:
    target = torch.zeros((4, 2), dtype=torch.float32)
    predictions = {
        name: torch.full((4, 2), 1.0, requires_grad=True)
        for name in v3.ROLE_CELL_NAMES
    }
    result = v3.role_causal_ranking_objective(
        predictions,
        target,
        sigma_weights=torch.full((4,), 0.25),
        margin=0.25,
    )
    assert all(
        float(value.detach()) == pytest.approx(0.25)
        for value in result.hinge_by_negative.values()
    )
    result.loss.backward()
    assert all(value.grad is not None for value in predictions.values())
    # The correct branch is pulled down; each intervention is pushed up while
    # its hinge is violated.
    assert bool((predictions["correct"].grad > 0).all())
    assert all(
        bool((predictions[name].grad < 0).all())
        for name in v3.NEGATIVE_CELL_NAMES
    )


def test_one_step_recomputes_and_passes_strict_post_update_gates() -> None:
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    target = torch.zeros((4, 1), dtype=torch.float32)

    def forward_cells() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return (
            {
                name: parameter[index].expand(4, 1)
                for index, name in enumerate(v3.ROLE_CELL_NAMES)
            },
            target,
        )

    result = v3.run_causal_update(
        forward_cells=forward_cells,
        optimizer=optimizer,
        trainable_parameters=[parameter],
        sigma_weights=torch.full((4,), 0.25),
        margin=0.25,
    )
    assert result.gradient_norm > 0.0
    assert result.post_errors["correct"] < result.pre_errors["correct"]
    assert all(
        result.post_errors[name] > result.pre_errors[name]
        for name in v3.NEGATIVE_CELL_NAMES
    )
    assert result.gates["evaluation_timing"] == "fresh_forward_after_optimizer_step"
    assert result.gates["accepted"] is True
    assert all(result.gates["margin_gates"].values())
    assert all(result.gates["gap_gain_gates"].values())


def test_post_update_gate_fails_closed_on_one_bad_intervention() -> None:
    gates = v3.post_update_role_gates(
        pre_errors={"correct": 1.0, "reverse": 1.0, "wrong": 1.0, "off": 1.0},
        post_errors={"correct": 0.8, "reverse": 1.2, "wrong": 0.9, "off": 1.3},
        margin=0.25,
    )
    assert gates["margin_gates"] == {"reverse": True, "wrong": False, "off": True}
    assert gates["accepted"] is False


def test_static_contract_forbids_fixed_sigma_and_manual_rope_path() -> None:
    source = Path(v3.__file__).read_text()
    assert "transformer.patch_vae_latent" in source
    assert "none/V/I/VI" in source
    assert "TIMESTEP = 1000" not in source
    assert "SIGMA = 1.0" not in source
    assert "torch.no_grad():\n        post_predictions" in source
