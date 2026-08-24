from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_self_identity_orbit_v4 as v4
import source_self_native_ref_contrastive_v3 as v3


def _member(name: str, index: int, base: torch.Tensor) -> v4.IdentityOrbitMember:
    channel_offset = torch.linspace(-0.4, 0.4, 16).reshape(1, 16, 1, 1, 1)
    if index == 0:
        video = base.clone()
    elif index == 1:
        video = base * 1.1 + channel_offset
    else:
        video = base * 0.9 - channel_offset
    refs = tuple(
        (
            video[:, :, latent_index : latent_index + 1]
            + float(ref_position) * 0.001
        ).clone()
        for ref_position, latent_index in enumerate((0, 7, 13, 20))
    )
    return v4.IdentityOrbitMember(name, video.contiguous(), refs)  # type: ignore[arg-type]


def _orbit() -> v4.IdentityOrbit:
    generator = torch.Generator(device="cpu").manual_seed(17)
    base = torch.randn((1, 16, 21, 2, 2), generator=generator)
    return v4.IdentityOrbit(
        tuple(
            _member(name, index, base)
            for index, name in enumerate(v4.ORBIT_MEMBER_NAMES)
        ),  # type: ignore[arg-type]
        same_motion_attested=True,
        same_camera_attested=True,
        same_scene_attested=True,
        appearance_only_counterfactual_attested=True,
        independently_encoded_rgb_refs_attested=True,
    )


class FakeNativeTransformer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.source_ids: list[float] = []

    def patch_vae_latent(
        self, value: torch.Tensor, *, source_id: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.source_ids.append(float(source_id))
        tokens = int(value.shape[2] * value.shape[3] * value.shape[4])
        latent = torch.full((1, tokens, 4), float(source_id))
        rotary = torch.full((1, 2, tokens, 3), float(source_id))
        return latent, rotary


def test_orbit_builds_all_exact_positive_cartesian_cells() -> None:
    orbit = _orbit()
    cells = v4.build_identity_orbit_cells(orbit)
    assert len(cells) == 27
    assert len({cell.key for cell in cells}) == 27
    for cell in cells:
        expected_donor = v4.apply_temporal_transform(
            orbit.members[cell.key.donor_index].video_latent, cell.key.transform
        )
        expected_target = v4.apply_temporal_transform(
            orbit.members[cell.key.target_identity_index].video_latent,
            cell.key.transform,
        )
        assert torch.equal(cell.donor, expected_donor)
        assert torch.equal(cell.target, expected_target)
        assert cell.references is orbit.members[
            cell.key.target_identity_index
        ].image_references
    receipt = orbit.receipt()
    assert receipt["cartesian_donor_ref_target_pairs"] == 9
    assert receipt["reference_count_per_member"] == 4
    assert receipt["reference_rgb_indices"] == [0, 27, 53, 80]
    assert receipt["native_rv2v4_reference_contract_digest"] == (
        v3.native_rv2v4_reference_contract()["digest"]
    )
    assert receipt["wrong_cross_scene_in_training_objective"] is False
    assert receipt["native_schedule_digest"] == v3.native_unipc40_schedule_receipt()[
        "digest"
    ]


def test_temporal_registry_is_deterministic_and_target_exact() -> None:
    phases = torch.arange(21, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
    phases = phases.expand(1, 16, 21, 2, 2).contiguous()
    assert torch.equal(v4.apply_temporal_transform(phases, "identity"), phases)
    reverse = v4.apply_temporal_transform(phases, "reverse")
    assert reverse[0, 0, :, 0, 0].tolist() == list(reversed(range(21)))
    warp = v4.apply_temporal_transform(phases, "monotonic_slow_fast")
    assert warp[0, 0, :, 0, 0].tolist() == [
        float(index) for index in v4.TEMPORAL_INDEX_MAPS["monotonic_slow_fast"]
    ]
    assert v4.temporal_transform_receipt()["monotonic_warp_is_nondecreasing"] is True
    assert (
        v4.temporal_transform_receipt()["digest"]
        == v4.PINNED_TEMPORAL_TRANSFORM_DIGEST
    )


def test_source_carrier_is_fixed_whitened_orthogonal_and_norm_matched() -> None:
    generator = torch.Generator(device="cpu").manual_seed(29)
    source = torch.randn((1, 16, 21, 2, 2), generator=generator)
    epsilon = torch.randn((1, 16, 21, 2, 2), generator=generator)
    first = v4.build_source_carrier(source, epsilon, seed=123)
    second = v4.build_source_carrier(source, epsilon, seed=123)
    assert first.temporal_permutation == second.temporal_permutation
    assert first.temporal_permutation == v4.carrier_temporal_permutation(seed=123)
    assert first.temporal_permutation != tuple(range(21))
    assert torch.equal(first.value, second.value)
    inner = float((first.value.double() * epsilon.double()).sum().abs())
    denominator = float(first.value.double().norm() * epsilon.double().norm())
    assert inner / denominator < 5.0e-5
    assert float(first.value.double().norm()) == pytest.approx(
        float(epsilon.double().norm()), rel=5.0e-5
    )


def test_rho_zero_is_bit_exact_gaussian_and_native_flow() -> None:
    orbit = _orbit()
    cell = v4.build_identity_orbit_cells(orbit)[0]
    generator = torch.Generator(device="cpu").manual_seed(31)
    epsilon = torch.randn(cell.target.shape, generator=generator)
    schedule = v4.SourceRichRhoSchedule(max_rho=0.0)
    states = v4.build_orbit_cell_states(
        cell, epsilon, indices=(0, 20, 39), rho_schedule=schedule
    )
    assert states.rhos.tolist() == [0.0, 0.0, 0.0]
    assert all(torch.equal(states.noise_base[index], epsilon) for index in range(3))
    torch.testing.assert_close(
        states.target_velocity,
        (epsilon - cell.target).unsqueeze(0).expand_as(states.target_velocity),
    )
    inference = v4.build_inference_source_rich_noise(
        cell.target,
        epsilon,
        schedule_index=0,
        rho_schedule=schedule,
    )
    assert torch.equal(inference.value, epsilon)
    assert inference.receipt["rho_zero_strict_gaussian_verified"] is True
    assert inference.receipt["rho_positive_non_gaussian_declared"] is False


def test_registered_rho_path_has_correct_derivative_and_train_infer_match() -> None:
    orbit = _orbit()
    cell = v4.build_identity_orbit_cells(orbit)[0]
    generator = torch.Generator(device="cpu").manual_seed(37)
    epsilon = torch.randn(cell.target.shape, generator=generator)
    schedule = v4.SourceRichRhoSchedule()
    indices = (0, 32, 39)
    states = v4.build_orbit_cell_states(
        cell, epsilon, indices=indices, rho_schedule=schedule, carrier_seed=99
    )
    assert float(states.rhos[0]) == pytest.approx(schedule.max_rho)
    assert float(states.rho_derivatives[0]) == 0.0
    assert 0.0 < float(states.rhos[1]) < schedule.max_rho
    assert float(states.rho_derivatives[1]) > 0.0
    assert float(states.rhos[2]) == 0.0
    assert torch.equal(states.noise_base[2], epsilon)

    for position, schedule_index in enumerate(indices):
        inference = v4.build_inference_source_rich_noise(
            cell.target,
            epsilon,
            schedule_index=schedule_index,
            rho_schedule=schedule,
            carrier_seed=99,
        )
        torch.testing.assert_close(inference.value, states.noise_base[position])
        torch.testing.assert_close(
            inference.derivative, states.noise_derivative[position]
        )

    sigma = float(v3.NATIVE_UNIPC40_SIGMAS[32])
    delta = 1.0e-5
    rho_plus = schedule.rho_and_derivative(sigma + delta)[0]
    rho_minus = schedule.rho_and_derivative(sigma - delta)[0]
    numerical = (rho_plus - rho_minus) / (2.0 * delta)
    assert schedule.rho_and_derivative(sigma)[1] == pytest.approx(
        numerical, rel=1.0e-5
    )
    naive_velocity = states.noise_base[1] - cell.target
    assert not torch.allclose(states.target_velocity[1], naive_velocity)
    receipt = states.receipt()
    assert receipt["training_inference_schedule_identical"] is True
    assert receipt["rho_positive_is_non_gaussian"] is True
    rho_receipt = schedule.receipt()
    assert rho_receipt["schema_version"] == v4.SCHEMA_VERSION
    assert rho_receipt["distribution_at_rho_zero"] == "strict_original_gaussian_values"
    assert rho_receipt["distribution_at_positive_rho"] == "source_conditioned_non_gaussian"
    assert len(rho_receipt) == len(set(rho_receipt))


def test_orbit_cell_pack_delegates_to_v3_native_vi_plus_i() -> None:
    orbit = _orbit()
    cell = next(
        cell
        for cell in v4.build_identity_orbit_cells(orbit)
        if cell.key == v4.OrbitCellKey(2, 1, "reverse")
    )
    generator = torch.Generator(device="cpu").manual_seed(41)
    epsilon = torch.randn(cell.target.shape, generator=generator)
    states = v4.build_orbit_cell_states(cell, epsilon, indices=(0,))
    transformer = FakeNativeTransformer()
    pack = v4.pack_orbit_cell_at_sigma(
        transformer, cell, states, sigma_position=0
    )
    assert isinstance(pack, v3.NativeRV2VPack)
    assert transformer.source_ids == [
        1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0
    ]
    assert pack.video_image.source_ids == (1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
    assert pack.image.source_ids == (1.0, 2.0, 3.0, 4.0, 0.0)


def test_full_orbit_objective_has_only_defined_targets_and_backpropagates() -> None:
    orbit = _orbit()
    cells = v4.build_identity_orbit_cells(orbit)
    generator = torch.Generator(device="cpu").manual_seed(43)
    epsilon = torch.randn(cells[0].target.shape, generator=generator)
    parameters = torch.nn.Parameter(torch.full((27,), 0.1))
    supervision: dict[v4.OrbitCellKey, v4.SourceRichStates] = {}
    predictions: dict[v4.OrbitCellKey, torch.Tensor] = {}
    for index, cell in enumerate(cells):
        states = v4.build_orbit_cell_states(
            cell,
            epsilon,
            indices=(0, 32, 39),
            carrier_seed=101,
        )
        supervision[cell.key] = states
        predictions[cell.key] = states.target_velocity + parameters[index]
    result = v4.identity_orbit_objective(
        predictions,
        supervision,
        orbit,
        reference_margin=0.001,
    )
    assert result.loss.requires_grad
    assert float(result.reconstruction_loss) == pytest.approx(0.01, rel=1.0e-5)
    assert float(result.donor_invariance_loss) == pytest.approx(0.0, abs=1.0e-10)
    assert float(result.motion_equivariance_loss) == pytest.approx(0.0, abs=1.0e-10)
    result.loss.backward()
    assert parameters.grad is not None
    assert bool(torch.isfinite(parameters.grad).all())
    assert bool((parameters.grad.abs() > 0).all())


def test_registered_microbatch_cycle_has_exact_weighted_coverage() -> None:
    cycle = v4.registered_orbit_microbatch_cycle()
    assert len(cycle) == 36
    assert tuple(step.ordinal for step in cycle) == tuple(range(36))
    assert tuple(step.step_type for step in cycle) == ("A", "C", "B", "C") * 9
    assert sum(step.step_type == "A" for step in cycle) == 9
    assert sum(step.step_type == "B" for step in cycle) == 9
    assert sum(step.step_type == "C" for step in cycle) == 18

    raw: dict[v4.OrbitCellKey, int] = {}
    weighted: dict[v4.OrbitCellKey, float] = {}
    for step in cycle:
        for key, weight in zip(step.keys, step.reconstruction_cell_weights):
            raw[key] = raw.get(key, 0) + 1
            weighted[key] = weighted.get(key, 0.0) + weight
    assert len(raw) == 27
    assert all(
        count == (4 if key.transform == "identity" else 3)
        for key, count in raw.items()
    )
    assert all(value == pytest.approx(1.0) for value in weighted.values())

    receipt = v4.orbit_microbatch_cycle_receipt()
    assert receipt["weighted_reconstruction_each_cell_exactly_once"] is True
    assert receipt["donor_invariance_group_count"] == 9
    assert receipt["reference_selection_group_count"] == 9
    assert receipt["equivariance_pair_count"] == 18
    assert receipt["dynamic_cell_selection_allowed"] is False
    assert receipt["trainer_must_force_gradient_checkpointing_disabled"] is True
    assert receipt["adapter_route_lifetime"] == "forward_only_no_backward_recompute"
    assert v4.validate_microbatch_runtime(
        gradient_checkpointing_enabled=False
    )["accepted"] is True
    with pytest.raises(v4.IdentityOrbitV4Error, match="must disable"):
        v4.validate_microbatch_runtime(gradient_checkpointing_enabled=True)


def test_sum_of_local_cycle_objectives_equals_full_objective() -> None:
    orbit = _orbit()
    cells = v4.build_identity_orbit_cells(orbit)
    cell_by_key = {cell.key: cell for cell in cells}
    generator = torch.Generator(device="cpu").manual_seed(53)
    epsilon = torch.randn(cells[0].target.shape, generator=generator)
    ordered_keys = sorted(cell_by_key)
    parameter = torch.nn.Parameter(
        torch.linspace(0.01, 0.27, len(ordered_keys), dtype=torch.float32)
    )
    supervision: dict[v4.OrbitCellKey, v4.SourceRichStates] = {}
    predictions: dict[v4.OrbitCellKey, torch.Tensor] = {}
    for index, key in enumerate(ordered_keys):
        states = v4.build_orbit_cell_states(
            cell_by_key[key],
            epsilon,
            indices=(0, 32, 39),
            carrier_seed=131,
        )
        supervision[key] = states
        predictions[key] = states.target_velocity + parameter[index]

    full = v4.identity_orbit_objective(
        predictions,
        supervision,
        orbit,
        reference_margin=0.001,
    )
    local_results = []
    for microbatch in v4.registered_orbit_microbatch_cycle():
        local_results.append(
            (
                microbatch,
                v4.identity_orbit_microbatch_objective(
                    microbatch,
                    {key: predictions[key] for key in microbatch.keys},
                    {key: supervision[key] for key in microbatch.keys},
                    orbit,
                    reference_margin=0.001,
                ),
            )
        )
    reconstruction = torch.stack(
        [result.reconstruction_cycle_contribution for _, result in local_results]
    ).sum()
    donor = torch.stack(
        [
            result.factor_cycle_contribution
            for step, result in local_results
            if step.step_type == "A"
        ]
    ).sum()
    reference = torch.stack(
        [
            result.factor_cycle_contribution
            for step, result in local_results
            if step.step_type == "B"
        ]
    ).sum()
    equivariance = torch.stack(
        [
            result.factor_cycle_contribution
            for step, result in local_results
            if step.step_type == "C"
        ]
    ).sum()
    local_loss = torch.stack([result.loss for _, result in local_results]).sum()
    torch.testing.assert_close(reconstruction, full.reconstruction_loss)
    torch.testing.assert_close(donor, full.donor_invariance_loss)
    torch.testing.assert_close(reference, full.reference_selection_loss)
    torch.testing.assert_close(equivariance, full.motion_equivariance_loss)
    torch.testing.assert_close(local_loss, full.loss)


def test_objective_rejects_missing_cartesian_positive() -> None:
    orbit = _orbit()
    cells = v4.build_identity_orbit_cells(orbit)
    generator = torch.Generator(device="cpu").manual_seed(47)
    epsilon = torch.randn(cells[0].target.shape, generator=generator)
    supervision = {
        cell.key: v4.build_orbit_cell_states(cell, epsilon, indices=(0,))
        for cell in cells[:-1]
    }
    predictions = {
        key: states.target_velocity.clone().requires_grad_(True)
        for key, states in supervision.items()
    }
    with pytest.raises(v4.IdentityOrbitV4Error, match="full registered 27-cell"):
        v4.identity_orbit_objective(predictions, supervision, orbit)


def test_cross_scene_wrong_refs_are_heldout_only_without_target_loss() -> None:
    target = torch.zeros((1, 16, 21, 2, 2))
    correct = torch.full_like(target, 0.01)
    wrong = torch.full_like(target, 1.0)
    gate = v4.heldout_wrong_scene_gate(
        correct_prediction_clean=correct,
        wrong_scene_prediction_clean=wrong,
        exact_orbit_target=target,
        maximum_correct_error=0.001,
        minimum_wrong_scene_sensitivity=0.5,
    )
    assert gate["used_by_training_objective"] is False
    assert gate["wrong_scene_target_defined"] is False
    assert gate["wrong_scene_error_term_computed"] is False
    assert gate["accepted"] is True


def test_static_contract_reuses_v3_and_contains_no_negative_inflation_loss() -> None:
    source = Path(v4.__file__).read_text()
    assert "import source_self_native_ref_contrastive_v3 as v3" in source
    assert "v3.build_native_rv2v_pack" in source
    assert "v3.NATIVE_UNIPC40_SIGMAS" in source
    assert "correct/reverse/wrong/off" not in source
    assert "wrong_cross_scene_in_training_objective\": False" in source
    assert "epsilon-clean" in source
