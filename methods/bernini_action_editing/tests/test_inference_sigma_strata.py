from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import inference_sigma_strata as strata  # noqa: E402


class _Device:
    type = "cpu"

    def __str__(self) -> str:
        return self.type


class _FakeTensor:
    def __init__(self, values, *, dtype: str, device_type: str = "cpu") -> None:
        self._values = list(values)
        self.dtype = dtype
        self.device = _Device()
        self.device.type = device_type
        self.ndim = 1

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeScalarTensor:
    def __init__(self, value, *, dtype="torch.float32", device_type="cpu") -> None:
        self._value = value
        self.dtype = dtype
        self.device = _Device()
        self.device.type = device_type

    def detach(self):
        return self

    def numel(self):
        return 1

    def cpu(self):
        return self

    def item(self):
        return self._value


def _config(**overrides):
    values = {
        "_class_name": "UniPCMultistepScheduler",
        "num_train_timesteps": 1000,
        "flow_shift": 5.0,
        "prediction_type": "flow_prediction",
        "predict_x0": True,
        "use_flow_sigmas": True,
        "thresholding": False,
        "solver_order": 2,
        "solver_type": "bh2",
        "final_sigmas_type": "zero",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Scheduler:
    def __init__(self, *, config=None, timesteps=None, sigmas=None) -> None:
        self.config = config or _config()
        self._timesteps = list(timesteps or strata.PINNED_TIMESTEPS)
        self._sigmas = list(sigmas or (*strata.PINNED_POSITIVE_SIGMAS, 0.0))
        self.calls = []

    def set_timesteps(self, steps: int) -> None:
        self.calls.append(steps)
        self.timesteps = _FakeTensor(self._timesteps, dtype="torch.int64")
        self.sigmas = _FakeTensor(self._sigmas, dtype="torch.float32")


class ExactScheduleTests(unittest.TestCase):
    def test_captured_schedule_and_hash_are_exact(self) -> None:
        self.assertEqual(len(strata.PINNED_TIMESTEPS), 40)
        self.assertEqual(len(strata.PINNED_POSITIVE_SIGMAS), 40)
        self.assertEqual(strata.PINNED_TIMESTEPS[-1], 117)
        self.assertEqual(
            strata.PINNED_POSITIVE_SIGMAS[-1], 0.11765105277299881
        )
        self.assertNotEqual(strata.PINNED_POSITIVE_SIGMAS[-1], 5.0 / 44.0)
        self.assertTrue(
            all(a > b for a, b in zip(
                strata.PINNED_POSITIVE_SIGMAS,
                strata.PINNED_POSITIVE_SIGMAS[1:],
            ))
        )
        payload = strata._schedule_digest_payload()
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(digest, strata.SCHEDULE_SHA256)
        self.assertEqual(
            digest,
            "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03",
        )

    def test_selector_is_absolute_step_cycle_in_official_order(self) -> None:
        first = strata.select_sigma_stratum(0)
        last = strata.select_sigma_stratum(39)
        wrapped = strata.select_sigma_stratum(40)
        self.assertEqual((first.cycle_index, first.schedule_index), (0, 0))
        self.assertEqual((last.cycle_index, last.schedule_index), (0, 39))
        self.assertEqual((wrapped.cycle_index, wrapped.schedule_index), (1, 0))
        self.assertEqual(first.timestep, 999)
        self.assertEqual(last.timestep, 117)
        self.assertEqual(first.sigma_float32_be_hex, "3f7fffef")
        self.assertEqual(last.sigma_float32_be_hex, "3df0f309")
        self.assertEqual(
            {strata.select_sigma_stratum(step).schedule_index for step in range(40)},
            set(range(40)),
        )

    def test_invalid_step_fails_closed(self) -> None:
        for value in (-1, True, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises(
                strata.InferenceSigmaStrataError
            ):
                strata.select_sigma_stratum(value)

    def test_pair_certificate_detects_timestep_and_sigma_drift(self) -> None:
        selected = strata.select_sigma_stratum(7)
        strata.assert_selected_timestep_sigma(
            timestep=float(selected.timestep), sigma=selected.sigma, selected=selected
        )
        strata.assert_selected_timestep_sigma(
            timestep=_FakeScalarTensor(selected.timestep, dtype="torch.int64"),
            sigma=_FakeScalarTensor(selected.sigma),
            selected=selected,
        )
        with self.assertRaisesRegex(
            strata.InferenceSigmaStrataError, "timestep differs"
        ):
            strata.assert_selected_timestep_sigma(
                timestep=selected.timestep - 1,
                sigma=selected.sigma,
                selected=selected,
            )
        with self.assertRaisesRegex(
            strata.InferenceSigmaStrataError, "sigma differs"
        ):
            strata.assert_selected_timestep_sigma(
                timestep=selected.timestep,
                sigma=selected.sigma + 1.0e-5,
                selected=selected,
            )
        for overrides in (
            {"dtype": "torch.float64"},
            {"device_type": "cuda"},
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                strata.InferenceSigmaStrataError, "torch.float32 on cpu"
            ):
                strata.assert_selected_timestep_sigma(
                    timestep=selected.timestep,
                    sigma=_FakeScalarTensor(selected.sigma, **overrides),
                    selected=selected,
                )
        for overrides in (
            {"dtype": "torch.bfloat16"},
            {"device_type": "cuda", "dtype": "torch.int64"},
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                strata.InferenceSigmaStrataError, "torch.int64 on cpu"
            ):
                strata.assert_selected_timestep_sigma(
                    timestep=_FakeScalarTensor(selected.timestep, **overrides),
                    sigma=selected.sigma,
                    selected=selected,
                )


class RuntimeAuditTests(unittest.TestCase):
    def test_exact_runtime_scheduler_passes_and_is_initialized_once(self) -> None:
        scheduler = _Scheduler()
        receipt = strata.audit_runtime_unipc_schedule(scheduler)
        self.assertEqual(scheduler.calls, [40])
        self.assertEqual(receipt["schedule_sha256"], strata.SCHEDULE_SHA256)
        self.assertEqual(receipt["timesteps"], list(strata.PINNED_TIMESTEPS))
        self.assertEqual(
            receipt["positive_sigmas_float32_be_hex"],
            list(strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX),
        )
        self.assertEqual(receipt["terminal_sigma_float32_be_hex"], "00000000")

    def test_preinitialized_scheduler_can_be_audited_without_reset(self) -> None:
        scheduler = _Scheduler()
        scheduler.set_timesteps(40)
        scheduler.calls.clear()
        strata.audit_runtime_unipc_schedule(scheduler, initialize=False)
        self.assertEqual(scheduler.calls, [])

    def test_runtime_config_drift_fails_closed(self) -> None:
        changes = {
            "flow_shift": 3.0,
            "prediction_type": "epsilon",
            "use_flow_sigmas": False,
            "final_sigmas_type": "sigma_min",
            "num_train_timesteps": 999,
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                scheduler = _Scheduler(config=_config(**{name: value}))
                with self.assertRaisesRegex(
                    strata.InferenceSigmaStrataError, f"config {name} differs"
                ):
                    strata.audit_runtime_unipc_schedule(scheduler)

    def test_one_bit_sigma_or_timestep_drift_fails_closed(self) -> None:
        sigmas = list(strata.PINNED_POSITIVE_SIGMAS) + [0.0]
        # Flip the least-significant bit of the first float32 value.
        sigmas[0] = strata._float_from_float32_hex("3f7fffee")
        with self.assertRaisesRegex(
            strata.InferenceSigmaStrataError, "sigma differs at schedule index 0"
        ):
            strata.audit_runtime_unipc_schedule(_Scheduler(sigmas=sigmas))
        timesteps = list(strata.PINNED_TIMESTEPS)
        timesteps[17] -= 1
        with self.assertRaisesRegex(
            strata.InferenceSigmaStrataError, "timestep differs at schedule index 17"
        ):
            strata.audit_runtime_unipc_schedule(_Scheduler(timesteps=timesteps))

    def test_terminal_and_tensor_representation_are_part_of_contract(self) -> None:
        sigmas = list(strata.PINNED_POSITIVE_SIGMAS) + [1.0e-8]
        with self.assertRaisesRegex(
            strata.InferenceSigmaStrataError, "sigma differs at schedule index 40"
        ):
            strata.audit_runtime_unipc_schedule(_Scheduler(sigmas=sigmas))

        scheduler = _Scheduler()
        scheduler.set_timesteps(40)
        scheduler.sigmas.dtype = "torch.bfloat16"
        with self.assertRaisesRegex(
            strata.InferenceSigmaStrataError, "torch.float32 on cpu"
        ):
            strata.audit_runtime_unipc_schedule(scheduler, initialize=False)

        scheduler = _Scheduler()
        scheduler.set_timesteps(40)
        scheduler.timesteps.device.type = "cuda"
        with self.assertRaisesRegex(
            strata.InferenceSigmaStrataError, "torch.int64 on cpu"
        ):
            strata.audit_runtime_unipc_schedule(scheduler, initialize=False)


class HistogramReceiptTests(unittest.TestCase):
    def test_64_updates_have_one_full_cycle_and_24_second_visits(self) -> None:
        counts = strata.histogram_for_optimizer_range(stop_step=64)
        self.assertEqual(counts[:24], (2,) * 24)
        self.assertEqual(counts[24:], (1,) * 16)
        self.assertEqual(sum(counts), 64)
        receipt = strata.build_sigma_strata_receipt(
            completed_optimizer_steps=64
        )
        self.assertEqual(receipt["complete_cycles"], 1)
        self.assertEqual(receipt["partial_cycle_steps"], 24)
        self.assertEqual(receipt["histogram_by_schedule_index"], list(counts))
        self.assertEqual(receipt["schedule"]["schedule_sha256"], strata.SCHEDULE_SHA256)
        self.assertEqual(len(receipt["receipt_digest"]), 64)

    def test_resume_uses_absolute_optimizer_step_without_cursor_state(self) -> None:
        total = strata.histogram_for_optimizer_range(stop_step=93)
        before = strata.histogram_for_optimizer_range(stop_step=64)
        resumed = strata.histogram_for_optimizer_range(start_step=64, stop_step=93)
        self.assertEqual(total, tuple(a + b for a, b in zip(before, resumed)))
        self.assertEqual(strata.select_sigma_stratum(64).schedule_index, 24)

    def test_receipt_is_deterministic_and_self_digest_excludes_itself(self) -> None:
        one = strata.build_sigma_strata_receipt(completed_optimizer_steps=40)
        two = strata.build_sigma_strata_receipt(completed_optimizer_steps=40)
        self.assertEqual(one, two)
        candidate = copy.deepcopy(one)
        declared = candidate.pop("receipt_digest")
        self.assertEqual(
            declared,
            hashlib.sha256(strata._canonical_json_bytes(candidate)).hexdigest(),
        )

    def test_invalid_histogram_range_fails_closed(self) -> None:
        for kwargs in (
            {"start_step": -1, "stop_step": 0},
            {"start_step": 2, "stop_step": 1},
            {"start_step": True, "stop_step": 2},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                strata.InferenceSigmaStrataError
            ):
                strata.histogram_for_optimizer_range(**kwargs)


if __name__ == "__main__":
    unittest.main()
