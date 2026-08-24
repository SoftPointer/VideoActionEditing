from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import caper_phase_action_quotient_probe as paq  # noqa: E402


PHASE_RANGES = {
    "start": [0, 2],
    "transition": [2, 4],
    "terminal": [4, 6],
    "hold": [6, 8],
}
CHECKPOINT_SHA256 = "a" * 64
SOURCE_REVISION_SHA256 = "b" * 64
POLICY_SHA256 = "e" * 64
SOURCE_EXPOSURE_REGISTRY_SHA256 = "f" * 64
INTERVENTION_SCALE = 0.5
SIGMA = 0.64


def _phase_tensor(
    phase_vectors: list[torch.Tensor],
    *,
    base: torch.Tensor,
    lexical: torch.Tensor,
) -> torch.Tensor:
    frames = []
    for value in phase_vectors:
        for _ in range(2):
            # Three spatial tokens make the pooling path part of every test.
            spatial = torch.stack((value, value + 0.01, value - 0.01), dim=0)
            frames.append(base[None, :] + lexical[None, :] + spatial)
    return torch.stack(frames, dim=0).to(dtype=torch.float32)


def make_bundle(
    *,
    lexical_only: bool = False,
    shuffled_phase: bool = False,
    common_source_motion: bool = False,
):
    hidden: dict[str, torch.Tensor] = {}
    states: dict[str, torch.Tensor] = {}
    records = []
    cohorts = (
        ("disc-0", "discovery", "identity-d0", "scene-d0", "seed-d0"),
        ("disc-1", "discovery", "identity-d1", "scene-d1", "seed-d1"),
        ("held-0", "admission", "identity-h0", "scene-h0", "seed-h0"),
        ("held-1", "admission", "identity-h1", "scene-h1", "seed-h1"),
    )
    lexical_directions = {
        name: torch.nn.functional.one_hot(torch.tensor(index), num_classes=8).float() * 25.0
        for index, name in enumerate(paq.REQUIRED_VARIANTS)
    }
    for cohort_index, (cohort_id, split, identity, scene, seed) in enumerate(cohorts):
        base = torch.linspace(-2.0, 2.0, 8) + float(cohort_index)
        # Small identity/scene changes remain below the stability gate.
        action_direction = torch.tensor(
            [1.0, 0.18 + 0.01 * cohort_index, 0.10, 0.04, 0.0, 0.0, 0.0, 0.0]
        )
        wrong_direction = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0])
        zero = torch.zeros(8)
        target_scalars = [0.0, 0.40, 1.0, 1.0]
        if shuffled_phase and split == "admission":
            target_scalars = [0.0, 1.0, 0.35, 1.0]
        phase_values = {
            "target": [value * action_direction for value in target_scalars],
            "noop": [zero, zero, zero, zero],
            "reverse": [
                -value * action_direction for value in (0.0, 0.40, 1.0, 1.0)
            ],
            "incomplete": [
                value * action_direction for value in (0.0, 0.40, 0.15, 0.15)
            ],
            "wrong_action": [
                value * wrong_direction for value in (0.0, 0.40, 1.0, 1.0)
            ],
        }
        if common_source_motion:
            source_direction = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            source_path = [value * source_direction for value in (0.0, 0.6, 1.2, 1.8)]
            phase_values = {
                variant: [phase + source for phase, source in zip(values, source_path)]
                for variant, values in phase_values.items()
            }
        state_key = f"state-{cohort_id}"
        generator = torch.Generator().manual_seed(100 + cohort_index)
        states[state_key] = torch.randn(2, 3, 4, generator=generator)
        for variant in paq.REQUIRED_VARIANTS:
            key = f"hidden-{cohort_id}-{variant}"
            if lexical_only:
                values = [zero, zero, zero, zero]
            else:
                values = phase_values[variant]
            hidden[key] = _phase_tensor(
                values,
                base=base,
                lexical=lexical_directions[variant],
            )
            text = f"closed condition for {variant} in {cohort_id}"
            records.append(
                {
                    "observation_id": f"obs-{cohort_id}-{variant}",
                    "cohort_id": cohort_id,
                    "split": split,
                    "variant": variant,
                    "identity_id": identity,
                    "scene_id": scene,
                    "seed_id": seed,
                    "requested_action_id": "sit-down-and-hold",
                    "action_family_id": "postural-transition-and-hold",
                    "action_family_member_id": f"family-member-{cohort_id}",
                    "source_exposure_id": f"source-exposure-{cohort_id}",
                    "condition_text": text,
                    "condition_text_sha256": paq.text_sha256(text),
                    "hook_site": "diff_dec.transformer.blocks.12.attn2.to_out.0:pre",
                    "hook_block_index": 12,
                    "diffusion_step": 7,
                    "sigma": SIGMA,
                    "sigma_float32_be_hex": paq.float32_bits(SIGMA, label="sigma"),
                    "phase_ranges": PHASE_RANGES,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "policy_sha256": POLICY_SHA256,
                    "source_exposure_registry_sha256": SOURCE_EXPOSURE_REGISTRY_SHA256,
                    "hidden_key": key,
                    "hidden_sha256": paq.tensor_sha256(hidden[key]),
                    "noisy_state_key": state_key,
                    "noisy_state_sha256": paq.tensor_sha256(states[state_key]),
                }
            )
    manifest = paq.seal_observation_manifest(
        records,
        probe_id="synthetic-paq-probe",
        checkpoint_sha256=CHECKPOINT_SHA256,
        policy_sha256=POLICY_SHA256,
        source_revision_sha256=SOURCE_REVISION_SHA256,
        source_exposure_registry_sha256=SOURCE_EXPOSURE_REGISTRY_SHA256,
        intervention_scale=INTERVENTION_SCALE,
    )
    return manifest, hidden, states


def reseal_records(records, manifest):
    return paq.seal_observation_manifest(
        records,
        probe_id=manifest["probe_id"],
        checkpoint_sha256=manifest["checkpoint_sha256"],
        policy_sha256=manifest["policy_sha256"],
        source_revision_sha256=manifest["source_revision_sha256"],
        source_exposure_registry_sha256=manifest[
            "source_exposure_registry_sha256"
        ],
        intervention_scale=manifest["intervention_scale"],
    )


def fake_decode_exact81(path: Path):
    return {
        "decoded_contract": paq.DECODED_MEDIA_CONTRACT,
        "decoded_frame_count": 81,
        "decoded_fps_numerator": 25,
        "decoded_fps_denominator": 1,
        "decoded_height": 4,
        "decoded_width": 4,
        "decoded_rgb24_sha256": hashlib.sha256(
            path.read_bytes() + b"-decoded-rgb24"
        ).hexdigest(),
    }


def _write_canonical_receipt(path: Path, receipt):
    value = copy.deepcopy(receipt)
    value.pop("receipt_payload_sha256", None)
    value["receipt_payload_sha256"] = paq.object_sha256(value)
    path.write_bytes(paq._canonical_json_bytes(value) + b"\n")
    return paq.CausalInterventionTrial(
        cohort_id=value["cohort_id"],
        receipt_path=str(path),
        receipt_file_sha256=paq.file_sha256(path),
        receipt_size_bytes=path.stat().st_size,
    )


def good_trials(manifest, candidate_code_sha256, root: Path):
    rows = {
        row["cohort_id"]: row
        for row in manifest["records"]
        if row["split"] == "admission"
    }
    trials = []
    for cohort_id, row in sorted(rows.items()):
        media = {}
        for role in paq.CAUSAL_MEDIA_ROLES:
            path = (root / f"{cohort_id}-{role}.media").resolve()
            path.write_bytes(f"{cohort_id}:{role}:sealed-exact81-fixture".encode("ascii"))
            media[role] = {
                "path": str(path),
                "file_sha256": paq.file_sha256(path),
                "size_bytes": path.stat().st_size,
                **fake_decode_exact81(path),
            }
        receipt = {
            "schema_version": paq.CAUSAL_RECEIPT_SCHEMA_VERSION,
            "receipt_id": f"causal-receipt-{cohort_id}",
            "manifest_sha256": manifest["manifest_sha256"],
            "cohort_id": cohort_id,
            "split": "admission",
            "identity_id": row["identity_id"],
            "scene_id": row["scene_id"],
            "seed_id": row["seed_id"],
            "requested_action_id": row["requested_action_id"],
            "action_family_id": row["action_family_id"],
            "action_family_member_id": row["action_family_member_id"],
            "source_exposure_id": row["source_exposure_id"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "policy_sha256": manifest["policy_sha256"],
            "source_revision_sha256": manifest["source_revision_sha256"],
            "source_exposure_registry_sha256": manifest[
                "source_exposure_registry_sha256"
            ],
            "hook_site": row["hook_site"],
            "hook_block_index": row["hook_block_index"],
            "diffusion_step": row["diffusion_step"],
            "sigma": row["sigma"],
            "sigma_float32_be_hex": row["sigma_float32_be_hex"],
            "phase_ranges": row["phase_ranges"],
            "candidate_code_sha256": candidate_code_sha256,
            "intervention_scale": manifest["intervention_scale"],
            "intervention_scale_bits": manifest["intervention_scale_bits"],
            "decoded_media_contract": paq.DECODED_MEDIA_CONTRACT,
            "decoded_media": media,
            "action_metrics": {
                "baseline_action_score": 0.20,
                "target_action_score": 0.55,
                "reverse_order_score": 0.92,
                "noop_effect": 0.01,
            },
            "preservation_axes": {
                axis: {"baseline_score": 0.90, "target_score": 0.90}
                for axis in paq.PRESERVATION_AXES
            },
            "no_weighted_preservation_aggregate": True,
        }
        receipt_path = (root / f"{cohort_id}.causal-receipt.json").resolve()
        trials.append(_write_canonical_receipt(receipt_path, receipt))
    return tuple(trials)


def rewrite_trial(trial, mutate):
    path = Path(trial.receipt_path)
    receipt = json.loads(path.read_text(encoding="ascii"))
    receipt.pop("receipt_payload_sha256")
    mutate(receipt)
    return _write_canonical_receipt(path, receipt)


class HookAndExtractionTests(unittest.TestCase):
    def test_hook_plan_names_real_to_out_modules_without_vendor_mutation(self) -> None:
        sites = paq.bernini_hook_plan(12)
        self.assertEqual(
            [site.site_id for site in sites],
            [
                "diff_dec.transformer.blocks.12.attn2.to_out.0:pre",
                "diff_dec.transformer.blocks.12.attn2.to_out.0:post",
                "diff_dec.transformer.blocks.12.attn1.to_out.0:pre",
            ],
        )
        with self.assertRaises(paq.PAQProbeError):
            paq.bernini_hook_plan(30)

    def test_start_anchor_keeps_terminal_hold_plateau(self) -> None:
        hidden = torch.tensor([10.0, 10.0, 11.0, 11.0, 13.0, 13.0, 13.0, 13.0])[:, None]
        path = paq.start_anchored_phase_path(
            hidden, paq.PhaseRanges.from_mapping(PHASE_RANGES)
        ).values[:, 0]
        self.assertTrue(torch.equal(path, torch.tensor([0.0, 1.0, 3.0, 3.0])))
        temporal_mean_centered = hidden[:, 0] - hidden[:, 0].mean()
        self.assertNotEqual(float(temporal_mean_centered[6]), 3.0)
        self.assertEqual(paq.ANCHOR_FORMULA, "hidden_t-minus-start_phase_mean_v1")


class ManifestTests(unittest.TestCase):
    def test_manifest_binds_every_hidden_and_same_noisy_state(self) -> None:
        manifest, hidden, states = make_bundle()
        rows = paq.validate_observation_manifest(manifest, hidden, states)
        self.assertEqual(len(rows), 20)
        self.assertEqual({row["split"] for row in rows}, {"discovery", "admission"})

        tampered = dict(manifest)
        tampered["probe_id"] = "tampered"
        with self.assertRaisesRegex(paq.PAQProbeError, "manifest_sha256"):
            paq.validate_observation_manifest(tampered, hidden, states)

        hidden_tampered = dict(hidden)
        first_key = sorted(hidden_tampered)[0]
        hidden_tampered[first_key] = hidden_tampered[first_key].clone()
        hidden_tampered[first_key][0, 0, 0] += 1.0
        with self.assertRaisesRegex(paq.PAQProbeError, "hidden tensor hash mismatch"):
            paq.validate_observation_manifest(manifest, hidden_tampered, states)

    def test_identity_scene_seed_and_state_leakage_are_rejected(self) -> None:
        manifest, hidden, states = make_bundle()
        records = [dict(row) for row in manifest["records"]]
        discovery_identity = next(
            row["identity_id"] for row in records if row["split"] == "discovery"
        )
        for row in records:
            if row["cohort_id"] == "held-0":
                row["identity_id"] = discovery_identity
        leaked = reseal_records(records, manifest)
        with self.assertRaisesRegex(paq.PAQProbeError, "split leakage.*identity_id"):
            paq.validate_observation_manifest(leaked, hidden, states)

        manifest, hidden, states = make_bundle()
        states["state-disc-1"] = states["state-disc-0"].clone()
        records = [dict(row) for row in manifest["records"]]
        for row in records:
            if row["cohort_id"] == "disc-1":
                row["noisy_state_sha256"] = paq.tensor_sha256(states["state-disc-1"])
        reused_state = reseal_records(records, manifest)
        with self.assertRaisesRegex(paq.PAQProbeError, "noisy state may belong"):
            paq.validate_observation_manifest(reused_state, hidden, states)

    def test_all_cohorts_share_one_action_and_coordinate_authority(self) -> None:
        attacks = {
            "requested_action_id": lambda row: row.update(
                requested_action_id="stand-up-and-hold"
            ),
            "action_family_id": lambda row: row.update(
                action_family_id="locomotion-and-hold"
            ),
            "hook_site": lambda row: row.update(
                hook_site="diff_dec.transformer.blocks.13.attn2.to_out.0:pre",
                hook_block_index=13,
            ),
            "diffusion_step": lambda row: row.update(diffusion_step=8),
            "sigma": lambda row: row.update(
                sigma=0.65,
                sigma_float32_be_hex=paq.float32_bits(0.65, label="sigma"),
            ),
            "phase_ranges": lambda row: row.update(
                phase_ranges={
                    "start": [0, 1],
                    "transition": [1, 3],
                    "terminal": [3, 6],
                    "hold": [6, 8],
                }
            ),
        }
        for expected_field, mutate in attacks.items():
            with self.subTest(field=expected_field):
                manifest, hidden, states = make_bundle()
                records = [dict(row) for row in manifest["records"]]
                for row in records:
                    if row["cohort_id"] == "held-0":
                        mutate(row)
                attacked = reseal_records(records, manifest)
                with self.assertRaisesRegex(
                    paq.PAQProbeError, rf"global {expected_field}"
                ):
                    paq.validate_observation_manifest(attacked, hidden, states)

    def test_checkpoint_policy_and_source_registry_are_record_bound(self) -> None:
        for field in (
            "checkpoint_sha256",
            "policy_sha256",
            "source_exposure_registry_sha256",
        ):
            with self.subTest(field=field):
                manifest, hidden, states = make_bundle()
                records = [dict(row) for row in manifest["records"]]
                for row in records:
                    if row["cohort_id"] == "held-0":
                        row[field] = "1" * 64
                attacked = reseal_records(records, manifest)
                with self.assertRaisesRegex(paq.PAQProbeError, field):
                    paq.validate_observation_manifest(attacked, hidden, states)

    def test_action_family_members_are_split_isolated(self) -> None:
        manifest, hidden, states = make_bundle()
        records = [dict(row) for row in manifest["records"]]
        discovery_member = next(
            row["action_family_member_id"]
            for row in records
            if row["cohort_id"] == "disc-0"
        )
        for row in records:
            if row["cohort_id"] == "held-0":
                row["action_family_member_id"] = discovery_member
        attacked = reseal_records(records, manifest)
        with self.assertRaisesRegex(
            paq.PAQProbeError, "split leakage.*action_family_member_id"
        ):
            paq.validate_observation_manifest(attacked, hidden, states)


class CausalMediaDecoderTests(unittest.TestCase):
    def test_real_media_is_decoded_to_exact81_rgb24_at_fps25(self) -> None:
        import av
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = (Path(directory).resolve() / "exact81.mp4")
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=25)
                stream.width = 16
                stream.height = 16
                stream.pix_fmt = "yuv420p"
                for index in range(81):
                    rgb = np.full((16, 16, 3), index % 251, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            observed = paq._decode_exact81_media(path)
            self.assertEqual(observed["decoded_frame_count"], 81)
            self.assertEqual(observed["decoded_fps_numerator"], 25)
            self.assertEqual(observed["decoded_fps_denominator"], 1)
            self.assertEqual(observed["decoded_height"], 16)
            self.assertEqual(observed["decoded_width"], 16)
            self.assertRegex(observed["decoded_rgb24_sha256"], r"^[0-9a-f]{64}$")


class AdmissionTests(unittest.TestCase):
    def test_trial_api_accepts_only_a_real_receipt_reference(self) -> None:
        self.assertEqual(
            tuple(paq.CausalInterventionTrial.__dataclass_fields__),
            (
                "cohort_id",
                "receipt_path",
                "receipt_file_sha256",
                "receipt_size_bytes",
            ),
        )
        source = Path(paq.__file__).read_text(encoding="utf-8")
        self.assertIn("with av.open(str(path), mode=\"r\")", source)
        self.assertIn("frame_count != 81", source)
        self.assertNotIn("baseline_preservation_score", source)
        self.assertNotIn("target_preservation_score", source)

    def test_observational_candidate_without_intervention_is_zero_training(self) -> None:
        manifest, hidden, states = make_bundle()
        audit = paq.observational_audit(manifest, hidden, states)
        self.assertTrue(audit.passed, audit.reasons)
        self.assertIsNotNone(audit.candidate_code)
        self.assertFalse(audit.candidate_is_training_eligible)

        decision = paq.decide_phase_action_quotient(manifest, hidden, states)
        self.assertTrue(decision.observational_candidate_passed)
        self.assertFalse(decision.causal_intervention_passed)
        self.assertFalse(decision.admitted_code)
        self.assertEqual(decision.status, paq.NO_ADMISSION_STATUS)
        self.assertEqual(decision.training_updates_authorized, 0)
        self.assertEqual(decision.parameter_updates_executed, 0)
        self.assertIn("missing_frozen_causal_intervention_trials", decision.reasons)

    def test_common_original_motion_is_quotiented_not_mistaken_for_active_noop(self) -> None:
        manifest, hidden, states = make_bundle(common_source_motion=True)
        audit = paq.observational_audit(manifest, hidden, states)
        self.assertTrue(audit.passed, audit.reasons)
        self.assertGreater(
            audit.metrics["diagnostic_maximum_noop_to_target_energy_ratio"],
            0.50,
        )
        decision = paq.decide_phase_action_quotient(manifest, hidden, states)
        self.assertFalse(decision.admitted_code)
        self.assertEqual(decision.status, paq.NO_ADMISSION_STATUS)

    def test_real_sealed_decoded_receipts_are_conjunctive(self) -> None:
        manifest, hidden, states = make_bundle()
        audit = paq.observational_audit(manifest, hidden, states)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            trials = good_trials(manifest, audit.candidate_code_sha256, root)
            with mock.patch.object(
                paq, "_decode_exact81_media", side_effect=fake_decode_exact81
            ):
                admitted = paq.decide_phase_action_quotient(
                    manifest, hidden, states, causal_trials=trials
                )
        self.assertTrue(admitted.admitted_code)
        self.assertEqual(admitted.training_updates_authorized, 1)
        # This is a read-only probe even after admission.
        self.assertEqual(admitted.parameter_updates_executed, 0)

    def test_action_reverse_and_noop_metrics_fail_from_receipt_content(self) -> None:
        mutations = {
            "target": lambda receipt: receipt["action_metrics"].update(
                target_action_score=0.21
            ),
            "reverse": lambda receipt: receipt["action_metrics"].update(
                reverse_order_score=0.20
            ),
            "noop": lambda receipt: receipt["action_metrics"].update(
                noop_effect=0.20
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(gate=name), tempfile.TemporaryDirectory() as directory:
                manifest, hidden, states = make_bundle()
                audit = paq.observational_audit(manifest, hidden, states)
                trials = good_trials(
                    manifest, audit.candidate_code_sha256, Path(directory).resolve()
                )
                bad = rewrite_trial(trials[0], mutate)
                with mock.patch.object(
                    paq, "_decode_exact81_media", side_effect=fake_decode_exact81
                ):
                    decision = paq.decide_phase_action_quotient(
                        manifest,
                        hidden,
                        states,
                        causal_trials=(bad, trials[1]),
                    )
                self.assertFalse(decision.admitted_code)
                self.assertEqual(decision.status, paq.NO_ADMISSION_STATUS)
                self.assertEqual(decision.training_updates_authorized, 0)
                self.assertEqual(decision.parameter_updates_executed, 0)

    def test_each_preservation_axis_is_a_zero_drop_hard_gate(self) -> None:
        self.assertNotIn(
            "maximum_preservation_drop", paq.CausalThresholds.__dataclass_fields__
        )
        for axis in paq.PRESERVATION_AXES:
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as directory:
                manifest, hidden, states = make_bundle()
                audit = paq.observational_audit(manifest, hidden, states)
                trials = good_trials(
                    manifest, audit.candidate_code_sha256, Path(directory).resolve()
                )

                def lower_one_axis(receipt, selected=axis):
                    # A 0.001 loss is smaller than the removed 0.02 budget but
                    # must still fail exact per-axis noninferiority.
                    receipt["preservation_axes"][selected]["target_score"] = 0.899

                bad = rewrite_trial(trials[0], lower_one_axis)
                with mock.patch.object(
                    paq, "_decode_exact81_media", side_effect=fake_decode_exact81
                ):
                    decision = paq.decide_phase_action_quotient(
                        manifest,
                        hidden,
                        states,
                        causal_trials=(bad, trials[1]),
                    )
                self.assertFalse(decision.admitted_code)
                self.assertIn(
                    f"held-0:preservation_{axis}_worse", decision.reasons
                )

    def test_fake_receipt_hash_and_symlink_are_rejected(self) -> None:
        manifest, hidden, states = make_bundle()
        audit = paq.observational_audit(manifest, hidden, states)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            trials = good_trials(manifest, audit.candidate_code_sha256, root)
            fake_hash = replace(trials[0], receipt_file_sha256="0" * 64)
            with mock.patch.object(
                paq, "_decode_exact81_media", side_effect=fake_decode_exact81
            ), self.assertRaisesRegex(paq.PAQProbeError, "actual hash/size"):
                paq.decide_phase_action_quotient(
                    manifest, hidden, states, causal_trials=(fake_hash, trials[1])
                )

            link = root / "receipt-link.json"
            link.symlink_to(Path(trials[0].receipt_path))
            symlinked = replace(trials[0], receipt_path=str(link))
            with mock.patch.object(
                paq, "_decode_exact81_media", side_effect=fake_decode_exact81
            ), self.assertRaisesRegex(paq.PAQProbeError, "canonical, regular, and non-symlink"):
                paq.decide_phase_action_quotient(
                    manifest, hidden, states, causal_trials=(symlinked, trials[1])
                )

    def test_receipt_binds_candidate_checkpoint_policy_source_and_scale(self) -> None:
        mutations = {
            "candidate_code_sha256": lambda receipt: receipt.update(
                candidate_code_sha256="0" * 64
            ),
            "checkpoint_sha256": lambda receipt: receipt.update(
                checkpoint_sha256="1" * 64
            ),
            "policy_sha256": lambda receipt: receipt.update(policy_sha256="2" * 64),
            "source_exposure_registry_sha256": lambda receipt: receipt.update(
                source_exposure_registry_sha256="3" * 64
            ),
            "intervention_scale": lambda receipt: receipt.update(
                intervention_scale=0.6,
                intervention_scale_bits=paq.float64_bits(
                    0.6, label="intervention_scale"
                ),
            ),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                manifest, hidden, states = make_bundle()
                audit = paq.observational_audit(manifest, hidden, states)
                trials = good_trials(
                    manifest, audit.candidate_code_sha256, Path(directory).resolve()
                )
                bad = rewrite_trial(trials[0], mutate)
                with mock.patch.object(
                    paq, "_decode_exact81_media", side_effect=fake_decode_exact81
                ), self.assertRaises(paq.PAQProbeError):
                    paq.decide_phase_action_quotient(
                        manifest,
                        hidden,
                        states,
                        causal_trials=(bad, trials[1]),
                    )

    def test_decoded_media_hash_and_exact81_evidence_are_revalidated(self) -> None:
        manifest, hidden, states = make_bundle()
        audit = paq.observational_audit(manifest, hidden, states)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            trials = good_trials(manifest, audit.candidate_code_sha256, root)
            receipt = json.loads(Path(trials[0].receipt_path).read_text(encoding="ascii"))
            target_media = Path(receipt["decoded_media"]["target"]["path"])
            target_media.write_bytes(target_media.read_bytes() + b"tampered")
            with mock.patch.object(
                paq, "_decode_exact81_media", side_effect=fake_decode_exact81
            ), self.assertRaisesRegex(paq.PAQProbeError, "media actual hash/size"):
                paq.decide_phase_action_quotient(
                    manifest, hidden, states, causal_trials=trials
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            trials = good_trials(manifest, audit.candidate_code_sha256, root)

            def fake_decoded_count(receipt):
                receipt["decoded_media"]["target"]["decoded_frame_count"] = 80

            bad = rewrite_trial(trials[0], fake_decoded_count)
            with mock.patch.object(
                paq, "_decode_exact81_media", side_effect=fake_decode_exact81
            ), self.assertRaisesRegex(paq.PAQProbeError, "decoded exact81 evidence"):
                paq.decide_phase_action_quotient(
                    manifest, hidden, states, causal_trials=(bad, trials[1])
                )

    def test_lexically_separable_static_hidden_is_not_a_paq(self) -> None:
        manifest, hidden, states = make_bundle(lexical_only=True)
        rows = manifest["records"]
        target = hidden[next(row["hidden_key"] for row in rows if row["variant"] == "target")]
        noop = hidden[next(row["hidden_key"] for row in rows if row["variant"] == "noop")]
        # A raw linear/centroid classifier has an enormous lexical signal.
        raw_centroid_distance = torch.linalg.vector_norm(target.mean((0, 1)) - noop.mean((0, 1)))
        self.assertGreater(float(raw_centroid_distance), 20.0)

        audit = paq.observational_audit(manifest, hidden, states)
        self.assertFalse(audit.passed)
        self.assertIsNone(audit.candidate_code)
        self.assertIn("target_contrast_too_small", audit.reasons)
        decision = paq.decide_phase_action_quotient(manifest, hidden, states)
        self.assertFalse(decision.admitted_code)
        self.assertEqual(decision.status, paq.NO_ADMISSION_STATUS)
        self.assertEqual(decision.parameter_updates_executed, 0)

    def test_wrong_phase_order_fails_heldout_progression(self) -> None:
        manifest, hidden, states = make_bundle(shuffled_phase=True)
        audit = paq.observational_audit(manifest, hidden, states)
        self.assertFalse(audit.passed)
        self.assertIn("terminal_does_not_follow_transition", audit.reasons)


if __name__ == "__main__":
    unittest.main()
