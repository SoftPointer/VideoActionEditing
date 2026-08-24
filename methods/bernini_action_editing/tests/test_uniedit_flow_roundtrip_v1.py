from __future__ import annotations

from pathlib import Path
import copy
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import uniedit_flow_roundtrip_v1 as contract  # noqa: E402


ARCHIVE_SHA = "1" * 64
LPIPS_WEIGHTS_SHA = "2" * 64
REFERENCE_RGB_SHA = "3" * 64
PROMPT_EMBEDDING_SHA = "4" * 64


def _resign_record(value: dict) -> None:
    value.pop("digest", None)
    value["digest"] = contract.object_sha256(value)


def _resign_receipt(value: dict) -> None:
    value.pop("receipt_digest", None)
    value["receipt_digest"] = contract.object_sha256(value)


def _slice_metrics(
    start: int,
    end: int,
    *,
    quintile_index: int | None,
    backends: dict,
    psnr: float,
    ssim: float,
    lpips: float,
) -> dict:
    metrics = {}
    values = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    for name in ("psnr", "ssim", "lpips"):
        backend_digest = backends[name]["digest"]
        metrics[name] = (
            contract.available_metric(
                name, values[name], backend_digest=backend_digest
            )
            if backends[name]["available"]
            else contract.pending_metric(
                name,
                f"{name} backend unavailable",
                backend_digest=backend_digest,
            )
        )
    return {
        "frame_start_inclusive": start,
        "frame_end_exclusive": end,
        "frame_count": end - start,
        "quintile_index": quintile_index,
        **metrics,
    }


def _metric_packet(
    arm: str,
    *,
    backends: dict,
    candidate_rgb_sha: str,
    psnr: float,
    ssim: float,
    lpips: float,
) -> dict:
    return {
        "schema_version": contract.METRIC_PACKET_SCHEMA,
        "arm": arm,
        "reference": "resized_source_rgb",
        "measurement_domain": "in_memory_vae_decode_before_video_encoding",
        "reference_rgb_tensor_sha256": REFERENCE_RGB_SHA,
        "candidate_rgb_tensor_sha256": candidate_rgb_sha,
        "full_video": _slice_metrics(
            0,
            contract.FRAME_COUNT,
            quintile_index=None,
            backends=backends,
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
        ),
        "temporal_quintiles": [
            _slice_metrics(
                start,
                end,
                quintile_index=index,
                backends=backends,
                psnr=psnr,
                ssim=ssim,
                lpips=lpips,
            )
            for index, (start, end) in enumerate(
                contract.TEMPORAL_QUINTILE_BOUNDS
            )
        ],
        "backends": copy.deepcopy(backends),
    }


def _source(clean_sha: str) -> dict:
    return contract.finalize_evidence_record(
        {
            "schema_version": contract.SOURCE_SCHEMA,
            "iid": contract.IID,
            "source_video_sha256": contract.SOURCE_VIDEO_SHA256,
            "source_dataset_spec_sha256": contract.SOURCE_DATASET_SPEC_SHA256,
            "source_dataset_receipt_sha256": contract.SOURCE_DATASET_RECEIPT_SHA256,
            "source_dataset_receipt_digest": contract.SOURCE_DATASET_RECEIPT_DIGEST,
            "orbit_dataset_spec_sha256": contract.ORBIT_DATASET_SPEC_SHA256,
            "orbit_dataset_receipt_sha256": contract.ORBIT_DATASET_RECEIPT_SHA256,
            "orbit_dataset_receipt_digest": contract.ORBIT_DATASET_RECEIPT_DIGEST,
            "orbit_row_digest": contract.ORBIT_ROW_DIGEST,
            "clean_latent_sha256": clean_sha,
            "clean_latent_shape": list(contract.LATENT_SHAPE),
            "clean_latent_dtype": "torch.float32",
            "clean_latent_coordinate": "normalized_bernini_vae_V0_video",
            "resized_source_rgb_tensor_sha256": REFERENCE_RGB_SHA,
            "frame_count": contract.FRAME_COUNT,
            "fps": contract.FPS,
            "height": contract.HEIGHT,
            "width": contract.WIDTH,
        }
    )


def _model() -> dict:
    return contract.finalize_evidence_record(
        {
            "schema_version": contract.MODEL_SCHEMA,
            "renderer": "Bernini-R-1.3B-transformer_1",
            "bernini_commit": contract.EXPECTED_BERNINI_COMMIT,
            "veomni_commit": contract.EXPECTED_VEOMNI_COMMIT,
            "checkpoint_tree_sha256": contract.EXPECTED_CHECKPOINT_TREE_SHA256,
            "checkpoint_manifest_sha256": contract.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "model_state_sha256": contract.EXPECTED_MODEL_STATE_SHA256,
            "transformer_count": 1,
            "transformer_block_count": 30,
            "transformer_frozen_eval": True,
            "vae_identity_digest": contract.PINNED_VAE_IDENTITY_DIGEST,
            "vae_file_sha256": dict(contract.EXPECTED_VAE_FILE_SHA256),
            "vae_frozen_eval": True,
        }
    )


def _parallel() -> dict:
    return contract.finalize_evidence_record(
        {
            "schema_version": contract.PARALLEL_SCHEMA,
            "topology": "WORLD4_DP1_SP4",
            "world_size": 4,
            "sequence_parallel_size": 4,
            "data_parallel_size": 1,
            "distributed_invocation_count": 1,
            "world4_consensus": True,
            "single_model_replica_per_rank": True,
        }
    )


def _prompt() -> dict:
    return contract.finalize_evidence_record(
        {
            "schema_version": contract.PROMPT_SCHEMA,
            "kind": "single_blank_t2v_condition",
            "raw_text": "",
            "utf8_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "utf8_bytes": 0,
            "embedding_sha256": PROMPT_EMBEDDING_SHA,
            "embedding_shape": [1, 512, 4096],
            "embedding_dtype": "torch.bfloat16",
            "encoder_call_count": 1,
            "guidance_branch_count": 1,
            "same_for_e0_u0_inverse_reconstruct": True,
            "cfg_used": False,
            "apg_used": False,
        }
    )


def _media(arm: str, decode_sha: str) -> dict:
    decoded_sha = contract.object_sha256({"decoded": arm})
    return contract.finalize_evidence_record(
        {
            "schema_version": contract.MEDIA_SCHEMA,
            "arm": arm,
            "decode_input_latent_sha256": decode_sha,
            "decoded_rgb_tensor_sha256": decoded_sha,
            "mp4_sha256": contract.object_sha256({"mp4": arm}),
            "frame_count": contract.FRAME_COUNT,
            "fps": contract.FPS,
            "height": contract.HEIGHT,
            "width": contract.WIDTH,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "vae_decode_count": 1,
            "vae_frozen_eval": True,
            "metrics_measured_before_video_encoding": True,
        }
    )


def _bindings(receipt: dict) -> dict:
    return {
        "source_digest": receipt["source"]["digest"],
        "model_digest": receipt["model"]["digest"],
        "parallel_digest": receipt["parallel"]["digest"],
        "prompt_digest": receipt["prompt"]["digest"],
        "prompt_embedding_sha256": receipt["prompt"]["embedding_sha256"],
        "schedule_digest": receipt["schedule"]["digest"],
        "solver_contract_digest": receipt["solver_contract"]["digest"],
    }


def _valid_unsigned_receipt(*, lpips_available: bool = True) -> dict:
    initial_state = 2.0
    clean_sha = contract.object_sha256(initial_state)
    solver_results = {
        arm: contract.run_solver_arm(initial_state, lambda state, query: 1.0, arm=arm)
        for arm in contract.SOLVER_ARMS
    }
    unsigned = {
        "schema_version": contract.RECEIPT_SCHEMA,
        "complete": True,
        "experiment": {
            "stage": "A0",
            "scope": "roundtrip_control_only",
            "read_only": True,
            "training": False,
            "editing": False,
        },
        "authority": {
            "paper_url": contract.PAPER_URL,
            "arxiv_algorithm_url": contract.ARXIV_ALGORITHM_URL,
            "official_repository_url": contract.OFFICIAL_REPOSITORY_URL,
            "official_repository_commit": contract.OFFICIAL_REPOSITORY_COMMIT,
            "official_uniinv_scheduler_git_blob": contract.OFFICIAL_UNIINV_SCHEDULER_GIT_BLOB,
        },
        "source": _source(clean_sha),
        "model": _model(),
        "parallel": _parallel(),
        "prompt": _prompt(),
        "schedule": contract.EXACT40_ROUNDTRIP_SCHEDULE.receipt(),
        "solver_contract": contract.solver_contract_receipt(),
        "execution": {
            "model_forward_calls_by_arm": dict(contract.ARM_MODEL_FORWARD_CALLS),
            "conditional_model_forward_calls": 161,
            "unconditional_model_forward_calls": 0,
            "cfg_combinations": 0,
            "apg_combinations": 0,
            "model_load_count": 1,
            "vae_load_count": 1,
            "vae_decode_count_by_arm": {arm: 1 for arm in contract.ARMS},
            "total_vae_decode_count": 3,
            "media_output_count": 3,
            "scheduler_instance_count": 0,
            "scheduler_step_count": 0,
            "optimizer_instance_count": 0,
            "optimizer_steps": 0,
            "adapter_forward_calls": 0,
        },
        "arms": {},
        "hard_gate": {},
        "dependencies": contract.dependency_receipt(ARCHIVE_SHA),
        "runtime_versions": contract.runtime_versions_receipt(),
        "prohibitions": {
            "optimizer_created": False,
            "optimizer_steps": 0,
            "adapter_loaded": False,
            "adapter_parameters": 0,
            "cfg_used": False,
            "apg_used": False,
            "uni_edit_a1_used": False,
            "scheduler_object_used_by_solver": False,
            "flow_shift_reapplied": False,
            "automatic_visual_claim": False,
            "semantic_method_success_claimed": False,
        },
        "visual_review": {
            "status": "pending",
            "automatic_claim": False,
            "human_review_required": True,
        },
        "method_source_revision": "0" * 40,
        "method_source_archive_sha256": ARCHIVE_SHA,
    }
    backends = contract.metric_backend_receipts(
        ARCHIVE_SHA,
        lpips_available=lpips_available,
        lpips_weights_sha256=LPIPS_WEIGHTS_SHA if lpips_available else None,
    )
    values = {
        "c0_vae_ceiling": (40.0, 0.99, 0.01),
        "e0_vanilla_euler_roundtrip": (20.0, 0.70, 0.30),
        "u0_uni_inv_roundtrip": (25.0, 0.80, 0.20),
    }
    c0_media = _media("c0_vae_ceiling", clean_sha)
    c0_metrics = _metric_packet(
        "c0_vae_ceiling",
        backends=backends,
        candidate_rgb_sha=c0_media["decoded_rgb_tensor_sha256"],
        psnr=values["c0_vae_ceiling"][0],
        ssim=values["c0_vae_ceiling"][1],
        lpips=values["c0_vae_ceiling"][2],
    )
    unsigned["arms"]["c0_vae_ceiling"] = contract.finalize_evidence_record(
        {
            "schema_version": contract.ARM_SCHEMA,
            "arm": "c0_vae_ceiling",
            "role": "frozen_vae_codec_ceiling",
            "bindings": _bindings(unsigned),
            "prompt_consumed": False,
            "transformer_forward_calls": 0,
            "inversion_trace": None,
            "reconstruction_trace": None,
            "input_clean_latent_sha256": clean_sha,
            "decode_input_latent_sha256": clean_sha,
            "media": c0_media,
            "metrics": c0_metrics,
        }
    )
    for arm in contract.SOLVER_ARMS:
        inversion, reconstruction = solver_results[arm]
        inversion_receipt = inversion.receipt(digest_state=contract.object_sha256)
        reconstruction_receipt = reconstruction.receipt(
            digest_state=contract.object_sha256
        )
        media = _media(arm, reconstruction_receipt["final_state_sha256"])
        psnr, ssim, lpips = values[arm]
        metrics = _metric_packet(
            arm,
            backends=backends,
            candidate_rgb_sha=media["decoded_rgb_tensor_sha256"],
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
        )
        chain = {
            "source_clean_latent_sha256": clean_sha,
            "inversion_initial_state_sha256": clean_sha,
            "inversion_final_state_sha256": inversion_receipt["final_state_sha256"],
            "reconstruction_initial_state_sha256": inversion_receipt["final_state_sha256"],
            "reconstruction_final_state_sha256": reconstruction_receipt["final_state_sha256"],
            "decode_input_latent_sha256": reconstruction_receipt["final_state_sha256"],
            "inversion_to_reconstruction_exact": True,
            "reconstruction_to_decode_exact": True,
        }
        unsigned["arms"][arm] = contract.finalize_evidence_record(
            {
                "schema_version": contract.ARM_SCHEMA,
                "arm": arm,
                "role": "numerical_roundtrip_control",
                "bindings": _bindings(unsigned),
                "prompt_consumed": True,
                "transformer_forward_calls": contract.ARM_MODEL_FORWARD_CALLS[arm],
                "inversion_trace": inversion_receipt,
                "reconstruction_trace": reconstruction_receipt,
                "state_chain": chain,
                "media": media,
                "metrics": metrics,
            }
        )
    return unsigned


def _valid_receipt(*, lpips_available: bool = True) -> dict:
    return contract.finalize_receipt(
        _valid_unsigned_receipt(lpips_available=lpips_available)
    )


class UniEditFlowAuthorityTests(unittest.TestCase):
    def test_authority_schedule_and_forward_budget_are_exact(self) -> None:
        self.assertEqual(
            contract.OFFICIAL_REPOSITORY_COMMIT,
            "dc9edb465545352bbd9d674010ac8683e554c97d",
        )
        self.assertEqual(
            contract.OFFICIAL_UNIINV_SCHEDULER_GIT_BLOB,
            "dde8c59d811a2064a6b07a5d21457f4aef636a3e",
        )
        schedule = contract.EXACT40_ROUNDTRIP_SCHEDULE
        receipt = schedule.receipt()
        self.assertEqual(schedule.interval_count, 40)
        self.assertEqual(receipt["flow_shift_declared_by_source_schedule"], 5.0)
        self.assertEqual(receipt["flow_shift_application_count_in_a0_runtime"], 0)
        self.assertEqual(
            receipt["digest"],
            "b87e5fbc313e5e900bbd3b5f5adac11514a14fc07d779f567a83dc77f3b8e625",
        )
        self.assertEqual(
            contract.ARM_MODEL_FORWARD_CALLS,
            {
                "c0_vae_ceiling": 0,
                "e0_vanilla_euler_roundtrip": 80,
                "u0_uni_inv_roundtrip": 81,
            },
        )
        self.assertEqual(contract.TOTAL_MODEL_FORWARD_CALLS, 161)


class UniEditFlowRecurrenceTests(unittest.TestCase):
    def test_exact40_query_order_and_forward_counts(self) -> None:
        calls = []

        def velocity(state, query):
            calls.append((query.phase, query.coordinate.ascending_index))
            return 1.0

        for arm, inverse_count in (
            ("e0_vanilla_euler_roundtrip", 40),
            ("u0_uni_inv_roundtrip", 41),
        ):
            calls.clear()
            inverse, reconstruct = contract.run_solver_arm(2.0, velocity, arm=arm)
            self.assertEqual(inverse.model_forward_calls, inverse_count)
            self.assertEqual(reconstruct.model_forward_calls, 40)
            self.assertEqual(reconstruct.final_state, 2.0)
            self.assertEqual(
                [index for phase, index in calls[:inverse_count]],
                list(range(inverse_count)),
            )
            self.assertEqual(
                [index for phase, index in calls[inverse_count:]],
                list(range(40, 0, -1)),
            )

    def test_callback_cannot_reuse_and_overwrite_previous_velocity_buffer(self) -> None:
        shared = [0.0]

        def hostile_velocity(state, query):
            shared[0] = 1.0 + query.coordinate.sigma
            return shared

        with self.assertRaisesRegex(
            contract.UniEditFlowRoundtripError,
            "mutated the previous velocity buffer",
        ):
            contract.uni_inv_predictor_corrector([2.0], hostile_velocity)

    def test_callbacks_cannot_mutate_state_or_axpy_inputs(self) -> None:
        def mutating_velocity(state, query):
            state[0] += 1.0
            return [0.0]

        with self.assertRaisesRegex(
            contract.UniEditFlowRoundtripError, "mutated the recurrence state"
        ):
            contract.vanilla_inverse_euler([1.0], mutating_velocity)

        def mutating_advance(state, velocity, delta):
            velocity[0] += delta
            return [state[0]]

        with self.assertRaisesRegex(
            contract.UniEditFlowRoundtripError, "mutated its velocity input"
        ):
            contract.vanilla_inverse_euler(
                [1.0], lambda state, query: [1.0], advance=mutating_advance
            )


class UniEditFlowGateAndReceiptTests(unittest.TestCase):
    def test_metrics_alone_never_authorize_a1_but_closed_receipt_does(self) -> None:
        unsigned = _valid_unsigned_receipt()
        metrics = {arm: unsigned["arms"][arm]["metrics"] for arm in contract.ARMS}
        metric_only = contract.evaluate_hard_gate(metrics)
        self.assertEqual(metric_only["status"], "pass")
        self.assertIs(metric_only["evidence_closure_complete"], False)
        self.assertIs(metric_only["a1_prompt_paired_correction_authorized"], False)
        with self.assertRaises(TypeError):
            contract.evaluate_hard_gate(  # type: ignore[call-arg]
                metrics, evidence_closure_complete=True
            )
        receipt = contract.finalize_receipt(unsigned)
        self.assertIs(
            receipt["hard_gate"]["a1_prompt_paired_correction_authorized"], True
        )
        self.assertIs(contract.validate_receipt(receipt), receipt)

    def test_lpips_backend_unavailable_forces_pending_and_no_a1(self) -> None:
        receipt = _valid_receipt(lpips_available=False)
        self.assertEqual(receipt["hard_gate"]["status"], "pending")
        self.assertIs(
            receipt["hard_gate"]["a1_prompt_paired_correction_authorized"], False
        )

    def test_re_signed_nested_schema_and_execution_attacks_fail(self) -> None:
        attacks = []
        extra_source = _valid_receipt()
        extra_source["source"]["extra"] = "forged"
        _resign_record(extra_source["source"])
        _resign_receipt(extra_source)
        attacks.append(extra_source)

        model = _valid_receipt()
        model["model"]["transformer_block_count"] = 29
        _resign_record(model["model"])
        _resign_receipt(model)
        attacks.append(model)

        execution = _valid_receipt()
        execution["execution"]["shadow_forward_count"] = 999
        _resign_receipt(execution)
        attacks.append(execution)

        versions = _valid_receipt()
        versions["runtime_versions"]["torch"] = "forged"
        _resign_record(versions["runtime_versions"])
        _resign_receipt(versions)
        attacks.append(versions)

        parallel = _valid_receipt()
        parallel["parallel"]["extra"] = "forged"
        _resign_record(parallel["parallel"])
        _resign_receipt(parallel)
        attacks.append(parallel)

        solver = _valid_receipt()
        solver["solver_contract"]["scheduler_step_calls"] = 1
        _resign_record(solver["solver_contract"])
        _resign_receipt(solver)
        attacks.append(solver)

        dependencies = _valid_receipt()
        dependencies["dependencies"]["local_source_sha256"][
            "dclr_runtime_contract.py"
        ] = "6" * 64
        _resign_record(dependencies["dependencies"])
        _resign_receipt(dependencies)
        attacks.append(dependencies)

        for hostile in attacks:
            with self.subTest(keys=hostile.keys()):
                with self.assertRaises(contract.UniEditFlowRoundtripError):
                    contract.validate_receipt(hostile)

    def test_re_signed_arm_binding_backend_and_a1_attacks_fail(self) -> None:
        binding_attacks = []
        for binding_key in (
            "source_digest",
            "model_digest",
            "parallel_digest",
            "prompt_digest",
            "prompt_embedding_sha256",
            "schedule_digest",
            "solver_contract_digest",
        ):
            hostile = _valid_receipt()
            hostile["arms"]["e0_vanilla_euler_roundtrip"]["bindings"][
                binding_key
            ] = "9" * 64
            _resign_record(hostile["arms"]["e0_vanilla_euler_roundtrip"])
            _resign_receipt(hostile)
            binding_attacks.append(hostile)

        backend = _valid_receipt()
        for arm in contract.ARMS:
            packet = backend["arms"][arm]["metrics"]
            lpips_backend = packet["backends"]["lpips"]
            lpips_backend["package_version"] = None
            lpips_backend["weights_sha256"] = None
            _resign_record(lpips_backend)
            for metric_slice in [packet["full_video"], *packet["temporal_quintiles"]]:
                metric_slice["lpips"]["backend_digest"] = lpips_backend["digest"]
            _resign_record(backend["arms"][arm])
        _resign_receipt(backend)

        unavailable = _valid_receipt(lpips_available=False)
        for arm in contract.ARMS:
            packet = unavailable["arms"][arm]["metrics"]
            for metric_slice in [packet["full_video"], *packet["temporal_quintiles"]]:
                metric_slice["lpips"] = contract.available_metric(
                    "lpips",
                    0.1,
                    backend_digest=packet["backends"]["lpips"]["digest"],
                )
            _resign_record(unavailable["arms"][arm])
        _resign_receipt(unavailable)

        forged_a1 = _valid_receipt(lpips_available=False)
        forged_a1["hard_gate"] = copy.deepcopy(_valid_receipt()["hard_gate"])
        _resign_receipt(forged_a1)

        for hostile in (*binding_attacks, backend, unavailable, forged_a1):
            with self.assertRaises(contract.UniEditFlowRoundtripError):
                contract.validate_receipt(hostile)

    def test_re_signed_trace_state_media_and_c0_attacks_fail(self) -> None:
        trace = _valid_receipt()
        arm = trace["arms"]["u0_uni_inv_roundtrip"]
        arm["inversion_trace"]["records"][2]["previous_velocity_sha256"] = "8" * 64
        _resign_record(arm["inversion_trace"])
        _resign_record(arm)
        _resign_receipt(trace)

        state = _valid_receipt()
        arm = state["arms"]["e0_vanilla_euler_roundtrip"]
        arm["state_chain"]["decode_input_latent_sha256"] = "7" * 64
        _resign_record(arm)
        _resign_receipt(state)

        media = _valid_receipt()
        arm = media["arms"]["u0_uni_inv_roundtrip"]
        arm["media"]["frame_count"] = 80
        _resign_record(arm["media"])
        _resign_record(arm)
        _resign_receipt(media)

        c0 = _valid_receipt()
        c0["arms"]["c0_vae_ceiling"]["transformer_forward_calls"] = 1
        _resign_record(c0["arms"]["c0_vae_ceiling"])
        _resign_receipt(c0)

        for hostile in (trace, state, media, c0):
            with self.assertRaises(contract.UniEditFlowRoundtripError):
                contract.validate_receipt(hostile)


if __name__ == "__main__":
    unittest.main()
