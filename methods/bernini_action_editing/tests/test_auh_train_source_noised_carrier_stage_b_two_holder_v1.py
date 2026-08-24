from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    METHOD_ROOT
    / "scripts"
    / "auh_train_source_noised_carrier_stage_b_two_holder_v1.sh"
)


class SourceNoisedCarrierStageBTwoHolderControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = CONTROLLER.read_text(encoding="utf-8")

    def test_shell_and_safe_usage_path(self) -> None:
        subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)
        result = subprocess.run(
            ["bash", str(CONTROLLER)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "usage: auh_train_source_noised_carrier_stage_b_two_holder_v1.sh run",
            result.stderr,
        )

    def test_holders_are_exact_and_retained_parent_has_no_child_path(self) -> None:
        for fragment in (
            "retained_job=135412",
            "retained_node=auh7-1b-gpu-293",
            "train_job0=135407",
            "train_node0=auh7-1b-gpu-260",
            "train_job1=135411",
            "train_node1=auh7-1b-gpu-214",
            'assert_parent_running "${retained_job}"',
            'assert_parent_running "${train_job0}"',
            'assert_parent_running "${train_job1}"',
        ):
            self.assertIn(fragment, self.text)
        self.assertNotIn("materialize_job", self.text)
        self.assertNotIn("materialize_node", self.text)
        self.assertNotIn("__materialize_exec", self.text)
        self.assertNotIn('launch_child "${retained_job}"', self.text)
        self.assertEqual(
            self.text.count('--gres=gpu:mi210:2'),
            1,
        )

    def test_existing_sealed_dataset_and_all_three_pins_are_mandatory(self) -> None:
        for fragment in (
            "BERNINI_SNC_STAGE_B_EXISTING_MATERIALIZED:?",
            "BERNINI_SNC_STAGE_B_EXISTING_PARQUET_SHA256:?",
            "BERNINI_SNC_STAGE_B_EXISTING_RECEIPT_SHA256:?",
            "BERNINI_SNC_STAGE_B_EXISTING_RECEIPT_DIGEST:?",
            'assert {p.name for p in root.iterdir()} == {"dataset.parquet","receipt.json"}',
            "assert parquet_sha == parquet_pin",
            "assert hashlib.sha256(receipt_raw).hexdigest() == receipt_pin",
            "assert d == digest_pin",
            "materialization_mode=reused_sealed_existing",
        ):
            self.assertIn(fragment, self.text)
        self.assertNotIn("BERNINI_SSR_EXISTING_MATERIALIZED", self.text)
        self.assertNotIn("created_on_holder", self.text)
        self.assertNotIn("if parquet_pin or receipt_pin or digest_pin", self.text)

    def test_release_is_r4_exact_twelve_member_content_closure(self) -> None:
        for fragment in (
            '"bernini-source-noised-carrier-stage-b-release-v1"',
            'manifest["release_generation"] == "r4"',
            'manifest["file_count"] == 12',
            'manifest["revision_kind"] == "content-closure-sha1"',
            'manifest["git_commit_claimed"] is False',
            'manifest["exact_member_closure"] is True',
            '"inference_sigma_strata.py"',
            '"source_noised_ladder_v1.py"',
            '"train_source_noised_carrier_strata_v1.py"',
            'assert [row["path"] for row in rows] == expected_relatives',
        ):
            self.assertIn(fragment, self.text)
        self.assertNotIn("bernini-source-self-training-release-v1", self.text)

    def test_world4_two_node_two_rank_stage_b_cli_is_exact(self) -> None:
        for fragment in (
            "--nnodes=2",
            "--nproc_per_node=2",
            '--node_rank="${node_rank}"',
            "--parallel-topology world4-dp1-sp4",
            "--mode source-carrier-strata-v1",
            "--adapter-block-scope early-mid-0-22",
            "--ack-upstream-training-use-forbidden",
            "--ack-forward-noising-is-not-inversion",
            "--no_python",
            "BERNINI_HELDOUT_RANK_CACHE_TOKEN",
            "f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5",
        ):
            self.assertIn(fragment, self.text)
        self.assertNotIn("--mode engineering-canary", self.text)
        self.assertNotIn("--rho", self.text)
        self.assertNotIn('optimizer_steps"]==1', self.text)

    def test_exact_four_step_postflight_and_claim_fences(self) -> None:
        for fragment in (
            'r["optimizer_steps"]==4',
            'r["positive_gradient_steps"]==4',
            'r["registered_schedule_indices"]==[16,29,35,38]',
            'r["optimizer_step_per_registered_sigma"] is True',
            'r["all_registered_strata_optimizer_authorized"] is True',
            'r["late_or_low_sigma_zero_update_gate_present"] is False',
            'r["distributed"]["checkpoint_recomputation_route_context_replayed"] is True',
            'placement=={"nodes":2,"local_world_size":2,"ranks_per_node":2,"sp4_crosses_nodes":True,"preferred_world4_placement":True}',
            'r["forward_noising"]["same_epsilon_target_and_donor_verified_every_logical_record"] is True',
            'r["forward_noising"]["same_sigma_target_and_donor_verified_every_logical_record"] is True',
            'r["forward_noising"]["different_epsilon_across_eight_logical_step_samples"] is True',
            'r["forward_noising"]["clean_source_references_routed_every_logical_record"] is True',
            'r["forward_noising"]["inversion_claimed"] is False',
            'r["forward_noising"]["reverse_ode_executed"] is False',
            'r["forward_noising"]["solver_state_replayed"] is False',
            'r["forward_noising"]["exact_roundtrip_claimed"] is False',
            '"method_success_claimed"',
            "COMPLETE_STAGE_B_PRETEXT_ONLY",
            "method_success_claimed=false inversion_claimed=false",
        ):
            self.assertIn(fragment, self.text)
        self.assertIn(
            'assert len(h["steps"])==4 and [step["schedule_index"] for step in h["steps"]]==[16,29,35,38]',
            self.text,
        )
        for fragment in (
            'assert r["initial_adapter_sha256"] != r["final_adapter_sha256"]',
            'assert step["parameter_sha256_before_step"]==before',
            "assert after != before",
            'assert before==r["final_adapter_sha256"] and len(records)==8',
            'assert len({record["epsilon_sha256"] for record in records})==8',
            'binding=dict(record["shared_noise_binding"]); digest=binding.pop("digest")',
            "assert hashlib.sha256(canonical).hexdigest()==digest",
            'binding["epsilon_sha256"]==record["epsilon_sha256"]==record["tensor_identities"]["epsilon"]',
            '"same_epsilon_object_reused_during_target_and_donor_construction"',
            '"target_formula_recomputed_and_equal"',
            '"donor_formula_recomputed_and_equal"',
            '"same_sigma_registered_coordinate_reused"',
            '"clean_source_references_routed"',
        ):
            self.assertIn(fragment, self.text)

    def test_only_identity_bound_child_srun_pids_can_be_signalled(self) -> None:
        lowered = self.text.lower()
        for forbidden in (
            "s" + "cancel",
            "scontrol " + "release",
            "scontrol " + "requeue",
            "p" + "kill",
            "kill" + "all",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertIn('register_child_pid "${pid}" "${job}" "${node}"', self.text)
        self.assertIn("child_identity_matches", self.text)
        self.assertIn("child_starttime", self.text)
        self.assertIn("child_cmdline_sha", self.text)
        self.assertIn("REFUSE_SIGNAL identity mismatch", self.text)
        self.assertEqual(self.text.count('kill -"$2" "$1"'), 1)
        self.assertNotIn("kill -9", lowered)

    def test_memory_gate_is_prelaunch_sampled_and_sacct_crosschecked(self) -> None:
        for fragment in (
            "memory.current",
            "resolve_cgroup2_memory_current",
            "/proc/self/mountinfo",
            "55834574848",
            "bernini-snc-stage-b-child-cgroup-memory-sampled-v1",
            "bernini-snc-stage-b-child-memory-crosscheck-v1",
            "sacct_max_rss_bytes",
            'fields["node_rank"] == expected_node_rank',
            '"${peak}" -lt "${memory_peak_limit_bytes}"',
            "sampled < bound and sacct_peak < bound",
        ):
            self.assertIn(fragment, self.text)
        self.assertNotIn("memory.usage_in_bytes", self.text)
        self.assertLess(
            self.text.index(
                'cgroup_counter="$(resolve_cgroup2_memory_current'
            ),
            self.text.index('"$@" &'),
        )
        self.assertGreater(
            self.text.rindex("assert_all_parents_running"),
            self.text.index("verify_step_memory"),
        )

    def test_fresh_output_and_no_parent_lifecycle_mutation(self) -> None:
        self.assertIn(
            '[[ ! -e "${run_root}" && ! -L "${run_root}" ]]',
            self.text,
        )
        self.assertIn('readonly training="${run_root}/stage_b_four_strata"', self.text)
        self.assertIn(
            'assert {p.name for p in root.iterdir()} == {"adapter.safetensors","optimizer.pt","history.json","receipt.json"}',
            self.text,
        )
        self.assertNotIn("scontrol suspend", self.text.lower())
        self.assertNotIn("scontrol hold", self.text.lower())


if __name__ == "__main__":
    unittest.main()
