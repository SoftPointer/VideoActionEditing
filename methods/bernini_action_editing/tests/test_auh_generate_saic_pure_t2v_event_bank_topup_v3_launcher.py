from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.sbatch"
)


class AUHSAICPureT2VEventBankTopupV3LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_scientific_runtime_and_topup_spec_are_unchanged(self) -> None:
        self.assertIn(
            "readonly expected_runtime_sha256=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36",
            self.source,
        )
        self.assertIn(
            'runtime="${method_root}/generate_saic_pure_t2v_event_bank_topup_v2.py"',
            self.source,
        )
        self.assertIn("saic_pure_t2v_event_bank_topup_v2.json", self.source)
        self.assertIn("branches=incomplete,camera_only,appearance_only", self.source)
        self.assertIn("candidates_per_group=30", self.source)
        self.assertIn("attempts=60", self.source)
        self.assertIn("seed_selection=false", self.source)
        self.assertIn("optimizer=false", self.source)

    def test_uses_kernel_atomic_dynamic_port_and_no_arithmetic_master_port(self) -> None:
        self.assertIn("--rdzv-backend=c10d", self.source)
        self.assertIn("--rdzv-endpoint=127.0.0.1:0", self.source)
        self.assertIn("--local-addr=127.0.0.1", self.source)
        self.assertIn("--max-restarts=0", self.source)
        self.assertNotIn("job_mod=", self.source)
        self.assertNotIn("base_port", self.source)
        self.assertNotIn("candidate_port", self.source)
        self.assertNotIn("--master_port", self.source)
        self.assertNotIn("--master-port", self.source)

    def test_inherited_shared_store_disable_poison_is_overridden_to_zero(self) -> None:
        assignment = "export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0"
        self.assertIn(assignment, self.source)
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1; '
                + assignment
                + '; test "${TORCH_DISABLE_SHARE_RDZV_TCP_STORE}" = 0',
            ],
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn('"torch_disable_share_rdzv_tcp_store": "0"', self.source)
        self.assertIn('"shared_tcp_store_bootstrap_required": True', self.source)

    def test_run_id_binds_job_group_fixed_index_candidate_and_launch(self) -> None:
        self.assertIn(
            'rdzv_id_prefix="saic-${SLURM_JOB_ID}-${group_id}-c$(printf',
            self.source,
        )
        self.assertIn('${candidate_digest:0:16}', self.source)
        self.assertIn('rdzv_id="${rdzv_id_prefix}-l$(printf', self.source)
        self.assertIn('--expected-rdzv-id "${rdzv_id}"', self.source)
        self.assertIn('"rdzv_id_prefix": rdzv_id_prefix', self.source)

    def test_permanent_claim_and_exact_world4_guard_are_on_real_path(self) -> None:
        self.assertIn('mkdir -m 0700 -- "${output_root}/logs/rendezvous/port-claims"', self.source)
        self.assertIn('port_claim_root_identity="$(stat -Lc', self.source)
        self.assertIn('"${rendezvous_guard}" worker', self.source)
        self.assertIn('--claim-root "${port_claim_root}"', self.source)
        self.assertIn('--lifecycle-dir "${launch_dir}"', self.source)
        self.assertIn('"${rendezvous_guard}" assemble', self.source)
        self.assertIn(
            'rendezvous_guard="${method_root}/saic_t2v_rendezvous_guard_v2.py"',
            self.source,
        )
        self.assertIn("SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256", self.source)

    def test_shared_filesystem_publication_and_groups_fail_fast(self) -> None:
        self.assertIn("umask 077", self.source)
        self.assertIn("Bash 5.1+ is required for fail-fast wait -n -p", self.source)
        self.assertIn(
            'wait -n -p finished_pid "${dog_pid}" "${human_pid}"', self.source
        )
        self.assertIn('kill "${dog_pid}" "${human_pid}"', self.source)
        self.assertIn("sibling terminated", self.source)
        self.assertIn(
            '"schema_version": "saic-t2v-topup-rendezvous-dynamic-plan-v2"',
            self.source,
        )

    def test_retry_is_bounded_and_collision_only_without_partial_reuse(self) -> None:
        self.assertIn("readonly maximum_rendezvous_launches=16", self.source)
        self.assertIn("launch_ordinal<=maximum_rendezvous_launches", self.source)
        self.assertIn('"${rendezvous_guard}" admit-collision', self.source)
        self.assertIn("collision retry budget exhausted", self.source)
        self.assertIn("failed without retry-authorizing collision", self.source)
        self.assertGreaterEqual(
            self.source.count(
                '[[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]]'
            ),
            2,
        )
        self.assertNotIn("--resume", self.source)
        self.assertNotIn("partial reuse", self.source.lower())

    def test_all8_remains_two_concurrent_world4_groups(self) -> None:
        self.assertIn('#SBATCH --gres=gpu:mi210:8', self.source)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.source)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.source)
        self.assertIn("dog_pid=$!", self.source)
        self.assertIn("human_pid=$!", self.source)
        self.assertIn('--nproc-per-node=4', self.source)
        self.assertNotIn('--nproc-per-node=8', self.source)

    def test_existing_allocation_mode_is_exactly_one_idle_world4_group(self) -> None:
        self.assertIn(
            'group_select="${SAIC_T2V_V3_GROUP_SELECT:-all8}"', self.source
        )
        self.assertIn(
            '"${group_select}" == sp4-a || "${group_select}" == sp4-b',
            self.source,
        )
        self.assertIn('allowed_counts = {4, 8}', self.source)
        self.assertIn('allowed_counts = {8}', self.source)
        self.assertIn('checked_indices = list(range(4, 8))', self.source)
        self.assertIn('if count not in allowed_counts', self.source)
        self.assertIn(
            'if any(rows[index]["used_bytes"] > 1024**3', self.source
        )
        self.assertIn(
            'run_group "${group_select}" "${selected_visible}"', self.source
        )
        self.assertIn('selected_visible="4,5,6,7"', self.source)
        self.assertIn('dishonest four-GPU sp4-b remap', self.source)
        self.assertIn(
            '"status": "exact30_world4_group_complete_pending_disjoint_merge_and_full_audit"',
            self.source,
        )
        self.assertIn('"candidate_count": 30', self.source)
        self.assertIn('"training": False', self.source)
        self.assertIn('merge_required=true', self.source)
        self.assertIn('os.fchmod(handle.fileno(), 0o444)', self.source)
        self.assertIn(
            'attempt_dir / "saic-event-topup-generation-receipt.json"',
            self.source,
        )

    def test_dynamic_plan_is_fixed_before_any_generation(self) -> None:
        plan = 'rendezvous_plan="${output_root}/logs/${group_id}-rendezvous-dynamic-plan-v1.json"'
        start = 'echo "[saic-t2v-topup-v3:${group_id}] START'
        self.assertIn(plan, self.source)
        self.assertIn("fixed_order", self.source)
        self.assertIn("lexicographic_envelope_basename", self.source)
        self.assertIn("numeric_ports_preregistered", self.source)
        self.assertIn("scientific_candidate_set_or_order_changed_by_retry", self.source)
        self.assertIn(
            "candidate envelope mode is not terminal-auditor compatible", self.source
        )
        self.assertLess(self.source.index(plan), self.source.index(start))

    def test_terminal_rendezvous_and_original_scientific_audits_both_gate_pass(self) -> None:
        rendezvous_audit = '"${rendezvous_guard}" audit-job'
        scientific_audit = '"${runtime}" audit-bank'
        final_banner = "SAIC_PURE_T2V_EVENT_BANK_TOPUP_ALL8_V3_STRONG_AUDIT_OK"
        self.assertIn(rendezvous_audit, self.source)
        self.assertIn(scientific_audit, self.source)
        self.assertIn("rendezvous-job-audit-v1.json", self.source)
        self.assertIn(final_banner, self.source)
        self.assertLess(self.source.index(rendezvous_audit), self.source.index(scientific_audit))
        self.assertLess(self.source.index(scientific_audit), self.source.index(final_banner))

    def test_v3_inputs_and_fresh_output_are_independent(self) -> None:
        for name in (
            "SAIC_T2V_V3_SOURCE_ARCHIVE",
            "SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256",
            "SAIC_T2V_V3_SOURCE_REVISION",
            "SAIC_T2V_V3_SOURCE_MANIFEST",
            "SAIC_T2V_V3_EVENT_SPEC",
            "SAIC_T2V_V3_OUTPUT_ROOT",
            "SAIC_T2V_V3_PYTHON_BIN",
            "SAIC_T2V_V3_STATIC_FFMPEG",
        ):
            self.assertIn(name, self.source)
        self.assertIn("output root must be fresh", self.source)
        self.assertNotIn("SAIC_T2V_V2_OUTPUT_ROOT", self.source)

    def test_archive_requires_v2_guard_launcher_and_tests(self) -> None:
        for basename in (
            "saic_t2v_rendezvous_guard_v2.py",
            "test_saic_t2v_rendezvous_guard_v2.py",
            "test_auh_generate_saic_pure_t2v_event_bank_topup_v3_launcher.py",
            "auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.sbatch",
        ):
            self.assertGreaterEqual(self.source.count(basename), 1)


if __name__ == "__main__":
    unittest.main()
