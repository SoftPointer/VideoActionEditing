from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_generate_saic_pure_t2v_event_bank_topup_all8_v2.sbatch"
)


class AUHSAICPureT2VEventBankTopupV2LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_v2_is_independent_and_uses_all8_as_two_world4_sp4_groups(self) -> None:
        self.assertIn("#SBATCH --qos=bgqos", self.source)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.source)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.source)
        self.assertIn("dog_pid=$!", self.source)
        self.assertIn("human_pid=$!", self.source)
        self.assertIn('wait "${dog_pid}"', self.source)
        self.assertIn('wait "${human_pid}"', self.source)
        self.assertIn("--nproc_per_node=4", self.source)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertIn("SAIC_T2V_V2_OUTPUT_ROOT", self.source)
        self.assertNotIn("SAIC_T2V_OUTPUT_ROOT:?", self.source)
        self.assertIn("output root must be fresh", self.source)
        self.assertNotIn("--resume", self.source)

    def test_each_sequential_torchrun_gets_a_unique_parity_partitioned_port(self) -> None:
        self.assertIn("dog_base_port=$((41000 + 2 * job_mod))", self.source)
        self.assertIn("human_base_port=$((dog_base_port + 1))", self.source)
        self.assertIn("readonly rendezvous_port_stride=2", self.source)
        self.assertIn("readonly candidates_per_group=30", self.source)
        self.assertIn(
            "candidate_port=$((base_master_port + rendezvous_port_stride * candidate_index))",
            self.source,
        )
        self.assertIn('--master_port="${candidate_port}"', self.source)
        self.assertNotIn('--master_port="${master_port}"', self.source)
        self.assertIn("candidate_index=$((candidate_index + 1))", self.source)
        self.assertIn("candidate-indexed rendezvous ports exceed 65535", self.source)

        job_mod_values = (0, 1, 482, 9999)
        for job_mod in job_mod_values:
            dog_base = 41000 + 2 * job_mod
            human_base = dog_base + 1
            dog_ports = {dog_base + 2 * index for index in range(30)}
            human_ports = {human_base + 2 * index for index in range(30)}
            self.assertEqual(len(dog_ports), 30)
            self.assertEqual(len(human_ports), 30)
            self.assertTrue(dog_ports.isdisjoint(human_ports))
            self.assertLessEqual(max(human_ports), 65535)

    def test_candidate_index_and_fixed_plan_order_are_receipted(self) -> None:
        self.assertIn("saic-t2v-topup-rendezvous-port-plan-v1", self.source)
        self.assertIn('"candidate_index": index', self.source)
        self.assertIn('"master_port": port', self.source)
        self.assertIn('"fixed_order": "lexicographic_envelope_basename"', self.source)
        self.assertIn('"same_group_port_reuse": False', self.source)
        self.assertIn('"resume_or_partial_reuse": False', self.source)
        self.assertIn(
            'sorted(plan_dir.glob("*.json"), key=lambda path: path.name)',
            self.source,
        )
        self.assertIn(
            "group did not execute exactly 30 fixed-order candidates",
            self.source,
        )

    def test_only_sixty_hard_negative_topups_are_rendered(self) -> None:
        self.assertIn("top_up_only=true", self.source)
        self.assertIn("attempts=60", self.source)
        self.assertIn(
            "branches=incomplete,camera_only,appearance_only", self.source
        )
        self.assertNotIn("branches=forward,reverse,noop", self.source)
        self.assertIn("candidates_per_group=30", self.source)
        self.assertIn("merged_branches=6", self.source)
        self.assertIn("seed_selection=false", self.source)
        self.assertIn("event_audit=pending", self.source)
        self.assertIn("optimizer=false", self.source)

    def test_launcher_is_exact81_text_only_and_black_proxy_only(self) -> None:
        self.assertIn("exact81=true", self.source)
        self.assertIn("semantic_inputs=text_only", self.source)
        self.assertIn("black_proxy_only=true", self.source)
        self.assertIn("real_source_rgb=false", self.source)
        self.assertIn("source_latent=false", self.source)
        self.assertIn("source_noise=false", self.source)
        self.assertIn("target=false", self.source)
        self.assertIn("reference=false", self.source)
        self.assertIn("donor=false", self.source)
        self.assertIn("materialize-proxies", self.source)
        self.assertIn("geometry-proxy-receipt.json", self.source)
        self.assertNotIn("source_video.mp4", self.source)
        self.assertNotIn("--target-video", self.source)
        self.assertNotIn("--motion-donor", self.source)

    def test_archive_bound_pyav_and_compute_visible_static_ffmpeg_are_required(self) -> None:
        self.assertIn("tools/ffprobe_pyav_saic.py", self.source)
        self.assertIn('ffprobe_bin="${method_root}/tools/ffprobe_pyav_saic.py"', self.source)
        self.assertIn("SAIC_T2V_V2_STATIC_FFMPEG", self.source)
        self.assertIn("ffmpeg must be a compute-visible static build", self.source)
        self.assertIn('SAIC_COMPUTE_STATIC_FFMPEG="${ffmpeg_bin}"', self.source)
        self.assertIn('IMAGEIO_FFMPEG_EXE="${ffmpeg_bin}"', self.source)
        self.assertIn('FFMPEG_BINARY="${ffmpeg_bin}"', self.source)
        self.assertNotIn("SAIC_T2V_FFPROBE:?", self.source)

    def test_shared_ffprobe_gets_only_validated_runtime_python_alias(self) -> None:
        validation = '[[ -x "${python_bin}" && -f "${python_bin}" ]]'
        alias = 'export SAIC_T2V_PYTHON_BIN="${python_bin}"'
        materialize = '"${python_bin}" -B "${contract_runtime}" materialize-proxies'
        self.assertIn(alias, self.source)
        self.assertLess(self.source.index(validation), self.source.index(alias))
        self.assertLess(self.source.index(alias), self.source.index(materialize))
        self.assertNotIn(
            'export SAIC_T2V_PYTHON_BIN="${SAIC_T2V_V2_PYTHON_BIN}"',
            self.source,
        )

    def test_archive_and_both_specs_are_immutable_bound(self) -> None:
        self.assertIn("git get-tar-commit-id", self.source)
        self.assertIn("source_archive_sha256", self.source)
        self.assertIn("saic_pure_t2v_event_bank_v1.json", self.source)
        self.assertIn("saic_pure_t2v_event_bank_topup_v2.json", self.source)
        self.assertIn("sealed-base-saic-t2v-event-v1-spec.json", self.source)
        self.assertIn("--base-v1-spec", self.source)
        self.assertGreaterEqual(self.source.count("--root-spec"), 2)
        self.assertGreaterEqual(self.source.count("--source-manifest"), 4)
        self.assertIn(
            "generate_saic_pure_t2v_event_bank_topup_v2.py", self.source
        )
        self.assertIn("audit-bank", self.source)
        self.assertIn(
            "saic-pure-t2v-event-bank-topup-receipt.json", self.source
        )


if __name__ == "__main__":
    unittest.main()
