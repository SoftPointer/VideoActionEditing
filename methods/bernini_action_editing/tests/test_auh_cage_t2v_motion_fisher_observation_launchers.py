from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOPUP = METHOD_ROOT / "scripts/auh_generate_cage_motion_fisher_action_topup_dual4.sbatch"
PROBE = METHOD_ROOT / "scripts/auh_probe_cage_t2v_motion_fisher_observations_dual4.sbatch"


class AUHMotionFisherLauncherTests(unittest.TestCase):
    def test_action_topup_uses_all_eight_gpus_and_only_four_actions(self) -> None:
        text = TOPUP.read_text()
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn('run_group sp4-a "0,1,2,3"', text)
        self.assertIn('run_group sp4-b "4,5,6,7"', text)
        self.assertIn("materialize-topup-plan", text)
        self.assertIn("audit-topup", text)
        self.assertIn("candidates=4 exact81=true steps=40 event_audit_pending=true", text)
        self.assertNotIn("--audit-bank", text)
        self.assertNotIn("optimizer.step", text)

    def test_observation_probe_preflights_events_before_dual_sp4(self) -> None:
        text = PROBE.read_text()
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        first_preflight = text.index('"${probe}" preflight')
        first_launch = text.index("run_probe sp4-a")
        self.assertLess(first_preflight, first_launch)
        self.assertIn("--event-index", text)
        self.assertIn("--topup-output-dir", text)
        self.assertIn('run_probe sp4-a "0,1,2,3"', text)
        self.assertIn('run_probe sp4-b "4,5,6,7"', text)
        self.assertIn("clips=8 views=6 blocks=30 exact81=true optimizer=false", text)
        self.assertIn("frozen_flow_log_ratio_to_same_state_noop_negative_control_v1", text)
        self.assertIn("frozen_text_conditioned_temporal_event_critic_raw_score_vjp_v1", text)
        self.assertNotIn("sbatch ", text)
        self.assertNotIn("optimizer.step", text)


if __name__ == "__main__":
    unittest.main()

