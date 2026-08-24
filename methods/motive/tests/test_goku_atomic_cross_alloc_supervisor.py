from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from motive import goku_atomic_cross_alloc_supervisor as supervisor


INITIAL_JOBS = [135096, 135151, 135152, 135153, 135154, 135155, 135156, 135157]


class CrossAllocationPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.epochs = [
            supervisor.Epoch(
                index=index,
                target=128 if index < 8 else 104,
                lane=(index - 1) % 2,
                predecessor=index - 2 if index > 2 else None,
                initial_job_id=INITIAL_JOBS[index - 1],
                run_root=self.root / f"run_{index:04d}",
                selected=self.root / f"selected_{index:04d}.jsonl",
                selected_sha256="a" * 64,
            )
            for index in range(1, 9)
        ]
        self.config = {
            "global_target": 1000,
            "merge_job_id": 135161,
            "recovery_root": str(self.root / "recovery"),
        }
        self.state = {
            "sequence": 0,
            "attempts": {
                str(epoch.index): {
                    "attempt": 0,
                    "job_id": epoch.initial_job_id,
                    "run_root": str(epoch.run_root),
                    "selected_manifest": str(epoch.selected),
                    "selected_manifest_sha256": epoch.selected_sha256,
                }
                for epoch in self.epochs
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def observations(self) -> dict[int, dict[str, object]]:
        result = {}
        for epoch in self.epochs:
            state = "RUNNING" if epoch.index in (1, 2) else "PENDING"
            dependency = (
                f"afterok:{INITIAL_JOBS[epoch.index - 3]}"
                if epoch.predecessor is not None
                else None
            )
            result[epoch.initial_job_id] = {
                "job_id": epoch.initial_job_id,
                "state": state,
                "reason": "",
                "dependency": dependency,
            }
        return result

    @mock.patch.object(supervisor, "validate_epoch_complete", return_value=False)
    def test_live_two_lane_jobs_are_read_only(self, _validate: mock.Mock) -> None:
        before = sorted(self.root.rglob("*"))
        plan = supervisor.build_plan(
            self.config, self.epochs, self.state, self.observations()
        )
        self.assertEqual(plan["action"], {"kind": "none", "reason": "all epochs active or complete"})
        self.assertFalse(plan["mutations_performed"])
        self.assertEqual(sorted(self.root.rglob("*")), before)
        self.assertEqual([(e.lane, e.predecessor) for e in self.epochs], [
            (0, None), (1, None), (0, 1), (1, 2),
            (0, 3), (1, 4), (0, 5), (1, 6),
        ])

    @mock.patch.object(supervisor, "validate_epoch_complete", return_value=False)
    def test_terminal_epoch_produces_fresh_attempt_plan(self, _validate: mock.Mock) -> None:
        observations = self.observations()
        observations[135096]["state"] = "TIMEOUT"
        plan = supervisor.build_plan(self.config, self.epochs, self.state, observations)
        action = plan["action"]
        self.assertEqual(action["kind"], "prepare_submit_retry")
        self.assertEqual(action["epoch_index"], 1)
        self.assertEqual(action["failed_job_id"], 135096)
        self.assertTrue(action["same_run_root_resume_forbidden"])
        self.assertEqual(action["recovery_semantics"], "fresh_job_fresh_run_root")
        self.assertIn("attempt_001", action["new_attempt_root"])
        self.assertTrue(action["downstream_dependency_rebind_required"])
        self.assertTrue(action["merge_redeployment_required"])
        self.assertFalse(action["execution_enabled"])

    @mock.patch.object(supervisor, "validate_epoch_complete")
    def test_hash_validated_complete_epoch_is_never_retried(
        self, validate: mock.Mock
    ) -> None:
        validate.side_effect = lambda epoch, _root: epoch.index == 1
        observations = self.observations()
        observations[135096]["state"] = "TIMEOUT"
        plan = supervisor.build_plan(self.config, self.epochs, self.state, observations)
        self.assertEqual(plan["action"]["kind"], "none")
        first = plan["epochs"][0]
        self.assertTrue(first["complete"])
        self.assertEqual(first["slurm_state"], "TIMEOUT")

    @mock.patch.object(supervisor, "validate_epoch_complete", return_value=False)
    def test_replacement_requires_pending_direct_dependent_rebind(
        self, _validate: mock.Mock
    ) -> None:
        replacement_job = 200001
        epoch1 = self.state["attempts"]["1"]
        epoch1.update(
            attempt=1,
            job_id=replacement_job,
            run_root=str(self.root / "recovery/epoch_0001/attempt_001/run"),
        )
        observations = self.observations()
        observations.pop(135096)
        observations[replacement_job] = {
            "job_id": replacement_job,
            "state": "RUNNING",
            "reason": "",
            "dependency": None,
        }
        plan = supervisor.build_plan(self.config, self.epochs, self.state, observations)
        action = plan["action"]
        self.assertEqual(action["kind"], "rebind_dependency")
        self.assertEqual(action["job_id"], 135152)
        self.assertEqual(action["current_dependency"], "afterok:135096")
        self.assertEqual(action["desired_dependency"], "afterok:200001")
        self.assertFalse(action["execution_enabled"])

    @mock.patch.object(supervisor, "validate_epoch_complete", return_value=False)
    def test_already_rebound_dependency_needs_no_action(self, _validate: mock.Mock) -> None:
        replacement_job = 200001
        self.state["attempts"]["1"].update(
            attempt=1,
            job_id=replacement_job,
            run_root=str(self.root / "recovery/epoch_0001/attempt_001/run"),
        )
        observations = self.observations()
        observations.pop(135096)
        observations[replacement_job] = {
            "job_id": replacement_job,
            "state": "RUNNING",
            "reason": "",
            "dependency": None,
        }
        observations[135152]["dependency"] = "afterok:200001"
        plan = supervisor.build_plan(self.config, self.epochs, self.state, observations)
        self.assertEqual(plan["action"]["kind"], "none")

    def test_cli_has_no_apply_switch(self) -> None:
        parser = supervisor.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-once", "--config", "/tmp/config.json", "--apply"])


if __name__ == "__main__":
    unittest.main()
