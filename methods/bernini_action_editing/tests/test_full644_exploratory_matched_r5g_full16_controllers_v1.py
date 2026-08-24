from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = MODULE_ROOT / "scripts"

STATIC = SCRIPT_ROOT / (
    "auh_gate_full644_exploratory_matched_r5g_full16_static_"
    "job143812_node293_once_v1.sh"
)
BOOTSTRAP = SCRIPT_ROOT / (
    "auh_gate_full644_exploratory_matched_r5g_full16_root_bootstrap_"
    "job143812_node293_once_v1.sh"
)
CONSUMPTION = SCRIPT_ROOT / (
    "auh_gate_full644_exploratory_matched_r5g_full16_cpu_consumption_"
    "job143812_node293_once_v1.sh"
)
GPU_V1 = SCRIPT_ROOT / (
    "auh_launch_full644_exploratory_matched_r5g_full16_"
    "job143812_node293_once_v1.sh"
)
GPU_V2 = SCRIPT_ROOT / (
    "auh_launch_full644_exploratory_matched_r5g_full16_"
    "job143812_node293_once_v2.sh"
)

EXPECTED_GATE_PINS = {
    "static_receipt": (
        "4403602ae1e3e7c50f5c22fdc2c397b5a16e78358ce7c84f88f4f759fa886c43",
        "5b1e08d7814b26afd98a475067e938fe56145e616d21103893e558b7f08c015e",
    ),
    "static_evidence": (
        "1ba054cdd23c638d09ceca1923be84bcfdcf54ee09f1c7fcc91550087cd5d3b1",
        "63938e0edb3fa273964ac7ba5b8b73b0fc97a60ee38e39c5bc4ac245fd8e1b05",
    ),
    "bootstrap_receipt": (
        "69b4a8bb4e7d66224560f1df911745770519f2f7209d3c7f4a4e075e9f32e3b0",
        "74507c60534016a0d998b48c6ab85650b338ab3abfb1763cf478154a20a91f58",
    ),
    "bootstrap_evidence": (
        "4b3dad51a6d73221a6ad9952543d7b04c418625eb83a509d5c119b751b5d4869",
        "5fb90052c58ac1a6c0b3ada6ea46895cd1ec900d5d35ca5121fa862fe1698f51",
    ),
    "consumption_receipt": (
        "9829c803e25c4ac0d85db3c915789c4d20de2855b957caa9c2484f9bab441936",
        "ca5be8045b753ede8d2481e07792f9e67ab09c5f202476cd0e3d4aa17cedd824",
    ),
    "consumption_evidence": (
        "e061be50f6f92c0d5f4795fc4ee5c5ded3f1438c6c3bd93525e57d199165d76c",
        "2570f96d70171843c67d4d3d5cab34a7d4195e430419e903e34abef61bc2272e",
    ),
}


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def embedded_python(path: Path) -> str:
    value = source(path)
    return value.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def assignment_literal(code: str, name: str):
    tree = ast.parse(code)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment absent: {name}")


class R5GFull16ControllerContractTests(unittest.TestCase):
    def test_every_controller_is_bash_valid_and_embedded_python_compiles(self) -> None:
        for path in (STATIC, BOOTSTRAP, CONSUMPTION, GPU_V2):
            with self.subTest(path=path.name):
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                code = embedded_python(path)
                compile(code, str(path) + ":embedded", "exec", optimize=0)
                compile(code, str(path) + ":embedded-O", "exec", optimize=1)

    def test_cpu_controllers_are_ready_full16_and_have_no_placeholders(self) -> None:
        expected = {
            STATIC: ("f644-r5g-full16-static-v1", "R5G_FULL16_STATIC_NOMODEL_PASS"),
            BOOTSTRAP: (
                "f644-r5g-full16-bootstrap-v1",
                "R5G_FULL16_ROOT_BOOTSTRAP_CPU_PROBE_PASS",
            ),
            CONSUMPTION: (
                "f644-r5g-full16-consume-v1",
                "R5G_FULL16_CPU_CONSUMPTION_HELDFD_GATE_PASS",
            ),
        }
        for path, tokens in expected.items():
            with self.subTest(path=path.name):
                value = source(path)
                self.assertIn("readonly R5G_CONTROLLER_STATE=READY", value)
                self.assertIn("full16-production", value)
                self.assertNotIn("__R5G_", value)
                self.assertNotIn("case00-pair-canary", value)
                for token in tokens:
                    self.assertIn(token, value)

    def test_old_gpu_controller_is_unconditionally_held(self) -> None:
        value = source(GPU_V1)
        hold = value.index("HOLD_SUPERSEDED_BY_V2")
        refusal = value.index("GPU v1 is HOLD")
        srun = value.index("exec /usr/bin/srun")
        self.assertLess(hold, refusal)
        self.assertLess(refusal, srun)
        self.assertNotIn("readonly R5G_GPU_CONTROLLER_STATE=READY", value)

    def test_gpu_v2_pins_all_six_gate_objects_exactly(self) -> None:
        value = source(GPU_V2)
        code = embedded_python(GPU_V2)
        pins = assignment_literal(code, "GATE_PINS")
        self.assertEqual(set(pins), set(EXPECTED_GATE_PINS))
        for key, (file_sha, object_digest) in EXPECTED_GATE_PINS.items():
            with self.subTest(key=key):
                self.assertEqual(pins[key]["sha256"], file_sha)
                digest_key = "receipt_digest" if key.endswith("receipt") else "evidence_digest"
                self.assertEqual(pins[key][digest_key], object_digest)
        self.assertIn("readonly R5G_GPU_CONTROLLER_STATE=READY", value)
        self.assertNotIn("__R5G_", value)

    def test_gpu_v2_requires_gate_closure_and_commits_attempt_before_only_srun(self) -> None:
        value = source(GPU_V2)
        self.assertNotIn('(\"evidence\", \"evidence\")', value)
        for token in (
            "r5g_full16_static_nomodel_probe.sacct-and-replay.json",
            "r5g_full16_root_bootstrap_cpu_probe.sacct-and-replay.json",
            "r5g_full16_cpu_consumption_probe.sacct-and-replay.json",
            "cpu_consumption_probe_work_r1",
            "r5g_full16_gpu_attempt_v2.json",
            "os.O_EXCL",
            "ATTEMPT_CLAIMED_BEFORE_SRUN",
            "retry_allowed\": False",
        ):
            self.assertIn(token, value)
        self.assertEqual(value.count("exec /usr/bin/srun"), 1)
        self.assertLess(value.index("create_attempt_marker("), value.index("exec /usr/bin/srun"))
        self.assertIn('--exclusive --exact --immediate=10', value)
        self.assertIn('--export=NONE --time=03:00:00', value)
        self.assertIn('10#$SLURM_STEP_ID > 211', value)


if __name__ == "__main__":
    unittest.main()
