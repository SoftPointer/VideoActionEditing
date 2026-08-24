#!/usr/bin/env python3
"""Byte-exact admission tests for the case01 exact5 READY GPU controller."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import subprocess
import unittest


METHOD = Path(__file__).resolve().parents[1]
HOLD = (
    METHOD / "scripts/"
    "auh_launch_case01_source_bone_exact5_r64_gpu_job143808_node292_once_v1.HOLD.sh"
)
READY = (
    METHOD / "scripts/"
    "auh_launch_case01_source_bone_exact5_r64_gpu_job143808_node292_once_v2.READY.sh"
)
HOLD_SHA256 = "7553b9c2a079421176bd5e851a57a44b675fa94505269dca77a8a2827dabe179"
HOLD_SIZE = 41_977
READY_SHA256 = "39c3a99c24f1988d408ed864b8815accb72453bf2f9a5ddb98f4ab0b01fff308"
READY_SIZE = 42_375

REPLACEMENTS = (
    (
        "readonly CONTROLLER_STATE=HOLD_PENDING_EXACT_CPU_GATE_PINS",
        "readonly CONTROLLER_STATE=READY",
    ),
    (
        '"sha256": "HOLD_PACKAGE_RECEIPT_SHA256"',
        '"sha256": "0561608208e5a155028d4f8ec876b91a096189e7bd16bf71b8c72ee609e0433b"',
    ),
    (
        '"receipt_digest": "HOLD_PACKAGE_RECEIPT_DIGEST"',
        '"receipt_digest": "2c43394fc759fc6f71ea9f7f1058adb3b3d1944158e77fe20296e49547e26738"',
    ),
    (
        '"sha256": "HOLD_PRODUCTION_LAUNCH_RECEIPT_SHA256"',
        '"sha256": "50b3c6a7f5c637d113808e6c18e444e5908fa5043ac03f6e6b4936f419bf5c69"',
    ),
    (
        '"receipt_digest": "HOLD_PRODUCTION_LAUNCH_RECEIPT_DIGEST"',
        '"receipt_digest": "a031593969df468e39d5c2defa6a6a12dcd7b9f173b3033bf306f71740242404"',
    ),
    (
        '"sha256": "HOLD_PRODUCTION_PAYLOAD_SHA256"',
        '"sha256": "07dda24d944ec27bf32b93db54079360d5d5067193ed3031b796490bc712271a"',
    ),
    ('"size": "HOLD_PRODUCTION_PAYLOAD_SIZE"', '"size": 29204'),
    (
        '"sha256": "HOLD_STATIC_RECEIPT_SHA256"',
        '"sha256": "b435fb39c481ac34732e754532f34b7e6c2eb679cf4b44352b50d1e52f3908cc"',
    ),
    (
        '"receipt_digest": "HOLD_STATIC_RECEIPT_DIGEST"',
        '"receipt_digest": "4797313bb637e22047195604e0e802dc8c98cd87bc126833f44e8512b7fc00a0"',
    ),
    (
        '"sha256": "HOLD_STATIC_EVIDENCE_SHA256"',
        '"sha256": "4e6274e58c5831f49d1a0dfc7c87ac0556e59d78f20c9ebf732eefc630ce4cf8"',
    ),
    (
        '"evidence_digest": "HOLD_STATIC_EVIDENCE_DIGEST"',
        '"evidence_digest": "6b5a5abdfb8efb32f5f362ea03f1910470411d751e11e011665c86e7fe3f5042"',
    ),
    (
        '"sha256": "HOLD_ROOT_FAKE_RECEIPT_SHA256"',
        '"sha256": "611dffed84a645415666ab36719dd7d0ace63571d4b68ae8536a67cf7301a02c"',
    ),
    (
        '"receipt_digest": "HOLD_ROOT_FAKE_RECEIPT_DIGEST"',
        '"receipt_digest": "a584a61f8d9add3fd1704eb92b689f206981c9fdd633eb4e981bf2179d0b90d1"',
    ),
    (
        '"sha256": "HOLD_ROOT_FAKE_EVIDENCE_SHA256"',
        '"sha256": "ae8f5f15bd3b44ada2fb44287f2c07bdcdc115937e702c985178cf3e65cda7f8"',
    ),
    (
        '"evidence_digest": "HOLD_ROOT_FAKE_EVIDENCE_DIGEST"',
        '"evidence_digest": "ad5ae788eda8d97640f6eedb60d58bbdf442efcfdf2f2c2364ee9d083bf2589e"',
    ),
)

EXPECTED_PINS = {
    "package": {
        "sha256": "0561608208e5a155028d4f8ec876b91a096189e7bd16bf71b8c72ee609e0433b",
        "receipt_digest": "2c43394fc759fc6f71ea9f7f1058adb3b3d1944158e77fe20296e49547e26738",
    },
    "launch_receipt": {
        "sha256": "50b3c6a7f5c637d113808e6c18e444e5908fa5043ac03f6e6b4936f419bf5c69",
        "receipt_digest": "a031593969df468e39d5c2defa6a6a12dcd7b9f173b3033bf306f71740242404",
    },
    "payload": {
        "sha256": "07dda24d944ec27bf32b93db54079360d5d5067193ed3031b796490bc712271a",
        "size": 29_204,
    },
    "static_receipt": {
        "sha256": "b435fb39c481ac34732e754532f34b7e6c2eb679cf4b44352b50d1e52f3908cc",
        "receipt_digest": "4797313bb637e22047195604e0e802dc8c98cd87bc126833f44e8512b7fc00a0",
    },
    "static_evidence": {
        "sha256": "4e6274e58c5831f49d1a0dfc7c87ac0556e59d78f20c9ebf732eefc630ce4cf8",
        "evidence_digest": "6b5a5abdfb8efb32f5f362ea03f1910470411d751e11e011665c86e7fe3f5042",
    },
    "root_fake_receipt": {
        "sha256": "611dffed84a645415666ab36719dd7d0ace63571d4b68ae8536a67cf7301a02c",
        "receipt_digest": "a584a61f8d9add3fd1704eb92b689f206981c9fdd633eb4e981bf2179d0b90d1",
    },
    "root_fake_evidence": {
        "sha256": "ae8f5f15bd3b44ada2fb44287f2c07bdcdc115937e702c985178cf3e65cda7f8",
        "evidence_digest": "ad5ae788eda8d97640f6eedb60d58bbdf442efcfdf2f2c2364ee9d083bf2589e",
    },
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def heredocs(source: str) -> list[str]:
    return [part.split("\nPY", 1)[0] for part in source.split("<<'PY'\n")[1:]]


def literal_assignment(source: str, name: str):
    tree = ast.parse(heredocs(source)[0])
    rows = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(rows) != 1:
        raise AssertionError(f"{name} assignment closure differs")
    return ast.literal_eval(rows[0])


class ReadyControllerTests(unittest.TestCase):
    def test_hold_bytes_are_preserved_and_still_fail_before_effects(self) -> None:
        raw = HOLD.read_bytes()
        self.assertEqual((sha256(raw), len(raw)), (HOLD_SHA256, HOLD_SIZE))
        completed = subprocess.run(
            ["/bin/bash", "-p", "-s"], input=raw,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 88)
        self.assertIn(b"HOLD pending exact CPU receipt/evidence pins", completed.stderr)

    def test_ready_is_only_the_state_and_exact14_pin_transform(self) -> None:
        hold = HOLD.read_text("utf-8")
        ready = READY.read_text("utf-8")
        self.assertEqual(len(REPLACEMENTS), 15)
        transformed = hold
        for before, after in REPLACEMENTS:
            self.assertEqual(transformed.count(before), 1, before)
            transformed = transformed.replace(before, after)
        self.assertEqual(ready, transformed)
        self.assertEqual(
            (sha256(READY.read_bytes()), len(READY.read_bytes())),
            (READY_SHA256, READY_SIZE),
        )

    def test_ready_has_no_unresolved_hold_placeholder_and_exact_pins(self) -> None:
        source = READY.read_text("utf-8")
        self.assertIn("readonly CONTROLLER_STATE=READY\n", source)
        self.assertNotIn("HOLD_PENDING_EXACT_CPU_GATE_PINS", source)
        self.assertEqual(re.findall(r'"HOLD_[A-Z0-9_]+"', source), [])
        for before, _ in REPLACEMENTS[1:]:
            self.assertNotIn(before, source)
        self.assertEqual(literal_assignment(source, "PINS"), EXPECTED_PINS)
        self.assertEqual(sum(len(row) for row in EXPECTED_PINS.values()), 14)

    def test_shell_embedded_python_and_one_shot_order(self) -> None:
        for path in (HOLD, READY):
            completed = subprocess.run(
                ["/bin/bash", "-n", str(path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            blocks = heredocs(path.read_text("utf-8"))
            self.assertEqual(len(blocks), 2)
            for optimize in (0, 2):
                for index, block in enumerate(blocks):
                    compile(block, f"{path.name}:heredoc{index}", "exec", optimize=optimize)

        source = READY.read_text("utf-8")
        srun = source.index("/usr/bin/srun")
        self.assertEqual(source.count("/usr/bin/srun"), 1)
        self.assertLess(source.index('"status": "ATTEMPT_CLAIMED_BEFORE_SRUN"'), srun)
        self.assertIn("print(max(gate_rows))", source)
        self.assertIn('(( 10#$SLURM_STEP_ID > 10#$1 ))', source)
        self.assertIn('"$MAX_GATE_STEP" <&"$PAYLOAD_FD"', source)
        self.assertNotIn("MAX_GATE_STEP=432", source)


if __name__ == "__main__":
    unittest.main()
