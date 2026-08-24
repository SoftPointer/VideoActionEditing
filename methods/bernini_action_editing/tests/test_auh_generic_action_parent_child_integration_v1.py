from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_train_generic_source_anchored_action_world4_holder_v1.sh"
)
FIXED_SOURCE_SHA = (
    "128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d"
)
FIXED_CHECKPOINT_MANIFEST_SHA = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)


class GenericActionParentChildIntegrationTests(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_executable(path: Path, source: str) -> None:
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        path.chmod(0o700)

    def _run(self, *, late_busy: bool) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory(prefix="gsa-parent-child-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        prefix = root / "vast" / "users" / "guangyi.chen"
        method_root = prefix / "release" / "methods" / "bernini_action_editing"
        scripts = method_root / "scripts"
        inputs = prefix / "inputs"
        mock_bin = root / "mock-bin"
        run_parent = prefix / "runs"
        for path in (scripts, inputs, mock_bin, run_parent):
            path.mkdir(parents=True, exist_ok=True)

        source_manifest = inputs / "source.json"
        # This fixed launcher input is not subject to the /vast path allowlist;
        # keeping it outside the rewritten prefix avoids a second text rewrite.
        checkpoint_manifest = root / "checkpoint.sha256"
        authority_plan = inputs / "smoke.plan.json"
        method_archive = inputs / "source.tar"
        method_manifest = inputs / "source.manifest.json"
        for path, raw in (
            (source_manifest, b"source\n"),
            (checkpoint_manifest, b"checkpoint\n"),
            (authority_plan, b"plan\n"),
            (method_archive, b"archive\n"),
            (method_manifest, b"manifest\n"),
            (method_root / "train_generic_source_anchored_action_v1.py", b"trainer\n"),
            (method_root / "generic_source_anchored_action_v1.py", b"core\n"),
            (method_root / "generic_source_anchored_action_pair_controller_v1.py", b"controller\n"),
            (scripts / "auh_generic_source_anchored_action_rank_exec_v1.sh", b"#!/bin/bash\n"),
        ):
            path.write_bytes(raw)

        source = LAUNCHER.read_text(encoding="utf-8")
        source = source.replace(
            "readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256",
            f"readonly checkpoint_manifest={checkpoint_manifest}",
        )
        source = source.replace(
            f"readonly expected_checkpoint_manifest_sha={FIXED_CHECKPOINT_MANIFEST_SHA}",
            f"readonly expected_checkpoint_manifest_sha={self._sha(checkpoint_manifest)}",
        )
        source = source.replace(
            f"readonly source_authority_sha={FIXED_SOURCE_SHA}",
            f"readonly source_authority_sha={self._sha(source_manifest)}",
        )
        source = source.replace("/vast/users/guangyi.chen", str(prefix))
        launcher = scripts / LAUNCHER.name
        launcher.write_text(source, encoding="utf-8")
        launcher.chmod(0o700)

        trace = root / "trace.log"
        squeue_count = root / "squeue.count"
        python_bin = prefix / "bin" / "python3"
        python_bin.parent.mkdir(parents=True)
        self._write_executable(
            python_bin,
            """\
            #!/bin/bash
            printf 'python:%s\n' "$*" >>"${GSA_TEST_TRACE}"
            previous=
            output=
            for argument in "$@"; do
              if [[ "${previous}" == --output ]]; then output="${argument}"; fi
              previous="${argument}"
            done
            if [[ -n "${output}" ]]; then
              mkdir -p "${output}"
              printf '{}\n' >"${output}/run_receipt.json"
              printf 'trainer-receipt\n' >>"${GSA_TEST_TRACE}"
            fi
            exit 0
            """,
        )
        mocks = {
            "sha256sum": """\
                #!/bin/bash
                exec /usr/bin/shasum -a 256 "$@"
            """,
            "realpath": """\
                #!/bin/bash
                [[ "${1:-}" == -m ]] && shift
                [[ "${1:-}" == -- ]] && shift
                printf '%s\n' "$1"
            """,
            "scontrol": """\
                #!/bin/bash
                echo 'JobId=136309 JobState=RUNNING UserId=guangyi.chen(1) NodeList=auh7-1b-gpu-280'
            """,
            "squeue": """\
                #!/bin/bash
                count=0
                [[ -f "${GSA_TEST_SQUEUE_COUNT}" ]] && count="$(cat "${GSA_TEST_SQUEUE_COUNT}")"
                count=$((count + 1))
                printf '%s\n' "${count}" >"${GSA_TEST_SQUEUE_COUNT}"
                if [[ -n "${SLURM_STEP_ID:-}" ]]; then
                  printf '%s.%s\n' "${SLURM_JOB_ID}" "${SLURM_STEP_ID}"
                  printf 'squeue-child-self\n' >>"${GSA_TEST_TRACE}"
                elif [[ "${GSA_TEST_LATE_BUSY:-0}" == 1 && "${count}" == 2 ]]; then
                  printf '136309.9\n'
                  printf 'squeue-parent-late-busy\n' >>"${GSA_TEST_TRACE}"
                else
                  printf 'squeue-parent-empty\n' >>"${GSA_TEST_TRACE}"
                fi
            """,
            "ssh": """\
                #!/bin/bash
                remote=
                for argument in "$@"; do remote="${argument}"; done
                if [[ "${remote}" == *showpids* ]]; then
                  for gpu in 0 1 2 3 4 5 6 7; do
                    echo 'GPU use (%): 0'
                    echo 'GPU Memory Allocated (VRAM%): 0'
                  done
                elif [[ "${remote}" == *showtopo* ]]; then
                  for gpu in 0 1 2 3 4 5 6 7; do
                    echo "GPU${gpu} XGMI XGMI XGMI PCIE PCIE PCIE PCIE"
                  done
                  for gpu in 0 1 2 3; do echo "GPU[${gpu}] Topology Numa Node: 0"; done
                  for gpu in 4 5 6 7; do echo "GPU[${gpu}] Topology Numa Node: 1"; done
                fi
            """,
            "sleep": """\
                #!/bin/bash
                exit 0
            """,
            "hostname": """\
                #!/bin/bash
                echo auh7-1b-gpu-280
            """,
            "rocm-smi": """\
                #!/bin/bash
                for gpu in 0 1 2 3; do echo 'GPU use (%): 0'; done
            """,
            "srun": """\
                #!/bin/bash
                printf 'srun:%s\n' "$*" >>"${GSA_TEST_TRACE}"
                [[ " $* " == *' --immediate=5 '* ]] || exit 91
                while [[ $# -gt 0 && "$1" != env ]]; do shift; done
                [[ "${1:-}" == env ]] || exit 92
                export SLURM_JOB_ID=136309 SLURM_STEP_ID=4 SLURM_STEP_GPUS=0,1,2,3
                printf 'srun-child-exec\n' >>"${GSA_TEST_TRACE}"
                exec "$@"
            """,
        }
        for name, body in mocks.items():
            self._write_executable(mock_bin / name, body)

        run_root = run_parent / "smoke-r"
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{mock_bin}:{environment['PATH']}",
                "GSA_TEST_TRACE": str(trace),
                "GSA_TEST_SQUEUE_COUNT": str(squeue_count),
                "GSA_TEST_LATE_BUSY": "1" if late_busy else "0",
                "GSA_CONFIRM_CHILD": "launch-approved-generic-pair-136309-136141",
                "GSA_ARM_ID": "smoke_r",
                "GSA_HOLDER_JOB": "136309",
                "GSA_HOLDER_NODE": "auh7-1b-gpu-280",
                "GSA_EXECUTION_PROFILE": "smoke-r",
                "GSA_CARRIER_POLICY": "installed_trainable_disposable",
                "GSA_RUN_ROOT": str(run_root),
                "GSA_MASTER_PORT": "33609",
                "GSA_AUTHORITY_PLAN": str(authority_plan),
                "GSA_AUTHORITY_PLAN_SHA256": self._sha(authority_plan),
                "GSA_METHOD_ROOT": str(method_root),
                "GSA_TRAINER_SHA256": self._sha(method_root / "train_generic_source_anchored_action_v1.py"),
                "GSA_CORE_SHA256": self._sha(method_root / "generic_source_anchored_action_v1.py"),
                "GSA_LAUNCHER_SHA256": self._sha(launcher),
                "GSA_METHOD_ARCHIVE": str(method_archive),
                "GSA_METHOD_ARCHIVE_SHA256": self._sha(method_archive),
                "GSA_METHOD_MANIFEST": str(method_manifest),
                "GSA_METHOD_MANIFEST_SHA256": self._sha(method_manifest),
                "GSA_SOURCE_MANIFEST": str(source_manifest),
                "GSA_SOURCE_MANIFEST_SHA256": self._sha(source_manifest),
                "GSA_PYTHON_BIN": str(python_bin),
            }
        )
        for optional in (
            "GSA_MANIFEST_VALIDATOR_SHA256",
            "GSA_REPRESENTATION_MANIFEST",
            "GSA_REPRESENTATION_MANIFEST_SHA256",
            "GSA_SOURCE_PAIR_MANIFEST",
            "GSA_SOURCE_PAIR_MANIFEST_SHA256",
            "GSA_RESUME_CHECKPOINT",
            "GSA_RESUME_CHECKPOINT_SHA256",
            "GSA_RESUME_RECEIPT",
            "GSA_RESUME_RECEIPT_SHA256",
        ):
            environment.pop(optional, None)
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=str(root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
            check=False,
        )
        return result, run_root, trace

    def test_parent_recurses_through_srun_child_and_late_busy_is_create_only(self) -> None:
        success, run_root, trace_path = self._run(late_busy=False)
        trace = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
        self.assertEqual(success.returncode, 0, success.stdout + trace)
        self.assertIn("srun-child-exec", trace)
        self.assertIn("squeue-child-self", trace)
        self.assertIn("trainer-receipt", trace)
        self.assertTrue((run_root / "controller.SMOKE_COMPLETE").is_file())

        busy, busy_root, busy_trace_path = self._run(late_busy=True)
        busy_trace = (
            busy_trace_path.read_text(encoding="utf-8")
            if busy_trace_path.exists()
            else ""
        )
        self.assertEqual(busy.returncode, 2, busy.stdout + busy_trace)
        self.assertIn("holder acquired a numbered child before run-root creation", busy.stdout)
        self.assertNotIn("srun:", busy_trace)
        self.assertFalse(busy_root.exists())


if __name__ == "__main__":
    unittest.main()
