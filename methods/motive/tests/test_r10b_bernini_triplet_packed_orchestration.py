from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_r10b_bernini_triplet_packed.sbatch"
)
AUTHORIZATION = {
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tree_record(root: Path, *, excluded: set[str] | None = None) -> dict:
    excluded = excluded or set()
    rows = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        size = path.stat().st_size
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
        total_bytes += size
    encoded = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return {
        "path": str(root),
        "files": len(rows),
        "bytes": total_bytes,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_pretty(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class R10BBerniniTripletPackedOrchestrationTests(unittest.TestCase):
    def test_resources_contract_and_closed_scope_are_explicit(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --partition=faculty",
            "#SBATCH --account=test-acc",
            "#SBATCH --qos=bgqos",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=3",
            "#SBATCH --ntasks-per-node=3",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=384G",
            "#SBATCH --gres=gpu:mi210:3",
            "#SBATCH --time=08:00:00",
            "#SBATCH --job-name=motive-r10b-bernini-triplet",
        ):
            self.assertIn(directive, text)

        for marker in (
            "MOTIVE_R10B_SOURCE_SNAPSHOT",
            "MOTIVE_R10B_SOURCE_TREE_SHA256",
            "MOTIVE_R10B_QWEN_PACKED_ROOT",
            "MOTIVE_R10B_QWEN_PACKED_COMMIT_DIGEST",
            "MOTIVE_R10B_TRACK_CACHE",
            "MOTIVE_R10B_TRACK_CACHE_SHA256",
            "MOTIVE_R10B_MODEL_PATH",
            "MOTIVE_R10B_MODEL_REVISION",
            "MOTIVE_R10B_MODEL_TREE_SHA256",
            "MOTIVE_R10B_BERNINI_REPO",
            "MOTIVE_R10B_BERNINI_SOURCE_COMMIT",
            "MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256",
            "MOTIVE_R10B_OUTPUT_ROOT",
            'canonical_manifest="${pilot_dir}/manifest.jsonl"',
            'original_manifest="${variant_dir}/original.jsonl"',
            (
                'cross_family_manifest="${variant_dir}/'
                'cross_family_shuffle.jsonl"'
            ),
            '"canonical"',
            '"original"',
            '"cross_family"',
            "--ntasks=3",
            "--ntasks-per-node=3",
            "--gpus-per-task=1",
            "--gpu-bind=single:1",
            "probe_bound_gpu",
            "torch.cuda.device_count() != 1",
            '"MI210" not in device_name.upper()',
            "-m motive.r10b_bernini_tangent_extract",
            "-m motive.r10b_bernini_retrieval_audit",
            'artifact_kind="controlled_retrieval_pilot"',
            'resize_mode="aspect_preserving_center_crop"',
            'num_frames="17"',
            'scheduler_steps="50"',
            'projection_seed_1="260108851"',
            'projection_seed_2="260108852"',
            'verify_bound_inputs "before"',
            'verify_bound_inputs "after_tangent"',
            'verify_bound_inputs "after_retrieval"',
            "validate_controlled_pilot_commit",
            "validate_prompt_variants",
            "pipeline_summary.json",
            "pipeline_done.json",
            "output_tree_without_done",
            "prompt_variants_commit_digest",
            "missing_independent_dino_appearance_control_artifact",
            '"videos_decoded_for_measurement": measurement_video_decodes',
            '"videos_copied": 0',
            '"videos_rendered": 0',
            '"renderer_calls": 0',
            '"optimizer_steps": 0',
            '"representation_gate_passed": False',
            '"renderer_probe_authorized": False',
            '"generation_authorized": False',
            '"editor_training_authorized": False',
            '"training_authorized": False',
            '"downstream_submitted": False',
            "atomic_write_new(summary_path, summary_payload)",
            "atomic_write_new(done_path, done_payload)",
        ):
            self.assertIn(marker, text)

        self.assertEqual(
            text.count("-m motive.r10b_bernini_tangent_extract"),
            2,
        )
        self.assertEqual(
            text.count("-m motive.r10b_bernini_retrieval_audit"),
            2,
        )
        self.assertNotIn("--max-samples", text)
        self.assertNotIn("sbatch ", text)
        self.assertNotIn("ffmpeg", text)
        self.assertNotIn("rsync", text)
        self.assertNotIn("torchrun", text)
        self.assertNotIn("deepspeed", text)
        self.assertNotRegex(text, r"(?m)^\s*(?:cp|scp)\s")

    def test_script_and_embedded_python_have_valid_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY",
            SCRIPT.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        self.assertEqual(len(blocks), 5)
        for block in blocks:
            ast.parse(block)

    def _build_fixture(
        self,
        root: Path,
        *,
        balanced: bool,
    ) -> tuple[dict[str, str], dict[str, Path | str]]:
        snapshot = root / "snapshot"
        code_root = snapshot / "methods" / "motive"
        module_root = code_root / "motive"
        scripts_root = code_root / "scripts"
        qwen = root / "qwen-packed"
        track = root / "tracks.npz"
        model = root / "model"
        bernini = root / "bernini"
        output = root / "triplet-output"
        scratch = root / "scratch"
        fake_bin = root / "bin"
        fake_python = root / "fake-python"
        fake_srun = fake_bin / "srun"
        log = root / "python.log"
        srun_marker = root / "srun.marker"

        for directory in (
            module_root,
            scripts_root,
            model,
            bernini,
            scratch,
            fake_bin,
            qwen,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (snapshot / "SOURCE_FILES.jsonl").write_text(
            '{"fixture":"frozen"}\n',
            encoding="utf-8",
        )
        for name in (
            "attribution.py",
            "r10b_bernini_pilot_manifest.py",
            "r10b_bernini_prompt_variants.py",
            "r10b_bernini_retrieval_audit.py",
            "r10b_bernini_tangent_extract.py",
            "r10b_lucy_tangent_extract.py",
            "r10b_tangent_core.py",
        ):
            (module_root / name).write_text(
                "# frozen fixture\n",
                encoding="utf-8",
            )
        (scripts_root / "action_source_snapshot.py").write_text(
            "# frozen fixture\n",
            encoding="utf-8",
        )
        track.write_bytes(b"immutable-track-cache")

        component_names = (
            "gpu_probes",
            "shard_queues",
            "shard_audits",
            "merged_audit",
            "pilot_manifest",
            "prompt_variants",
        )
        for name in component_names:
            (qwen / name).mkdir()
        for name in (
            "gpu_probes",
            "shard_queues",
            "shard_audits",
            "merged_audit",
        ):
            (qwen / name / "fixture.json").write_text(
                '{"fixture":true}\n',
                encoding="utf-8",
            )

        pilot = qwen / "pilot_manifest"
        variants = qwen / "prompt_variants"
        pilot_lines = [
            json.dumps(
                {"iid": f"iid-{index:02d}", "prompt": "canonical"},
                sort_keys=True,
                separators=(",", ":"),
            )
            for index in range(20)
        ]
        (pilot / "manifest.jsonl").write_text(
            "\n".join(pilot_lines) + "\n",
            encoding="utf-8",
        )
        _write_pretty(
            pilot / "summary.json",
            {"balanced_pilot_ready": balanced},
        )
        _write_pretty(pilot / "shortfalls.json", {"shortfalls": []})
        _write_pretty(pilot / "done.json", {"status": "complete"})
        for filename, prompt in (
            ("original.jsonl", "original"),
            ("cross_family_shuffle.jsonl", "cross"),
        ):
            lines = [
                json.dumps(
                    {"iid": f"iid-{index:02d}", "prompt": prompt},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for index in range(20)
            ]
            (variants / filename).write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
        _write_pretty(variants / "summary.json", {"status": "complete"})
        _write_pretty(variants / "done.json", {"status": "complete"})

        controlled_digest = "d" * 64
        qwen_summary = {
            "schema_version": "motive-r10b-qwen-v2-packed-pipeline-v1",
            "status": "ready_for_commit",
            "inputs": {
                "source_snapshot": {
                    "path": str(snapshot),
                    "tree_sha256": "1" * 64,
                }
            },
            "execution": {
                "real_gpu_tasks": 8,
                "mi210_gpus": 8,
                "generation_error_rows": 0,
                "backend_execution": {
                    "mode": "production_local_qwen",
                    "cuda_only": True,
                },
            },
            "outputs": {
                name: _tree_record(qwen / name)
                for name in component_names
            },
            "controlled_pilot_commit_digest": controlled_digest,
            "balanced_pilot_ready": balanced,
            "prompt_variants_materialized": balanced,
            "videos_copied": 0,
            "videos_rendered": 0,
            "optimizer_steps": 0,
            "formal_evidence": False,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "generation_authorized": False,
            "training_authorized": False,
            "downstream_submitted": False,
            "authorization": dict(AUTHORIZATION),
        }
        _write_pretty(qwen / "pipeline_summary.json", qwen_summary)
        summary_raw = (qwen / "pipeline_summary.json").read_bytes()
        summary_record = {
            "bytes": len(summary_raw),
            "sha256": hashlib.sha256(summary_raw).hexdigest(),
        }
        output_tree = _tree_record(qwen)
        binding = {
            "pipeline_summary": summary_record,
            "output_tree_without_done": output_tree,
            "controlled_pilot_commit_digest": controlled_digest,
        }
        qwen_done = {
            "schema_version": (
                "motive-r10b-qwen-v2-packed-pipeline-done-v1"
            ),
            "status": "complete",
            "pipeline_summary": summary_record,
            "output_tree_without_done": output_tree,
            "commit_digest": _canonical_digest(binding),
            "controlled_pilot_commit_digest": controlled_digest,
            "balanced_pilot_ready": balanced,
            "real_gpu_tasks_verified": 8,
            "production_backend_cuda_only": True,
            "generation_error_rows": 0,
            "authorization": dict(AUTHORIZATION),
        }
        _write_pretty(qwen / "pipeline_done.json", qwen_done)

        fake_srun.write_text(
            """#!/usr/bin/env bash
set -Eeuo pipefail
: > "${FAKE_SRUN_MARKER:?}"
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "bash" ]]; then
    break
  fi
  shift
done
if [[ "${1:-}" != "bash" ]]; then
  echo "fake srun did not find worker command" >&2
  exit 90
fi
for rank in 0 1 2; do
  SLURM_PROCID="${rank}" SLURM_LOCALID="${rank}" SLURM_NTASKS=3 "$@"
done
""",
            encoding="utf-8",
        )
        fake_srun.chmod(
            fake_srun.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

        real_python = repr(sys.executable)
        fake_python.write_text(
            f"""#!{sys.executable}
from pathlib import Path
import hashlib
import json
import os
import shlex
import sys

REAL_PYTHON = {real_python}
args = sys.argv[1:]
with Path(os.environ["FAKE_PYTHON_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(shlex.join(args) + "\\n")

def value_after(name):
    index = args.index(name)
    return args[index + 1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if args and args[0] == "-":
    payload = args[1:]
    if payload and "/gpu_probes/" in payload[0]:
        sys.stdin.read()
        target = Path(payload[0])
        tag = payload[1]
        rank = int(payload[2])
        names = (
            "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "SLURM_GPUS_ON_NODE",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_NTASKS",
            "SLURM_JOB_ID",
            "SLURMD_NODENAME",
            "HOSTNAME",
        )
        probe = {{
            "schema_version": (
                "motive-r10b-bernini-triplet-packed-gpu-probe-v1"
            ),
            "status": "passed",
            "tag": tag,
            "rank": rank,
            "local_rank": rank,
            "slurm_ntasks": 3,
            "device_count": 1,
            "device_index": 0,
            "device_name": "AMD Instinct MI210",
            "device_uuid": None,
            "tensor_probe": {{
                "device": "cuda:0",
                "sum_of_squares": 14.0,
                "passed": True,
            }},
            "visibility": {{name: os.environ.get(name) for name in names}},
        }}
        target.write_text(
            json.dumps(probe, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
        raise SystemExit(0)
    if payload and payload[0] in (
        os.environ["MOTIVE_R10B_QWEN_PACKED_ROOT"],
        os.environ["MOTIVE_R10B_OUTPUT_ROOT"],
    ):
        os.execv(REAL_PYTHON, [REAL_PYTHON, *args])
    sys.stdin.read()
    raise SystemExit(0)

if args and args[0].endswith("/action_source_snapshot.py"):
    raise SystemExit(0)
if len(args) < 2 or args[0] != "-m":
    raise SystemExit("unexpected fake-python call: " + shlex.join(args))

module = args[1]
validate = "--validate-only" in args
if module == "motive.r10b_bernini_tangent_extract":
    if validate:
        raise SystemExit(0)
    output = Path(value_after("--output-dir"))
    manifest = Path(value_after("--manifest")).resolve()
    track = Path(value_after("--track-cache")).resolve()
    model = Path(value_after("--model-path")).resolve()
    tag = output.name
    output.mkdir()
    run_contract = {{
        "artifact_kind": "controlled_retrieval_pilot",
        "scheduler": {{
            "class": "UniPCMultistepScheduler",
            "steps": 50,
            "index": 25,
            "selection": "fixed_index",
            "timestep": 500.0,
            "sigma": 0.5,
        }},
        "noise": {{
            "mode": "iid_spatiotemporal",
            "seed": 260108853,
        }},
        "resize": {{
            "mode": "aspect_preserving_center_crop",
            "width": 256,
            "height": 256,
            "num_frames": 17,
        }},
    }}
    summary = {{
        "status": "complete",
        "artifact_kind": "controlled_retrieval_pilot",
        "measurement": {{
            "run_contract": run_contract,
            "projection_dimension_per_role": 2048,
            "projection_seeds": [260108851, 260108852],
        }},
        "data": {{
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
            "track_cache": str(track),
            "track_cache_sha256": digest(track),
            "rows": 20,
            "videos_read": 40,
        }},
        "model": {{
            "model_path": str(model),
            "huggingface_revision": (
                "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
            ),
            "checkpoint_manifest": {{"tree_sha256": "2" * 64}},
        }},
        "official_bernini_source": {{
            "commit": "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "bundle_sha256": "3" * 64,
        }},
        "source_tree_sha256": "1" * 64,
        "runtime": {{"width": 256, "height": 256, "num_frames": 17}},
        "parameter_manifest_sha256": "4" * 64,
        "safety": {{
            "optimizer_created": False,
            "optimizer_steps": 0,
            "scheduler_steps_executed": 0,
            "renderer_calls": 0,
            "videos_decoded_for_measurement": 40,
            "videos_rendered": 0,
            "videos_copied": 0,
            "checkpoint_mutated": False,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        }},
    }}
    done = {{
        "status": "complete",
        "artifact_kind": "controlled_retrieval_pilot",
        "artifact_digest": hashlib.sha256(tag.encode()).hexdigest(),
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }}
    (output / "features.npz").write_bytes(b"fake-feature")
    (output / "rows.jsonl").write_text(
        "{{\\"iid\\":\\"fixture\\"}}\\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    (output / "done.json").write_text(
        json.dumps(done, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    raise SystemExit(0)

if module == "motive.r10b_bernini_retrieval_audit":
    if validate:
        raise SystemExit(0)
    output = Path(value_after("--output-dir"))
    output.mkdir()
    artifact_values = [
        args[index + 1]
        for index, value in enumerate(args)
        if value == "--artifact"
    ]
    feature_artifacts = []
    for value in artifact_values:
        tag, path_raw = value.split("=", 1)
        path = Path(path_raw).resolve()
        tangent_done = json.loads(
            (path / "done.json").read_text(encoding="utf-8")
        )
        feature_artifacts.append(
            {{
                "tag": tag,
                "path": str(path),
                "artifact_digest": tangent_done["artifact_digest"],
            }}
        )
    audit = {{
        "status": "complete",
        "pilot": {{"commit_digest": "d" * 64}},
        "feature_artifacts": feature_artifacts,
        "decision": {{
            "development_signal_requires_sigma_noise_dimension_holdout": False
        }},
        "leakage_readouts": {{
            "appearance": {{
                "available": False,
                "sufficient_for_gate": False,
                "passed": False,
            }}
        }},
        "media_io": {{
            "video_files_read": 0,
            "video_files_copied": 0,
            "video_files_rendered": 0,
            "feature_arrays_read_only": True,
        }},
        "authorization": {{
            "human_label": False,
            "formal_evidence": False,
            "representation_promoted": False,
            "renderer_probe_authorized": False,
            "generation_authorized": False,
            "training_authorized": False,
        }},
        "formal_evidence": False,
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "generation_authorized": False,
        "editor_training_authorized": False,
        "training_authorized": False,
    }}
    audit_raw = (
        json.dumps(audit, indent=2, sort_keys=True) + "\\n"
    ).encode()
    (output / "retrieval_audit.json").write_bytes(audit_raw)
    done = {{
        "status": "complete",
        "artifact_digest": hashlib.sha256(audit_raw).hexdigest(),
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }}
    (output / "done.json").write_text(
        json.dumps(done, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    raise SystemExit(0)

raise SystemExit("unexpected fake module: " + module)
""",
            encoding="utf-8",
        )
        fake_python.chmod(
            fake_python.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

        environment = dict(os.environ)
        environment.update(
            {
                "MOTIVE_R10B_SOURCE_SNAPSHOT": str(snapshot),
                "MOTIVE_R10B_SOURCE_TREE_SHA256": "1" * 64,
                "MOTIVE_R10B_QWEN_PACKED_ROOT": str(qwen),
                "MOTIVE_R10B_QWEN_PACKED_COMMIT_DIGEST": qwen_done[
                    "commit_digest"
                ],
                "MOTIVE_R10B_TRACK_CACHE": str(track),
                "MOTIVE_R10B_TRACK_CACHE_SHA256": _sha256(track),
                "MOTIVE_R10B_MODEL_PATH": str(model),
                "MOTIVE_R10B_MODEL_REVISION": (
                    "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
                ),
                "MOTIVE_R10B_MODEL_TREE_SHA256": "2" * 64,
                "MOTIVE_R10B_BERNINI_REPO": str(bernini),
                "MOTIVE_R10B_BERNINI_SOURCE_COMMIT": (
                    "2d2b4591ac053ec25c6371b01a5a6746679e5793"
                ),
                "MOTIVE_R10B_BERNINI_SOURCE_BUNDLE_SHA256": "3" * 64,
                "MOTIVE_R10B_OUTPUT_ROOT": str(output),
                "MOTIVE_R10B_PYTHON_BIN": str(fake_python),
                "FAKE_PYTHON_LOG": str(log),
                "FAKE_SRUN_MARKER": str(srun_marker),
                "SLURM_NTASKS": "3",
                "SLURM_JOB_ID": "888",
                "SLURM_TMPDIR": str(scratch),
                "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
            }
        )
        paths: dict[str, Path | str] = {
            "output": output,
            "qwen": qwen,
            "log": log,
            "srun_marker": srun_marker,
            "qwen_commit": qwen_done["commit_digest"],
        }
        return environment, paths

    def test_fake_srun_executes_three_variants_and_commits_exact_closure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            environment, paths = self._build_fixture(root, balanced=True)
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(Path(paths["srun_marker"]).is_file())

            calls = Path(paths["log"]).read_text(
                encoding="utf-8"
            ).splitlines()
            tangent_calls = [
                shlex.split(call)
                for call in calls
                if call.startswith("-m motive.r10b_bernini_tangent_extract ")
                and "--validate-only" not in call
            ]
            tangent_validations = [
                shlex.split(call)
                for call in calls
                if call.startswith("-m motive.r10b_bernini_tangent_extract ")
                and "--validate-only" in call
            ]
            self.assertEqual(len(tangent_calls), 3)
            self.assertEqual(len(tangent_validations), 3)
            routed = {}
            for call in tangent_calls:
                output = Path(call[call.index("--output-dir") + 1])
                manifest = Path(call[call.index("--manifest") + 1])
                routed[output.name] = manifest.name
                self.assertIn(
                    [
                        "--projection-seeds",
                        "260108851",
                        "260108852",
                    ],
                    [call[index : index + 3] for index in range(len(call) - 2)],
                )
                self.assertEqual(
                    call[call.index("--artifact-kind") + 1],
                    "controlled_retrieval_pilot",
                )
                self.assertEqual(
                    call[call.index("--resize-mode") + 1],
                    "aspect_preserving_center_crop",
                )
                self.assertEqual(
                    call[call.index("--noise-mode") + 1],
                    "iid_spatiotemporal",
                )
                self.assertNotIn("--max-samples", call)
            self.assertEqual(
                routed,
                {
                    "canonical": "manifest.jsonl",
                    "original": "original.jsonl",
                    "cross_family": "cross_family_shuffle.jsonl",
                },
            )

            retrieval_calls = [
                shlex.split(call)
                for call in calls
                if call.startswith("-m motive.r10b_bernini_retrieval_audit ")
                and "--validate-only" not in call
            ]
            self.assertEqual(len(retrieval_calls), 1)
            retrieval = retrieval_calls[0]
            artifact_values = [
                retrieval[index + 1]
                for index, value in enumerate(retrieval)
                if value == "--artifact"
            ]
            self.assertEqual(
                {value.split("=", 1)[0] for value in artifact_values},
                {"canonical", "original", "cross_family"},
            )

            output = Path(paths["output"])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "artifacts",
                    "gpu_probes",
                    "retrieval_audit",
                    "pipeline_summary.json",
                    "pipeline_done.json",
                },
            )
            self.assertEqual(
                {path.name for path in (output / "artifacts").iterdir()},
                {"canonical", "original", "cross_family"},
            )
            summary = json.loads(
                (output / "pipeline_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            done = json.loads(
                (output / "pipeline_done.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "ready_for_commit")
            self.assertEqual(summary["execution"]["real_gpu_tasks"], 3)
            self.assertEqual(summary["execution"]["mi210_gpus"], 3)
            self.assertEqual(
                summary["execution"]["rank_to_prompt_variant"],
                {
                    "0": "canonical",
                    "1": "original",
                    "2": "cross_family",
                },
            )
            self.assertEqual(summary["videos_decoded_for_measurement"], 120)
            self.assertEqual(summary["videos_copied"], 0)
            self.assertEqual(summary["videos_rendered"], 0)
            self.assertEqual(summary["renderer_calls"], 0)
            self.assertEqual(summary["optimizer_steps"], 0)
            self.assertIs(
                summary["retrieval"]["appearance_control_available"],
                False,
            )
            self.assertIs(
                summary["retrieval"][
                    "development_signal_promotion_eligible"
                ],
                False,
            )
            self.assertEqual(
                summary["inputs"]["qwen_packed"]["commit_digest"],
                paths["qwen_commit"],
            )
            self.assertRegex(
                summary["inputs"]["qwen_packed"][
                    "prompt_variants_commit_digest"
                ],
                r"^[0-9a-f]{64}$",
            )
            for field in (
                "formal_evidence",
                "representation_gate_passed",
                "renderer_probe_authorized",
                "generation_authorized",
                "editor_training_authorized",
                "training_authorized",
                "downstream_submitted",
            ):
                self.assertIs(summary[field], False)
                self.assertIs(done[field], False)

            summary_raw = (output / "pipeline_summary.json").read_bytes()
            self.assertEqual(
                done["pipeline_summary"],
                {
                    "bytes": len(summary_raw),
                    "sha256": hashlib.sha256(summary_raw).hexdigest(),
                },
            )
            self.assertEqual(
                done["output_tree_without_done"],
                _tree_record(output, excluded={"pipeline_done.json"}),
            )
            expected_binding = {
                "pipeline_summary": done["pipeline_summary"],
                "output_tree_without_done": done["output_tree_without_done"],
                "qwen_packed_commit_digest": done[
                    "qwen_packed_commit_digest"
                ],
                "tangent_artifact_digests": done[
                    "tangent_artifact_digests"
                ],
                "retrieval_artifact_digest": done[
                    "retrieval_artifact_digest"
                ],
            }
            self.assertEqual(
                done["commit_digest"],
                _canonical_digest(expected_binding),
            )
            videos = [
                path
                for path in output.rglob("*")
                if path.suffix.lower() in {".mp4", ".webm", ".mov", ".avi"}
            ]
            self.assertEqual(videos, [])

    def test_unbalanced_packed_commit_fails_before_srun_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            environment, paths = self._build_fixture(root, balanced=False)
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "balanced/runtime/closed-scope gate differs",
                completed.stderr,
            )
            self.assertFalse(Path(paths["srun_marker"]).exists())
            self.assertFalse(Path(paths["output"]).exists())


if __name__ == "__main__":
    unittest.main()
