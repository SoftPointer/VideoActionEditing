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
    / "auh_r10b_qwen_v2_packed.sbatch"
)


class R10BQwenV2PackedOrchestrationTests(unittest.TestCase):
    def test_resources_eight_real_tasks_and_closed_scope_are_explicit(
        self,
    ) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --partition=faculty",
            "#SBATCH --account=test-acc",
            "#SBATCH --qos=bgqos",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=8",
            "#SBATCH --ntasks-per-node=8",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=1024G",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --time=02:00:00",
            "#SBATCH --job-name=motive-r10b-qwen-v2-packed",
        ):
            self.assertIn(directive, text)

        for marker in (
            "MOTIVE_R10B_SOURCE_SNAPSHOT",
            "MOTIVE_R10B_SOURCE_TREE_SHA256",
            "MOTIVE_R10B_FULL_QUEUE_DIR",
            "MOTIVE_R10B_DATA_ROOT",
            "MOTIVE_R10B_QWEN_MODEL",
            "MOTIVE_R10B_OUTPUT_ROOT",
            "MOTIVE_R10B_PYTHON_BIN",
            "motive/attribution.py",
            "motive/r10b_bernini_retrieval_audit.py",
            "motive/r10b_bernini_tangent_extract.py",
            'split_strategy="round_robin"',
            "motive/r10b_lucy_tangent_extract.py",
            "motive/r10b_tangent_core.py",
            "shard_queues",
            "shard_audits",
            "gpu_probes",
            "merged_audit",
            "pilot_manifest",
            "prompt_variants",
            "pipeline_summary.json",
            "pipeline_done.json",
            "--gpus-per-task=1",
            "--gpu-bind=single:1",
            "SLURM_PROCID",
            "probe_bound_gpu",
            "torch.cuda.is_available()",
            "torch.cuda.device_count()",
            '"MI210" not in device_name.upper()',
            'device="cuda:0"',
            "os.replace(temporary, probe_path)",
            "atomic_write_new(summary_path, payload)",
            "atomic_write_new(done_path, done_payload)",
            "output_tree_without_done",
            '"commit_digest": canonical_digest(commit_binding)',
            "validate_controlled_pilot_commit",
            "require_complete_qwen_audit",
            "production_local_qwen",
            "backend_execution",
            '"cuda_only"',
            "generation_error_rows",
            "resolved_blob_name",
            "32 * 1024 * 1024",
            'inventory["sha256"]',
            '"status": "ready_for_commit"',
            '"status": "complete"',
            'printf -v shard_name "shard_%03d"',
            'nframes="${MOTIVE_R10B_NFRAMES:-12}"',
            'max_pixels="${MOTIVE_R10B_MAX_PIXELS:-294912}"',
            'max_new_tokens=512',
            'attn_implementation="sdpa"',
            'qwen_visual_calls_per_row" "1"',
            'qwen_text_calls_per_row" "0"',
            '"qwen_text_calls_per_row": 0',
            'verify_bound_inputs "before"',
            'verify_bound_inputs "after"',
            'if [[ "${balanced}" == "true" ]]',
            "not_built_unbalanced_pilot",
            '"downstream_submitted": False',
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "WANDB_DISABLED=true",
            "PYTHONNOUSERSITE=1",
            "bash -c '",
        ):
            self.assertIn(marker, text)

        self.assertNotIn("#SBATCH --array", text)
        self.assertNotIn("text_only", text)
        self.assertNotIn("generate_text", text)
        self.assertNotIn("sbatch ", text)
        self.assertNotIn("ffmpeg", text)
        self.assertNotIn("rsync", text)
        self.assertNotIn("torchrun", text)
        self.assertNotIn("deepspeed", text)
        self.assertNotIn("bash -lc", text)
        self.assertNotIn("object_digest(inventory)", text)
        self.assertNotRegex(text, r"(?m)^\s*(?:cp|scp)\s")

    def test_cli_chain_and_runtime_prompt_contract_are_explicit(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        expected_cli_fragments = (
            "-m motive.r10b_qwen_audit_shards split",
            "--shard-count \"${shard_count}\"",
            "--strategy \"${split_strategy}\"",
            "-m motive.r10b_qwen_audit_shards validate-split",
            "-m motive.r10b_family_qwen_audit",
            "--nframes \"${MOTIVE_PACKED_NFRAMES}\"",
            "--max-pixels \"${MOTIVE_PACKED_MAX_PIXELS}\"",
            "--max-new-tokens \"${MOTIVE_PACKED_MAX_NEW_TOKENS}\"",
            "--attn-implementation \"${MOTIVE_PACKED_ATTN_IMPLEMENTATION}\"",
            "--validate-only",
            "-m motive.r10b_qwen_audit_shards merge",
            "-m motive.r10b_qwen_audit_shards validate-merge",
            "-m motive.r10b_bernini_pilot_manifest finalize",
            "--audit-records \"${merged_audit}/adapters.jsonl\"",
            "-m motive.r10b_bernini_prompt_variants build",
            "-m motive.r10b_bernini_prompt_variants validate",
        )
        for fragment in expected_cli_fragments:
            self.assertIn(fragment, text)
        self.assertEqual(
            text.count("-m motive.r10b_family_qwen_audit"),
            2,
        )
        self.assertIn(
            "qwen.get(\"qwen_prompt_sha256\") != PROMPT_CONTRACT_SHA256",
            text,
        )
        # The queue must be checked against whatever contract is frozen in the
        # source snapshot; an orchestration file must not pin an older digest.
        self.assertNotRegex(
            text,
            r"qwen_prompt_sha256[^\n]*[\"'][0-9a-f]{64}[\"']",
        )

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
        self.assertEqual(len(blocks), 6)
        for block in blocks:
            ast.parse(block)

    def test_temporal_overrides_must_be_positive_integers(self) -> None:
        base_environment = dict(os.environ)
        base_environment.update(
            {
                "MOTIVE_R10B_SOURCE_SNAPSHOT": "/tmp/fixture-snapshot",
                "MOTIVE_R10B_SOURCE_TREE_SHA256": "1" * 64,
                "MOTIVE_R10B_FULL_QUEUE_DIR": "/tmp/fixture-queue",
                "MOTIVE_R10B_DATA_ROOT": "/tmp/fixture-data",
                "MOTIVE_R10B_QWEN_MODEL": "/tmp/fixture-model",
                "MOTIVE_R10B_OUTPUT_ROOT": "/tmp/fixture-output",
                "MOTIVE_R10B_PYTHON_BIN": "/bin/true",
                "SLURM_NTASKS": "8",
            }
        )
        for name, value in (
            ("MOTIVE_R10B_NFRAMES", "0"),
            ("MOTIVE_R10B_NFRAMES", "12.5"),
            ("MOTIVE_R10B_MAX_PIXELS", "-1"),
            ("MOTIVE_R10B_MAX_PIXELS", "pixels"),
        ):
            with self.subTest(name=name, value=value):
                environment = dict(base_environment)
                environment[name] = value
                completed = subprocess.run(
                    ["bash", str(SCRIPT)],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"{name} must be one positive integer: {value}",
                    completed.stderr,
                )

    def test_fake_srun_executes_eight_distinct_shards_and_stops_unbalanced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            snapshot = root / "snapshot"
            code_root = snapshot / "methods" / "motive"
            module_root = code_root / "motive"
            scripts_root = code_root / "scripts"
            queue = root / "full_queue"
            data = root / "data"
            revision = "c" * 40
            model = root / revision
            output = root / "packed_output"
            scratch = root / "scratch"
            fake_bin = root / "bin"
            fake_python = root / "fake-python"
            python_log = root / "python.log"
            fake_srun = fake_bin / "srun"

            for directory in (
                module_root,
                scripts_root,
                queue,
                data,
                model,
                scratch,
                fake_bin,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (snapshot / "SOURCE_FILES.jsonl").write_text(
                '{"fixture":"frozen"}\n',
                encoding="utf-8",
            )
            for name in (
                "attribution.py",
                "qwen_filter.py",
                "r10b_bernini_retrieval_audit.py",
                "r10b_bernini_tangent_extract.py",
                "r10b_family_qwen_audit.py",
                "r10b_lucy_tangent_extract.py",
                "r10b_qwen_audit_shards.py",
                "r10b_bernini_pilot_manifest.py",
                "r10b_bernini_prompt_variants.py",
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
            (queue / "qwen_audit_queue.jsonl").write_text(
                '{"iid":"fixture"}\n',
                encoding="utf-8",
            )
            (queue / "summary.json").write_text(
                json.dumps(
                    {
                        "qwen_audit": {
                            "qwen_model_id": (
                                "Qwen/Qwen2.5-VL-7B-Instruct@" + revision
                            )
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (queue / "done.json").write_text(
                '{"status":"complete"}\n',
                encoding="utf-8",
            )
            (model / "config.json").write_text(
                '{"model_type":"qwen2_5_vl"}\n',
                encoding="utf-8",
            )

            fake_srun.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
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
for rank in 0 1 2 3 4 5 6 7; do
  SLURM_PROCID="${rank}" SLURM_LOCALID="${rank}" SLURM_NTASKS=8 "$@"
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

            real_python = shlex.quote(sys.executable)
            fake_python.write_text(
                f"""#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\\n' "$*" >> "${{FAKE_PYTHON_LOG:?}}"

value_after() {{
  local target="$1"
  shift
  local previous=""
  local argument
  for argument in "$@"; do
    if [[ "${{previous}}" == "${{target}}" ]]; then
      printf '%s\\n' "${{argument}}"
      return 0
    fi
    previous="${{argument}}"
  done
  return 1
}}

if [[ "${{1:-}}" == "-" ]]; then
  if [[ "${{2:-}}" == */gpu_probes/shard_*.json ]]; then
    cat >/dev/null
    target="${{2}}"
    rank="${{3}}"
    printf '%s\\n' '{{"device_count":1,"device_index":0,"device_name":"AMD Instinct MI210","device_uuid":null,"local_rank":'"${{rank}}"',"rank":'"${{rank}}"',"schema_version":"motive-r10b-qwen-v2-gpu-probe-v1","slurm_ntasks":8,"status":"passed","tensor_probe":{{"device":"cuda:0","passed":true,"sum_of_squares":14.0}},"visibility":{{"CUDA_VISIBLE_DEVICES":"0","GPU_DEVICE_ORDINAL":null,"HIP_VISIBLE_DEVICES":null,"HOSTNAME":"fake-node","ROCR_VISIBLE_DEVICES":"0","SLURMD_NODENAME":"fake-node","SLURM_GPUS_ON_NODE":"8","SLURM_JOB_GPUS":"0,1,2,3,4,5,6,7","SLURM_JOB_ID":"777","SLURM_LOCALID":"'"${{rank}}"'","SLURM_NTASKS":"8","SLURM_PROCID":"'"${{rank}}"'","SLURM_STEP_GPUS":"'"${{rank}}"'"}}}}' > "${{target}}"
    exit 0
  fi
  if [[ "$#" -ge 8 ]]; then
    exec {real_python} "$@"
  fi
  if [[ "${{2:-}}" == */summary.json && "$#" -eq 3 ]]; then
    exec {real_python} "$@"
  fi
  cat >/dev/null
  if [[ "${{2:-}}" == "{model}" ]]; then
    printf '%064d\\n' 0 | tr '0' 'e'
  elif [[ "${{2:-}}" == */pilot_manifest ]]; then
    printf '%064d\\n' 0 | tr '0' 'd'
  fi
  exit 0
fi
if [[ "${{1:-}}" == "-c" ]]; then
  printf '%s\\n' "${{FAKE_BALANCED:-false}}"
  exit 0
fi
if [[ "${{1:-}}" == *"/action_source_snapshot.py" ]]; then
  exit 0
fi
if [[ "${{1:-}}" != "-m" ]]; then
  echo "unexpected fake-python call: $*" >&2
  exit 91
fi

module="${{2:-}}"
command="${{3:-}}"
case "${{module}}:${{command}}" in
  motive.r10b_qwen_audit_shards:split)
    target="$(value_after --output-root "$@")"
    mkdir "${{target}}"
    : > "${{target}}/shards_summary.json"
    : > "${{target}}/shards_done.json"
    for rank in 0 1 2 3 4 5 6 7; do
      printf -v shard "shard_%03d" "${{rank}}"
      mkdir "${{target}}/${{shard}}"
      : > "${{target}}/${{shard}}/qwen_audit_queue.jsonl"
      : > "${{target}}/${{shard}}/summary.json"
      : > "${{target}}/${{shard}}/done.json"
    done
    ;;
  motive.r10b_qwen_audit_shards:merge)
    target="$(value_after --output-dir "$@")"
    mkdir "${{target}}"
    : > "${{target}}/records.jsonl"
    : > "${{target}}/adapters.jsonl"
    : > "${{target}}/done.json"
    printf '%s\\n' '{{"formal_evidence":false,"generation_authorized":false,"generation_error_rows":0,"model":{{"inventory":{{"sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}}}},"prompt_contract":{{"qwen_text_calls_per_row":0,"qwen_visual_calls_per_row":1,"stage2":"deterministic_python_no_model_call"}},"renderer_probe_authorized":false,"representation_gate_passed":false,"rows":8,"runtime":{{"nframes":16,"max_pixels":589824,"backend_execution":{{"cpu_offload_detected":false,"cuda_available":true,"cuda_only":true,"current_device":0,"device_count":1,"disk_offload_detected":false,"inspection_performed":true,"meta_offload_detected":false,"mode":"production_local_qwen","model_device":"cuda:0","parameter_devices":["cuda:0"],"production_backend":true,"schema_version":"motive-r10b-qwen-backend-execution-v1","test_backend":false,"verified_after_model_load":true}}}},"schema_error_rows":0,"status":"complete","successful_rows":8,"training_authorized":false}}' > "${{target}}/summary.json"
    ;;
  motive.r10b_qwen_audit_shards:validate-split|motive.r10b_qwen_audit_shards:validate-merge)
    ;;
  motive.r10b_family_qwen_audit:*)
    target="$(value_after --output-dir "$@")"
    if [[ " $* " != *" --validate-only "* ]]; then
      mkdir "${{target}}"
      : > "${{target}}/records.jsonl"
      : > "${{target}}/adapters.jsonl"
      printf '%s\\n' '{{"generation_error_rows":0,"rows":1,"runtime":{{"nframes":16,"max_pixels":589824,"backend_execution":{{"cpu_offload_detected":false,"cuda_available":true,"cuda_only":true,"current_device":0,"device_count":1,"disk_offload_detected":false,"inspection_performed":true,"meta_offload_detected":false,"mode":"production_local_qwen","model_device":"cuda:0","parameter_devices":["cuda:0"],"production_backend":true,"schema_version":"motive-r10b-qwen-backend-execution-v1","test_backend":false,"verified_after_model_load":true}}}},"schema_error_rows":0,"status":"complete","successful_rows":1}}' > "${{target}}/summary.json"
      : > "${{target}}/done.json"
    fi
    ;;
  motive.r10b_bernini_pilot_manifest:finalize)
    target="$(value_after --output-dir "$@")"
    mkdir "${{target}}"
    : > "${{target}}/manifest.jsonl"
    : > "${{target}}/shortfalls.json"
    : > "${{target}}/done.json"
    printf '%s\\n' '{{"balanced_pilot_ready":false}}' > "${{target}}/summary.json"
    ;;
  motive.r10b_bernini_prompt_variants:build)
    target="$(value_after --output-dir "$@")"
    mkdir "${{target}}"
    : > "${{target}}/summary.json"
    ;;
  motive.r10b_bernini_prompt_variants:validate)
    ;;
  *)
    echo "unexpected fake module call: $*" >&2
    exit 92
    ;;
esac
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
                    "MOTIVE_R10B_FULL_QUEUE_DIR": str(queue),
                    "MOTIVE_R10B_DATA_ROOT": str(data),
                    "MOTIVE_R10B_QWEN_MODEL": str(model),
                    "MOTIVE_R10B_OUTPUT_ROOT": str(output),
                    "MOTIVE_R10B_PYTHON_BIN": str(fake_python),
                    "MOTIVE_R10B_NFRAMES": "16",
                    "MOTIVE_R10B_MAX_PIXELS": "589824",
                    "FAKE_PYTHON_LOG": str(python_log),
                    "FAKE_BALANCED": "false",
                    "SLURM_NTASKS": "8",
                    "SLURM_JOB_ID": "777",
                    "SLURM_TMPDIR": str(scratch),
                    "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
                }
            )
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "motive_r10b_qwen_v2_packed.nframes=16",
                completed.stdout,
            )
            self.assertIn(
                "motive_r10b_qwen_v2_packed.max_pixels=589824",
                completed.stdout,
            )

            calls = python_log.read_text(encoding="utf-8").splitlines()
            inference_calls = [
                call
                for call in calls
                if call.startswith("-m motive.r10b_family_qwen_audit ")
                and "--validate-only" not in call
            ]
            validation_calls = [
                call
                for call in calls
                if call.startswith("-m motive.r10b_family_qwen_audit ")
                and "--validate-only" in call
            ]
            self.assertEqual(len(inference_calls), 8)
            self.assertEqual(len(validation_calls), 8)
            expected_shards = {
                f"shard_{index:03d}" for index in range(8)
            }
            inferred_shards = {
                match.group(1)
                for call in inference_calls
                if (
                    match := re.search(
                        r"--queue-dir \S+/(shard_\d{3})(?:\s|$)",
                        call,
                    )
                )
            }
            self.assertEqual(inferred_shards, expected_shards)
            for call in inference_calls:
                self.assertIn("--nframes 16", call)
                self.assertIn("--max-pixels 589824", call)
                self.assertIn("--max-new-tokens 512", call)
                self.assertIn("--attn-implementation sdpa", call)

            self.assertTrue((output / "merged_audit").is_dir())
            self.assertTrue((output / "pilot_manifest").is_dir())
            self.assertFalse((output / "prompt_variants").exists())
            self.assertFalse(
                any(
                    "motive.r10b_bernini_prompt_variants build" in call
                    for call in calls
                )
            )
            probe_calls = [
                call
                for call in calls
                if re.match(
                    r"^- \S+/gpu_probes/shard_\d{3}\.json [0-7] [0-7] 8$",
                    call,
                )
            ]
            self.assertEqual(len(probe_calls), 8)
            completion_gate_calls = [
                call
                for call in calls
                if re.match(
                    r"^- \S+/(?:shard_\d{3}|merged_audit)/summary\.json "
                    r"(?:shard_\d{3}|merged_audit) 16 589824$",
                    call,
                )
            ]
            self.assertEqual(len(completion_gate_calls), 9)
            self.assertTrue(
                any(
                    call == f"- {output / 'pilot_manifest'}"
                    for call in calls
                )
            )

            expected_root_names = {
                "gpu_probes",
                "shard_queues",
                "shard_audits",
                "merged_audit",
                "pilot_manifest",
                "pipeline_summary.json",
                "pipeline_done.json",
            }
            self.assertEqual(
                {path.name for path in output.iterdir()},
                expected_root_names,
            )
            probe_root = output / "gpu_probes"
            expected_probe_names = {
                f"shard_{index:03d}.json" for index in range(8)
            }
            self.assertEqual(
                {path.name for path in probe_root.iterdir()},
                expected_probe_names,
            )
            for rank in range(8):
                probe = json.loads(
                    (
                        probe_root / f"shard_{rank:03d}.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(probe["rank"], rank)
                self.assertEqual(probe["local_rank"], rank)
                self.assertEqual(probe["device_count"], 1)
                self.assertIn("MI210", probe["device_name"])
                self.assertEqual(
                    probe["tensor_probe"],
                    {
                        "device": "cuda:0",
                        "passed": True,
                        "sum_of_squares": 14.0,
                    },
                )
                self.assertEqual(
                    probe["visibility"]["SLURM_PROCID"],
                    str(rank),
                )

            summary = json.loads(
                (output / "pipeline_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "ready_for_commit")
            self.assertEqual(summary["execution"]["real_gpu_tasks"], 8)
            self.assertEqual(
                len(summary["execution"]["gpu_probe_evidence"]),
                8,
            )
            self.assertEqual(summary["execution"]["queue_shards"], 8)
            self.assertEqual(summary["execution"]["nframes"], 16)
            self.assertEqual(summary["execution"]["max_pixels"], 589824)
            self.assertEqual(
                summary["execution"]["qwen_text_calls_per_row"],
                0,
            )
            self.assertEqual(summary["execution"]["successful_rows"], 8)
            self.assertEqual(summary["execution"]["schema_error_rows"], 0)
            self.assertEqual(
                summary["execution"]["generation_error_rows"],
                0,
            )
            self.assertIs(
                summary["execution"]["backend_execution"]["cuda_only"],
                True,
            )
            self.assertEqual(
                summary["inputs"]["qwen_model"]["inventory_sha256"],
                "e" * 64,
            )
            self.assertEqual(
                summary["controlled_pilot_commit_digest"],
                "d" * 64,
            )
            self.assertEqual(
                set(summary["outputs"]),
                {
                    "gpu_probes",
                    "shard_queues",
                    "shard_audits",
                    "merged_audit",
                    "pilot_manifest",
                },
            )
            self.assertIs(summary["balanced_pilot_ready"], False)
            self.assertIs(summary["prompt_variants_materialized"], False)
            for field in (
                "formal_evidence",
                "representation_gate_passed",
                "renderer_probe_authorized",
                "generation_authorized",
                "training_authorized",
                "downstream_submitted",
            ):
                self.assertIs(summary[field], False)

            summary_path = output / "pipeline_summary.json"
            done_path = output / "pipeline_done.json"
            summary_raw = summary_path.read_bytes()
            done = json.loads(done_path.read_text(encoding="utf-8"))
            self.assertEqual(done["status"], "complete")
            self.assertEqual(done["real_gpu_tasks_verified"], 8)
            self.assertEqual(done["nframes"], 16)
            self.assertEqual(done["max_pixels"], 589824)
            self.assertEqual(done["generation_error_rows"], 0)
            self.assertIs(done["production_backend_cuda_only"], True)
            self.assertEqual(
                done["pipeline_summary"],
                {
                    "bytes": len(summary_raw),
                    "sha256": hashlib.sha256(summary_raw).hexdigest(),
                },
            )

            rows = []
            total_bytes = 0
            for path in sorted(
                output.rglob("*"),
                key=lambda item: item.as_posix(),
            ):
                if path.is_dir():
                    continue
                relative = path.relative_to(output).as_posix()
                if relative == "pipeline_done.json":
                    continue
                raw = path.read_bytes()
                rows.append(
                    {
                        "relative_path": relative,
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                total_bytes += len(raw)
            tree_raw = "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ).encode("utf-8")
            expected_tree = {
                "path": str(output),
                "files": len(rows),
                "bytes": total_bytes,
                "tree_sha256": hashlib.sha256(tree_raw).hexdigest(),
            }
            self.assertEqual(
                done["output_tree_without_done"],
                expected_tree,
            )
            commit_binding = {
                "pipeline_summary": done["pipeline_summary"],
                "output_tree_without_done": expected_tree,
                "controlled_pilot_commit_digest": "d" * 64,
            }
            expected_commit_digest = hashlib.sha256(
                json.dumps(
                    commit_binding,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(done["commit_digest"], expected_commit_digest)
            self.assertFalse(
                any(".tmp." in path.name for path in output.rglob("*"))
            )


if __name__ == "__main__":
    unittest.main()
