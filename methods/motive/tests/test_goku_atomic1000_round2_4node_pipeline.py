from __future__ import annotations

import re
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
PIPELINE = ROOT / "tmp" / "launch_goku_atomic1000_round2_4node_pipeline.sh"
WAN = ROOT / "tmp" / "launch_fullmotion_v16_wan_stream_round2_4node.sh"
HOLDER = ROOT / "tmp" / "goku_atomic1000_round2_4node_holder.sbatch"


def embedded(text: str, marker: str) -> str:
    opening = f"<<'{marker}'"
    start = text.index("\n", text.index(opening)) + 1
    end = text.index(f"\n{marker}\n", start)
    return text[start:end] + "\n"


class Round2FourNodePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = PIPELINE.read_text(encoding="utf-8")
        cls.wan = WAN.read_text(encoding="utf-8")
        cls.holder = HOLDER.read_text(encoding="utf-8")

    def test_shell_and_embedded_python_syntax(self) -> None:
        for path in (PIPELINE, WAN, HOLDER):
            result = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        for text in (self.pipeline, self.wan):
            for marker in re.findall(r"<<'([A-Z][A-Z0-9_]*)'", text):
                compile(embedded(text, marker), marker, "exec")

    def test_exact_four_node_topology(self) -> None:
        self.assertIn("#SBATCH --nodes=4", self.holder)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.holder)
        self.assertIn('"${SLURM_JOB_NUM_NODES:-}" == 4', self.holder)
        for text in (self.pipeline, self.wan):
            self.assertIn('"NumNodes=4" "gres/gpu:mi210=32"', text)
            self.assertNotIn('"NumNodes=8" "gres/gpu:mi210=64"', text)
        self.assertIn("MOTIVE_ATOMIC_PLANNER_WORKERS:-8", self.pipeline)
        self.assertIn("MOTIVE_ATOMIC_LABEL_WORKERS:-8", self.pipeline)
        self.assertIn("workers == nodes * 2", self.pipeline)
        self.assertIn('"${planner_workers}" 4 "${planner_root}"', self.pipeline)
        self.assertIn("--gpus-per-task=4 --gpu-bind=none", self.pipeline)
        self.assertIn('wan_nodes=(\n  "${allocated_nodes[0]}"', self.wan)
        self.assertIn("max_concurrent\": 4", self.wan)
        self.assertIn("--gpus-per-task=8 --gpu-bind=none", self.wan)

    def test_dynamic_job_and_fail_closed_hash_bindings(self) -> None:
        self.assertIn("MOTIVE_ATOMIC_JOB_ID:-", self.pipeline)
        self.assertIn("MOTIVE_ATOMIC_JOB_NAME:-", self.pipeline)
        self.assertIn("MOTIVE_ATOMIC_WAN_LAUNCHER_SHA256", self.pipeline)
        self.assertIn("Wan launcher bytes differ from explicit digest", self.pipeline)
        self.assertIn("MOTIVE_FULL_MOTION_WAN_LAUNCHER_SHA256", self.wan)
        self.assertIn("round2 Wan launcher bytes differ from bound digest", self.wan)
        self.assertIn('"holder_control": holder_control', self.pipeline)
        self.assertIn("MOTIVE_ATOMIC_HOLDER_READY_SCHEMA", self.pipeline)
        self.assertIn("MOTIVE_ATOMIC_HOLDER_RELEASE_SCHEMA", self.pipeline)
        self.assertIn('value["schema_version"] != sys.argv[4]', self.pipeline)
        self.assertIn('"schema_version": sys.argv[3]', self.pipeline)

    def test_exact_target_and_failure_replacement_admission(self) -> None:
        self.assertIn("MOTIVE_ATOMIC_GLOBAL_FINAL_TARGET:-1000", self.pipeline)
        self.assertIn("global_final_target == 1000", self.pipeline)
        self.assertIn("expected_rows >= smoke_batch_rows", self.pipeline)
        self.assertNotIn("expected_rows >= 1000", self.pipeline)
        self.assertIn("minimum_final_success <= expected_rows", self.pipeline)
        self.assertIn("admission_cap=$((required_new + prior_error))", self.pipeline)
        cap = embedded(self.pipeline, "PY_ADMISSION_CAP")
        self.assertIn("payload = b\"\".join(lines[:cap])", cap)
        self.assertIn('progress["atomic_ok_rows"] = min(len(lines), cap)', cap)
        final = embedded(self.pipeline, "PY_FINAL")
        self.assertIn("rows = rows[:minimum]", final)
        self.assertIn("len(rows) != minimum", final)
        self.assertIn("source_timebase = video_probe(source_video)", final)
        self.assertIn("target_timebase = video_probe(target_video)", final)
        self.assertIn('stream.get("avg_frame_rate") != "25/1"', final)
        self.assertIn("frames != 81", final)
        self.assertIn(
            'sha_file(source_video) != admission.get("source_video_sha256")', final
        )
        self.assertIn(
            "png_pixels(conditioning_anchor), source_frame_zero(source_video)",
            final,
        )
        self.assertIn(
            "conditioning_npy_pixels(conditioning_npy), png_pixels(conditioning_png)",
            final,
        )
        self.assertIn('"decoded_target_frame0_override_required": True', final)
        self.assertIn(
            '"target_mp4_decoded_frame0_pixel_equality_claimed": False', final
        )

    def test_admission_cap_helper_materializes_exact_prefix(self) -> None:
        helper = embedded(self.pipeline, "PY_ADMISSION_CAP")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "atomic.jsonl"
            rows = [f'{{"iid":"iid{index:04d}"}}\n'.encode() for index in range(1005)]
            raw = b"".join(rows)
            source.write_bytes(raw)
            progress = root / "progress.json"
            import hashlib
            import json

            progress.write_text(
                json.dumps(
                    {
                        "atomic_manifest": str(source),
                        "atomic_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                        "atomic_ok_rows": 1005,
                        "status": "target_reached",
                        "progress_digest": "placeholder",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "capped.jsonl"
            output_progress = root / "capped_progress.json"
            result = subprocess.run(
                [
                    "python3", "-", str(source), str(progress), str(output),
                    str(output_progress), "1002", "0",
                ],
                input=helper,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(output.read_bytes().splitlines()), 1002)
            capped_progress = json.loads(output_progress.read_text(encoding="utf-8"))
            self.assertEqual(capped_progress["atomic_ok_rows"], 1002)
            self.assertEqual(capped_progress["atomic_manifest"], str(output))

    def test_admission_cap_helper_accepts_zero_atomic_passes(self) -> None:
        helper = embedded(self.pipeline, "PY_ADMISSION_CAP")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "atomic.jsonl"
            source.write_bytes(b"")
            progress = root / "progress.json"
            import hashlib
            import json

            progress.write_text(
                json.dumps(
                    {
                        "atomic_manifest": str(source),
                        "atomic_manifest_sha256": hashlib.sha256(b"").hexdigest(),
                        "atomic_ok_rows": 0,
                        "status": "continue",
                        "progress_digest": "placeholder",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "capped.jsonl"
            output_progress = root / "capped_progress.json"
            result = subprocess.run(
                [
                    "python3", "-", str(source), str(progress), str(output),
                    str(output_progress), "8", "0",
                ],
                input=helper,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), b"")
            capped_progress = json.loads(output_progress.read_text(encoding="utf-8"))
            self.assertEqual(capped_progress["atomic_ok_rows"], 0)
            self.assertEqual(capped_progress["status"], "continue")
            self.assertEqual(capped_progress["atomic_manifest"], str(output))

    def test_independent_round_and_three_day_holder(self) -> None:
        self.assertIn("#SBATCH --time=3-00:00:00", self.holder)
        self.assertIn("round2 is independent and forbids legacy reuse", self.pipeline)
        self.assertIn("MOTIVE_ATOMIC_HOLDER_CONTROL", self.holder)
        self.assertIn("MOTIVE_ATOMIC_EPOCH_LOCK", self.holder)
        self.assertIn('flock -n "${epoch_lock_fd}"', self.holder)
        self.assertIn("release_holder_${SLURM_JOB_ID}.json", self.holder)


if __name__ == "__main__":
    unittest.main()
