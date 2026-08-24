from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_full_motion_wan_dispatch_existing_job.sh"
)
NODES = [f"auh{i:03d}" for i in range(1, 9)]
OFFICIAL_COMMIT = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ordered(values: list[str]) -> str:
    return _sha(b"".join(value.encode() + b"\n" for value in values))


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_shards(root: Path) -> Path:
    output = root / "shard_manifest"
    shards = output / "shards"
    shards.mkdir(parents=True)
    rows = [
        {"iid": f"iid{index:03d}", "group_id": f"group-{index:03d}"}
        for index in range(256)
    ]
    lines = [_canonical(row) + b"\n" for row in rows]
    primary = root / "primary_256.jsonl"
    primary_raw = b"".join(lines)
    primary.write_bytes(primary_raw)
    descriptors = []
    artifacts: dict[str, dict[str, object]] = {}
    for index in range(32):
        start = index * 8
        end = start + 8
        relative = f"shards/shard_{index:03d}.jsonl"
        raw = b"".join(lines[start:end])
        (output / relative).write_bytes(raw)
        iids = [row["iid"] for row in rows[start:end]]
        descriptor = {
            "shard_index": index,
            "shard_id": f"shard_{index:03d}",
            "path": relative,
            "root_row_start_zero_based": start,
            "root_row_end_exclusive": end,
            "root_row_indices_zero_based": list(range(start, end)),
            "rows": 8,
            "bytes": len(raw),
            "sha256": _sha(raw),
            "ordered_iids": iids,
            "ordered_iids_sha256": _ordered(iids),
            "ordered_row_sha256": _ordered(
                [_sha(_canonical(row)) for row in rows[start:end]]
            ),
        }
        descriptors.append(descriptor)
        artifacts[relative] = {"sha256": _sha(raw), "bytes": len(raw), "rows": 8}
    jobs_raw = (
        "header\n" + "".join(f"{index}\n" for index in range(32))
    ).encode()
    (output / "jobs.tsv").write_bytes(jobs_raw)
    source = {
        "primary_path": str(primary),
        "primary_sha256": _sha(primary_raw),
        "primary_bytes": len(primary_raw),
        "primary_rows": 256,
    }
    implementation = {"fixture": _sha(b"fixture")}
    input_digest = _sha(_canonical(source))
    summary = {
        "schema_version": "motive-goku-full-motion-shard-manifest-v1",
        "status": "complete",
        "source": source,
        "input_digest": input_digest,
        "layout": {
            "root_rows": 256,
            "rows_per_shard": 8,
            "shard_count": 32,
            "complete_nonoverlapping_coverage": True,
        },
        "shards": descriptors,
        "shards_digest": _sha(_canonical(descriptors)),
        "jobs": {
            "sha256": _sha(jobs_raw),
            "bytes": len(jobs_raw),
            "rows_excluding_header": 32,
        },
        "implementation": implementation,
        "implementation_digest": _sha(_canonical(implementation)),
    }
    summary_raw = _write_json(output / "summary.json", summary)
    artifacts["jobs.tsv"] = {
        "sha256": _sha(jobs_raw),
        "bytes": len(jobs_raw),
        "rows": 32,
    }
    artifacts["summary.json"] = {
        "sha256": _sha(summary_raw),
        "bytes": len(summary_raw),
        "rows": 1,
    }
    done = {
        "schema_version": "motive-goku-full-motion-shard-manifest-done-v1",
        "status": "complete",
        "source": source,
        "input_digest": input_digest,
        "implementation": implementation,
        "implementation_digest": _sha(_canonical(implementation)),
        "artifacts": artifacts,
        "artifact_digest": _sha(_canonical(artifacts)),
    }
    done["done_digest"] = _sha(_canonical(done))
    _write_json(output / "done.json", done)
    return output


FAKE_SRUN = r'''#!/usr/bin/env python3
import hashlib,json,os,sys
from pathlib import Path

args=sys.argv[1:]
node=next((x.split("=",1)[1] for x in args if x.startswith("--nodelist=")),"")
log=Path(os.environ["FAKE_DISPATCH_LOG"])
def append(text):
    with log.open("a") as handle: handle.write(text+"\n")
if "--overlap" not in args:
    append("unsafe\tmissing_overlap")
    raise SystemExit(90)
if "--exclusive" in args or any(x.startswith("--kill-on-bad-exit") for x in args):
    append("unsafe\tstep_control")
    raise SystemExit(91)
if "rocm-smi" in " ".join(args):
    append("idle\t"+node)
    if os.environ.get("FAKE_IDLE_FAIL_NODE") == node: raise SystemExit(19)
    raise SystemExit(0)
bindings={x.split("=",1)[0]:x.split("=",1)[1] for x in args if x.startswith("MOTIVE_") and "=" in x}
if "MOTIVE_WAN22_OUTPUT_ROOT" not in bindings: raise SystemExit(0)
for expected in (
    "ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
    "HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
):
    if expected not in args: raise SystemExit(92)
out=Path(bindings["MOTIVE_WAN22_OUTPUT_ROOT"])
manifest=Path(bindings["MOTIVE_GOKU_ACTION_GENERATION_MANIFEST"])
shard=int(out.name.rsplit("_",1)[1])
append(f"run\t{node}\t{shard}\t{bindings.get('MOTIVE_WAN22_MAX_NEW_SAMPLES','')}\t{manifest}")
if bindings.get("MOTIVE_WAN22_MAX_NEW_SAMPLES") != "8": raise SystemExit(20)
if os.environ.get("FAKE_FAIL_SHARD") == str(shard): raise SystemExit(42)
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
def write_json(path,value): path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
manifest_raw=manifest.read_bytes(); rows=[json.loads(x) for x in manifest_raw.splitlines()]
out.mkdir(parents=True,exist_ok=True)
contract={"schema_version":"motive-wan22-i2v-batch-run-v1","manifest":{"sha256":sha(manifest_raw),"bytes":len(manifest_raw),"row_count":8,"selected_row_count":8,"max_samples":8},"selected_inputs":[{"iid":x["iid"]} for x in rows],"distributed_execution":{"world_size":8,"max_new_samples_per_allocation":8},"generation_parameters":{"frame_num":81},"authorization":{"mode":"sshsig_full_motion_root_contiguous8_release_v3"},"temporal_policy":{"frame_count":81,"frame_rate":"25/1"}}
contract["contract_digest"]=sha(canon(contract)); write_json(out/"run_contract.json",contract)
generated=[]; result_digests=[]
for index,row in enumerate(rows):
    iid=row["iid"]; sample=out/"samples"/iid; sample.mkdir(parents=True,exist_ok=True)
    files={"preview_mp4":b"preview"+iid.encode(),"conditioning_anchor_original":b"anchor"+iid.encode(),"conditioning_frame0_float32":b"float"+iid.encode(),"conditioning_frame0_png":b"png"+iid.encode()}
    names={"preview_mp4":"preview.mp4","conditioning_anchor_original":"anchor.png","conditioning_frame0_float32":"frame.npy","conditioning_frame0_png":"frame.png"}
    outputs={}
    for key,raw in files.items():
        (sample/names[key]).write_bytes(raw); outputs[key]=names[key]; outputs[key+"_sha256"]=sha(raw)
    result={"schema_version":"motive-wan22-i2v-sample-v1","iid":iid,"manifest_sha256":sha(manifest_raw),"contract_digest":contract["contract_digest"],"outputs":outputs}
    result["result_digest"]=sha(canon(result)); write_json(sample/"result.json",result)
    result_digests.append(result["result_digest"])
    generated.append({"schema_version":"motive-wan22-i2v-generated-target-v1","iid":iid,"result_json":str(sample/"result.json"),"result_digest":result["result_digest"],"target_preview_mp4":str(sample/"preview.mp4"),"target_preview_mp4_sha256":outputs["preview_mp4_sha256"]})
generated_raw=b"".join(canon(x)+b"\n" for x in generated); (out/"generated_manifest.jsonl").write_bytes(generated_raw)
complete={"schema_version":"motive-wan22-i2v-batch-complete-v1","contract_digest":contract["contract_digest"],"manifest_sha256":sha(manifest_raw),"selected_sample_count":8,"completed_sample_count":8,"generated_manifest":"generated_manifest.jsonl","generated_manifest_sha256":sha(generated_raw),"temporal_policy":contract["temporal_policy"],"sample_result_digests":result_digests}
complete["complete_digest"]=sha(canon(complete)); write_json(out/"run_complete.json",complete)
'''


class FullMotionWanDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.log = self.root / "calls.tsv"
        self.log.write_text("")
        self.snapshot = self.root / "snapshot"
        code = self.snapshot / "methods" / "motive"
        (code / "motive").mkdir(parents=True)
        (code / "scripts").mkdir()
        (self.snapshot / "SOURCE_FILES.jsonl").write_text("{}\n")
        for name in (
            "wan22_i2v_batch.py",
            "wan22_signed_release.py",
            "wan22_full_motion_signed_release.py",
            "goku_full_motion_finalize.py",
        ):
            (code / "motive" / name).write_text("# fixture\n")
        (code / "scripts" / "auh_wan22_i2v_full.sbatch").write_text(
            "#!/usr/bin/env bash\nexit 99\n"
        )
        self.shards = _make_shards(self.root)
        self.release = self.root / "root_release.json"
        self.release.write_text('{"schema_version":"fixture"}\n')
        self.wan = self.root / "Wan2.2"
        (self.wan / ".git").mkdir(parents=True)
        (self.wan / "wan").mkdir()
        (self.wan / "generate.py").write_text("# fixture\n")
        (self.wan / "wan" / "image2video.py").write_text("# fixture\n")
        self.ckpt = self.root / "checkpoint"
        for relative in (
            ".cache/huggingface/download/.keep",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "high_noise_model/config.json",
            "high_noise_model/diffusion_pytorch_model.safetensors.index.json",
            "low_noise_model/config.json",
            "low_noise_model/diffusion_pytorch_model.safetensors.index.json",
        ):
            path = self.ckpt / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        self.ffprobe = self.root / "ffprobe"
        _write_executable(self.ffprobe, "#!/usr/bin/env bash\nexit 0\n")
        self.python = self.root / "python"
        _write_executable(
            self.python,
            """#!/usr/bin/env python3
import os,sys
if len(sys.argv)>2 and sys.argv[1:3]==['-m','motive.wan22_full_motion_signed_release']:
    raise SystemExit(0)
os.execv(os.environ['REAL_PYTHON'],[os.environ['REAL_PYTHON'],*sys.argv[1:]])
""",
        )
        _write_executable(self.fake_bin / "srun", FAKE_SRUN)
        _write_executable(
            self.fake_bin / "scontrol",
            """#!/usr/bin/env python3
import os,sys
if sys.argv[1:3]==['show','job']:
 print('JobId=123 JobState=RUNNING NodeList=auh[001-008] gres/gpu:mi210=64')
elif sys.argv[1:3]==['show','hostnames']:
 print('\\n'.join(os.environ['FAKE_NODES'].split(',')))
else: raise SystemExit(2)
""",
        )
        _write_executable(
            self.fake_bin / "squeue",
            """#!/usr/bin/env python3
import sys
if '-s' in sys.argv: print('123.0')
else: print('auh[001-008]')
""",
        )
        _write_executable(
            self.fake_bin / "git",
            f"#!/usr/bin/env bash\nprintf '%s\\n' '{OFFICIAL_COMMIT}'\n",
        )
        _write_executable(self.fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
        self.output = self.root / "wan_output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _env(self, **updates: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "PATH": f"{self.fake_bin}:{env['PATH']}",
                "REAL_PYTHON": sys.executable,
                "FAKE_NODES": ",".join(NODES),
                "FAKE_DISPATCH_LOG": str(self.log),
                "MOTIVE_EXISTING_SLURM_JOB_ID": "123",
                "MOTIVE_FULL_MOTION_WAN_NODES": ",".join(NODES),
                "MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT": str(self.snapshot),
                "MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR": str(self.shards),
                "MOTIVE_FULL_MOTION_ROOT_SIGNED_RELEASE": str(self.release),
                "MOTIVE_WAN22_CODE_ROOT": str(self.wan),
                "MOTIVE_WAN22_CKPT_DIR": str(self.ckpt),
                "MOTIVE_WAN22_PYTHON_BIN": str(self.python),
                "MOTIVE_WAN22_FFPROBE_BIN": str(self.ffprobe),
                "MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT": str(self.output),
                "MOTIVE_FULL_MOTION_WAN_IDLE_PROBE_INTERVAL_SECONDS": "0",
                "LC_ALL": "C",
            }
        )
        env.update(updates)
        return env

    def _run(self, **updates: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env=self._env(**updates),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )

    def _calls(self, kind: str) -> list[list[str]]:
        return [
            line.split("\t")
            for line in self.log.read_text().splitlines()
            if line.startswith(kind + "\t")
        ]

    def test_double_idle_audit_and_rank_stride_dispatch(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preserving existing steps: 123.0", result.stdout)
        idle = self._calls("idle")
        runs = self._calls("run")
        self.assertEqual(len(idle), 16)
        self.assertEqual({node: sum(row[1] == node for row in idle) for node in NODES}, {node: 2 for node in NODES})
        self.assertEqual(len(runs), 32)
        self.assertTrue(all(row[3] == "8" for row in runs))
        self.assertTrue(
            all(
                Path(row[4])
                == self.shards / "shards" / f"shard_{int(row[2]):03d}.jsonl"
                for row in runs
            )
        )
        self.assertEqual(len({row[4] for row in runs}), 32)
        observed = {node: sorted(int(row[2]) for row in runs if row[1] == node) for node in NODES}
        expected = {node: [rank, rank + 8, rank + 16, rank + 24] for rank, node in enumerate(NODES)}
        self.assertEqual(observed, expected)
        done = json.loads((self.output / "dispatch_complete.json").read_text())
        payload = dict(done)
        digest = payload.pop("complete_digest")
        self.assertEqual(digest, _sha(_canonical(payload)))
        self.assertEqual(len(done["completed_shards"]), 32)

    def test_holder_overlap_policy_is_non_destructive_and_exact_zero(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("preserving existing steps", text)
        self.assertIn("for audit in 1 2", text)
        self.assertIn('label == "GPU use (%)"', text)
        self.assertIn('label == "GPU Memory Allocated (VRAM%)"', text)
        self.assertIn('label == "VRAM Total Used Memory (B)"', text)
        self.assertIn("seen == 8", text)
        self.assertIn("1073741824", text)
        self.assertIn("rocm-smi --showpids --csv", text)
        self.assertIn("(gpu_flag+0) != 0", text)
        self.assertIn("(vram+0) != 0", text)
        self.assertIn("if srun --overlap", text)
        self.assertIn("--mem=0", text)
        self.assertIn("ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", text)
        self.assertNotIn("--exclusive", text)
        self.assertNotIn("--kill-on-bad-exit", text)
        self.assertNotIn("scancel", text)

    def test_one_failure_is_nonzero_but_peer_and_later_shards_continue(self) -> None:
        result = self._run(FAKE_FAIL_SHARD="5")
        self.assertNotEqual(result.returncode, 0)
        runs = self._calls("run")
        observed = {int(row[2]) for row in runs}
        self.assertEqual(observed, set(range(32)))
        for later in (13, 21, 29):
            self.assertTrue(
                (self.output / "wan_shards" / f"shard_{later:03d}" / "run_complete.json").is_file()
            )
        self.assertFalse((self.output / "dispatch_complete.json").exists())

    def test_verified_complete_shards_are_skipped_on_resume(self) -> None:
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        (self.output / "dispatch_complete.json").unlink()
        self.log.write_text("")
        second = self._run()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self._calls("run"), [])
        for rank in range(8):
            state = (
                self.output / "state" / f"worker_{rank:02d}.tsv"
            ).read_text()
            self.assertEqual(state.count("skipped_verified_complete"), 4)

    def test_idle_failure_aborts_before_any_generation_step(self) -> None:
        result = self._run(FAKE_IDLE_FAIL_NODE=NODES[3])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._calls("run"), [])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
