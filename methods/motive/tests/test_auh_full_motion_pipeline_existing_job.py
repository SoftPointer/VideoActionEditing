from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_full_motion_pipeline_existing_job.sh"
)
NODES = [f"auh{index:03d}" for index in range(1, 9)]
QWEN_V6_LINEAGE = {
    "record": "goku-full-motion-qwen-record-v6",
    "hard_gate": "goku-full-motion-hard-gate-v6",
    "provenance": "goku-full-motion-qwen-provenance-v6",
    "source_inventory_alignment": (
        "motive-goku-full-motion-source-inventory-alignment-v4"
    ),
    "change_region_proposals": (
        "motive-goku-full-motion-change-region-proposals-v1"
    ),
    "coverage_authority": "motive-goku-full-motion-coverage-authority-v2",
    "coverage_authority_inventory": (
        "motive-goku-full-motion-coverage-authority-inventory-v1"
    ),
    "coverage_authority_assignments": (
        "motive-goku-full-motion-coverage-authority-assignments-v1"
    ),
    "coverage_authority_allowed_owner_map": (
        "motive-goku-full-motion-coverage-authority-allowed-owner-map-v1"
    ),
    "coverage_authority_alignment": (
        "motive-goku-full-motion-coverage-authority-alignment-v2"
    ),
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_sha(value: object) -> str:
    return _sha(_canonical(value))


def _hard_pass_binding(iid: str) -> dict[str, str]:
    value = {
        "iid": iid,
        "record_schema_version": QWEN_V6_LINEAGE["record"],
        "provenance_schema_version": QWEN_V6_LINEAGE["provenance"],
        "hard_gate_schema_version": QWEN_V6_LINEAGE["hard_gate"],
        "change_region_proposals_schema_version": QWEN_V6_LINEAGE[
            "change_region_proposals"
        ],
        "coverage_authority_schema_version": QWEN_V6_LINEAGE[
            "coverage_authority"
        ],
        "coverage_authority_inventory_schema_version": QWEN_V6_LINEAGE[
            "coverage_authority_inventory"
        ],
        "coverage_authority_assignments_schema_version": QWEN_V6_LINEAGE[
            "coverage_authority_assignments"
        ],
        "source_inventory_alignment_schema_version": QWEN_V6_LINEAGE[
            "source_inventory_alignment"
        ],
        "coverage_authority_alignment_schema_version": QWEN_V6_LINEAGE[
            "coverage_authority_alignment"
        ],
    }
    for field in (
        "media_verification_sha256",
        "change_region_proposals_sha256",
        "coverage_authority_inventory_prompt_sha256",
        "coverage_authority_inventory_visual_input_sha256",
        "coverage_authority_inventory_sha256",
        "coverage_authority_assignments_prompt_sha256",
        "coverage_authority_assignments_visual_input_sha256",
        "coverage_authority_assignments_sha256",
        "coverage_authority_sha256",
        "i0_grounding_sha256",
        "primary_source_census_sha256",
        "secondary_source_census_sha256",
        "source_inventory_alignment_sha256",
        "coverage_authority_alignment_sha256",
        "hard_gate_sha256",
        "result_sha256",
        "provenance_sha256",
    ):
        value[field] = _sha(f"{iid}:{field}".encode())
    return value


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


FAKE_PYTHON = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
if args[:2] == ["-m", "motive.goku_full_motion_smoke_gate"]:
    output = Path(args[args.index("--output") + 1])
    shutil.copyfile(os.environ["FAKE_REBUILT_GATE"], output)
    raise SystemExit(0)
if args[:2] == ["-m", "motive.wan22_full_motion_signed_release"]:
    if os.environ.get("FAKE_RELEASE_FAIL") == "1":
        raise SystemExit(23)
    if len(args) < 7 or args[2] != "verify":
        raise SystemExit(24)
    release_path = Path(args[args.index("--release") + 1]).resolve(strict=True)
    manifest = Path(args[args.index("--manifest") + 1]).resolve(strict=True)
    primary = Path(os.environ["MOTIVE_FULL_MOTION_GENERATION_PRIMARY"]).resolve(
        strict=True
    )
    try:
        if not manifest.stem.startswith("shard_"):
            raise ValueError(manifest.stem)
        shard_index = int(manifest.stem[len("shard_") :])
    except ValueError:
        raise SystemExit(25)
    if not 0 <= shard_index < 32:
        raise SystemExit(26)
    primary_lines = primary.read_bytes().splitlines(keepends=True)
    manifest_lines = manifest.read_bytes().splitlines(keepends=True)
    if (
        len(primary_lines) != 256
        or len(manifest_lines) != 8
        or manifest_lines
        != primary_lines[shard_index * 8 : (shard_index + 1) * 8]
    ):
        raise SystemExit(27)
    log = os.environ.get("FAKE_RELEASE_LOG")
    if log:
        with Path(log).open("a", encoding="utf-8") as handle:
            handle.write(str(manifest) + "\n")
    start = shard_index * 8
    if os.environ.get("FAKE_RELEASE_BAD_BINDING_INDEX") == str(shard_index):
        start += 8
    binding = {
        "path": str(release_path),
        "release_id": os.environ["MOTIVE_FULL_MOTION_RELEASE_ID"],
        "payload_sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
        "signer_key_fingerprint": (
            "SHA256:A6zKKVBr6MSG29PO5J7A91aJYKcORNOkidofuI+jf6Y"
        ),
        "root_manifest_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "root_manifest_rows": 256,
        "root_row_start_zero_based": start,
        "root_row_stop_exclusive": start + 8,
    }
    print(json.dumps(binding, sort_keys=True))
    raise SystemExit(0)
os.execv(sys.executable, [sys.executable, *args])
'''


FAKE_CONTROLLER = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_with_digest(path, value, field):
    value[field] = sha(canonical(value))
    path.write_bytes(canonical(value) + b"\n")


def append(*values):
    with Path(os.environ["FAKE_STAGE_LOG"]).open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(value) for value in values) + "\n")


stage = sys.argv[1]
job = os.environ["MOTIVE_EXISTING_SLURM_JOB_ID"]

if stage == "watcher":
    nodes = os.environ["MOTIVE_FULL_MOTION_PIPELINE_NODES"]
    qwen_nodes = os.environ["MOTIVE_FULL_MOTION_QWEN_NODES"]
    finalize_node = os.environ["MOTIVE_FULL_MOTION_FINALIZE_NODE"]
    append("watcher_start", nodes, qwen_nodes, finalize_node)
    ready = {
        "schema_version": "motive-goku-full-motion-finalize-release-watcher-ready-v1",
        "status": "ready",
        "slurm_job_id": job,
        "nodes": nodes.split(","),
        "qwen_nodes": qwen_nodes.split(","),
        "finalize_node": finalize_node,
    }
    ready_path = Path(
        os.environ["MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_READY"]
    )
    json_with_digest(ready_path, ready, "receipt_digest")
    ready_path.chmod(0o400)
    qwen_done = Path(os.environ["MOTIVE_FULL_MOTION_FULL_QWEN_DONE"])
    deadline = time.monotonic() + 10
    while not qwen_done.is_file():
        if time.monotonic() >= deadline:
            raise SystemExit(32)
        time.sleep(0.01)

    final_pool = Path(os.environ["MOTIVE_FULL_MOTION_FINAL_POOL"])
    final_pool.mkdir()
    primary = final_pool / "primary_256.jsonl"
    primary.write_bytes(b"".join(
        canonical({"iid": f"iid{index:03d}"}) + b"\n" for index in range(256)
    ))
    final_done = {
        "schema_version": "motive-goku-full-motion-finalize-done-v1",
        "status": "complete",
    }
    json_with_digest(final_pool / "done.json", final_done, "done_digest")
    for name in ("reserve_64.jsonl", "review_candidates.jsonl", "summary.json"):
        (final_pool / name).write_text("{}\n")
    shard_root = Path(os.environ["MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR"])
    leaf = shard_root / "shards"
    leaf.mkdir(parents=True)
    primary_lines = primary.read_bytes().splitlines(keepends=True)
    for index in range(32):
        (leaf / f"shard_{index:03d}.jsonl").write_bytes(
            b"".join(primary_lines[index * 8 : (index + 1) * 8])
        )
    shard_done = {
        "schema_version": "motive-goku-full-motion-shard-manifest-done-v1",
        "status": "complete",
    }
    json_with_digest(shard_root / "done.json", shard_done, "done_digest")
    request = Path(os.environ["MOTIVE_FULL_MOTION_RELEASE_REQUEST"])
    request.write_text("fixture release request\n")
    release = Path(os.environ["MOTIVE_FULL_MOTION_RELEASE_DIR"])
    (release / "root_signed_release.json").write_text("fixture signed release\n")
    full_input = Path(os.environ["MOTIVE_FULL_MOTION_FULL_INPUT"]).resolve()
    terminal = {
        "schema_version": "motive-goku-full-motion-finalize-release-watcher-v1",
        "status": "complete",
        "slurm_job_id": job,
        "nodes": nodes.split(","),
        "finalize_node": finalize_node,
        "source_snapshot": os.environ["MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT"],
        "full_input": {
            "path": str(full_input),
            "sha256": sha(full_input.read_bytes()),
        },
        "qwen_done": {
            "path": str(qwen_done.resolve()),
            "sha256": sha(qwen_done.read_bytes()),
        },
        "final_pool": str(final_pool),
        "shard_manifest_root": str(shard_root),
        "release_request": {
            "path": str(request),
            "sha256": sha(request.read_bytes()),
        },
        "completed_at_utc": "2026-08-01T12:35:56Z",
    }
    terminal_path = Path(
        os.environ["MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_RECEIPT"]
    )
    json_with_digest(terminal_path, terminal, "receipt_digest")
    terminal_path.chmod(0o400)
    append("watcher_done", nodes)

elif stage == "qwen":
    nodes = os.environ["MOTIVE_FULL_MOTION_NODES"]
    append(stage, nodes, os.environ["MOTIVE_FULL_MOTION_SMOKE_GATE"])
    if os.environ.get("FAKE_QWEN_FAIL") == "1":
        raise SystemExit(31)
    root = Path(os.environ["MOTIVE_FULL_MOTION_FULL_QWEN_ROOT"])
    root.mkdir()
    for index in range(8):
        (root / f"qwen_shard_{index:03d}.jsonl").write_text("{}\n")
        (root / f"qwen_shard_{index:03d}.receipt.json").write_text("{}\n")
    input_path = Path(os.environ["MOTIVE_FULL_MOTION_FULL_INPUT"]).resolve()
    done = Path(os.environ["MOTIVE_FULL_MOTION_FULL_QWEN_DONE"])
    done.write_text("\n".join([
        "schema=motive-goku-full-motion-qwen-controller-v1",
        "status=complete",
        "input=" + str(input_path),
        "input_sha256=" + sha(input_path.read_bytes()),
        "output_root=" + str(root.resolve()),
        "slurm_job_id=" + job,
        "nodes=" + nodes,
        "completed_at_utc=2026-08-01T12:34:56Z",
    ]) + "\n")
    done.chmod(0o400)

elif stage == "wan":
    nodes = os.environ["MOTIVE_FULL_MOTION_WAN_NODES"]
    append(stage, nodes)
    output = Path(os.environ["MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT"])
    (output / "wan_shards").mkdir(parents=True)
    receipt = {
        "schema_version": "motive-full-motion-wan-existing-allocation-dispatch-v1",
        "status": "complete",
        "slurm_job_id": job,
        "nodes": nodes.split(","),
        "shard_manifest_dir": os.environ["MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR"],
        "root_signed_release": os.environ["MOTIVE_FULL_MOTION_ROOT_SIGNED_RELEASE"],
        "output_root": str(output),
        "completed_shards": [
            {"shard_index": index} for index in range(32)
        ],
    }
    json_with_digest(
        Path(os.environ["MOTIVE_FULL_MOTION_WAN_DISPATCH_RECEIPT"]),
        receipt,
        "complete_digest",
    )
    if os.environ.get("FAKE_INVALID_WAN_RECEIPT") == "1":
        value = json.loads(
            Path(os.environ["MOTIVE_FULL_MOTION_WAN_DISPATCH_RECEIPT"]).read_text()
        )
        value["complete_digest"] = "0" * 64
        Path(os.environ["MOTIVE_FULL_MOTION_WAN_DISPATCH_RECEIPT"]).write_text(
            json.dumps(value, sort_keys=True) + "\n"
        )

elif stage == "post":
    nodes = os.environ["MOTIVE_FULL_MOTION_POSTCHECK_NODES"]
    append(stage, nodes)
    output = Path(os.environ["MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT"])
    output.mkdir()
    status = output / "status.tsv"
    status.write_text("shard\tstatus\n")
    receipt = {
        "schema_version": "motive-goku-full-motion-postcheck-dispatch-receipt-v2",
        "status": "complete",
        "slurm_job_id": job,
        "nodes": nodes.split(","),
        "source_snapshot": os.environ["MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT"],
        "generation_shard_dir": os.environ["MOTIVE_FULL_MOTION_GENERATION_SHARD_DIR"],
        "wan_shards_root": os.environ["MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT"],
        "model": os.environ["MOTIVE_FULL_MOTION_POSTCHECK_MODEL"],
        "completed_shards": 32,
        "failed_shards": [],
        "shards": [{"shard_index": index} for index in range(32)],
        "status_tsv": str(status),
        "status_tsv_sha256": sha(status.read_bytes()),
    }
    json_with_digest(
        Path(os.environ["MOTIVE_FULL_MOTION_POSTCHECK_DISPATCH_RECEIPT"]),
        receipt,
        "receipt_digest",
    )

elif stage == "select":
    append(stage, "")
    output = Path(os.environ["MOTIVE_FULL_MOTION_EXACT128_OUTPUT"])
    output.mkdir()
    (output / "dataset_manifest.jsonl").write_bytes(b"{}\n" * 128)
    (output / "done.json").write_text('{"status":"complete"}\n')
    receipt = {
        "schema_version": "motive-goku-full-motion-select128-controller-receipt-v1",
        "status": "complete",
        "config": {"exact_size": 128, "min_multi_unit": 32},
        "source_snapshot": os.environ["MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT"],
        "generation": {
            "primary": {"path": os.environ["MOTIVE_FULL_MOTION_GENERATION_PRIMARY"]},
            "done": {"path": os.environ["MOTIVE_FULL_MOTION_GENERATION_DONE"]},
        },
        "shard_manifest_root": os.environ["MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR"],
        "wan_shards_root": os.environ["MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT"],
        "postcheck_dispatch": {
            "receipt": {
                "path": os.environ["MOTIVE_FULL_MOTION_POSTCHECK_DISPATCH_RECEIPT"]
            }
        },
        "output": {"root": str(output)},
    }
    json_with_digest(
        Path(os.environ["MOTIVE_FULL_MOTION_EXACT128_RECEIPT"]),
        receipt,
        "receipt_digest",
    )
else:
    raise SystemExit("unknown fake stage: " + stage)
'''


class FullMotionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.run_root = self.root / "run"
        self.run_root.mkdir()
        self.production = self.run_root / "production"
        self.release_dir = self.production / "release"
        self.release_dir.mkdir(parents=True)
        self.stage_log = self.root / "stages.tsv"
        self.stage_log.write_text("")
        self.release_log = self.root / "release_verifications.txt"
        self.release_log.write_text("")

        self.snapshot = self.root / "snapshot"
        package = self.snapshot / "methods" / "motive" / "motive"
        scripts = self.snapshot / "methods" / "motive" / "scripts"
        package.mkdir(parents=True)
        scripts.mkdir()
        (self.snapshot / "SOURCE_FILES.jsonl").write_text("{}\n")
        (package / "__init__.py").write_text("")
        (package / "goku_full_motion_contract.py").write_text(
            "import hashlib,json\n"
            "def object_sha256(value):\n"
            " return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()\n"
        )
        self.implementation = {"fixture": "pipeline-test", "version": 1}
        (package / "goku_full_motion_qwen.py").write_text(
            f"RECORD_SCHEMA = {QWEN_V6_LINEAGE['record']!r}\n"
            f"HARD_GATE_SCHEMA = {QWEN_V6_LINEAGE['hard_gate']!r}\n"
            f"PROVENANCE_SCHEMA = {QWEN_V6_LINEAGE['provenance']!r}\n"
            "SOURCE_INVENTORY_ALIGNMENT_SCHEMA = "
            f"{QWEN_V6_LINEAGE['source_inventory_alignment']!r}\n"
            "CHANGE_REGION_PROPOSALS_SCHEMA = "
            f"{QWEN_V6_LINEAGE['change_region_proposals']!r}\n"
            "COVERAGE_AUTHORITY_SCHEMA = "
            f"{QWEN_V6_LINEAGE['coverage_authority']!r}\n"
            "COVERAGE_AUTHORITY_INVENTORY_SCHEMA = "
            f"{QWEN_V6_LINEAGE['coverage_authority_inventory']!r}\n"
            "COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA = "
            f"{QWEN_V6_LINEAGE['coverage_authority_assignments']!r}\n"
            "COVERAGE_AUTHORITY_ALLOWED_OWNER_MAP_SCHEMA = "
            f"{QWEN_V6_LINEAGE['coverage_authority_allowed_owner_map']!r}\n"
            "COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA = "
            f"{QWEN_V6_LINEAGE['coverage_authority_alignment']!r}\n"
            "def _implementation_bundle():\n"
            f" return {self.implementation!r}\n"
        )
        (package / "goku_full_motion_smoke_gate.py").write_text(
            "# Frozen smoke-gate implementation fixture.\n"
        )
        self.pipeline = scripts / "pipeline.sh"
        shutil.copyfile(SCRIPT, self.pipeline)
        self.pipeline.chmod(0o444)
        self.snapshot_tool = self.snapshot / "verify_snapshot.py"
        self.snapshot_tool.write_text("raise SystemExit(0)\n")

        self.driver = self.root / "fake_controller.py"
        _write_executable(self.driver, FAKE_CONTROLLER)
        self.controllers: dict[str, Path] = {}
        for stage in ("qwen", "watcher", "wan", "post", "select"):
            controller = scripts / f"{stage}.sh"
            controller.write_text(
                "#!/bin/bash\n"
                'exec "$FAKE_REAL_PYTHON" "$FAKE_CONTROLLER_DRIVER" '
                + stage
                + "\n",
                encoding="utf-8",
            )
            controller.chmod(0o444)
            self.controllers[stage] = controller

        self.fake_python = self.fake_bin / "python"
        _write_executable(self.fake_python, FAKE_PYTHON)
        for name in ("ffprobe", "ffmpeg"):
            _write_executable(self.fake_bin / name, "#!/bin/sh\nexit 0\n")

        _write_executable(
            self.fake_bin / "squeue",
            "#!/bin/sh\nprintf '%s\\n' fake-allocation\n",
        )
        _write_executable(
            self.fake_bin / "scontrol",
            "#!/bin/sh\n"
            'if [ "$1 $2" = "show hostnames" ]; then\n'
            + "".join(f" printf '%s\\n' {node}\n" for node in NODES)
            + " exit 0\nfi\n"
            + "printf '%s\\n' \"JobId=$FAKE_JOB UserId=$FAKE_USER(1000) "
            "JobState=RUNNING NumNodes=8 gres/gpu:mi210=64\"\n",
        )

        data = self.root / "data"
        data.mkdir()
        self.smoke_input = data / "smoke.jsonl"
        self.smoke_input.write_bytes(b"{}\n" * 8)
        self.full_input = data / "full.jsonl"
        self.full_input.write_bytes(b"{}\n" * 768)
        self.smoke_root = data / "smoke_qwen"
        self.smoke_root.mkdir()
        self.qwen_model = self.root / "qwen_model"
        self.qwen_model.mkdir()
        (self.qwen_model / "config.json").write_text("{}\n")
        self.post_model = self.root / "post_model"
        self.post_model.mkdir()
        (self.post_model / "config.json").write_text("{}\n")
        self.wan_code = self.root / "wan_code"
        self.wan_code.mkdir()
        self.wan_checkpoint = self.root / "wan_checkpoint"
        self.wan_checkpoint.mkdir()

        self.raw_gate = data / "gate.json"
        hard_pass_iids = ["1dbe39537c984690", "iid-b", "iid-c"]
        gate = {
            "schema_version": "motive-goku-full-motion-qwen-smoke-gate-v6",
            "status": "pass",
            "input": {
                "path": str(self.smoke_input.resolve()),
                "sha256": _sha(self.smoke_input.read_bytes()),
                "rows": 8,
            },
            "qwen_lineage": dict(QWEN_V6_LINEAGE),
            "qwen_runtime": {
                "implementation_digest": _object_sha(self.implementation),
                "model_path": str(self.qwen_model.resolve()),
                "run_config": {"schemas": dict(QWEN_V6_LINEAGE)},
            },
            "hard_passes": len(hard_pass_iids),
            "hard_pass_iids": hard_pass_iids,
            "hard_pass_bindings": [
                _hard_pass_binding(iid) for iid in hard_pass_iids
            ],
            "canary": {
                "iid": "1dbe39537c984690",
                "qwen_record_schema_version": QWEN_V6_LINEAGE["record"],
                "qwen_hard_gate_schema_version": QWEN_V6_LINEAGE["hard_gate"],
                "qwen_provenance_schema_version": QWEN_V6_LINEAGE[
                    "provenance"
                ],
                "coverage_authority_inventory_schema_version": (
                    QWEN_V6_LINEAGE["coverage_authority_inventory"]
                ),
                "coverage_authority_assignments_schema_version": (
                    QWEN_V6_LINEAGE["coverage_authority_assignments"]
                ),
                "source_inventory_alignment_schema_version": QWEN_V6_LINEAGE[
                    "source_inventory_alignment"
                ],
                "coverage_authority_alignment_schema_version": QWEN_V6_LINEAGE[
                    "coverage_authority_alignment"
                ],
            },
        }
        gate["gate_digest"] = _object_sha(gate)
        self.raw_gate.write_bytes(_canonical(gate) + b"\n")

        self.paths = {
            "verified": self.run_root / "verified_gate.json",
            "qwen_root": self.run_root / "full_qwen",
            "qwen_done": self.run_root / "full_qwen.done",
            "final_pool": self.run_root / "final_pool",
            "primary": self.run_root / "final_pool" / "primary_256.jsonl",
            "generation_done": self.run_root / "final_pool" / "done.json",
            "shards": self.production / "generation_shards",
            "release_request": self.release_dir / "release_request.json",
            "release": self.release_dir / "root_signed_release.json",
            "watcher_ready": self.production / "watcher.ready.json",
            "watcher_receipt": self.production / "watcher.done.json",
            "watcher_log": self.run_root / "watcher.log",
            "wan": self.run_root / "wan",
            "post": self.run_root / "postcheck",
            "exact": self.run_root / "exact128",
            "exact_receipt": self.run_root / "exact128.receipt.json",
            "pipeline_receipt": self.run_root / "pipeline.receipt.json",
        }

        self.env = os.environ.copy()
        self.env.update(
            {
                "PATH": str(self.fake_bin) + os.pathsep + self.env["PATH"],
                "FAKE_JOB": "123",
                "FAKE_USER": subprocess.check_output(
                    ["id", "-un"], text=True
                ).strip(),
                "FAKE_REAL_PYTHON": sys.executable,
                "FAKE_CONTROLLER_DRIVER": str(self.driver),
                "FAKE_STAGE_LOG": str(self.stage_log),
                "FAKE_RELEASE_LOG": str(self.release_log),
                "FAKE_REBUILT_GATE": str(self.raw_gate),
                "MOTIVE_EXISTING_SLURM_JOB_ID": "123",
                "MOTIVE_FULL_MOTION_PIPELINE_NODES": ",".join(NODES),
                "MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT": str(self.snapshot),
                "MOTIVE_FULL_MOTION_SOURCE_TREE_SHA256": "1" * 64,
                "MOTIVE_FULL_MOTION_SNAPSHOT_TOOL": str(self.snapshot_tool),
                "MOTIVE_FULL_MOTION_SMOKE_INPUT": str(self.smoke_input),
                "MOTIVE_FULL_MOTION_SMOKE_INPUT_SHA256": _sha(
                    self.smoke_input.read_bytes()
                ),
                "MOTIVE_FULL_MOTION_SMOKE_QWEN_ROOT": str(self.smoke_root),
                "MOTIVE_FULL_MOTION_SMOKE_GATE": str(self.raw_gate),
                "MOTIVE_FULL_MOTION_SMOKE_GATE_SHA256": _sha(
                    self.raw_gate.read_bytes()
                ),
                "MOTIVE_FULL_MOTION_VERIFIED_GATE": str(self.paths["verified"]),
                "MOTIVE_FULL_MOTION_CANARY_IID": "1dbe39537c984690",
                "MOTIVE_FULL_MOTION_MINIMUM_HARD_PASSES": "3",
                "MOTIVE_FULL_MOTION_MINIMUM_CANARY_DYNAMIC_UNITS": "2",
                "MOTIVE_FULL_MOTION_FULL_INPUT": str(self.full_input),
                "MOTIVE_FULL_MOTION_FULL_INPUT_SHA256": _sha(
                    self.full_input.read_bytes()
                ),
                "MOTIVE_FULL_MOTION_FULL_QWEN_ROOT": str(self.paths["qwen_root"]),
                "MOTIVE_FULL_MOTION_FULL_QWEN_DONE": str(self.paths["qwen_done"]),
                "MOTIVE_FULL_MOTION_QWEN_MODEL": str(self.qwen_model),
                "MOTIVE_FULL_MOTION_QWEN_MODEL_METADATA_SHA256": self._model_metadata_sha(
                    self.qwen_model
                ),
                "MOTIVE_FULL_MOTION_QWEN_PYTHON": str(self.fake_python),
                "MOTIVE_FULL_MOTION_QWEN_DISTRIBUTED_CONTROLLER": str(
                    self.controllers["qwen"]
                ),
                "MOTIVE_FULL_MOTION_GATE_WAIT_SECONDS": "1",
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER": str(
                    self.controllers["watcher"]
                ),
                "MOTIVE_FULL_MOTION_FINALIZE_NODE": NODES[0],
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_RECEIPT": str(
                    self.paths["watcher_receipt"]
                ),
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_READY": str(
                    self.paths["watcher_ready"]
                ),
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_LOG": str(
                    self.paths["watcher_log"]
                ),
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_STARTUP_WAIT_SECONDS": "5",
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WAIT_SECONDS": "10",
                "MOTIVE_FULL_MOTION_FINALIZE_RELEASE_POLL_SECONDS": "1",
                "MOTIVE_FULL_MOTION_FINALIZE_CPUS": "4",
                "MOTIVE_FULL_MOTION_FINAL_POOL": str(self.paths["final_pool"]),
                "MOTIVE_FULL_MOTION_GENERATION_PRIMARY": str(self.paths["primary"]),
                "MOTIVE_FULL_MOTION_GENERATION_DONE": str(
                    self.paths["generation_done"]
                ),
                "MOTIVE_FULL_MOTION_PRODUCTION_ROOT": str(self.production),
                "MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR": str(self.paths["shards"]),
                "MOTIVE_FULL_MOTION_GENERATION_SHARD_DIR": str(
                    self.paths["shards"] / "shards"
                ),
                "MOTIVE_FULL_MOTION_ROOT_SIGNED_RELEASE": str(
                    self.paths["release"]
                ),
                "MOTIVE_FULL_MOTION_RELEASE_DIR": str(self.release_dir),
                "MOTIVE_FULL_MOTION_RELEASE_REQUEST": str(
                    self.paths["release_request"]
                ),
                "MOTIVE_FULL_MOTION_RELEASE_ID": "fixture-release-v1",
                "MOTIVE_FULL_MOTION_RELEASE_CHALLENGE": "2" * 64,
                "MOTIVE_FULL_MOTION_RELEASE_PYTHON": str(self.fake_python),
                "MOTIVE_FULL_MOTION_RELEASE_WAIT_SECONDS": "1",
                "MOTIVE_FULL_MOTION_RELEASE_POLL_SECONDS": "1",
                "MOTIVE_FULL_MOTION_WAN_DISPATCHER": str(self.controllers["wan"]),
                "MOTIVE_WAN22_CODE_ROOT": str(self.wan_code),
                "MOTIVE_WAN22_CKPT_DIR": str(self.wan_checkpoint),
                "MOTIVE_WAN22_PYTHON_BIN": str(self.fake_python),
                "MOTIVE_WAN22_FFPROBE_BIN": str(self.fake_bin / "ffprobe"),
                "MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT": str(self.paths["wan"]),
                "MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT": str(
                    self.paths["wan"] / "wan_shards"
                ),
                "MOTIVE_FULL_MOTION_WAN_DISPATCH_RECEIPT": str(
                    self.paths["wan"] / "dispatch_complete.json"
                ),
                "MOTIVE_FULL_MOTION_WAN_STEP_CPUS": "32",
                "MOTIVE_FULL_MOTION_WAN_IDLE_PROBE_INTERVAL_SECONDS": "0",
                "MOTIVE_WAN22_FRAME_NUM": "81",
                "MOTIVE_WAN22_SAMPLE_STEPS": "20",
                "MOTIVE_WAN22_SAMPLE_SHIFT": "5.0",
                "MOTIVE_WAN22_SIZE": "832*480",
                "MOTIVE_WAN22_BASE_SEED": "99",
                "MOTIVE_FULL_MOTION_POSTCHECK_DISPATCHER": str(
                    self.controllers["post"]
                ),
                "MOTIVE_FULL_MOTION_POSTCHECK_MODEL": str(self.post_model),
                "MOTIVE_FULL_MOTION_POSTCHECK_PYTHON": str(self.fake_python),
                "MOTIVE_FULL_MOTION_POSTCHECK_FFPROBE": str(
                    self.fake_bin / "ffprobe"
                ),
                "MOTIVE_FULL_MOTION_POSTCHECK_FFMPEG": str(
                    self.fake_bin / "ffmpeg"
                ),
                "MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT": str(
                    self.paths["post"]
                ),
                "MOTIVE_FULL_MOTION_POSTCHECK_DISPATCH_RECEIPT": str(
                    self.paths["post"] / "dispatcher_receipt.json"
                ),
                "MOTIVE_FULL_MOTION_POSTCHECK_CPUS": "16",
                "MOTIVE_FULL_MOTION_POSTCHECK_IDLE_RECHECK_SECONDS": "1",
                "MOTIVE_FULL_MOTION_SELECT128_CONTROLLER": str(
                    self.controllers["select"]
                ),
                "MOTIVE_FULL_MOTION_SELECT_PYTHON": str(self.fake_python),
                "MOTIVE_FULL_MOTION_FFPROBE": str(self.fake_bin / "ffprobe"),
                "MOTIVE_FULL_MOTION_FFMPEG": str(self.fake_bin / "ffmpeg"),
                "MOTIVE_FULL_MOTION_EXACT128_OUTPUT": str(self.paths["exact"]),
                "MOTIVE_FULL_MOTION_EXACT128_RECEIPT": str(
                    self.paths["exact_receipt"]
                ),
                "MOTIVE_FULL_MOTION_EXACT128_WAIT_SECONDS": "1",
                "MOTIVE_FULL_MOTION_EXACT128_POLL_SECONDS": "1",
                "MOTIVE_FULL_MOTION_PIPELINE_PYTHON": str(self.fake_python),
                "MOTIVE_FULL_MOTION_PIPELINE_CONTROLLER": str(self.pipeline),
                "MOTIVE_FULL_MOTION_PIPELINE_RECEIPT": str(
                    self.paths["pipeline_receipt"]
                ),
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _model_metadata_sha(root: Path) -> str:
        rows = []
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            metadata = path.stat()
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": metadata.st_size,
                    "mtime_ns": metadata.st_mtime_ns,
                    "sha256": _sha(path.read_bytes()),
                }
            )
        return _object_sha(rows)

    def _run(self, **extra: str) -> subprocess.CompletedProcess[str]:
        env = dict(self.env)
        env.update(extra)
        return subprocess.run(
            ["bash", str(self.pipeline)],
            text=True,
            capture_output=True,
            env=env,
            timeout=20,
            check=False,
        )

    def test_success_orders_stages_splits_nodes_and_publishes_receipt(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [line.split("\t") for line in self.stage_log.read_text().splitlines()]
        self.assertEqual(
            [line[0] for line in calls],
            ["watcher_start", "qwen", "watcher_done", "wan", "post", "select"],
        )
        self.assertEqual(calls[0][1], ",".join(NODES))
        self.assertEqual(calls[0][2], ",".join(NODES[:4]))
        self.assertEqual(calls[1][1], ",".join(NODES[:4]))
        self.assertEqual(calls[1][2], str(self.paths["verified"]))
        self.assertEqual(calls[3][1], ",".join(NODES))
        self.assertEqual(calls[4][1], ",".join(NODES[:4]))
        verified_shards = self.release_log.read_text().splitlines()
        self.assertEqual(
            verified_shards,
            [
                str(
                    (
                        self.paths["shards"]
                        / "shards"
                        / f"shard_{index:03d}.jsonl"
                    ).resolve()
                )
                for index in range(32)
            ],
        )
        self.assertEqual(self.paths["verified"].read_bytes(), self.raw_gate.read_bytes())
        self.assertEqual(stat.S_IMODE(self.paths["verified"].stat().st_mode), 0o400)

        receipt = json.loads(self.paths["pipeline_receipt"].read_text())
        digest = receipt.pop("receipt_digest")
        self.assertEqual(digest, _object_sha(receipt))
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["nodes"], NODES)
        self.assertEqual(receipt["first_four_nodes"], NODES[:4])

        before = self.stage_log.read_text()
        rerun = self._run()
        self.assertNotEqual(rerun.returncode, 0)
        self.assertIn("create-only verified smoke gate already exists", rerun.stderr)
        self.assertEqual(self.stage_log.read_text(), before)

    def test_raw_gate_byte_mismatch_stops_before_verified_publish(self) -> None:
        rebuilt = self.root / "different_gate.json"
        rebuilt.write_bytes(self.raw_gate.read_bytes() + b" ")
        result = self._run(FAKE_REBUILT_GATE=str(rebuilt))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh smoke gate bytes differ", result.stderr)
        self.assertFalse(self.paths["verified"].exists())
        self.assertEqual(self.stage_log.read_text(), "")
        self.assertFalse(self.paths["pipeline_receipt"].exists())

    def test_pre_v6_gate_is_rejected_before_verified_publish(self) -> None:
        legacy = self.root / "legacy_v4_gate.json"
        value = json.loads(self.raw_gate.read_text())
        value["schema_version"] = (
            "motive-goku-full-motion-qwen-smoke-gate-v4"
        )
        value.pop("gate_digest")
        value["gate_digest"] = _object_sha(value)
        legacy.write_bytes(_canonical(value) + b"\n")
        result = self._run(
            FAKE_REBUILT_GATE=str(legacy),
            MOTIVE_FULL_MOTION_SMOKE_GATE=str(legacy),
            MOTIVE_FULL_MOTION_SMOKE_GATE_SHA256=_sha(legacy.read_bytes()),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw gate identity or digest differs", result.stderr)
        self.assertFalse(self.paths["verified"].exists())
        self.assertEqual(self.stage_log.read_text(), "")
        self.assertFalse(self.paths["pipeline_receipt"].exists())

    def test_v6_gate_with_pre_v6_record_lineage_is_rejected(self) -> None:
        legacy = self.root / "legacy_record_gate.json"
        value = json.loads(self.raw_gate.read_text())
        value["qwen_lineage"]["record"] = "goku-full-motion-qwen-record-v4"
        value["qwen_runtime"]["run_config"]["schemas"]["record"] = (
            "goku-full-motion-qwen-record-v4"
        )
        for binding in value["hard_pass_bindings"]:
            binding["record_schema_version"] = (
                "goku-full-motion-qwen-record-v4"
            )
        value["canary"]["qwen_record_schema_version"] = (
            "goku-full-motion-qwen-record-v4"
        )
        value.pop("gate_digest")
        value["gate_digest"] = _object_sha(value)
        legacy.write_bytes(_canonical(value) + b"\n")
        result = self._run(
            FAKE_REBUILT_GATE=str(legacy),
            MOTIVE_FULL_MOTION_SMOKE_GATE=str(legacy),
            MOTIVE_FULL_MOTION_SMOKE_GATE_SHA256=_sha(legacy.read_bytes()),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not the frozen two-stage Qwen v6", result.stderr)
        self.assertFalse(self.paths["verified"].exists())
        self.assertEqual(self.stage_log.read_text(), "")

    def test_qwen_failure_stops_all_downstream_stages(self) -> None:
        result = self._run(FAKE_QWEN_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("distributed full-Qwen controller failed", result.stderr)
        self.assertEqual(
            [line.split("\t")[0] for line in self.stage_log.read_text().splitlines()],
            ["watcher_start", "qwen"],
        )
        self.assertFalse(self.paths["wan"].exists())
        self.assertFalse(self.paths["post"].exists())
        self.assertFalse(self.paths["exact"].exists())
        self.assertFalse(self.paths["pipeline_receipt"].exists())

    def test_signed_partition_verifier_failure_stops_before_wan(self) -> None:
        result = self._run(FAKE_RELEASE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "root signed release exact-32x8 partition verification failed",
            result.stderr,
        )
        self.assertEqual(
            [line.split("\t")[0] for line in self.stage_log.read_text().splitlines()],
            ["watcher_start", "qwen", "watcher_done"],
        )
        self.assertFalse(self.paths["wan"].exists())
        self.assertFalse(self.paths["post"].exists())
        self.assertFalse(self.paths["exact"].exists())

    def test_signed_partition_root_index_mismatch_stops_before_wan(self) -> None:
        result = self._run(FAKE_RELEASE_BAD_BINDING_INDEX="7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "release verifier root/partition binding differs for shard 7",
            result.stderr,
        )
        self.assertEqual(len(self.release_log.read_text().splitlines()), 8)
        self.assertEqual(
            [line.split("\t")[0] for line in self.stage_log.read_text().splitlines()],
            ["watcher_start", "qwen", "watcher_done"],
        )
        self.assertFalse(self.paths["wan"].exists())

    def test_invalid_wan_receipt_stops_postcheck_and_selection(self) -> None:
        result = self._run(FAKE_INVALID_WAN_RECEIPT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Wan dispatcher terminal receipt is invalid", result.stderr)
        self.assertEqual(
            [line.split("\t")[0] for line in self.stage_log.read_text().splitlines()],
            ["watcher_start", "qwen", "watcher_done", "wan"],
        )
        self.assertFalse(self.paths["post"].exists())
        self.assertFalse(self.paths["exact"].exists())
        self.assertFalse(self.paths["pipeline_receipt"].exists())


if __name__ == "__main__":
    unittest.main()
