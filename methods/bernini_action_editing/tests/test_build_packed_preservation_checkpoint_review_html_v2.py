#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import clean_source_visual_context_checkpoint_review_contract_v1 as authoring  # noqa: E402
import packed_preservation_checkpoint_review_v2 as review  # noqa: E402
import build_packed_preservation_checkpoint_review_html_v2 as builder  # noqa: E402


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _sha_bytes(value)


def _media(relative: str, sha256: str) -> dict:
    return {
        "relative_mp4": relative,
        "mp4_sha256": sha256,
        "frame_count": review.FRAME_COUNT,
        "fps": review.FPS,
        "codec": "h264",
        "width": 64,
        "height": 48,
        "probe_backend": "unit-test-declaration",
    }


def _resign(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="ascii"))
    mutate(value)
    value.pop("receipt_digest", None)
    value["receipt_digest"] = review.object_sha256(value)
    path.write_bytes(review.canonical_json_bytes(value) + b"\n")


class PackedPreservationHtmlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.manifest_path = self.root / "review-manifest.json"
        self.training_receipt_path = self.root / "training" / "receipt.json"
        self.training_receipt_path.parent.mkdir()
        self.training_receipt_path.write_bytes(b"terminal-training-receipt\n")
        self.manifest, self.source_bytes = self._build_manifest()
        self.manifest_path.write_bytes(review.canonical_json_bytes(self.manifest) + b"\n")
        self.manifest_sha = review.file_sha256(self.manifest_path)
        self.training_sha = review.file_sha256(self.training_receipt_path)
        self.authority = self._build_authority()
        self.shard_root = self.root / "shards"
        self.shard_root.mkdir()
        self._build_shards()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build_manifest(self) -> tuple[dict, dict[str, bytes]]:
        sentinels = []
        source_bytes: dict[str, bytes] = {}
        for index, sentinel_id in enumerate(authoring.SENTINEL_ORDER):
            payload = f"source-video-{sentinel_id}".encode("ascii")
            source_bytes[sentinel_id] = payload
            instruction = (
                f"Make the complete target action happen for {sentinel_id}, while keeping "
                "the original subject, clothing, objects, and scene recognizable."
            )
            sentinels.append(
                {
                    "sentinel_id": sentinel_id,
                    "diversity_role": ("animal", "human", "hand-object", "emitter")[index],
                    "source_entity_type": ("dog", "person", "person-hand", "fireworks")[index],
                    "iid": f"iid-{index}",
                    "source_caption": f"A fixed source description for {sentinel_id}.",
                    "source_video_sha256": _sha_bytes(payload),
                    "seed": 52005001 + index,
                    "instructions": {"forward": instruction},
                    "instruction_sha256": {"forward": _sha_bytes(instruction.encode("utf-8"))},
                }
            )
        unsigned = {
            "schema_version": "unit-test-manifest",
            "source_only_manifest": {
                "file_sha256": review.SOURCE_ONLY_MANIFEST_SHA256,
            },
            "sentinel_order": list(authoring.SENTINEL_ORDER),
            "sentinels": sentinels,
        }
        return {
            **unsigned,
            "manifest_digest": review.object_sha256(unsigned),
        }, source_bytes

    def _build_authority(self) -> review.TrainingAuthority:
        training = self.training_receipt_path.parent
        checkpoints = []
        for step in review.CHECKPOINT_STEPS:
            directory = training / "checkpoints" / f"checkpoint-{step:08d}"
            checkpoints.append(
                review.CheckpointAuthority(
                    step=step,
                    directory=directory,
                    adapter=directory / "adapter.pt",
                    adapter_sha256=_sha_bytes(f"adapter-{step}".encode()),
                    metadata=directory / "metadata.json",
                    metadata_sha256=_sha_bytes(f"metadata-{step}".encode()),
                    parameter_sha256=_sha_bytes(f"parameter-{step}".encode()),
                    inventory_sha256=_sha_bytes(f"inventory-{step}".encode()),
                )
            )
        return review.TrainingAuthority(
            receipt=self.training_receipt_path,
            receipt_file_sha256=self.training_sha,
            receipt_digest=_sha_bytes(b"terminal-digest"),
            lora_scope="all-attention",
            source_only_manifest_sha256=review.SOURCE_ONLY_MANIFEST_SHA256,
            checkpoints=tuple(checkpoints),
        )

    @staticmethod
    def _tensor_identity(label: str) -> dict:
        return {
            "all_rank_exact": True,
            "identity": {
                "label": label,
                "sha256": _sha_bytes(label.encode("ascii")),
            },
        }

    @staticmethod
    def _trace(sentinel_id: str, step: int) -> dict:
        return {
            "calls": review.NUM_INFERENCE_STEPS * 10,
            "source_calls": review.NUM_INFERENCE_STEPS * 9,
            "target_calls": review.NUM_INFERENCE_STEPS,
            "source_tokens": 1000,
            "target_tokens": 100,
            "rows_sha256": _sha_bytes(f"trace-{sentinel_id}-{step}".encode()),
            "source_id_zero_is_target": True,
            "source_id_positive_is_source": True,
            "native_rotary_unchanged": True,
        }

    def _build_shards(self) -> None:
        manifest_by_id = {
            row["sentinel_id"]: row for row in self.manifest["sentinels"]
        }
        stable_base = {
            "path": "/unit/base",
            "tree_sha256": "1" * 64,
            "opened_read_only": True,
        }
        def runtime_for_step(step: int) -> dict:
            release_root = f"/unit/runtime-step-{step}"
            unsigned = {
                "method_root": f"{release_root}/methods/bernini_action_editing",
                "archive": f"{release_root}/method.tar",
                "archive_sha256": "3" * 64,
                "manifest": f"{release_root}/manifest.json",
                "manifest_sha256": "7" * 64,
                "manifest_digest": "8" * 64,
                "method_revision": "2" * 40,
                "exact_member_count": len(builder.release_contract.FILES_AND_MODES),
                "archive_members_verified": True,
                "executed_root_exact_closure_verified": True,
                "executed_file_bound": True,
                "executed_launcher_bound": True,
            }
            return {
                **unsigned,
                "digest": review.object_sha256(unsigned),
                "launcher_sha256": "4" * 64,
            }
        stable_pins = {"bernini_commit": "5" * 40, "veomni_commit": "6" * 40}
        stable_architecture = {"scope": "all-attention", "rank": 256}
        stable_installation = {"strict": True, "projection_count": 240}
        stable_preprocessing = {
            sentinel_id: {"source_sha256": manifest_by_id[sentinel_id]["source_video_sha256"]}
            for sentinel_id in authoring.SENTINEL_ORDER
        }
        stable_prompts = {
            sentinel_id: {
                "full_instruction": manifest_by_id[sentinel_id]["instructions"]["forward"],
                "instruction_utf8_sha256": manifest_by_id[sentinel_id]["instruction_sha256"]["forward"],
                "native_prompt_utf8_sha256": _sha_bytes(f"prompt-{sentinel_id}".encode()),
            }
            for sentinel_id in authoring.SENTINEL_ORDER
        }
        stable_versions = {"torch": "unit", "media": {"probe": "unit"}}
        for step in review.CHECKPOINT_STEPS:
            shard = self.shard_root / f"step-{step:08d}"
            shard.mkdir()
            checkpoint = self.authority.checkpoint(step)
            source_records = []
            decode_records = []
            for sentinel_id in authoring.SENTINEL_ORDER:
                sentinel = manifest_by_id[sentinel_id]
                source_rel = f"media/{sentinel_id}__source.mp4"
                source_sha = _write(shard / source_rel, self.source_bytes[sentinel_id])
                source_records.append(
                    {
                        "sentinel_id": sentinel_id,
                        "iid": sentinel["iid"],
                        "diversity_role": sentinel["diversity_role"],
                        "source_entity_type": sentinel["source_entity_type"],
                        "source_caption": sentinel["source_caption"],
                        "source_video_sha256": source_sha,
                        "full_instruction": sentinel["instructions"]["forward"],
                        "instruction_utf8_sha256": sentinel["instruction_sha256"]["forward"],
                        "seed": sentinel["seed"],
                        **_media(source_rel, source_sha),
                    }
                )
                gaussian_sha = _sha_bytes(f"gaussian-{sentinel_id}".encode())
                gaussian_identity = self._tensor_identity(f"gaussian-{sentinel_id}")
                native_payload = f"native-output-{sentinel_id}".encode("ascii")
                endpoint = self._tensor_identity(f"endpoint-{sentinel_id}-{step}")
                if step == 0:
                    endpoint = self._tensor_identity(f"endpoint-{sentinel_id}-zero")
                    native_rel = f"media/{sentinel_id}__native.mp4"
                    native_sha = _write(shard / native_rel, native_payload)
                    decode_records.append(
                        {
                            "endpoint_identity": endpoint,
                            "initial_gaussian_sha256": gaussian_sha,
                            "initial_gaussian_identity": gaussian_identity,
                            "initial_gaussian_call_count": 1,
                            "sentinel_id": sentinel_id,
                            "arm": "native",
                            "checkpoint_step": None,
                            "adapter_loaded": False,
                            "native_patch_route": None,
                            **_media(native_rel, native_sha),
                        }
                    )
                adapted_payload = (
                    native_payload
                    if step == 0
                    else f"adapted-output-{sentinel_id}-{step}".encode("ascii")
                )
                adapted_rel = f"media/{sentinel_id}__step-{step}.mp4"
                adapted_sha = _write(shard / adapted_rel, adapted_payload)
                decode_records.append(
                    {
                        "endpoint_identity": endpoint,
                        "initial_gaussian_sha256": gaussian_sha,
                        "initial_gaussian_identity": gaussian_identity,
                        "initial_gaussian_call_count": 1,
                        "sentinel_id": sentinel_id,
                        "arm": f"step-{step}",
                        "checkpoint_step": step,
                        "adapter_loaded": True,
                        "checkpoint_parameter_sha256": checkpoint.parameter_sha256,
                        "native_patch_route": self._trace(sentinel_id, step),
                        "step_zero_native_endpoint_equal": step == 0,
                        **_media(adapted_rel, adapted_sha),
                    }
                )
            unsigned = {
                "schema_version": builder.SHARD_SCHEMA,
                "method": builder.SHARD_METHOD,
                "complete": True,
                "lora_scope": "all-attention",
                "execution_scope": "exact80",
                "checkpoint_step": step,
                "sentinel_order": list(authoring.SENTINEL_ORDER),
                "smoke_only": False,
                "review_manifest": {
                    "path": str(self.manifest_path),
                    "file_sha256": self.manifest_sha,
                    "manifest_digest": self.manifest["manifest_digest"],
                },
                "checkpoint_authority": checkpoint.receipt(),
                "continuous_training_run": str(self.training_receipt_path.parent),
                "terminal_training_receipt_bound": True,
                "terminal_training_authority": self.authority.as_receipt(),
                "base_checkpoint": stable_base,
                "runtime_source": runtime_for_step(step),
                "pinned_sources": stable_pins,
                "architecture": stable_architecture,
                "lora_installation": stable_installation,
                "checkpoint_load": {
                    "step": step,
                    "adapter_sha256": checkpoint.adapter_sha256,
                    "parameter_sha256": checkpoint.parameter_sha256,
                    "inventory_sha256": checkpoint.inventory_sha256,
                    "parameter_unchanged_during_inference": True,
                },
                "source_preprocessing": stable_preprocessing,
                "prompts": stable_prompts,
                "source_records": source_records,
                "decode_records": decode_records,
                "execution": {
                    "world_size": review.WORLD_SIZE,
                    "sequence_parallel_size": review.SP_SIZE,
                    "num_inference_steps": review.NUM_INFERENCE_STEPS,
                    "frame_count": review.FRAME_COUNT,
                    "fps": review.FPS,
                    "native_rv2v_four_references": True,
                    "same_source_instruction_seed_all_columns": True,
                    "step_zero_endpoint_byte_exact_native": step == 0,
                    "step_zero_media_byte_copy_from_native": step == 0,
                    "parent_allocation_released": False,
                },
                "runtime_versions": stable_versions,
                "optimizer_present": False,
                "backward_performed": False,
                "parameter_update_performed": False,
                "feature_evaluator_present": False,
                "vlm_evaluator_present": False,
                "automatic_ranking_present": False,
                "candidate_selection_present": False,
                "quality_claimed": False,
                "scientific_claim_authorized": False,
                "manual_review_pending": True,
            }
            receipt = {**unsigned, "receipt_digest": review.object_sha256(unsigned)}
            (shard / "receipt.json").write_bytes(
                review.canonical_json_bytes(receipt) + b"\n"
            )

    def _build(self, output_name: str = "review") -> dict:
        with mock.patch.object(
            builder.authoring, "load_manifest", return_value=self.manifest
        ) as load_manifest, mock.patch.object(
            builder.review, "load_training_authority", return_value=self.authority
        ) as load_training:
            result = builder.build_review(
                manifest_path=self.manifest_path,
                expected_manifest_sha256=self.manifest_sha,
                training_receipt_path=self.training_receipt_path,
                expected_training_receipt_sha256=self.training_sha,
                lora_scope=self.authority.lora_scope,
                shard_root=self.shard_root,
                output_dir=self.root / output_name,
                verify_manifest_files=False,
                verify_training_files=False,
                verify_media=False,
            )
        load_manifest.assert_called_once_with(
            self.manifest_path,
            expected_file_sha256=self.manifest_sha,
            verify_files=False,
        )
        load_training.assert_called_once_with(
            self.training_receipt_path,
            expected_file_sha256=self.training_sha,
            expected_lora_scope=self.authority.lora_scope,
            verify_files=False,
        )
        return dict(result)

    def test_builds_four_sentinel_seven_column_content_addressed_packet(self) -> None:
        result = self._build()
        output = Path(result["output"])
        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertEqual(page.count('class="sentinel"'), 4)
        for label in (
            "Source", "Native", "Optimizer update 0", "Optimizer update 20",
            "Optimizer update 40", "Optimizer update 60", "Optimizer update 80",
        ):
            self.assertIn(f">{label}<", page)
        for sentinel in self.manifest["sentinels"]:
            self.assertIn(sentinel["sentinel_id"], page)
            self.assertIn(sentinel["source_caption"], page)
            self.assertIn(sentinel["instructions"]["forward"], page)
            self.assertIn(str(sentinel["seed"]), page)
        lowered = page.lower()
        for forbidden in ('<script src=', "http://", "https://"):
            self.assertNotIn(forbidden, lowered)
        self.assertEqual(page.count("<script>"), 1)
        self.assertEqual(page.count("</script>"), 1)
        self.assertEqual(page.count('class="sync-controls"'), 4)
        self.assertEqual(page.count('<button type="button" data-sync-action="play">'), 4)
        self.assertEqual(page.count('<button type="button" data-sync-action="pause">'), 4)
        self.assertEqual(page.count('<button type="button" data-sync-action="restart">'), 4)
        self.assertEqual(page.count('<video controls muted playsinline'), 28)
        self.assertNotIn('<video controls loop', page)
        self.assertEqual(page.count('class="sync-status" role="status" aria-live="polite"'), 4)
        self.assertEqual(page.count("同时播放 / Play together"), 4)
        self.assertEqual(page.count("同时暂停 / Pause together"), 4)
        self.assertEqual(page.count("从头同步 / Restart together"), 4)
        self.assertIn("FRAME_SECONDS = 1 / 25", page)
        self.assertIn("SOFT_DRIFT_SECONDS = 0.012", page)
        self.assertIn("requestVideoFrameCallback", page)
        self.assertIn("requestAnimationFrame", page)
        self.assertNotIn("setInterval", page)
        self.assertIn('section.querySelectorAll("video")', page)
        self.assertIn("Promise.allSettled", page)
        self.assertIn("seekTogether(requested)", page)
        self.assertIn("Math.abs(drift) > FRAME_SECONDS", page)
        self.assertIn('video.addEventListener("pause"', page)
        self.assertIn('video.addEventListener("seeking"', page)
        self.assertIn('video.addEventListener("waiting"', page)
        self.assertIn('video.addEventListener("ratechange"', page)
        self.assertIn('document.addEventListener("visibilitychange"', page)
        self.assertIn('section.setAttribute("aria-busy"', page)
        inline_script = page.split("<script>", 1)[1].split("</script>", 1)[0].lower()
        for forbidden in ("fetch(", "xmlhttprequest", "websocket", "eval(", "import("):
            self.assertNotIn(forbidden, inline_script)
        self.assertIn("All-attention main", page)
        self.assertIn("188,946,432", page)
        self.assertIn("64 real source videos", page)
        self.assertIn("640 logical training records", page)
        self.assertIn("noop 256; cube 128; speed 128; tube 128", page)
        self.assertIn("not a score, value, reward, or quality judgment", page)
        self.assertIn("no automatic ranking or candidate selection", page)
        media = list((output / "media").glob("*.mp4"))
        self.assertEqual(len(media), 24)
        self.assertTrue(all(path.stem == review.file_sha256(path) for path in media))
        evidence = json.loads((output / "evidence.json").read_text(encoding="ascii"))
        self.assertEqual(evidence["training_authority"]["receipt_file_sha256"], self.training_sha)
        self.assertTrue(evidence["step_zero_native_byte_equality"])
        self.assertTrue(evidence["same_gaussian_per_sentinel_all_columns"])

    def test_sync_playback_state_machine_is_fail_closed_and_frame_aligned(self) -> None:
        script = builder._SYNC_PLAYBACK_SCRIPT[len("<script>") : -len("</script>")]
        harness = r'''
class EventTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(name, fn) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(fn);
  }
  removeEventListener(name, fn) {
    const rows = this.listeners.get(name) || [];
    this.listeners.set(name, rows.filter((row) => row !== fn));
  }
  dispatch(name) {
    for (const fn of [...(this.listeners.get(name) || [])]) fn({type: name, target: this});
  }
}
class Control extends EventTarget { click() { this.dispatch("click"); } }
let nextFrameId = 1;
class FakeVideo extends EventTarget {
  constructor() {
    super();
    this.readyState = 4;
    this.duration = 3.24;
    this._currentTime = 0;
    this.seeking = false;
    this.paused = true;
    this.ended = false;
    this.playbackRate = 1;
    this.rejectPlay = false;
    this.playCalls = 0;
    this.loadCalls = 0;
    this.deferPlay = false;
    this.resolvePlay = null;
    this.frames = new Map();
  }
  get currentTime() { return this._currentTime; }
  set currentTime(value) {
    this._currentTime = value;
    this.seeking = true;
    this.dispatch("seeking");
    queueMicrotask(() => { this.seeking = false; this.dispatch("seeked"); });
  }
  setRawTime(value) { this._currentTime = value; }
  load() { this.loadCalls += 1; }
  play() {
    this.playCalls += 1;
    if (this.rejectPlay) return Promise.reject(new Error("blocked"));
    this.paused = false;
    this.ended = false;
    if (this.deferPlay) {
      return new Promise((resolve) => { this.resolvePlay = resolve; });
    }
    return Promise.resolve();
  }
  pause() {
    if (!this.paused) {
      this.paused = true;
      this.dispatch("pause");
    }
  }
  requestVideoFrameCallback(fn) {
    const id = nextFrameId++;
    this.frames.set(id, fn);
    return id;
  }
  cancelVideoFrameCallback(id) { this.frames.delete(id); }
  fireFrame(mediaTime) {
    this.setRawTime(mediaTime);
    const callbacks = [...this.frames.values()];
    this.frames.clear();
    callbacks.forEach((fn) => fn(0, {mediaTime}));
  }
}
class Section {
  constructor() {
    this.videos = Array.from({length: 7}, () => new FakeVideo());
    this.status = {textContent: ""};
    this.play = new Control();
    this.pause = new Control();
    this.restart = new Control();
    this.attributes = new Map();
  }
  setAttribute(name, value) { this.attributes.set(name, value); }
  querySelectorAll(selector) { return selector === "video" ? this.videos : []; }
  querySelector(selector) {
    if (selector === ".sync-status") return this.status;
    if (selector.includes('"play"')) return this.play;
    if (selector.includes('"pause"')) return this.pause;
    if (selector.includes('"restart"')) return this.restart;
    return null;
  }
}
const sections = [new Section(), new Section()];
const documentTarget = new EventTarget();
global.document = Object.assign(documentTarget, {
  hidden: false,
  querySelectorAll: (selector) => selector === ".sentinel" ? sections : [],
});
global.HTMLMediaElement = {HAVE_METADATA: 1, HAVE_CURRENT_DATA: 2};
global.window = {
  setTimeout,
  clearTimeout,
  requestAnimationFrame: (fn) => setTimeout(() => fn(0), 0),
  cancelAnimationFrame: clearTimeout,
};
const flush = () => new Promise((resolve) => setTimeout(resolve, 5));
const settle = async () => { await flush(); await flush(); await flush(); };
const check = (value, message) => { if (!value) throw new Error(message); };
'''
        assertions = r'''
(async () => {
  const first = sections[0];
  const second = sections[1];

  first.play.click();
  await settle();
  check(first.videos.every((video) => !video.paused), "seven-way play did not start");

  first.videos.slice(1).forEach((video) => video.setRawTime(0));
  first.videos[0].fireFrame(0.5);
  await settle();
  check(first.videos.slice(1).every((video) => Math.abs(video.currentTime - 0.5) <= 0.04),
        "hard resync exceeded one frame");
  first.videos[1].setRawTime(0.52);
  first.videos[0].fireFrame(0.5);
  check(first.videos[1].playbackRate < 1 && first.videos[1].playbackRate >= 0.97,
        "soft drift did not use bounded rate correction");

  first.pause.click();
  await settle();
  check(first.videos.every((video) => video.paused), "pause did not stop all videos");
  first.restart.click();
  await settle();
  check(first.videos.every((video) => !video.paused && Math.abs(video.currentTime) < 0.001),
        "restart did not seek and play all videos");

  first.videos.forEach((video) => video.setRawTime(3));
  first.videos[0].ended = true;
  first.videos[0].paused = true;
  first.videos[0].dispatch("pause");
  first.videos[0].dispatch("ended");
  await settle();
  check(first.videos.every((video) => !video.paused && Math.abs(video.currentTime) < 0.001),
        "master-ended loop was not centralized");

  second.videos[0].requestVideoFrameCallback = null;
  second.videos[0].cancelVideoFrameCallback = null;
  second.videos.forEach((video) => { video.readyState = 0; });
  second.play.click();
  await flush();
  check(first.videos.every((video) => video.paused), "starting another sample did not pause the first");
  check(second.videos.every((video) => video.playCalls === 0), "readiness barrier was bypassed");
  second.videos.forEach((video) => { video.readyState = 2; video.dispatch("loadeddata"); });
  await settle();
  check(second.videos.every((video) => !video.paused), "delayed readiness did not resume group start");
  second.videos.slice(1).forEach((video) => video.setRawTime(0));
  second.videos[0].setRawTime(0.5);
  await settle();
  check(second.videos.slice(1).every((video) => Math.abs(video.currentTime - 0.5) <= 0.04),
        "requestAnimationFrame fallback did not align followers");

  second.videos[2].pause();
  await settle();
  check(second.videos.every((video) => video.paused), "manual pause was not fail-closed");
  second.videos[3].rejectPlay = true;
  second.restart.click();
  await settle();
  check(second.videos.every((video) => video.paused), "play rejection left another video running");
  check(second.status.textContent.includes("failed") || second.status.textContent.includes("无法"),
        "play rejection was not announced");

  second.videos[3].rejectPlay = false;
  second.play.click();
  await settle();
  second.videos[4].currentTime = 1.2;
  await settle();
  check(second.videos.every((video) => video.paused), "manual seek was not fail-closed");

  second.play.click();
  await settle();
  second.videos[0].playbackRate = 1.5;
  second.videos[0].dispatch("ratechange");
  await settle();
  check(second.videos.every((video) => video.paused), "manual playback-rate change was not fail-closed");

  second.play.click();
  await settle();
  second.videos[5].dispatch("waiting");
  await settle();
  check(second.videos.every((video) => video.paused), "buffer waiting was not fail-closed");

  second.play.click();
  await settle();
  document.hidden = true;
  document.dispatch("visibilitychange");
  await settle();
  check(second.videos.every((video) => video.paused), "hidden page did not pause active group");

  document.hidden = false;
  first.videos[6].deferPlay = true;
  first.play.click();
  await flush();
  first.videos[2].pause();
  first.videos[6].resolvePlay();
  await settle();
  check(first.videos.every((video) => video.paused), "pending play promise resurrected a cancelled group");
  console.log("SYNC_STATE_MACHINE_PASS");
})().catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
'''
        runtime = self.root / "sync-playback-runtime-test.js"
        runtime.write_text(harness + "\n" + script + "\n" + assertions, encoding="utf-8")
        completed = subprocess.run(
            ["node", str(runtime)],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("SYNC_STATE_MACHINE_PASS", completed.stdout)

    def test_self_attention_control_has_unambiguous_scope_and_parameter_count(self) -> None:
        self.authority = replace(self.authority, lora_scope="self-attention")
        for step in review.CHECKPOINT_STEPS:
            path = self.shard_root / f"step-{step:08d}" / "receipt.json"

            def mutate(value: dict) -> None:
                value["lora_scope"] = "self-attention"
                value["terminal_training_authority"] = self.authority.as_receipt()
                value["architecture"]["scope"] = "self-attention"

            _resign(path, mutate)
        result = self._build("self-review")
        page = (Path(result["output"]) / "index.html").read_text("utf-8")
        self.assertIn("Self-attention control", page)
        self.assertIn("94,574,592", page)

    def test_rejects_shard_not_bound_to_terminal_checkpoint(self) -> None:
        path = self.shard_root / "step-00000040" / "receipt.json"

        def mutate(value: dict) -> None:
            value["checkpoint_authority"]["parameter_sha256"] = "9" * 64

        _resign(path, mutate)
        with self.assertRaisesRegex(
            builder.PackedPreservationHtmlError, "terminal training authority"
        ):
            self._build()

    def test_rejects_missing_terminal_training_authority(self) -> None:
        path = self.shard_root / "step-00000020" / "receipt.json"

        def mutate(value: dict) -> None:
            value["terminal_training_receipt_bound"] = False
            value["terminal_training_authority"] = None

        _resign(path, mutate)
        with self.assertRaisesRegex(
            builder.PackedPreservationHtmlError, "formal decode authority"
        ):
            self._build()

    def test_rejects_tampered_terminal_training_authority(self) -> None:
        path = self.shard_root / "step-00000060" / "receipt.json"

        def mutate(value: dict) -> None:
            value["terminal_training_authority"]["training_receipt_digest"] = "9" * 64

        _resign(path, mutate)
        with self.assertRaisesRegex(
            builder.PackedPreservationHtmlError, "formal decode authority"
        ):
            self._build()

    def test_rejects_gaussian_drift_across_checkpoint(self) -> None:
        path = self.shard_root / "step-00000060" / "receipt.json"

        def mutate(value: dict) -> None:
            value["decode_records"][0]["initial_gaussian_sha256"] = "9" * 64

        _resign(path, mutate)
        with self.assertRaisesRegex(builder.PackedPreservationHtmlError, "same official Gaussian"):
            self._build()

    def test_rejects_step_zero_native_media_drift(self) -> None:
        path = self.shard_root / "step-00000000" / "receipt.json"

        def mutate(value: dict) -> None:
            row = value["decode_records"][1]
            media_path = path.parent / row["relative_mp4"]
            row["mp4_sha256"] = _write(media_path, b"different-step-zero-output")

        _resign(path, mutate)
        with self.assertRaisesRegex(builder.PackedPreservationHtmlError, "equality gate"):
            self._build()

    def test_rejects_source_instruction_or_seed_drift(self) -> None:
        path = self.shard_root / "step-00000020" / "receipt.json"

        def mutate(value: dict) -> None:
            value["source_records"][0]["seed"] += 1

        _resign(path, mutate)
        with self.assertRaisesRegex(builder.PackedPreservationHtmlError, "source/instruction/seed"):
            self._build()

    def test_rejects_media_bytes_changed_after_receipt(self) -> None:
        receipt = json.loads(
            (self.shard_root / "step-00000080" / "receipt.json").read_text(encoding="ascii")
        )
        row = receipt["decode_records"][0]
        (self.shard_root / "step-00000080" / row["relative_mp4"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(builder.PackedPreservationHtmlError, "MP4 bytes differ"):
            self._build()


if __name__ == "__main__":
    unittest.main()
