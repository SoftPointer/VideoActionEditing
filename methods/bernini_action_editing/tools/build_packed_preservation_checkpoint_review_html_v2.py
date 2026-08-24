#!/usr/bin/env python3
"""Build a sealed native/0/20/40/60/80 packed-preservation review packet.

The builder is deliberately a provenance and presentation step.  It accepts
only the five formal exact80 decode shards, binds every checkpoint identity to
the completed training receipt, verifies the fixed four-sentinel source /
instruction / seed registry, and copies every referenced video into a fresh
content-addressed packet.  It does not evaluate or choose outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_checkpoint_review_contract_v1 as authoring  # noqa: E402
import packed_preservation_checkpoint_review_v2 as review  # noqa: E402
import packed_preservation_checkpoint_review_release_v2 as release_contract  # noqa: E402


SCHEMA_VERSION = "bernini-packed-preservation-checkpoint-review-html-v2"
SHARD_SCHEMA = "bernini-packed-preservation-checkpoint-decode-shard-v2"
SHARD_METHOD = "bernini-packed-preservation-checkpoint-review-v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_SCOPE_PRESENTATION = {
    "all-attention": ("All-attention main", 188_946_432),
    "self-attention": ("Self-attention control", 94_574_592),
}
_SYNC_PLAYBACK_SCRIPT = """<script>
(() => {
  "use strict";

  const FRAME_SECONDS = 1 / 25;
  const SOFT_DRIFT_SECONDS = 0.012;
  const READY_TIMEOUT_MS = 12000;
  let activeController = null;

  const waitUntil = (video, predicate, events, label) => {
    if (predicate()) return Promise.resolve();
    return new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        events.forEach((event) => video.removeEventListener(event, onProgress));
        video.removeEventListener("error", onError);
        window.clearTimeout(timeout);
      };
      const finish = (error) => {
        if (settled) return;
        settled = true;
        cleanup();
        error ? reject(error) : resolve();
      };
      const onProgress = () => {
        if (predicate()) finish();
      };
      const onError = () => finish(new Error(`${label}: media error`));
      const timeout = window.setTimeout(
        () => finish(new Error(`${label}: timed out`)),
        READY_TIMEOUT_MS,
      );
      events.forEach((event) => video.addEventListener(event, onProgress));
      video.addEventListener("error", onError);
    });
  };

  document.querySelectorAll(".sentinel").forEach((section) => {
    const videos = Array.from(section.querySelectorAll("video"));
    if (videos.length !== 7) return;
    const master = videos[0];
    const status = section.querySelector(".sync-status");
    const playButton = section.querySelector('[data-sync-action="play"]');
    const pauseButton = section.querySelector('[data-sync-action="pause"]');
    const restartButton = section.querySelector('[data-sync-action="restart"]');
    const programmaticPause = new WeakSet();
    const programmaticSeek = new WeakSet();
    const programmaticRateChanges = new WeakMap();
    let active = false;
    let busy = false;
    let command = 0;
    let frameCallback = null;
    let frameCallbackKind = null;

    const setStatus = (message) => {
      status.textContent = message;
    };

    const setBusy = (value) => {
      busy = value;
      section.setAttribute("aria-busy", value ? "true" : "false");
      playButton.disabled = value;
      restartButton.disabled = value;
    };

    const cancelFrameSync = () => {
      if (frameCallback === null) return;
      if (frameCallbackKind === "video" && master.cancelVideoFrameCallback) {
        master.cancelVideoFrameCallback(frameCallback);
      } else if (frameCallbackKind === "animation") {
        window.cancelAnimationFrame(frameCallback);
      }
      frameCallback = null;
      frameCallbackKind = null;
    };

    const masterTime = () => {
      const value = master.currentTime;
      return Number.isFinite(value) ? value : 0;
    };

    const clampTime = (video, requested) => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) return Math.max(0, requested);
      return Math.max(0, Math.min(requested, Math.max(0, video.duration - FRAME_SECONDS)));
    };

    const setPlaybackRate = (video, value) => {
      if (Math.abs(video.playbackRate - value) < 0.0001) return;
      programmaticRateChanges.set(video, (programmaticRateChanges.get(video) || 0) + 1);
      video.playbackRate = value;
    };

    const pauseVideos = () => {
      videos.forEach((video) => {
        setPlaybackRate(video, 1);
        if (!video.paused) {
          programmaticPause.add(video);
          video.pause();
        }
      });
    };

    const seekOne = async (video, requested) => {
      await waitUntil(
        video,
        () => video.readyState >= HTMLMediaElement.HAVE_METADATA,
        ["loadedmetadata", "durationchange"],
        "metadata",
      );
      const target = clampTime(video, requested);
      if (!video.seeking && Math.abs(video.currentTime - target) < 0.001) return;
      programmaticSeek.add(video);
      video.currentTime = target;
      try {
        await waitUntil(video, () => !video.seeking, ["seeked"], "seek");
      } finally {
        programmaticSeek.delete(video);
      }
    };

    const seekTogether = async (requested) => {
      await Promise.all(videos.map((video) => seekOne(video, requested)));
    };

    const stop = (message, align = true) => {
      command += 1;
      active = false;
      setBusy(false);
      cancelFrameSync();
      pauseVideos();
      if (activeController === controller) activeController = null;
      if (align && videos.every((video) => video.readyState >= HTMLMediaElement.HAVE_METADATA)) {
        void seekTogether(masterTime()).catch(() => {});
      }
      setStatus(message);
    };

    const correctFollowers = (reportedMasterTime) => {
      if (!active) return;
      const reference = Number.isFinite(reportedMasterTime) ? reportedMasterTime : masterTime();
      videos.slice(1).forEach((video) => {
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
        const drift = video.currentTime - reference;
        if (Math.abs(drift) > FRAME_SECONDS) {
          programmaticSeek.add(video);
          video.currentTime = clampTime(video, reference);
        } else if (Math.abs(drift) > SOFT_DRIFT_SECONDS) {
          setPlaybackRate(video, Math.max(0.97, Math.min(1.03, 1 - drift * 0.75)));
        } else {
          setPlaybackRate(video, 1);
        }
      });
    };

    const scheduleFrameSync = () => {
      cancelFrameSync();
      if (!active) return;
      if (master.requestVideoFrameCallback) {
        frameCallbackKind = "video";
        frameCallback = master.requestVideoFrameCallback((_now, metadata) => {
          frameCallback = null;
          frameCallbackKind = null;
          correctFollowers(metadata && metadata.mediaTime);
          scheduleFrameSync();
        });
      } else {
        frameCallbackKind = "animation";
        frameCallback = window.requestAnimationFrame(() => {
          frameCallback = null;
          frameCallbackKind = null;
          correctFollowers(masterTime());
          scheduleFrameSync();
        });
      }
    };

    const start = async (requested, message) => {
      const token = ++command;
      if (activeController && activeController !== controller) {
        activeController.stop("已暂停：开始播放另一个样本。 / Paused: another sample started.");
      }
      activeController = controller;
      active = false;
      setBusy(true);
      cancelFrameSync();
      pauseVideos();
      setStatus("正在对齐 7 个视频… / Aligning 7 videos…");
      try {
        videos.forEach((video) => {
          if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) video.load();
        });
        await Promise.all(videos.map((video) => waitUntil(
          video,
          () => video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA,
          ["loadeddata", "canplay"],
          "media readiness",
        )));
        if (token !== command) return;
        await seekTogether(requested);
        if (token !== command) return;
        const results = await Promise.allSettled(videos.map((video) => video.play()));
        if (token !== command) {
          pauseVideos();
          return;
        }
        const rejected = results.filter((result) => result.status === "rejected");
        if (rejected.length || videos.some((video) => video.paused)) {
          const detail = rejected.length
            ? `${rejected.length} 路被浏览器阻止`
            : "至少一路未进入播放状态";
          stop(`无法同时播放：${detail}。 / Sync play failed.`, false);
          return;
        }
        setBusy(false);
        active = true;
        setStatus(`${message} 7 个视频。 / ${message === "从头同步播放" ? "Restarted" : "Playing"} 7 videos together.`);
        scheduleFrameSync();
      } catch (error) {
        if (token === command) {
          stop(`无法同时播放：${error.message} / Sync play failed.`, false);
        }
      }
    };

    const controller = { stop };

    playButton.addEventListener("click", () => {
      const requested = master.ended ? 0 : masterTime();
      void start(requested, "同步播放");
    });

    pauseButton.addEventListener("click", () => {
      stop("已同时暂停 7 个视频。 / Paused 7 videos together.");
    });

    restartButton.addEventListener("click", () => {
      void start(0, "从头同步播放");
    });

    videos.forEach((video) => {
      video.addEventListener("pause", () => {
        if (programmaticPause.delete(video) || video.ended || (!active && !busy)) return;
        stop("已全部暂停：其中一路被单独暂停。 / Paused: one video was paused.");
      });
      video.addEventListener("seeking", () => {
        if (programmaticSeek.has(video) || (!active && !busy)) return;
        stop("已全部暂停：其中一路被拖动。再次点击同时播放会对齐到 Source。 / Paused after a manual seek.", false);
      });
      video.addEventListener("seeked", () => programmaticSeek.delete(video));
      video.addEventListener("ratechange", () => {
        const pending = programmaticRateChanges.get(video) || 0;
        if (pending) {
          pending === 1
            ? programmaticRateChanges.delete(video)
            : programmaticRateChanges.set(video, pending - 1);
          return;
        }
        if (active || busy) {
          stop("已全部暂停：其中一路速度被更改。 / Paused after a playback-rate change.", false);
        }
      });
      video.addEventListener("error", () => {
        if (active || busy) stop("已全部暂停：媒体读取错误。 / Paused after a media error.", false);
      });
      video.addEventListener("stalled", () => {
        if (active || busy) stop("已全部暂停：其中一路加载停滞。 / Paused after a media stall.", false);
      });
      video.addEventListener("waiting", () => {
        if (active || busy) stop("已全部暂停：其中一路等待下一帧。 / Paused while one video buffers.", false);
      });
    });

    master.addEventListener("ended", () => {
      if (active) void start(0, "从头同步播放");
    });
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && activeController) {
      activeController.stop("页面隐藏，已全部暂停。 / Paused while the page is hidden.");
    }
  });
})();
</script>"""
_ROOT_FIELDS = {
    "schema_version",
    "method",
    "complete",
    "lora_scope",
    "execution_scope",
    "checkpoint_step",
    "sentinel_order",
    "smoke_only",
    "review_manifest",
    "checkpoint_authority",
    "continuous_training_run",
    "terminal_training_receipt_bound",
    "terminal_training_authority",
    "base_checkpoint",
    "runtime_source",
    "pinned_sources",
    "architecture",
    "lora_installation",
    "checkpoint_load",
    "source_preprocessing",
    "prompts",
    "source_records",
    "decode_records",
    "execution",
    "runtime_versions",
    "optimizer_present",
    "backward_performed",
    "parameter_update_performed",
    "feature_evaluator_present",
    "vlm_evaluator_present",
    "automatic_ranking_present",
    "candidate_selection_present",
    "quality_claimed",
    "scientific_claim_authorized",
    "manual_review_pending",
    "receipt_digest",
}
_MEDIA_FIELDS = {
    "relative_mp4",
    "mp4_sha256",
    "frame_count",
    "fps",
    "codec",
    "width",
    "height",
    "probe_backend",
}
_SOURCE_FIELDS = {
    "sentinel_id",
    "iid",
    "diversity_role",
    "source_entity_type",
    "source_caption",
    "source_video_sha256",
    "full_instruction",
    "instruction_utf8_sha256",
    "seed",
    *_MEDIA_FIELDS,
}
_DECODE_COMMON_FIELDS = {
    "endpoint_identity",
    "initial_gaussian_sha256",
    "initial_gaussian_identity",
    "initial_gaussian_call_count",
    "sentinel_id",
    "arm",
    "checkpoint_step",
    "adapter_loaded",
    "native_patch_route",
    *_MEDIA_FIELDS,
}
_DECODE_ADAPTER_FIELDS = {
    *_DECODE_COMMON_FIELDS,
    "checkpoint_parameter_sha256",
    "step_zero_native_endpoint_equal",
}


class PackedPreservationHtmlError(RuntimeError):
    """Raised before a partial or ambiguous review packet is published."""


def fail(message: str) -> NoReturn:
    raise PackedPreservationHtmlError(message)


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        fail(f"{label} must be non-empty text")
    return value


def _plain_root(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PackedPreservationHtmlError(f"{label} is unavailable") from error
    if resolved != path or not path.is_dir() or path.is_symlink():
        fail(f"{label} must be a canonical plain directory")
    return path


def _safe_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or _SAFE_NAME.fullmatch(path.name) is None
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        fail("output-dir must be one fresh safe absolute child of a plain parent")
    return path


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} must be a plain file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackedPreservationHtmlError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def _embedded_receipt_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = _sha(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if review.object_sha256(unsigned) != declared:
        fail(f"{label} embedded digest differs")
    return declared


def _probe_exact81(path: Path) -> Mapping[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,codec_name,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        root = json.loads(completed.stdout)
        streams = root.get("streams")
        stream = streams[0] if isinstance(streams, list) and len(streams) == 1 else None
        if not isinstance(stream, Mapping):
            fail(f"{path} must have one video stream")
        numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
        result = {
            "frame_count": int(stream["nb_read_frames"]),
            "fps": float(numerator) / float(denominator),
            "codec": str(stream["codec_name"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
        }
    except PackedPreservationHtmlError:
        raise
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError,
            TypeError, ValueError, ZeroDivisionError) as error:
        raise PackedPreservationHtmlError(f"cannot fully probe {path}") from error
    if (
        result["frame_count"] != review.FRAME_COUNT
        or abs(result["fps"] - float(review.FPS)) > 1.0e-9
        or result["width"] <= 0
        or result["height"] <= 0
        or not result["codec"]
    ):
        fail(f"{path} is not exact81 25-fps media")
    return result


def _media_path(
    *,
    shard: Path,
    record: Mapping[str, Any],
    label: str,
    verify_media: bool,
) -> Path:
    if not _MEDIA_FIELDS.issubset(record):
        fail(f"{label} media fields differ")
    sha = _sha(record.get("mp4_sha256"), label=f"{label} MP4 SHA")
    if (
        record.get("frame_count") != review.FRAME_COUNT
        or record.get("fps") != review.FPS
        or type(record.get("width")) is not int
        or int(record["width"]) <= 0
        or type(record.get("height")) is not int
        or int(record["height"]) <= 0
        or not isinstance(record.get("codec"), str)
        or not record["codec"]
        or not isinstance(record.get("probe_backend"), str)
        or not record["probe_backend"]
    ):
        fail(f"{label} exact81 media declaration differs")
    relative = Path(str(record.get("relative_mp4")))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".mp4":
        fail(f"{label} MP4 path is unsafe")
    try:
        root = shard.resolve(strict=True)
        path = (shard / relative).resolve(strict=True)
    except OSError as error:
        raise PackedPreservationHtmlError(f"{label} MP4 is unavailable") from error
    if path == root or root not in path.parents or path.is_symlink() or not path.is_file():
        fail(f"{label} MP4 escapes the shard")
    if review.file_sha256(path) != sha:
        fail(f"{label} MP4 bytes differ")
    if verify_media:
        observed = _probe_exact81(path)
        if any(observed[key] != record[key] for key in ("frame_count", "fps", "codec", "width", "height")):
            fail(f"{label} decoded media metadata differs")
    return path


def _validate_source_records(
    *,
    shard: Path,
    value: Any,
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    verify_media: bool,
) -> tuple[Mapping[str, Any], ...]:
    if (
        not isinstance(value, list)
        or tuple(row.get("sentinel_id") for row in value if isinstance(row, Mapping))
        != authoring.SENTINEL_ORDER
    ):
        fail("source record order differs")
    rows: list[Mapping[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _SOURCE_FIELDS:
            fail("source record fields differ")
        sentinel_id = str(raw["sentinel_id"])
        sentinel = manifest_by_id[sentinel_id]
        instruction = sentinel["instructions"]["forward"]
        expected = {
            "iid": sentinel["iid"],
            "diversity_role": sentinel["diversity_role"],
            "source_entity_type": sentinel["source_entity_type"],
            "source_caption": sentinel["source_caption"],
            "source_video_sha256": sentinel["source_video_sha256"],
            "full_instruction": instruction,
            "instruction_utf8_sha256": sentinel["instruction_sha256"]["forward"],
            "seed": sentinel["seed"],
        }
        if any(raw.get(key) != expected_value for key, expected_value in expected.items()):
            fail(f"{sentinel_id} source/instruction/seed differs from fixed manifest")
        if raw["mp4_sha256"] != raw["source_video_sha256"]:
            fail(f"{sentinel_id} self-contained source bytes differ")
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != raw["instruction_utf8_sha256"]:
            fail(f"{sentinel_id} full instruction bytes differ")
        _media_path(
            shard=shard,
            record=raw,
            label=f"{sentinel_id} source",
            verify_media=verify_media,
        )
        rows.append(dict(raw))
    return tuple(rows)


def _validate_decode_records(
    *,
    shard: Path,
    value: Any,
    step: int,
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    checkpoint_parameter_sha256: str,
    verify_media: bool,
) -> tuple[Mapping[str, Any], ...]:
    expected_arms = ("native", f"step-{step}") if step == 0 else (f"step-{step}",)
    expected_coordinates = tuple(
        (sentinel_id, arm)
        for sentinel_id in authoring.SENTINEL_ORDER
        for arm in expected_arms
    )
    if (
        not isinstance(value, list)
        or tuple(
            (row.get("sentinel_id"), row.get("arm"))
            for row in value
            if isinstance(row, Mapping)
        )
        != expected_coordinates
    ):
        fail(f"checkpoint {step} decode record order differs")
    rows: list[Mapping[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            fail(f"checkpoint {step} decode record differs")
        native = raw.get("arm") == "native"
        expected_fields = _DECODE_COMMON_FIELDS if native else _DECODE_ADAPTER_FIELDS
        if set(raw) != expected_fields:
            fail(f"checkpoint {step} decode record fields differ")
        sentinel_id = str(raw["sentinel_id"])
        sentinel = manifest_by_id[sentinel_id]
        _sha(raw.get("initial_gaussian_sha256"), label="initial Gaussian SHA")
        if (
            not isinstance(raw.get("endpoint_identity"), Mapping)
            or not isinstance(raw.get("initial_gaussian_identity"), Mapping)
            or raw.get("initial_gaussian_call_count") != 1
        ):
            fail(f"{sentinel_id} checkpoint {step} tensor provenance differs")
        if native:
            if (
                step != 0
                or raw.get("checkpoint_step") is not None
                or raw.get("adapter_loaded") is not False
                or raw.get("native_patch_route") is not None
            ):
                fail(f"{sentinel_id} native route declaration differs")
        else:
            trace = raw.get("native_patch_route")
            if (
                raw.get("checkpoint_step") != step
                or raw.get("adapter_loaded") is not True
                or raw.get("checkpoint_parameter_sha256") != checkpoint_parameter_sha256
                or raw.get("step_zero_native_endpoint_equal") is not (step == 0)
                or not isinstance(trace, Mapping)
                or trace.get("calls") != review.NUM_INFERENCE_STEPS * 10
                or trace.get("source_calls") != review.NUM_INFERENCE_STEPS * 9
                or trace.get("target_calls") != review.NUM_INFERENCE_STEPS
                or trace.get("source_id_zero_is_target") is not True
                or trace.get("source_id_positive_is_source") is not True
                or trace.get("native_rotary_unchanged") is not True
            ):
                fail(f"{sentinel_id} checkpoint {step} strict native patch route differs")
        _media_path(
            shard=shard,
            record=raw,
            label=f"{sentinel_id} {raw['arm']}",
            verify_media=verify_media,
        )
        rows.append(dict(raw))
    return tuple(rows)


def _validate_shard(
    *,
    root: Path,
    step: int,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    authority: review.TrainingAuthority,
    verify_media: bool,
) -> Mapping[str, Any]:
    shard = root / f"step-{step:08d}"
    if shard.is_symlink() or not shard.is_dir() or shard.resolve(strict=True) != shard:
        fail(f"checkpoint {step} shard root differs")
    value = _read_json(shard / "receipt.json", label=f"checkpoint {step} shard receipt")
    if set(value) != _ROOT_FIELDS:
        fail(f"checkpoint {step} shard fields differ")
    _embedded_receipt_digest(value, label=f"checkpoint {step} shard receipt")
    if (
        value.get("schema_version") != SHARD_SCHEMA
        or value.get("method") != SHARD_METHOD
        or value.get("complete") is not True
        or value.get("lora_scope") != authority.lora_scope
        or value.get("execution_scope") != "exact80"
        or value.get("checkpoint_step") != step
        or value.get("sentinel_order") != list(authoring.SENTINEL_ORDER)
        or value.get("smoke_only") is not False
        or value.get("terminal_training_receipt_bound") is not True
        or value.get("terminal_training_authority") != authority.as_receipt()
        or value.get("continuous_training_run") != str(authority.receipt.parent)
        or value.get("optimizer_present") is not False
        or value.get("backward_performed") is not False
        or value.get("parameter_update_performed") is not False
        or value.get("feature_evaluator_present") is not False
        or value.get("vlm_evaluator_present") is not False
        or value.get("automatic_ranking_present") is not False
        or value.get("candidate_selection_present") is not False
        or value.get("quality_claimed") is not False
        or value.get("scientific_claim_authorized") is not False
        or value.get("manual_review_pending") is not True
    ):
        fail(f"checkpoint {step} formal decode authority differs")
    runtime_source = value.get("runtime_source")
    runtime_fields = {
        "method_root", "archive", "archive_sha256", "manifest",
        "manifest_sha256", "manifest_digest", "method_revision",
        "exact_member_count", "archive_members_verified",
        "executed_root_exact_closure_verified", "executed_file_bound",
        "executed_launcher_bound", "digest", "launcher_sha256",
    }
    if not isinstance(runtime_source, Mapping) or set(runtime_source) != runtime_fields:
        fail(f"checkpoint {step} runtime release receipt differs")
    unsigned_runtime = dict(runtime_source)
    unsigned_runtime.pop("launcher_sha256", None)
    runtime_digest = unsigned_runtime.pop("digest", None)
    if (
        runtime_digest != review.object_sha256(unsigned_runtime)
        or runtime_source.get("exact_member_count") != len(release_contract.FILES_AND_MODES)
        or any(
            runtime_source.get(field) is not True
            for field in (
                "archive_members_verified", "executed_root_exact_closure_verified",
                "executed_file_bound", "executed_launcher_bound",
            )
        )
        or any(
            _SHA256.fullmatch(str(runtime_source.get(field))) is None
            for field in (
                "archive_sha256", "manifest_sha256", "manifest_digest",
                "launcher_sha256", "digest",
            )
        )
        or re.fullmatch(r"[0-9a-f]{40}", str(runtime_source.get("method_revision"))) is None
    ):
        fail(f"checkpoint {step} runtime release identity differs")
    if verify_media:
        try:
            opened_runtime = release_contract.validate_executed_release(
                executed_file=Path(str(runtime_source["method_root"])) / release_contract.RUNNER_MEMBER,
                executed_launcher=Path(str(runtime_source["method_root"])) / release_contract.LAUNCHER_MEMBER,
                manifest=runtime_source["manifest"],
                expected_manifest_sha256=runtime_source["manifest_sha256"],
                expected_archive_sha256=runtime_source["archive_sha256"],
                expected_method_revision=runtime_source["method_revision"],
            )
        except release_contract.ReviewReleaseError as error:
            raise PackedPreservationHtmlError(str(error)) from error
        if dict(opened_runtime) != {key: runtime_source[key] for key in opened_runtime}:
            fail(f"checkpoint {step} opened runtime receipt differs")
        launcher = Path(str(runtime_source["method_root"])) / release_contract.LAUNCHER_MEMBER
        if review.file_sha256(launcher) != runtime_source["launcher_sha256"]:
            fail(f"checkpoint {step} launcher bytes differ")
    manifest_row = value.get("review_manifest")
    if (
        not isinstance(manifest_row, Mapping)
        or set(manifest_row) != {"path", "file_sha256", "manifest_digest"}
        or manifest_row.get("file_sha256") != manifest_file_sha256
        or manifest_row.get("manifest_digest") != manifest.get("manifest_digest")
    ):
        fail(f"checkpoint {step} review manifest binding differs")
    checkpoint = authority.checkpoint(step)
    if value.get("checkpoint_authority") != checkpoint.receipt():
        fail(f"checkpoint {step} differs from terminal training authority")
    execution = value.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("world_size") != review.WORLD_SIZE
        or execution.get("sequence_parallel_size") != review.SP_SIZE
        or execution.get("num_inference_steps") != review.NUM_INFERENCE_STEPS
        or execution.get("frame_count") != review.FRAME_COUNT
        or execution.get("fps") != review.FPS
        or execution.get("native_rv2v_four_references") is not True
        or execution.get("same_source_instruction_seed_all_columns") is not True
        or execution.get("step_zero_endpoint_byte_exact_native") is not (step == 0)
        or execution.get("step_zero_media_byte_copy_from_native") is not (step == 0)
        or execution.get("parent_allocation_released") is not False
    ):
        fail(f"checkpoint {step} execution contract differs")
    architecture = value.get("architecture")
    if not isinstance(architecture, Mapping) or architecture.get("scope") != authority.lora_scope:
        fail(f"checkpoint {step} architecture scope differs")
    checkpoint_load = value.get("checkpoint_load")
    if (
        not isinstance(checkpoint_load, Mapping)
        or checkpoint_load.get("step") != step
        or checkpoint_load.get("adapter_sha256") != checkpoint.adapter_sha256
        or checkpoint_load.get("parameter_sha256") != checkpoint.parameter_sha256
        or checkpoint_load.get("inventory_sha256") != checkpoint.inventory_sha256
        or checkpoint_load.get("parameter_unchanged_during_inference") is not True
    ):
        fail(f"checkpoint {step} strict adapter load binding differs")
    manifest_by_id = {row["sentinel_id"]: row for row in manifest["sentinels"]}
    source_records = _validate_source_records(
        shard=shard,
        value=value.get("source_records"),
        manifest_by_id=manifest_by_id,
        verify_media=verify_media,
    )
    decode_records = _validate_decode_records(
        shard=shard,
        value=value.get("decode_records"),
        step=step,
        manifest_by_id=manifest_by_id,
        checkpoint_parameter_sha256=checkpoint.parameter_sha256,
        verify_media=verify_media,
    )
    prompts = value.get("prompts")
    if not isinstance(prompts, Mapping) or set(prompts) != set(authoring.SENTINEL_ORDER):
        fail(f"checkpoint {step} prompt registry differs")
    for sentinel_id in authoring.SENTINEL_ORDER:
        sentinel = manifest_by_id[sentinel_id]
        prompt = prompts.get(sentinel_id)
        if (
            not isinstance(prompt, Mapping)
            or prompt.get("full_instruction") != sentinel["instructions"]["forward"]
            or prompt.get("instruction_utf8_sha256")
            != sentinel["instruction_sha256"]["forward"]
            or _SHA256.fullmatch(str(prompt.get("native_prompt_utf8_sha256"))) is None
        ):
            fail(f"{sentinel_id} checkpoint {step} prompt bytes differ")
    return {**dict(value), "source_records": source_records, "decode_records": decode_records}


def load_shards(
    *,
    shard_root: Path,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    authority: review.TrainingAuthority,
    verify_media: bool,
) -> Mapping[int, Mapping[str, Any]]:
    expected_names = {f"step-{step:08d}" for step in review.CHECKPOINT_STEPS}
    if {path.name for path in shard_root.iterdir()} != expected_names:
        fail("shard root must contain exactly the five checkpoint directories")
    shards = {
        step: _validate_shard(
            root=shard_root,
            step=step,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            authority=authority,
            verify_media=verify_media,
        )
        for step in review.CHECKPOINT_STEPS
    }
    canonical_sources = tuple(
        {
            key: row[key]
            for key in _SOURCE_FIELDS
            if key not in {"relative_mp4", "probe_backend"}
        }
        for row in shards[0]["source_records"]
    )
    stable_fields = (
        "continuous_training_run",
        "base_checkpoint",
        "pinned_sources",
        "architecture",
        "lora_installation",
        "source_preprocessing",
        "prompts",
        "runtime_versions",
    )
    for step, shard in shards.items():
        normalized_sources = tuple(
            {
                key: row[key]
                for key in _SOURCE_FIELDS
                if key not in {"relative_mp4", "probe_backend"}
            }
            for row in shard["source_records"]
        )
        if normalized_sources != canonical_sources:
            fail(f"checkpoint {step} changed fixed source/instruction/seed records")
        if any(shard[field] != shards[0][field] for field in stable_fields):
            fail(f"checkpoint {step} changed fixed inference route or inputs")
    runtime_identity_fields = (
        "archive_sha256", "manifest_sha256", "manifest_digest", "method_revision",
        "exact_member_count", "archive_members_verified",
        "executed_root_exact_closure_verified", "executed_file_bound",
        "executed_launcher_bound", "launcher_sha256",
    )
    reference_runtime = shards[0]["runtime_source"]
    for step, shard in shards.items():
        if any(
            shard["runtime_source"][field] != reference_runtime[field]
            for field in runtime_identity_fields
        ):
            fail(f"checkpoint {step} changed authenticated runtime content identity")
    gaussian_hashes: dict[str, set[str]] = {
        sentinel_id: set() for sentinel_id in authoring.SENTINEL_ORDER
    }
    gaussian_identities: dict[str, set[str]] = {
        sentinel_id: set() for sentinel_id in authoring.SENTINEL_ORDER
    }
    for shard in shards.values():
        for row in shard["decode_records"]:
            sentinel_id = row["sentinel_id"]
            gaussian_hashes[sentinel_id].add(row["initial_gaussian_sha256"])
            gaussian_identities[sentinel_id].add(
                review.object_sha256(row["initial_gaussian_identity"])
            )
    if any(len(values) != 1 for values in gaussian_hashes.values()) or any(
        len(values) != 1 for values in gaussian_identities.values()
    ):
        fail("one sentinel did not reuse the same official Gaussian across checkpoints")
    zero = {
        (row["sentinel_id"], row["arm"]): row
        for row in shards[0]["decode_records"]
    }
    for sentinel_id in authoring.SENTINEL_ORDER:
        native = zero[(sentinel_id, "native")]
        adapted = zero[(sentinel_id, "step-0")]
        if (
            native["mp4_sha256"] != adapted["mp4_sha256"]
            or native["endpoint_identity"] != adapted["endpoint_identity"]
            or native["initial_gaussian_sha256"] != adapted["initial_gaussian_sha256"]
            or native["initial_gaussian_identity"] != adapted["initial_gaussian_identity"]
        ):
            fail(f"{sentinel_id} step-0/native equality gate failed")
    return shards


def _copy_content_addressed(
    *,
    source: Path,
    media_dir: Path,
    expected_sha256: str,
) -> str:
    sha = _sha(expected_sha256, label="published media SHA")
    target = media_dir / f"{sha}.mp4"
    if target.exists():
        if target.is_symlink() or not target.is_file() or review.file_sha256(target) != sha:
            fail("content-addressed media collision")
        return f"media/{target.name}"
    shutil.copyfile(source, target)
    if review.file_sha256(target) != sha:
        fail("copied content-addressed media bytes differ")
    os.chmod(target, 0o444)
    return f"media/{target.name}"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _video(relative: str, *, label: str) -> str:
    return (
        f'<video controls muted playsinline preload="metadata" aria-label="{_esc(label)}">'
        f'<source src="{_esc(relative)}" type="video/mp4"></video>'
    )


def _render_html(
    *,
    manifest: Mapping[str, Any],
    shards: Mapping[int, Mapping[str, Any]],
    media: Mapping[tuple[Any, ...], str],
    evidence_sha256: str,
) -> str:
    scope = shards[0]["lora_scope"]
    if scope not in _SCOPE_PRESENTATION:
        fail("HTML LoRA scope presentation differs")
    scope_label, trainable_parameters = _SCOPE_PRESENTATION[scope]
    manifest_by_id = {row["sentinel_id"]: row for row in manifest["sentinels"]}
    step_rows = {
        step: {
            row["sentinel_id"]: row
            for row in shard["decode_records"]
            if row["arm"] == f"step-{step}"
        }
        for step, shard in shards.items()
    }
    native_rows = {
        row["sentinel_id"]: row
        for row in shards[0]["decode_records"]
        if row["arm"] == "native"
    }
    sections: list[str] = []
    for sentinel_id in authoring.SENTINEL_ORDER:
        sentinel = manifest_by_id[sentinel_id]
        cells = [
            (
                "Source",
                media[("source", sentinel_id)],
                "Unchanged input video",
            ),
            (
                "Native",
                media[("native", sentinel_id)],
                f"Frozen Bernini · {_esc(native_rows[sentinel_id]['mp4_sha256'][:14])}…",
            ),
        ]
        cells.extend(
            (
                f"Optimizer update {step}",
                media[("step", step, sentinel_id)],
                f"Strict checkpoint · {_esc(step_rows[step][sentinel_id]['mp4_sha256'][:14])}…",
            )
            for step in review.CHECKPOINT_STEPS
        )
        cards = "".join(
            '<article class="cell">'
            f'<h3>{_esc(title)}</h3>{_video(relative, label=f"{sentinel_id} {title}")}'
            f'<p>{caption}</p></article>'
            for title, relative, caption in cells
        )
        sections.append(
            '<section class="sentinel">'
            f'<header><p class="eyebrow">{_esc(sentinel["diversity_role"])} · '
            f'{_esc(sentinel["source_entity_type"])}</p>'
            f'<h2>{_esc(sentinel_id)}</h2><p class="source-caption"><b>Source description</b> '
            f'{_esc(sentinel["source_caption"])}</p>'
            f'<p class="instruction"><b>Full editing instruction</b> '
            f'{_esc(sentinel["instructions"]["forward"])}</p>'
            f'<p class="meta"><b>Fixed seed</b> <code>{_esc(sentinel["seed"])}</code> · '
            f'<b>IID</b> <code>{_esc(sentinel["iid"])}</code> · <b>Source SHA-256</b> '
            f'<code>{_esc(sentinel["source_video_sha256"])}</code></p></header>'
            f'<div class="sync-controls" role="group" aria-label="Synchronized playback controls for {_esc(sentinel_id)}">'
            '<button type="button" data-sync-action="play">同时播放 / Play together</button>'
            '<button type="button" data-sync-action="pause">同时暂停 / Pause together</button>'
            '<button type="button" data-sync-action="restart">从头同步 / Restart together</button>'
            '<span class="sync-status" role="status" aria-live="polite">就绪：可同步播放 7 个视频。 / Ready.</span>'
            '</div>'
            f'<div class="strip">{cards}</div></section>'
        )
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(scope_label)} · packed preservation checkpoint review</title>
<style>
:root{{--bg:#080b12;--panel:#111827;--cell:#172033;--line:#2b3a55;--text:#eef5ff;--muted:#a8b5c8;--cyan:#61ddff;--amber:#ffc66d}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% -10%,#1d3554,#080b12 38%);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:2200px;margin:auto;padding:28px}}h1{{font-size:clamp(34px,5vw,68px);line-height:1.02;margin:.15em 0}}h2{{font-size:30px;margin:.15em 0}}h3{{margin:0 0 9px}}p{{margin:.45em 0}}button{{border:1px solid #4c668c;border-radius:8px;background:#173150;color:var(--text);font:inherit;font-weight:750;padding:8px 13px;cursor:pointer}}button:hover{{background:#21466f}}button:focus-visible{{outline:3px solid var(--cyan);outline-offset:2px}}code{{color:#c2efff;overflow-wrap:anywhere}}.hero,.sentinel{{background:rgba(17,24,39,.95);border:1px solid var(--line);border-radius:20px;padding:22px;margin-bottom:26px;box-shadow:0 18px 55px #0007}}.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:800;color:var(--cyan)}}.callout{{margin-top:16px;border-left:4px solid var(--amber);background:#0c1422;padding:13px 15px;border-radius:8px}}.instruction{{font-size:18px;color:#fff}}.source-caption,.meta,.cell p,.sync-status{{color:var(--muted)}}.sync-controls{{display:flex;align-items:center;flex-wrap:wrap;gap:9px;margin:16px 0 2px;padding:11px;background:#0c1422;border:1px solid var(--line);border-radius:11px}}.sync-status{{margin-left:4px}}.strip{{display:grid;grid-template-columns:repeat(7,minmax(270px,1fr));gap:13px;overflow-x:auto;padding:15px 2px 8px;scroll-snap-type:x proximity}}.cell{{background:var(--cell);border:1px solid var(--line);border-radius:13px;padding:11px;min-width:270px;scroll-snap-align:start}}video{{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#000;border-radius:9px}}footer{{color:var(--muted);padding:12px 0 30px}}@media(max-width:700px){{main{{padding:12px}}.hero,.sentinel{{padding:15px}}.sync-controls button{{flex:1 1 auto}}.sync-status{{flex-basis:100%;margin-left:0}}}}
</style></head><body><main>
<section class="hero"><p class="eyebrow">Fixed checkpoint evidence · manual viewing</p><h1>{_esc(scope_label)}</h1>
<p><b>Trajectory:</b> {_esc(scope)} LoRA · {_esc(f"{trainable_parameters:,}")} trainable parameters. The same four held-out source videos, complete instructions, fixed seeds, and official initial Gaussian are shown at native Bernini and optimizer updates 0, 20, 40, 60, and 80.</p>
<p><b>Training scale:</b> 64 real source videos, expanded into 640 logical training records = 80 optimizer updates × global batch 8. Training histogram: noop 256; cube 128; speed 128; tube 128.</p>
<div class="callout">Optimizer update N is a training-step index, not a score, value, reward, or quality judgment. Update 0 is required to be byte-identical to Native. Every later column is a strict load from the completed continuous exact80 trajectory; no automatic ranking or candidate selection is used. Compare complete 81-frame, 25-fps videos directly.</div>
<p>Evidence file SHA-256 <code>{_esc(evidence_sha256)}</code> · built {_esc(created)}</p></section>
{"".join(sections)}
<footer>Self-contained packet: every video is a relative content-addressed file inside this directory.</footer>
</main>{_SYNC_PLAYBACK_SCRIPT}</body></html>"""


def build_review(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    training_receipt_path: Path,
    expected_training_receipt_sha256: str,
    lora_scope: str,
    shard_root: Path,
    output_dir: Path,
    verify_manifest_files: bool = True,
    verify_training_files: bool = True,
    verify_media: bool = True,
) -> Mapping[str, Any]:
    manifest_sha = _sha(expected_manifest_sha256, label="expected manifest SHA")
    training_sha = _sha(
        expected_training_receipt_sha256, label="expected training receipt SHA"
    )
    try:
        manifest = authoring.load_manifest(
            manifest_path,
            expected_file_sha256=manifest_sha,
            verify_files=verify_manifest_files,
        )
        authority = review.load_training_authority(
            training_receipt_path,
            expected_file_sha256=training_sha,
            expected_lora_scope=lora_scope,
            verify_files=verify_training_files,
        )
    except (authoring.CheckpointReviewContractError, review.PackedPreservationReviewError) as error:
        raise PackedPreservationHtmlError(str(error)) from error
    source_authority = manifest.get("source_only_manifest")
    if (
        not isinstance(source_authority, Mapping)
        or source_authority.get("file_sha256") != authority.source_only_manifest_sha256
        or authority.source_only_manifest_sha256 != review.SOURCE_ONLY_MANIFEST_SHA256
    ):
        fail("review manifest and final training receipt source-only authority differ")
    root = _plain_root(shard_root, label="checkpoint shard root")
    output = _safe_output(output_dir)
    shards = load_shards(
        shard_root=root,
        manifest=manifest,
        manifest_file_sha256=manifest_sha,
        authority=authority,
        verify_media=verify_media,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        media_dir = stage / "media"
        media_dir.mkdir(mode=0o700)
        media_map: dict[tuple[Any, ...], str] = {}
        source_zero = {row["sentinel_id"]: row for row in shards[0]["source_records"]}
        native_zero = {
            row["sentinel_id"]: row
            for row in shards[0]["decode_records"]
            if row["arm"] == "native"
        }

        def admit(*, key: tuple[Any, ...], step: int, row: Mapping[str, Any]) -> None:
            relative = Path(str(row["relative_mp4"]))
            source = (root / f"step-{step:08d}" / relative).resolve(strict=True)
            media_map[key] = _copy_content_addressed(
                source=source,
                media_dir=media_dir,
                expected_sha256=row["mp4_sha256"],
            )

        for sentinel_id in authoring.SENTINEL_ORDER:
            admit(key=("source", sentinel_id), step=0, row=source_zero[sentinel_id])
            admit(key=("native", sentinel_id), step=0, row=native_zero[sentinel_id])
        for step, shard in shards.items():
            for row in shard["decode_records"]:
                if row["arm"] == f"step-{step}":
                    admit(key=("step", step, row["sentinel_id"]), step=step, row=row)

        checkpoint_bindings = [
            {
                "step": checkpoint.step,
                "adapter_sha256": checkpoint.adapter_sha256,
                "metadata_sha256": checkpoint.metadata_sha256,
                "parameter_sha256": checkpoint.parameter_sha256,
                "inventory_sha256": checkpoint.inventory_sha256,
            }
            for checkpoint in authority.checkpoints
        ]
        unsigned_evidence = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "trajectory_label": _SCOPE_PRESENTATION[authority.lora_scope][0],
            "trainable_parameters": _SCOPE_PRESENTATION[authority.lora_scope][1],
            "real_source_video_count": 64,
            "logical_training_record_count": 640,
            "optimizer_update_count": 80,
            "global_batch_size": 8,
            "training_histogram": {"noop": 256, "cube": 128, "speed": 128, "tube": 128},
            "training_authority": {
                "receipt_file_sha256": authority.receipt_file_sha256,
                "receipt_digest": authority.receipt_digest,
                "lora_scope": authority.lora_scope,
                "source_only_manifest_sha256": authority.source_only_manifest_sha256,
                "checkpoints": checkpoint_bindings,
            },
            "review_manifest": {
                "file_sha256": manifest_sha,
                "manifest_digest": manifest["manifest_digest"],
            },
            "checkpoint_steps": list(review.CHECKPOINT_STEPS),
            "sentinel_order": list(authoring.SENTINEL_ORDER),
            "shard_receipt_digests": {
                str(step): shard["receipt_digest"] for step, shard in shards.items()
            },
            "logical_video_reference_count": len(media_map),
            "physical_media_file_count": len(tuple(media_dir.glob("*.mp4"))),
            "step_zero_native_byte_equality": True,
            "same_gaussian_per_sentinel_all_columns": True,
            "self_contained": True,
            "manual_review_pending": True,
            "quality_claimed": False,
        }
        evidence = {
            **unsigned_evidence,
            "evidence_digest": review.object_sha256(unsigned_evidence),
        }
        evidence_path = stage / "evidence.json"
        evidence_path.write_bytes(review.canonical_json_bytes(evidence) + b"\n")
        evidence_sha = review.file_sha256(evidence_path)
        page = _render_html(
            manifest=manifest,
            shards=shards,
            media=media_map,
            evidence_sha256=evidence_sha,
        )
        lowered = page.lower()
        if (
            re.search(r"<script\b[^>]*\bsrc\s*=", page, flags=re.IGNORECASE)
            or "http://" in lowered
            or "https://" in lowered
        ):
            fail("HTML is not self-contained")
        (stage / "index.html").write_text(page, encoding="utf-8")
        os.chmod(stage / "index.html", 0o444)
        os.chmod(evidence_path, 0o444)
        os.chmod(media_dir, 0o555)
        os.rename(stage, output)
        os.chmod(output, 0o555)
    except Exception:
        if stage.exists() and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise
    return {
        "output": str(output),
        "index": str(output / "index.html"),
        "evidence_sha256": evidence_sha,
        "sentinels": len(authoring.SENTINEL_ORDER),
        "checkpoint_steps": list(review.CHECKPOINT_STEPS),
        "physical_media_files": unsigned_evidence["physical_media_file_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--lora-scope", choices=review.core.LORA_SCOPES, required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_review(
        manifest_path=Path(args.manifest).expanduser(),
        expected_manifest_sha256=args.expected_manifest_sha256,
        training_receipt_path=Path(args.training_receipt).expanduser(),
        expected_training_receipt_sha256=args.expected_training_receipt_sha256,
        lora_scope=args.lora_scope,
        shard_root=Path(args.shard_root).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PackedPreservationHtmlError",
    "SCHEMA_VERSION",
    "SHARD_SCHEMA",
    "build_parser",
    "build_review",
    "load_shards",
    "main",
]
