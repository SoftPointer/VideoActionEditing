#!/usr/bin/env python3
"""Build the two-trajectory landing page for packed-preservation review v2."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


LANES = {
    "all-attention": ("All-attention main", 188_946_432),
    "self-attention": ("Self-attention control", 94_574_592),
}


class TopIndexError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise TopIndexError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _lane(root: Path, scope: str) -> Mapping[str, Any]:
    lane = root / scope
    if lane.is_symlink() or not lane.is_dir() or lane.resolve(strict=True) != lane:
        fail(f"{scope} review directory differs")
    page = lane / "index.html"
    evidence_path = lane / "evidence.json"
    for path in (page, evidence_path):
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            fail(f"{scope} review artifact differs")
    try:
        raw = evidence_path.read_bytes()
        evidence = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TopIndexError(f"cannot read {scope} evidence") from error
    label, parameters = LANES[scope]
    training = evidence.get("training_authority") if isinstance(evidence, Mapping) else None
    if (
        not isinstance(evidence, Mapping)
        or raw != _canonical(evidence) + b"\n"
        or evidence.get("complete") is not True
        or evidence.get("trajectory_label") != label
        or evidence.get("trainable_parameters") != parameters
        or evidence.get("real_source_video_count") != 64
        or evidence.get("logical_training_record_count") != 640
        or evidence.get("optimizer_update_count") != 80
        or evidence.get("global_batch_size") != 8
        or evidence.get("training_histogram") != {"noop": 256, "cube": 128, "speed": 128, "tube": 128}
        or not isinstance(training, Mapping)
        or training.get("lora_scope") != scope
        or evidence.get("quality_claimed") is not False
        or evidence.get("manual_review_pending") is not True
    ):
        fail(f"{scope} evidence authority differs")
    return evidence


def build_top_index(root_value: str | Path) -> Mapping[str, Any]:
    root = Path(root_value).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        fail("review root must be one canonical directory")
    target = root / "index.html"
    if target.exists() or target.is_symlink():
        fail("top-level index must be fresh")
    evidence = {scope: _lane(root, scope) for scope in LANES}
    cards = "".join(
        f'<a class="lane" href="{scope}/index.html"><p>{html.escape(label)}</p>'
        f'<strong>{parameters:,} trainable parameters</strong><span>Open fixed Source / Native / optimizer update 0, 20, 40, 60, 80 review →</span></a>'
        for scope, (label, parameters) in LANES.items()
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Packed preservation checkpoint review</title><style>
body{{margin:0;background:#080b12;color:#eef5ff;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:48px 24px}}h1{{font-size:52px;line-height:1.05}}.note{{padding:18px;background:#111827;border:1px solid #2b3a55;border-radius:14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin-top:24px}}.lane{{display:flex;flex-direction:column;gap:10px;text-decoration:none;color:inherit;background:#172033;border:1px solid #2b3a55;border-radius:16px;padding:22px}}.lane:hover{{border-color:#61ddff}}.lane p{{font-size:25px;font-weight:800;margin:0}}.lane span{{color:#a8b5c8}}
</style></head><body><main><p>PACKED PRESERVATION · FIXED MANUAL REVIEW</p><h1>Two separate training trajectories</h1><div class="note">Both pages use the same 64 real source videos, 640 logical records, 80 optimizer updates × global batch 8, fixed four held-out sentinels, instructions, seeds, and official RV2V protocol. They differ only in trainable attention scope. Optimizer update is an index—not a score, value, reward, or quality judgment. No automatic ranking or candidate selection is used.</div><div class="grid">{cards}</div></main></body></html>"""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".index.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(page)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        os.chmod(target, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "index": str(target),
        "lanes": list(LANES),
        "all_attention_evidence_digest": evidence["all-attention"]["evidence_digest"],
        "self_attention_evidence_digest": evidence["self-attention"]["evidence_digest"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(json.dumps(build_top_index(Path(_parser().parse_args(argv).root).expanduser()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
