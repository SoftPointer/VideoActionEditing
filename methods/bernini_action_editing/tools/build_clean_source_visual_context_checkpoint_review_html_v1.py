#!/usr/bin/env python3
"""Build the self-contained Stage-B 0/20/40/60/80 decoded review page."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_checkpoint_review_contract_v1 as contract  # noqa: E402


class CheckpointReviewHtmlError(RuntimeError):
    """Raised before a partial or misleading HTML packet is published."""


def fail(message: str) -> NoReturn:
    raise CheckpointReviewHtmlError(message)


def _safe_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        fail("output-dir must be one fresh absolute child of an existing plain parent")
    return path


def _plain_root(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CheckpointReviewHtmlError(f"{label} is unavailable") from error
    if resolved != path or not path.is_dir() or path.is_symlink():
        fail(f"{label} must be a canonical plain directory")
    return path


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} must be a plain file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointReviewHtmlError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def load_shards(
    *,
    shard_root: Path,
    manifest: Mapping[str, Any],
    verify_media: bool,
) -> Mapping[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    expected_names = {f"step-{step:08d}" for step in contract.CHECKPOINT_STEPS}
    observed_names = {path.name for path in shard_root.iterdir()}
    if observed_names != expected_names:
        fail("shard root must contain exactly the five cadence directories")
    for step in contract.CHECKPOINT_STEPS:
        root = shard_root / f"step-{step:08d}"
        if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
            fail(f"checkpoint shard {step} root differs")
        receipt = _read_json(root / "receipt.json", label=f"checkpoint shard {step} receipt")
        try:
            result[step] = contract.validate_shard_receipt(
                receipt,
                expected_step=step,
                expected_manifest_digest=str(manifest["manifest_digest"]),
                manifest_value=manifest,
                media_root=root,
                verify_media=verify_media,
            )
        except contract.CheckpointReviewContractError as error:
            raise CheckpointReviewHtmlError(str(error)) from error
    memory_kinds = {row["memory_input_kind"] for row in result.values()}
    if len(memory_kinds) != 1:
        fail("five checkpoint shards do not share one training arm")
    canonical_sources = result[0]["source_records"]
    for step, shard in result.items():
        if shard["source_records"] != canonical_sources:
            fail(f"checkpoint {step} changed fixed source/caption/seed records")
    gaussian_hashes: dict[str, set[str]] = {
        sentinel: set() for sentinel in contract.SENTINEL_ORDER
    }
    for shard in result.values():
        for row in shard["logical_records"]:
            gaussian_hashes[row["sentinel_id"]].add(
                row["initial_gaussian_sha256"]
            )
    if any(len(values) != 1 for values in gaussian_hashes.values()):
        fail("one sentinel did not reuse the same official Gaussian everywhere")
    # The disabled edge is frozen Bernini and must not drift with adapter
    # weights.  At step zero the zero-init correct route and separately shown
    # native control must also be byte exact to it.
    carrier_hashes: dict[str, set[str]] = {
        sentinel: set() for sentinel in contract.SENTINEL_ORDER
    }
    for shard in result.values():
        for row in shard["logical_records"]:
            if row["arm"] == "carrier-off":
                carrier_hashes[row["sentinel_id"]].add(row["mp4_sha256"])
    if any(len(values) != 1 for values in carrier_hashes.values()):
        fail("carrier-off frozen-base decode drifted across checkpoints")
    step_zero = result[0]
    logical_zero = {
        (row["sentinel_id"], row["arm"]): row
        for row in step_zero["logical_records"]
    }
    native_zero = {row["sentinel_id"]: row for row in step_zero["native_records"]}
    for sentinel in contract.SENTINEL_ORDER:
        hashes = {
            logical_zero[(sentinel, "correct")]["mp4_sha256"],
            logical_zero[(sentinel, "forward")]["mp4_sha256"],
            logical_zero[(sentinel, "carrier-off")]["mp4_sha256"],
            native_zero[sentinel]["mp4_sha256"],
        }
        if len(hashes) != 1:
            fail(f"step-0 zero-init/native parity failed for {sentinel}")
        if (
            native_zero[sentinel]["initial_gaussian_sha256"]
            not in gaussian_hashes[sentinel]
        ):
            fail(f"step-0 native Gaussian differs for {sentinel}")
    return result


def _copy_media(
    *,
    source: Path,
    target: Path,
    expected_sha256: str,
) -> None:
    if source.is_symlink() or not source.is_file() or target.exists() or target.is_symlink():
        fail("media copy source/target differs")
    shutil.copyfile(source, target)
    if contract.file_sha256(target) != expected_sha256:
        fail("copied self-contained MP4 bytes differ")
    os.chmod(target, 0o444)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _video(relative: str, *, label: str) -> str:
    return (
        f'<video controls loop muted playsinline preload="metadata" '
        f'aria-label="{_esc(label)}"><source src="{_esc(relative)}" '
        'type="video/mp4"></video>'
    )


def _render_html(
    *,
    manifest: Mapping[str, Any],
    shards: Mapping[int, Mapping[str, Any]],
    media_map: Mapping[tuple[Any, ...], str],
    evidence_sha256: str,
) -> str:
    sentinel_by_id = {row["sentinel_id"]: row for row in manifest["sentinels"]}
    sentinel_by_iid = {row["iid"]: row for row in manifest["sentinels"]}
    sections: list[str] = []
    for sentinel_id in contract.SENTINEL_ORDER:
        sentinel = sentinel_by_id[sentinel_id]
        wrong_owner = sentinel_by_iid[sentinel["wrong_owner_iid"]]
        source_rel = media_map[("source", sentinel_id)]
        native_rel = media_map[("native", sentinel_id)]
        instructions = "".join(
            f'<dt>{_esc(branch)}</dt><dd>{_esc(sentinel["instructions"][branch])}</dd>'
            for branch in contract.TEXT_BRANCHES
        )
        checkpoint_sections: list[str] = []
        for step in contract.CHECKPOINT_STEPS:
            shard = shards[step]
            checkpoint = shard["checkpoint"]
            rows = [row for row in shard["logical_records"] if row["sentinel_id"] == sentinel_id]
            cards: list[str] = []
            for row in rows:
                relative = media_map[("logical", step, sentinel_id, row["arm"])]
                badge = "carrier axis" if row["axis"] == "source-control" else "instruction axis"
                memory = (
                    "extra carrier disabled; native Bernini full-video/reference source remains"
                    if row["source_control"] == "carrier-off"
                    else (
                        f"memory from {wrong_owner['sentinel_id']} "
                        f"(IID {wrong_owner['iid']}; {wrong_owner['diversity_role']} / "
                        f"{wrong_owner['source_entity_type']}); equal latent geometry, "
                        "different entity and scene"
                        if row["source_control"] == "wrong-owner"
                        else (
                            "same-owner memory with exact latent phases reversed 20→0"
                            if row["source_control"] == "order-permutation"
                            else "same-owner persistent clean/noised visual memory"
                        )
                    )
                )
                cards.append(
                    '<article class="card">'
                    f'<div class="card-head"><h4>{_esc(row["arm"])}</h4><span>{_esc(badge)}</span></div>'
                    + _video(relative, label=f"{sentinel_id} step {step} {row['arm']}")
                    + f'<p class="instruction"><b>Full instruction</b> {_esc(row["instruction"])}</p>'
                    + f'<p class="mechanism"><b>Actual binding</b> {_esc(memory)}; '
                    f'seed {_esc(row["seed"])}; source SHA <code>{_esc(row["source_video_sha256"][:16])}…</code>.</p>'
                    '</article>'
                )
            checkpoint_sections.append(
                '<section class="checkpoint">'
                f'<div class="checkpoint-head"><h3>Checkpoint step {step}</h3>'
                f'<p>{_esc(checkpoint["logical_records_seen"])} training records seen · '
                f'strict checkpoint load · <code>{_esc(checkpoint["file_sha256"][:18])}…</code></p></div>'
                '<div class="axis-label">Source-carrier controls (forward instruction held fixed)</div>'
                f'<div class="grid source-grid">{"".join(cards[:4])}</div>'
                '<div class="axis-label">Typed instruction controls (correct carrier held fixed)</div>'
                f'<div class="grid text-grid">{"".join(cards[4:])}</div>'
                '</section>'
            )
        sections.append(
            '<section class="sentinel">'
            f'<header><p class="eyebrow">{_esc(sentinel["diversity_role"])} · source entity: {_esc(sentinel["source_entity_type"])}</p>'
            f'<h2>{_esc(sentinel_id)} · IID {_esc(sentinel["iid"])}</h2>'
            f'<p>{_esc(sentinel["source_caption"])}</p>'
            f'<p class="identity">Fixed seed <code>{_esc(sentinel["seed"])}</code> · source SHA '
            f'<code>{_esc(sentinel["source_video_sha256"])}</code> · latent geometry '
            f'<code>{_esc("×".join(str(value) for value in sentinel["latent_shape"]))}</code><br>'
            f'Registered wrong-owner memory: <code>{_esc(wrong_owner["sentinel_id"])}</code> · '
            f'IID <code>{_esc(wrong_owner["iid"])}</code> · {_esc(wrong_owner["source_caption"])}</p></header>'
            '<div class="anchor-grid">'
            f'<article class="anchor"><h3>Source (unchanged input)</h3>{_video(source_rel, label=f"{sentinel_id} source")}</article>'
            f'<article class="anchor"><h3>Native frozen Bernini</h3>{_video(native_rel, label=f"{sentinel_id} native")}'
            f'<p>{_esc(sentinel["instructions"]["forward"])}</p></article></div>'
            '<details><summary>All six complete instructions</summary>'
            f'<dl>{instructions}</dl></details>'
            + "".join(checkpoint_sections)
            + '</section>'
        )
    memory_kind = shards[0]["memory_input_kind"]
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clean-source visual-context checkpoint review</title>
<style>
:root{{--bg:#080b12;--panel:#101624;--panel2:#151d2e;--text:#edf3ff;--muted:#9dacbf;--line:#29364d;--cyan:#58dbff;--amber:#ffc766;--red:#ff6174}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% -10%,#172944 0,#080b12 38%);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:1880px;margin:auto;padding:30px}}h1{{font-size:clamp(30px,5vw,72px);line-height:1;margin:.15em 0}}h2{{font-size:30px;margin:.15em 0}}h3,h4{{margin:.25em 0}}p{{margin:.45em 0}}code{{color:#b9ecff;overflow-wrap:anywhere}}.hero,.sentinel{{border:1px solid var(--line);background:rgba(16,22,36,.94);border-radius:22px;padding:24px;margin-bottom:28px;box-shadow:0 18px 60px #0008}}.hero-grid,.anchor-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.callout{{background:#0b1220;border-left:4px solid var(--cyan);padding:14px 16px;border-radius:9px}}.warning{{border-left-color:var(--amber)}}.eyebrow,.axis-label{{text-transform:uppercase;letter-spacing:.14em;font-weight:800;color:var(--cyan);font-size:12px}}.identity,.mechanism,.checkpoint-head p{{color:var(--muted)}}.anchor{{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px}}video{{display:block;width:100%;background:#000;border-radius:10px;aspect-ratio:16/10;object-fit:contain}}details{{margin:18px 0;padding:14px;border:1px solid var(--line);border-radius:12px}}summary{{cursor:pointer;font-weight:800}}dl{{display:grid;grid-template-columns:140px 1fr;gap:8px 14px}}dt{{color:var(--amber);font-weight:800}}dd{{margin:0}}.checkpoint{{border-top:1px solid var(--line);padding-top:22px;margin-top:28px}}.checkpoint-head{{display:flex;justify-content:space-between;gap:16px;align-items:baseline}}.grid{{display:grid;gap:14px;margin:9px 0 22px}}.source-grid{{grid-template-columns:repeat(4,minmax(0,1fr))}}.text-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}.card{{background:#0c1220;border:1px solid var(--line);border-radius:14px;padding:11px;min-width:0}}.card-head{{display:flex;justify-content:space-between;gap:8px;align-items:center}}.card-head span{{font-size:11px;color:var(--muted)}}.instruction{{font-size:13px}}.mechanism{{font-size:12px}}footer{{color:var(--muted);padding:20px 0}}@media(max-width:1100px){{.source-grid,.text-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:680px){{main{{padding:12px}}.hero-grid,.anchor-grid,.source-grid,.text-grid{{grid-template-columns:1fr}}.checkpoint-head{{display:block}}dl{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><p class="eyebrow">Decoded checkpoint evidence · not an evaluator</p><h1>Persistent source context across training</h1>
<p>Four fixed Stage-B held-out sentinels, one source/instruction/seed registry, and the exact same review grid at optimizer steps 0, 20, 40, 60 and 80. Training arm: <code>{_esc(memory_kind)}</code>.</p>
<div class="hero-grid"><div class="callout"><b>What “carrier-off” removes</b><br>Only the newly trained target-query → persistent source-memory attention edge is disabled. Bernini’s native full source video, four references, text prompt, initial Gaussian and scheduler remain present.</div>
<div class="callout"><b>What wrong-owner/order change</b><br>Wrong-owner swaps only the extra memory edge to the registered different held-out source with identical latent geometry. The replacement intentionally has a different entity and scene, so this is a stress control—not a pure owner-only causal contrast. Order-permutation keeps the correct owner but reverses its 21 latent phases before positional encoding. Native RV2V source conditions remain correct in both controls.</div>
<div class="callout"><b>Why there are two forward labels</b><br><code>correct</code> is the forward anchor on the carrier-control axis; <code>forward</code> is the same physical decode on the typed-instruction axis. Their MP4, trace and instruction are required to be identical.</div>
<div class="callout warning"><b>How to judge</b><br>Watch complete 81-frame, 25-fps videos. This packet intentionally contains no feature scalar, reward, ranking, automatic verdict or chosen winner.</div></div>
<p>Evidence manifest SHA-256: <code>{_esc(evidence_sha256)}</code> · built {_esc(created)}</p></section>
{"".join(sections)}
<footer>Self-contained packet: all video links are relative files inside this directory. Manual action and preservation review remains pending.</footer>
</main></body></html>"""


def build_review(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    shard_root: Path,
    output_dir: Path,
    verify_manifest_files: bool = True,
    verify_media: bool = True,
) -> Mapping[str, Any]:
    try:
        manifest = contract.load_manifest(
            manifest_path,
            expected_file_sha256=expected_manifest_sha256,
            verify_files=verify_manifest_files,
        )
    except contract.CheckpointReviewContractError as error:
        raise CheckpointReviewHtmlError(str(error)) from error
    root = _plain_root(shard_root, label="checkpoint shard root")
    output = _safe_output(output_dir)
    shards = load_shards(shard_root=root, manifest=manifest, verify_media=verify_media)
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists() or stage.is_symlink():
        fail("HTML staging path already exists")
    stage.mkdir(mode=0o700)
    media_dir = stage / "media"
    media_dir.mkdir(mode=0o700)
    media_map: dict[tuple[Any, ...], str] = {}
    # Content-addressed deduplication keeps the packet transportable.  It is
    # especially important for the required correct/forward alias, step-0
    # native parity and the frozen carrier-off control across five steps.
    copied: dict[str, str] = {}

    def admit(
        *,
        key: tuple[Any, ...],
        shard_step: int,
        relative: str,
        sha256: str,
        output_name: str,
    ) -> None:
        source = root / f"step-{shard_step:08d}" / relative
        source.resolve(strict=True)
        if sha256 in copied:
            media_map[key] = copied[sha256]
            return
        target = media_dir / output_name
        _copy_media(source=source, target=target, expected_sha256=sha256)
        web = f"media/{output_name}"
        copied[sha256] = web
        media_map[key] = web

    for row in shards[0]["source_records"]:
        admit(
            key=("source", row["sentinel_id"]),
            shard_step=0,
            relative=row["relative_mp4"],
            sha256=row["mp4_sha256"],
            output_name=f"{row['sentinel_id']}__source.mp4",
        )
    for row in shards[0]["native_records"]:
        admit(
            key=("native", row["sentinel_id"]),
            shard_step=0,
            relative=row["relative_mp4"],
            sha256=row["mp4_sha256"],
            output_name=f"{row['sentinel_id']}__native.mp4",
        )
    for step, shard in shards.items():
        for row in shard["logical_records"]:
            admit(
                key=("logical", step, row["sentinel_id"], row["arm"]),
                shard_step=step,
                relative=row["relative_mp4"],
                sha256=row["mp4_sha256"],
                output_name=f"step-{step:08d}__{row['sentinel_id']}__{row['arm']}.mp4",
            )
    aggregate_unsigned = {
        "schema_version": contract.AGGREGATE_SCHEMA,
        "complete": True,
        "review_manifest": {
            "file_sha256": expected_manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
        },
        "memory_input_kind": shards[0]["memory_input_kind"],
        "checkpoint_steps": list(contract.CHECKPOINT_STEPS),
        "sentinel_order": list(contract.SENTINEL_ORDER),
        "logical_arm_order": list(contract.LOGICAL_ARM_ORDER),
        "logical_record_count": sum(len(row["logical_records"]) for row in shards.values()),
        "physical_media_file_count": len(copied),
        "shard_receipt_digests": {
            str(step): shard["receipt_digest"] for step, shard in shards.items()
        },
        "self_contained": True,
        "manual_review_pending": True,
        "quality_claimed": False,
    }
    aggregate = {
        **aggregate_unsigned,
        "evidence_digest": contract.object_sha256(aggregate_unsigned),
    }
    evidence_path = stage / "evidence.json"
    evidence_path.write_bytes(contract.canonical_json_bytes(aggregate) + b"\n")
    evidence_sha = contract.file_sha256(evidence_path)
    html_text = _render_html(
        manifest=manifest,
        shards=shards,
        media_map=media_map,
        evidence_sha256=evidence_sha,
    )
    if any(token in html_text.lower() for token in ("<script", "http://", "https://")):
        fail("HTML is not self-contained")
    (stage / "index.html").write_text(html_text, encoding="utf-8")
    os.chmod(stage / "index.html", 0o444)
    os.chmod(evidence_path, 0o444)
    os.chmod(media_dir, 0o555)
    os.rename(stage, output)
    os.chmod(output, 0o555)
    return {
        "output": str(output),
        "index": str(output / "index.html"),
        "evidence_sha256": evidence_sha,
        "logical_records": aggregate["logical_record_count"],
        "physical_media_files": aggregate["physical_media_file_count"],
        "memory_input_kind": aggregate["memory_input_kind"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_review(
        manifest_path=Path(args.manifest).expanduser(),
        expected_manifest_sha256=args.expected_manifest_sha256,
        shard_root=Path(args.shard_root).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CheckpointReviewHtmlError",
    "build_parser",
    "build_review",
    "load_shards",
    "main",
]
