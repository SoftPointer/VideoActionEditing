#!/usr/bin/env python3
"""Build a portable offline review site for the real case01 trajectory exact5.

The builder is downstream of the postflight bundle and independent all-frame
audit.  It recomputes the strict report, requires byte equality with the
provided sealed report, and validates every referenced video and all-81 sheet
before creating the site target.  It never creates placeholder media or an
unreviewed result page.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

try:
    from methods.bernini_action_editing.tools import (
        case01_object_trajectory_exact5_postflight_v1 as postflight,
    )
except ModuleNotFoundError:  # Direct execution from this tools directory.
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from methods.bernini_action_editing.tools import (
        case01_object_trajectory_exact5_postflight_v1 as postflight,
    )


SITE_SCHEMA = "case01-object-trajectory-exact5-offline-review-site-v3"
SITE_MANIFEST = Path("site-manifest.json")


class SiteBuildError(RuntimeError):
    """The review evidence or portable-site closure differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SiteBuildError(message)


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _badge(status: str) -> str:
    css = "pass" if status == "PASS" else "fail"
    return f'<span class="badge {css}">{_h(status)}</span>'


def _reason_list(reasons: Any) -> str:
    if not isinstance(reasons, list) or not reasons:
        return '<p class="quiet">No failing reason recorded.</p>'
    return "<ul>" + "".join(f"<li>{_h(reason)}</li>" for reason in reasons) + "</ul>"


def _validate_strict_report(
    *,
    bundle_root: Path,
    observations_path: Path,
    report_path: Path,
    ffmpeg: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = postflight.validate_bundle(bundle_root, ffmpeg=ffmpeg)
    review = postflight.validate_observations(bundle, observations_path)
    expected = postflight._build_strict_report_from_validated(
        bundle,
        review,
    )
    report, report_file = postflight._load_canonical_json(
        report_path.resolve(strict=True),
        label="strict report",
        allowed_modes={0o400, 0o444},
    )
    _require(
        postflight._json_exact_equal(report, expected),
        "strict report is not the recomputed report",
    )
    _require(
        report.get("schema_version") == postflight.STRICT_REPORT_SCHEMA
        and report.get("status") == "COMPLETE_FAIL_CLOSED"
        and report.get("arm_order") == list(postflight.ARM_ORDER)
        and report.get("primary_arm") == postflight.PRIMARY_ARM,
        "strict report identity differs",
    )
    _require(
        report.get("review_design") == postflight.REVIEW_DESIGN,
        "strict report review design differs",
    )
    return bundle, review, report, report_file


def _render_arm(
    *,
    report: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> str:
    arm = str(report["variant"])
    gates = report["gates"]
    identity = gates["dog_identity_retention"]
    bone = gates["same_source_bone_reuse"]
    action = gates["ordered_source_bone_action"]
    stages = observation["action_trace"]["stages"]
    stage_rows = "".join(
        "<tr>"
        f"<td>{_h(stage['name'])}</td>"
        f"<td>{_badge('PASS' if stage['observed'] else 'FAIL')}</td>"
        f"<td>{_h(stage['frame_interval'])}</td>"
        f"<td>{_h(stage['evidence'])}</td>"
        "</tr>"
        for stage in stages
    )
    gate_rows = (
        ("All 81 frames + source/output pair", gates["review_coverage"]),
        ("Same source dog identity", identity),
        ("Same source bone#1 reuse/conservation", bone),
        ("Approach → contact → grip → lift → hold", action),
    )
    gate_html = "".join(
        "<div class=\"gate\">"
        f"<h4>{_h(title)} {_badge(str(gate['status']))}</h4>"
        f"{_reason_list(gate.get('reasons'))}"
        "</div>"
        for title, gate in gate_rows
    )
    primary = '<span class="primary">primary canary</span>' if arm == postflight.PRIMARY_ARM else ""
    return f"""
    <article class="arm" id="{_h(arm)}">
      <header>
        <div><code>{_h(arm)}</code> {primary}</div>
        <div>{_badge(str(report['status']))}</div>
      </header>
      <div class="media-grid">
        <figure>
          <video controls preload="metadata" src="bundle/media/{_h(arm)}.mp4"></video>
          <figcaption>Real verified 81-frame output</figcaption>
        </figure>
        <figure>
          <a href="bundle/sheets/{_h(arm)}-all81.jpg">
            <img loading="lazy" src="bundle/sheets/{_h(arm)}-all81.jpg"
                 alt="{_h(arm)} all 81 frames, 9 by 9 row-major">
          </a>
          <figcaption>All frames 0…80, row-major 9×9</figcaption>
        </figure>
      </div>
      <section class="gates">{gate_html}</section>
      <details>
        <summary>Five-stage observation trace</summary>
        <table><thead><tr><th>Stage</th><th>Observed</th><th>Frames</th><th>Evidence</th></tr></thead>
        <tbody>{stage_rows}</tbody></table>
      </details>
    </article>
    """


def render_html(
    *,
    bundle: Mapping[str, Any],
    review: Mapping[str, Any],
    report: Mapping[str, Any],
) -> str:
    reports = {row["variant"]: row for row in report["arms"]}
    observations = {
        row["variant"]: row for row in review["observations"]["arms"]
    }
    arms_html = "".join(
        _render_arm(report=reports[arm], observation=observations[arm])
        for arm in postflight.ARM_ORDER
    )
    passing = ", ".join(report["passing_arms"]) or "none"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case01 object-trajectory exact5 — strict all-frame review</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1018; --panel:#141c28; --line:#2a394d;
  --text:#ecf2fa; --quiet:#9fb0c5; --pass:#37d990; --fail:#ff6675; --accent:#78a9ff; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ width:min(1500px,96vw); margin:32px auto 80px }}
h1,h2,h3,h4,p {{ margin-top:0 }} code {{ color:#b9d2ff }}
.notice,.summary,.arm {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:18px; margin:18px 0 }}
.notice {{ border-left:5px solid #f2b84b }} .summary {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px }}
.metric {{ background:#0f1621; padding:13px; border-radius:9px }} .metric strong {{ display:block;font-size:23px }}
.arm>header {{ display:flex; align-items:center; justify-content:space-between; gap:14px; font-size:18px }}
.media-grid {{ display:grid; grid-template-columns:minmax(300px,.82fr) minmax(520px,1.5fr); gap:16px }}
figure {{ margin:0 }} video,img {{ display:block; width:100%; max-height:740px; object-fit:contain;
  background:#05070a; border:1px solid var(--line); border-radius:9px }} figcaption {{ color:var(--quiet);margin-top:6px }}
.gates {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; margin-top:16px }}
.gate {{ background:#0f1621;border:1px solid var(--line);padding:12px;border-radius:9px }}
.gate h4 {{ min-height:46px }} .badge {{ display:inline-block;border-radius:999px;padding:2px 9px;
  font-weight:750;font-size:12px }} .badge.pass {{ background:#123c2c;color:var(--pass) }}
.badge.fail {{ background:#481c25;color:var(--fail) }} .primary {{ color:#07111e;background:var(--accent);
  padding:2px 8px;border-radius:999px;font-size:12px }} .quiet,figcaption {{ color:var(--quiet) }}
table {{ width:100%;border-collapse:collapse;margin-top:10px }} th,td {{ text-align:left;padding:8px;
  border-bottom:1px solid var(--line);vertical-align:top }} summary {{ cursor:pointer;color:#c8d9f0;margin-top:14px }}
ul {{ padding-left:20px;margin-bottom:0 }} a {{ color:#9fc0ff }}
@media(max-width:900px) {{ .media-grid {{ grid-template-columns:1fr }} }}
</style>
</head>
<body><main>
<h1>Case01 object-trajectory exact5</h1>
<p class="quiet">IID {_h(postflight.IID)} · strict independent non-blind all-81-frame review</p>
<section class="notice">
  <h2>Claim boundary</h2>
  <p>This is a one-case, one-seed engineering-oracle canary using a hand-authored,
  zero-training trajectory scaffold. It does not demonstrate learned object-centric
  representation learning and does not authorize a causal or scientific claim.</p>
  <p>An arm passes only if all-frame coverage, source-dog identity, reuse/conservation
  of the same original <code>bone#1</code>, and the complete ordered action chain all pass.
  There is no averaging or compensation.</p>
  <p>Review design: <code>{_h(report['review_design'])}</code>. No randomized arm aliases
  or sealed alias key were used, so this audit is not described as blind.</p>
</section>
<section class="summary">
  <div class="metric"><span>Primary arm</span><strong>{_h(report['primary_canary_status'])}</strong><code>{_h(postflight.PRIMARY_ARM)}</code></div>
  <div class="metric"><span>Passing arms</span><strong>{_h(report['counts']['pass_count'])}/5</strong><span>{_h(passing)}</span></div>
  <div class="metric"><span>Evidence</span><strong>5 × 81</strong><span>real outputs + native receipts</span></div>
  <div class="metric"><span>Rule</span><strong>AND</strong><span>identity ∧ same bone ∧ 5-stage action</span></div>
</section>
<section class="arm">
  <header><div><code>source / exact_original</code></div></header>
  <div class="media-grid">
    <figure><video controls preload="metadata" src="bundle/source/exact_original.mp4"></video><figcaption>Original source</figcaption></figure>
    <figure><a href="bundle/sheets/source-exact_original-all81.jpg"><img src="bundle/sheets/source-exact_original-all81.jpg" alt="Source all 81 frames"></a><figcaption>Source frames 0…80, row-major 9×9</figcaption></figure>
  </div>
</section>
{arms_html}
<section class="notice">
  <h2>Evidence authority</h2>
  <p>Postflight manifest <code>{_h(bundle['manifest_sha256'])}</code></p>
  <p>Strict report digest <code>{_h(report['report_digest'])}</code></p>
  <p>Observations digest <code>{_h(review['observations']['observations_digest'])}</code></p>
</section>
</main></body></html>
"""


def _file_row(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    row = postflight._stable_file(path, label=role, allowed_modes={0o400, 0o444})
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "sha256": row["sha256"],
        "size": row["size"],
        "mode": f"0{row['mode']:03o}",
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    postflight._write_create_only_bytes(
        path,
        payload,
        final_mode=0o400,
        label=f"site create-only file {path.name}",
    )


def _bundle_file_modes(bundle_root: Path) -> dict[str, int]:
    """Return the exact already-validated portable bundle file closure."""

    rows: dict[str, int] = {}
    for path in sorted(bundle_root.rglob("*")):
        relative = path.relative_to(bundle_root).as_posix()
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            _require(not stat.S_ISLNK(info.st_mode), f"bundle directory resolves elsewhere: {relative}")
            continue
        _require(
            stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"bundle member is not a plain file: {relative}",
        )
        mode = stat.S_IMODE(info.st_mode)
        expected = (
            0o400
            if relative
            in {postflight.MANIFEST_REL.as_posix(), postflight.PUBLICATION_MARKER_REL.as_posix()}
            else 0o444
        )
        _require(mode == expected, f"bundle member mode differs: {relative}")
        rows[relative] = expected
    _require(len(rows) == 22, "embedded bundle file count differs")
    return rows


def _copy_full_bundle(source_root: Path, target_root: Path) -> dict[str, int]:
    modes = _bundle_file_modes(source_root)
    for relative, mode in modes.items():
        source = source_root / relative
        row = postflight._stable_file(
            source, label=f"embedded bundle source {relative}", allowed_modes={mode}
        )
        postflight._copy_verified(
            source,
            target_root / relative,
            sha256=row["sha256"],
            size=row["size"],
            final_mode=mode,
        )
    return modes


def build_site(
    *,
    bundle_root: str | Path,
    observations_path: str | Path,
    strict_report_path: str | Path,
    site_root: str | Path,
    ffmpeg: str | Path,
) -> dict[str, Any]:
    destination = Path(site_root).expanduser()
    _require(destination.is_absolute(), "site root is not absolute")
    _require(os.path.normpath(str(destination)) == str(destination), "site root is not canonical")
    _require(not os.path.lexists(destination), "site root already exists")
    bundle_path = postflight._canonical_existing_path(bundle_root, label="postflight bundle")
    observations = postflight._canonical_existing_path(observations_path, label="strict observations")
    strict_report = postflight._canonical_existing_path(strict_report_path, label="strict report")
    ffmpeg_path = postflight._canonical_existing_path(
        ffmpeg, label="site validation ffmpeg"
    )

    # Every evidence check completes before the destination or staging path is
    # created.  In particular, absent five-arm outputs cannot leave a site.
    bundle, review, report, report_file = _validate_strict_report(
        bundle_root=bundle_path,
        observations_path=observations,
        report_path=strict_report,
        ffmpeg=ffmpeg_path,
    )
    postflight._ensure_plain_parent(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    staging_owned = postflight._owned_identity(os.lstat(staging))
    try:
        bundle_modes = _copy_full_bundle(bundle["root"], staging / "bundle")
        # This is deliberately before HTML rendering: a copied manifest alone
        # is not a portable evidence bundle.
        embedded_bundle = postflight.validate_bundle(
            staging / "bundle", ffmpeg=ffmpeg_path
        )
        _require(
            embedded_bundle["manifest_sha256"] == bundle["manifest_sha256"],
            "embedded bundle authority differs after copy",
        )
        evidence_sources = (
            (observations, "evidence/strict-observations.json", review["observations_sha256"], review["observations_size"]),
            (strict_report, "evidence/strict-report.json", report_file["sha256"], report_file["size"]),
        )
        for source, relative, sha256, size in evidence_sources:
            postflight._copy_verified(source, staging / relative, sha256=sha256, size=size)
        html_payload = render_html(
            bundle=embedded_bundle, review=review, report=report
        ).encode("utf-8")
        _write_create_only(staging / "index.html", html_payload)
        artifacts: list[dict[str, Any]] = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            artifacts.append(_file_row(staging, path, role=path.relative_to(staging).as_posix()))
        manifest: dict[str, Any] = {
            "schema_version": SITE_SCHEMA,
            "status": "OFFLINE_REVIEW_SITE_COMPLETE",
            "case_id": postflight.CASE_ID,
            "iid": postflight.IID,
            "arm_order": list(postflight.ARM_ORDER),
            "primary_canary_status": report["primary_canary_status"],
            "review_design": report["review_design"],
            "postflight_manifest_sha256": embedded_bundle["manifest_sha256"],
            "postflight_manifest_digest": embedded_bundle["manifest"]["manifest_digest"],
            "postflight_manifest_status": embedded_bundle["manifest"]["status"],
            "postflight_manifest_schema": embedded_bundle["manifest"]["schema_version"],
            "observations_sha256": review["observations_sha256"],
            "observations_digest": review["observations"]["observations_digest"],
            "observations_status": review["observations"]["status"],
            "observations_schema": review["observations"]["schema_version"],
            "observations_role": report["observation_authority"]["role"],
            "strict_report_sha256": report_file["sha256"],
            "strict_report_digest": report["report_digest"],
            "strict_report_status": report["status"],
            "strict_report_schema": report["schema_version"],
            "real_output_count": 5,
            "all81_sheet_count": 6,
            "placeholder_media_count": 0,
            "embedded_bundle_file_count": len(bundle_modes),
            "relative_urls_only": True,
            "scientific_claim_authorized": False,
            "artifacts": artifacts,
        }
        manifest["manifest_digest"] = postflight.object_sha256(manifest)
        manifest_file = postflight._write_create_only_json(
            staging / SITE_MANIFEST,
            manifest,
            digest_field="manifest_digest",
        )
        validate_site(
            staging,
            ffmpeg=ffmpeg_path,
            require_publication_marker=False,
        )
        marker = postflight.build_publication_marker(
            kind="offline-review-site",
            authority_role="site-manifest",
            authority_path=SITE_MANIFEST.as_posix(),
            authority_sha256=manifest_file["sha256"],
            authority_digest=manifest["manifest_digest"],
        )
        published_owned = postflight._publish_directory_create_only(
            staging, destination, marker=marker
        )
        try:
            return validate_site(destination, ffmpeg=ffmpeg_path)
        except BaseException:
            postflight._cleanup_owned_directory(destination, published_owned)
            raise
    except BaseException:
        postflight._cleanup_owned_directory(staging, staging_owned)
        raise


def validate_site(
    site_root: str | Path, *, ffmpeg: str | Path,
    require_publication_marker: bool = True,
) -> dict[str, Any]:
    root = postflight._canonical_existing_path(site_root, label="site root")
    _require(root.is_dir() and not root.is_symlink(), "site root is not a plain directory")
    expected_root = {"bundle", "evidence", "index.html", SITE_MANIFEST.name}
    if require_publication_marker:
        expected_root.add(postflight.PUBLICATION_MARKER_REL.name)
    postflight._exact_names(root, expected_root, label="site root")
    postflight._exact_names(
        root / "evidence",
        {"strict-observations.json", "strict-report.json"},
        label="site evidence",
    )

    # Full bundle validation is the first semantic replay.  Site evidence may
    # not substitute a copied top-level postflight manifest for this closure.
    embedded = postflight.validate_bundle(root / "bundle", ffmpeg=ffmpeg)
    bundle_modes = _bundle_file_modes(root / "bundle")
    manifest, manifest_file = postflight._load_canonical_json(
        root / SITE_MANIFEST,
        label="site manifest",
        allowed_modes={0o400},
    )
    postflight._strict_digest(manifest, "manifest_digest", label="site manifest")
    expected_manifest_fields = {
        "schema_version", "status", "case_id", "iid", "arm_order",
        "primary_canary_status", "review_design", "postflight_manifest_sha256",
        "postflight_manifest_digest", "postflight_manifest_status",
        "postflight_manifest_schema", "observations_sha256", "observations_digest",
        "observations_status", "observations_schema", "observations_role",
        "strict_report_sha256", "strict_report_digest", "strict_report_status",
        "strict_report_schema", "real_output_count", "all81_sheet_count",
        "placeholder_media_count", "embedded_bundle_file_count",
        "relative_urls_only", "scientific_claim_authorized", "artifacts",
        "manifest_digest",
    }
    _require(set(manifest) == expected_manifest_fields, "site manifest schema differs")
    _require(
        manifest.get("schema_version") == SITE_SCHEMA
        and manifest.get("status") == "OFFLINE_REVIEW_SITE_COMPLETE"
        and manifest.get("case_id") == postflight.CASE_ID
        and manifest.get("iid") == postflight.IID
        and manifest.get("arm_order") == list(postflight.ARM_ORDER)
        and manifest.get("review_design") == postflight.REVIEW_DESIGN
        and manifest.get("observations_role") == postflight.OBSERVATION_ROLE,
        "site manifest closure differs",
    )
    for field, expected in (
        ("real_output_count", 5), ("all81_sheet_count", 6),
        ("placeholder_media_count", 0),
        ("embedded_bundle_file_count", 22),
    ):
        postflight._require_exact_int(
            manifest.get(field), expected, label=f"site {field}"
        )
    postflight._require_exact_bool(
        manifest.get("relative_urls_only"), True, label="site relative URLs"
    )
    postflight._require_exact_bool(
        manifest.get("scientific_claim_authorized"), False,
        label="site scientific claim",
    )

    artifacts = manifest.get("artifacts")
    expected_paths = {
        "index.html",
        "evidence/strict-observations.json",
        "evidence/strict-report.json",
        *(f"bundle/{relative}" for relative in bundle_modes),
    }
    _require(
        isinstance(artifacts, list)
        and len(artifacts) == 25
        and all(isinstance(row, Mapping) for row in artifacts)
        and all(isinstance(row.get("path"), str) for row in artifacts)
        and {row.get("path") for row in artifacts} == expected_paths,
        "site artifact path/count closure differs",
    )
    artifact_by_path = {str(row["path"]): row for row in artifacts}
    _require(len(artifact_by_path) == len(artifacts), "site artifact path reuse differs")
    for path, row in artifact_by_path.items():
        if path == "index.html":
            expected_mode = 0o400
        elif path.startswith("bundle/"):
            expected_mode = bundle_modes[path[len("bundle/"):]]
        else:
            expected_mode = 0o444
        postflight._validate_member(
            root,
            row,
            label=f"site artifact {path}",
            expected_role=path,
            expected_path=path,
            expected_mode=expected_mode,
        )

    observations_path = root / "evidence/strict-observations.json"
    report_path = root / "evidence/strict-report.json"
    review = postflight.validate_observations(embedded, observations_path)
    report, report_file = postflight._load_canonical_json(
        report_path,
        label="site copied strict report",
        allowed_modes={0o444},
        expected_sha256=artifact_by_path["evidence/strict-report.json"]["sha256"],
    )
    postflight._strict_digest(report, "report_digest", label="site copied strict report")
    expected_report = postflight._build_strict_report_from_validated(embedded, review)
    _require(
        postflight._json_exact_equal(report, expected_report),
        "site copied strict report is not deterministically recomputed",
    )
    observation_artifact = artifact_by_path["evidence/strict-observations.json"]
    bindings = {
        "postflight_manifest_sha256": embedded["manifest_sha256"],
        "postflight_manifest_digest": embedded["manifest"]["manifest_digest"],
        "postflight_manifest_status": embedded["manifest"]["status"],
        "postflight_manifest_schema": embedded["manifest"]["schema_version"],
        "observations_sha256": observation_artifact["sha256"],
        "observations_digest": review["observations"]["observations_digest"],
        "observations_status": review["observations"]["status"],
        "observations_schema": review["observations"]["schema_version"],
        "observations_role": report["observation_authority"]["role"],
        "strict_report_sha256": report_file["sha256"],
        "strict_report_digest": report["report_digest"],
        "strict_report_status": report["status"],
        "strict_report_schema": report["schema_version"],
    }
    for key, value in bindings.items():
        _require(
            manifest.get(key) == value and type(manifest.get(key)) is type(value),
            f"site top-level binding differs: {key}",
        )
    _require(
        manifest.get("primary_canary_status") == report["primary_canary_status"]
        and type(manifest.get("primary_canary_status")) is str,
        "site primary status/report binding differs",
    )
    _require(
        report.get("postflight_authority")
        == {
            "role": "postflight-manifest",
            "path": postflight.MANIFEST_REL.as_posix(),
            "sha256": embedded["manifest_sha256"],
            "manifest_digest": embedded["manifest"]["manifest_digest"],
        },
        "site strict-report/postflight authority differs",
    )

    expected_index = render_html(
        bundle=embedded,
        review=review,
        report=report,
    ).encode("utf-8")
    index_row = postflight._stable_file(
        root / "index.html",
        label="site deterministic index",
        expected_sha256=artifact_by_path["index.html"]["sha256"],
        expected_size=artifact_by_path["index.html"]["size"],
        allowed_modes={0o400},
        return_bytes=True,
    )
    _require(
        index_row["bytes"] == expected_index,
        "site index bytes differ from sealed evidence",
    )
    if require_publication_marker:
        postflight.validate_publication_marker(
            root,
            kind="offline-review-site",
            authority_role="site-manifest",
            authority_path=SITE_MANIFEST.as_posix(),
            authority_sha256=manifest_file["sha256"],
            authority_digest=manifest["manifest_digest"],
        )
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": root / SITE_MANIFEST,
        "manifest_sha256": manifest_file["sha256"],
        "index_path": root / "index.html",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--bundle", required=True)
    build.add_argument("--observations", required=True)
    build.add_argument("--strict-report", required=True)
    build.add_argument("--site", required=True)
    build.add_argument("--ffmpeg", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--site", required=True)
    verify.add_argument("--ffmpeg", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            value = build_site(
                bundle_root=args.bundle,
                observations_path=args.observations,
                strict_report_path=args.strict_report,
                site_root=args.site,
                ffmpeg=args.ffmpeg,
            )
            status = "OFFLINE_REVIEW_SITE_CREATED"
        else:
            value = validate_site(args.site, ffmpeg=args.ffmpeg)
            status = "OFFLINE_REVIEW_SITE_VALID"
    except (OSError, postflight.PostflightError, SiteBuildError) as error:
        print(json.dumps({"status": "EVIDENCE_INVALID", "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": status,
                "site": str(value["root"]),
                "manifest_sha256": value["manifest_sha256"],
                "index": str(value["index_path"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
