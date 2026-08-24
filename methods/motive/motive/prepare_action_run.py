"""Freeze input, checkpoint, configuration, and environment contracts for P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCHEMA = "motive-action-run-contract-v1"
INVENTORY_SCHEMA = "motive-action-input-inventory-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _inventory(root: Path) -> tuple[list[dict[str, object]], str]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing inventory root: {root}")
    rows: list[dict[str, object]] = []
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        dir_names.sort()
        for name in sorted(file_names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if not resolved.is_file():
                    raise ValueError(f"inventory symlink is not a file: {path}")
                kind = "symlink_file"
                target = os.readlink(path)
            elif path.is_file():
                resolved = path
                kind = "file"
                target = None
            else:
                raise ValueError(f"inventory entry is not a regular file: {path}")
            row: dict[str, object] = {
                "path": relative,
                "type": kind,
                "size": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
            if target is not None:
                row["symlink_target"] = target
            rows.append(row)
    rows.sort(key=lambda row: str(row["path"]))
    digest = hashlib.sha256(_canonical(rows)).hexdigest()
    return rows, digest


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _write_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(
    *,
    run_id: str,
    run_root: Path,
    runtime_repo: Path,
    source_snapshot: Path,
    source_tree_sha256: str,
    source_run: Path,
) -> dict[str, object]:
    run_root = run_root.expanduser().resolve()
    runtime_repo = runtime_repo.expanduser().resolve()
    source_snapshot = source_snapshot.expanduser().resolve()
    source_run = source_run.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root must already exist: {run_root}")
    contracts = run_root / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract_path = contracts / "run_contract.json"
    inventory_path = contracts / "input_inventory.json"
    if contract_path.exists() or inventory_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing run contract under {contracts}"
        )

    source_provenance_path = source_snapshot / "SOURCE_PROVENANCE.json"
    source_provenance = json.loads(
        source_provenance_path.read_text(encoding="utf-8")
    )
    if source_provenance.get("source_tree_sha256") != source_tree_sha256:
        raise ValueError("source snapshot tree digest does not match request")

    fused_manifest = source_run / "fused" / "all.jsonl"
    feature_root = source_run / "feature_snapshot"
    checkpoint_root = runtime_repo / "checkpoints" / "LucyEdit"
    plain_config = (
        runtime_repo
        / "lucy"
        / "configs"
        / "goku_motive_action_plain_lora_pilot.json"
    )
    router_config = (
        runtime_repo
        / "lucy"
        / "configs"
        / "goku_motive_action_router_pilot.json"
    )
    for path in (fused_manifest, plain_config, router_config):
        if not path.is_file():
            raise FileNotFoundError(f"missing run-contract input: {path}")

    feature_files, feature_digest = _inventory(feature_root)
    checkpoint_files, checkpoint_digest = _inventory(checkpoint_root)
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "fused_manifest": {
            "path": str(fused_manifest),
            "size": fused_manifest.stat().st_size,
            "sha256": _sha256(fused_manifest),
        },
        "feature_snapshot": {
            "root": str(feature_root),
            "files": feature_files,
            "inventory_digest": feature_digest,
        },
        "lucy_checkpoint": {
            "root": str(checkpoint_root.resolve()),
            "files": checkpoint_files,
            "inventory_digest": checkpoint_digest,
        },
    }
    _write_atomic(inventory_path, inventory)
    inventory_sha256 = _sha256(inventory_path)

    torch_version = _package_version("torch")
    torch_hip = None
    if torch_version is not None:
        import torch

        torch_hip = torch.version.hip
    contract = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_root": str(run_root),
        "runtime_repo": str(runtime_repo),
        "source_snapshot": str(source_snapshot),
        "source_tree_sha256": source_tree_sha256,
        "source_manifest_sha256": _sha256(
            source_snapshot / "SOURCE_FILES.jsonl"
        ),
        "source_git_base_commit": source_provenance.get("git_base_commit"),
        "source_git_status_short": source_provenance.get("git_status_short"),
        "input_inventory_sha256": inventory_sha256,
        "feature_inventory_digest": feature_digest,
        "lucy_checkpoint_inventory_digest": checkpoint_digest,
        "configs": {
            "plain": {
                "path": str(plain_config),
                "sha256": _sha256(plain_config),
            },
            "router": {
                "path": str(router_config),
                "sha256": _sha256(router_config),
            },
        },
        "training_contract": {
            "arms": [
                "e1_plain_lora",
                "e2_random_router",
                "e3_motive_frozen",
            ],
            "seed": 2026,
            "max_steps": 100,
            "per_process_batch_size": 1,
            "num_processes": 8,
            "gradient_accumulation_steps": 1,
            "effective_global_batch_size": 8,
            "max_concurrent_nodes": 2,
            "gpus_per_node": 8,
        },
        "scientific_scope": (
            "pseudo-label representation and Lucy wiring/training-stability "
            "pilot; human_approved=false; production_eligible=false"
        ),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch_version,
            "torch_hip": torch_hip,
            "accelerate_version": _package_version("accelerate"),
            "diffusers_version": _package_version("diffusers"),
            "numpy_version": _package_version("numpy"),
        },
    }
    _write_atomic(contract_path, contract)
    return contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--runtime-repo", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--source-run", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = prepare(
        run_id=args.run_id,
        run_root=args.run_root,
        runtime_repo=args.runtime_repo,
        source_snapshot=args.source_snapshot,
        source_tree_sha256=args.source_tree_sha256,
        source_run=args.source_run,
    )
    print(
        json.dumps(
            {
                "run_id": contract["run_id"],
                "source_tree_sha256": contract["source_tree_sha256"],
                "input_inventory_sha256": contract[
                    "input_inventory_sha256"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
