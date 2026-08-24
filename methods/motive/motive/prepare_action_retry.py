"""Prepare an immutable-code retry of the Motive/Lucy action experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCHEMA = "motive-action-retry-contract-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _link(target: Path, source: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace runtime link: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(strict=True), target_is_directory=source.is_dir())


def prepare_retry(
    *,
    run_id: str,
    run_root: Path,
    runtime_repo: Path,
    source_snapshot: Path,
    source_tree_sha256: str,
    live_repo: Path,
    base_run: Path,
    seed: int = 2026,
) -> dict[str, object]:
    run_root = run_root.expanduser().absolute()
    runtime_repo = runtime_repo.expanduser().absolute()
    source_snapshot = source_snapshot.expanduser().resolve()
    live_repo = live_repo.expanduser().resolve()
    base_run = base_run.expanduser().resolve()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"refusing to reuse retry run root: {run_root}")
    if runtime_repo.exists() or runtime_repo.is_symlink():
        raise FileExistsError(f"refusing to reuse runtime repo: {runtime_repo}")

    provenance_path = source_snapshot / "SOURCE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("source_tree_sha256") != source_tree_sha256:
        raise ValueError("source snapshot tree digest does not match request")

    prep = base_run / "prep"
    base_output = base_run / "lucy" / "e1_plain_lora" / f"seed_{seed}"
    base_log = base_output.with_suffix(".log")
    base_status = base_run / "status" / f"e1_plain_lora_seed_{seed}.json"
    final_checkpoint = base_output / "checkpoint_step_000100.pt"
    required = [
        prep / "lucy_train_manifest.jsonl",
        prep / "repr_seed_2026" / "prompt_action_encoder.pt",
        base_output / "run_config.json",
        base_log,
        base_status,
        final_checkpoint,
        live_repo / "data",
        live_repo / "checkpoints",
    ]
    required.extend(
        base_output / f"checkpoint_step_000100.rng_rank_{rank:03d}.pt"
        for rank in range(8)
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing retry input: {path}")

    run_root.mkdir(parents=True)
    runtime_repo.mkdir(parents=True)
    _link(runtime_repo / "lucy", source_snapshot / "lucy")
    _link(runtime_repo / "methods", source_snapshot / "methods")
    _link(runtime_repo / "data", live_repo / "data")
    _link(runtime_repo / "checkpoints", live_repo / "checkpoints")
    _link(run_root / "prep", prep)

    reused_output = run_root / "lucy" / "e1_plain_lora" / f"seed_{seed}"
    reused_output.mkdir(parents=True)
    for source in sorted(base_output.iterdir(), key=lambda item: item.name):
        _link(reused_output / source.name, source)
    _link(reused_output.with_suffix(".log"), base_log)
    _link(
        run_root / "status" / f"e1_plain_lora_seed_{seed}.json",
        base_status,
    )
    (run_root / "smoke" / "lucy").mkdir(parents=True)
    (run_root / "smoke" / "status").mkdir(parents=True)
    (run_root / "slurm").mkdir(parents=True)

    base_contract = base_run / "contracts" / "run_contract.json"
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
        "base_run": str(base_run),
        "base_run_contract": (
            {
                "path": str(base_contract),
                "sha256": _sha256(base_contract),
            }
            if base_contract.is_file()
            else None
        ),
        "reused_e1": {
            "checkpoint_path": str(final_checkpoint),
            "checkpoint_sha256": _sha256(final_checkpoint),
            "run_config_sha256": _sha256(base_output / "run_config.json"),
            "log_sha256": _sha256(base_log),
            "status_sha256": _sha256(base_status),
        },
        "training_contract": {
            "seed": int(seed),
            "max_steps": 100,
            "per_process_batch_size": 1,
            "num_processes": 8,
            "gradient_accumulation_steps": 1,
            "effective_global_batch_size": 8,
            "max_concurrent_nodes": 2,
            "gpus_per_node": 8,
            "arms": [
                "e1_plain_lora_reused",
                "e2_fixed_random",
                "e2_random_frozen_trunk_trainable_head",
                "e3_motive_frozen_trunk_trainable_head",
            ],
            "full_run_requires_all_six_smokes": True,
            "fresh_restart_for_failed_e2_e3": True,
        },
        "scientific_scope": (
            "Pseudo-label P0 representation ablation. E2 and E3 have matched "
            "frozen trunks and trainable component heads; the controlled "
            "difference is random versus Motive-pretrained trunk weights. A "
            "fixed-random routing arm separates gating from head learning."
        ),
    }
    _atomic_json(run_root / "contracts" / "retry_contract.json", contract)
    return contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--runtime-repo", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--live-repo", required=True, type=Path)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--seed", default=2026, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = prepare_retry(
        run_id=args.run_id,
        run_root=args.run_root,
        runtime_repo=args.runtime_repo,
        source_snapshot=args.source_snapshot,
        source_tree_sha256=args.source_tree_sha256,
        live_repo=args.live_repo,
        base_run=args.base_run,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "run_id": contract["run_id"],
                "run_root": contract["run_root"],
                "runtime_repo": contract["runtime_repo"],
                "source_tree_sha256": contract["source_tree_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
