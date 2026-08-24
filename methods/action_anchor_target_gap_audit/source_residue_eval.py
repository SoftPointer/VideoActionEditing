#!/usr/bin/env python3
"""MEV Qwen v3 trace-only audit with an explicit source-action residue gate.

The video is still observed exactly once without an instruction.  This module
reuses that frozen v2 neutral trace, then judges both the requested target
transitions and a metadata-derived harmful source-event behavior.  Source
identity, scene, layout, camera, and the initial pose are explicitly excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .audit import MANIFEST_SCHEMA, assert_not_protected_write, file_sha256
from .corrected_eval import (
    NativeVideoQwenJudge,
    RECORD_SCHEMA as V2_RECORD_SCHEMA,
    ROLES,
    TRACE_SCHEMA,
    YES_NO_UNCERTAIN,
    _extract_json,
    _validate_evidence,
    _winner,
    load_contracts,
    load_json,
    utc_now,
    validate_inputs,
    validate_trace,
    write_json,
    write_jsonl,
)


SOURCE_CONTRACT_SCHEMA = "mev-action-source-residue-contracts-v3"
OBSERVATION_SCHEMA = "mev-action-source-residue-observation-v3"
RECORD_SCHEMA = "mev-action-source-residue-record-v3"
SUMMARY_SCHEMA = "mev-action-source-residue-summary-v3"


SOURCE_ONLY_OBSERVATION_SCHEMA = "mev-action-source-only-observation-v3"


SOURCE_AWARE_SYSTEM_PROMPT = """You are a conservative TRACE-ONLY detector of
a forbidden source-event behavior. You do not see any video in this stage. The
sole visual evidence is an instruction-free neutral trace frozen before any
source annotation or candidate role was disclosed. Judge only the supplied
harmful behavioral predicate. Never use identity, scene, object presence,
layout, camera, clothing, or a single initial pose as evidence. The word
"residue" does NOT mean the behavior must remain at the end: when the supplied
predicate says an occurrence at any checkpoint is harmful, one clear occurrence
makes the result yes permanently, even if the subject later stops it or performs
another action. Conversely, when the predicate explicitly says a new transition
disqualifies residue, that transition makes the result no. Predicate ids are
opaque. Apply the full natural-language definition literally, including its
temporal quantifiers. Report uncertainty instead of inventing motion. Return
exactly one JSON object and no Markdown."""


SOURCE_AWARE_TEMPLATE = """Use only the frozen chronological trace below. You
do not have the video, target instruction, source video, real target video,
candidate role, layout comparison, or identity comparison. This is a separate
source-only negative gate; do not balance it against target completion.

The original MEV source event is context for naming the behavior, not visual
evidence. Never count the same person, object, scene, clothing, camera, spatial
layout, or the initial checkpoint alone. Scan the ENTIRE trace before deciding.
Apply the harmful predicate's temporal wording exactly:

- "at any checkpoint" / "performed or retained" is an ANY-OCCURRENCE rule. A
  clear matching checkpoint forces result=yes. Later standing upright, stopping
  the behavior, or doing a desired action cannot erase that violation.
- "across multiple checkpoints" / "only continues" is a PERSISTENCE rule. A
  single initial pose is insufficient; use the named transition that confirms
  or disqualifies persistence.
- If the definition says any new turn/transition disqualifies residue, one clear
  such transition forces result=no and a later held endpoint cannot restore it.
- A discontinuity cannot prove the physical transition crossing the cut.

Frozen instruction-free neutral trace:
{trace}

Original MEV source-event annotation (behavior name only):
{source_action}

Harmful source-event predicate:
- {source_residue_id}: {source_residue_description}

Return exactly this shape. Evidence must quote or closely paraphrase checkpoints
from the frozen trace and must support the chosen result:
{{
  "schema_version": "mev-action-source-only-observation-v3",
  "source_action_residue": {{
    "id": "supplied_source_residue_id",
    "result": "yes|no|uncertain",
    "evidence": [{{"phase": "early|middle|late", "observation": "trace-grounded behavioral evidence; never appearance/layout"}}]
  }},
  "temporal_rule_application": "state whether this is any-occurrence, persistence, or transition-disqualified logic and why the evidence implies the result",
  "summary": "one short source-only conclusion"
}}"""


def _format_items(items: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"- {item['id']}: {item['description']}" for item in items)


def load_source_contracts(
    path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Mapping[str, Any]]:
    payload = load_json(path)
    if payload.get("schema_version") != SOURCE_CONTRACT_SCHEMA:
        raise ValueError("source residue contract schema differs")
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("source residue contracts must contain exactly 16 samples")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        prefix = row.get("pair_prefix")
        if not isinstance(prefix, str) or prefix in indexed:
            raise ValueError("source residue contract prefix is missing or duplicated")
        caption = row.get("source_action_caption")
        residue = row.get("source_residue")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"{prefix} source action caption differs")
        if not isinstance(residue, dict) or set(residue) != {"id", "description"}:
            raise ValueError(f"{prefix} source residue shape differs")
        if any(not isinstance(residue[key], str) or not residue[key].strip() for key in residue):
            raise ValueError(f"{prefix} source residue value differs")
        indexed[prefix] = row
    if manifest is not None:
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("manifest schema differs")
        manifest_rows = {row["pair_prefix"]: row for row in manifest.get("samples", [])}
        if set(manifest_rows) != set(indexed):
            raise ValueError("manifest and source residue contract pair sets differ")
        for prefix, row in indexed.items():
            if row["source_action_caption"] != manifest_rows[prefix]["source_action_caption"]:
                raise ValueError(f"{prefix} source caption is not verbatim frozen MEV metadata")
    return indexed


def make_source_aware_prompt(
    target_contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    trace: Mapping[str, Any],
    reverse_order: bool,
) -> str:
    # target_contract and reverse_order remain in the public signature so old
    # callers cannot accidentally bypass input validation.  They are
    # deliberately not disclosed to the source-only judge.
    del target_contract, reverse_order
    validate_trace(trace)
    residue = source_contract["source_residue"]
    return SOURCE_AWARE_TEMPLATE.format(
        trace=json.dumps(trace, sort_keys=True, ensure_ascii=False),
        source_action=source_contract["source_action_caption"],
        source_residue_id=residue["id"],
        source_residue_description=residue["description"],
    )


def validate_source_only_observation(
    value: Any,
    source_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = {
        "schema_version", "source_action_residue", "temporal_rule_application",
        "summary",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("source-only observation keys differ")
    if value["schema_version"] != SOURCE_ONLY_OBSERVATION_SCHEMA:
        raise ValueError("source-only observation schema differs")
    for field in ("temporal_rule_application", "summary"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"source-only {field} differs")
    residue = value["source_action_residue"]
    expected_id = source_contract["source_residue"]["id"]
    if not isinstance(residue, dict) or set(residue) != {"id", "result", "evidence"}:
        raise ValueError("source-only residue shape differs")
    if residue["id"] != expected_id or residue["result"] not in YES_NO_UNCERTAIN:
        raise ValueError("source-only residue value differs")
    _validate_evidence(residue["evidence"])
    return value


def parse_source_only_observation(
    raw: str,
    source_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    return validate_source_only_observation(_extract_json(raw), source_contract)


def validate_observation(
    value: Any,
    target_contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = {
        "schema_version", "action_observable", "requested_action_complete",
        "required_predicates", "forbidden_behaviors", "source_action_residue",
        "summary",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("source-aware observation keys differ")
    if value["schema_version"] != OBSERVATION_SCHEMA:
        raise ValueError("source-aware observation schema differs")
    for field in ("action_observable", "requested_action_complete"):
        if value[field] not in YES_NO_UNCERTAIN:
            raise ValueError(f"{field} enum differs")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise ValueError("source-aware summary differs")
    for field in ("required_predicates", "forbidden_behaviors"):
        expected_ids = {item["id"] for item in target_contract[field]}
        rows = value[field]
        if not isinstance(rows, list) or len(rows) != len(expected_ids):
            raise ValueError(f"{field} count differs")
        if {row.get("id") for row in rows if isinstance(row, dict)} != expected_ids:
            raise ValueError(f"{field} ids differ")
        for row in rows:
            if set(row) != {"id", "result", "evidence"} or row["result"] not in YES_NO_UNCERTAIN:
                raise ValueError(f"{field} item differs")
            _validate_evidence(row["evidence"])
    residue = value["source_action_residue"]
    expected_residue_id = source_contract["source_residue"]["id"]
    if not isinstance(residue, dict) or set(residue) != {"id", "result", "evidence"}:
        raise ValueError("source residue observation shape differs")
    if residue["id"] != expected_residue_id or residue["result"] not in YES_NO_UNCERTAIN:
        raise ValueError("source residue observation value differs")
    _validate_evidence(residue["evidence"])
    return value


def parse_observation(
    raw: str,
    target_contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    return validate_observation(_extract_json(raw), target_contract, source_contract)


def observation_components(observation: Mapping[str, Any]) -> list[int]:
    score = {"yes": 4, "uncertain": 2, "no": 0}
    inverse = {"yes": 0, "uncertain": 2, "no": 4}
    components = [
        score[observation["action_observable"]],
        score[observation["requested_action_complete"]],
    ]
    components.extend(score[row["result"]] for row in observation["required_predicates"])
    components.extend(inverse[row["result"]] for row in observation["forbidden_behaviors"])
    # Harmful source action is a noncompensatory failure even when a subset of
    # the requested target behavior is also visible.
    components.append(inverse[observation["source_action_residue"]["result"]])
    return components


def observation_gate(observation: Mapping[str, Any]) -> int:
    return min(observation_components(observation))


def observation_coverage(observation: Mapping[str, Any]) -> float:
    return mean(observation_components(observation))


def _source_rows(paths: Sequence[str]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl_any(path))
    keys = [(row["pair_prefix"], row["role"], row["pass_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("source v2 records contain duplicate pair/role/pass keys")
    if any(row.get("schema_version") != V2_RECORD_SCHEMA for row in rows):
        raise ValueError("source record is not v2")
    if any(row.get("neutral_trace") is None for row in rows):
        raise ValueError("source record has no valid frozen neutral trace")
    return rows


def _read_jsonl_any(path: str | Path) -> list[Mapping[str, Any]]:
    """Read JSONL without assuming the v2 record schema.

    corrected_eval's private reader intentionally accepts only v2 records, so
    using it for a resumable v3 output or v3 summary rejects every valid row.
    Schema checks remain explicit at each v2/v3 call site below.
    """

    rows: list[Mapping[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL row is not an object")
            rows.append(value)
    return rows


def qwen_rescore(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve(strict=True)
    target_path = Path(args.target_contracts).resolve(strict=True)
    source_path = Path(args.source_contracts).resolve(strict=True)
    manifest = load_json(manifest_path)
    target_contracts = load_contracts(target_path)
    validate_inputs(manifest, target_contracts)
    source_contracts = load_source_contracts(source_path, manifest)
    if set(source_contracts) != set(target_contracts):
        raise ValueError("target and source residue contract pair sets differ")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index differs")
    all_rows = _source_rows(args.records)
    expected_total = len(target_contracts) * 8
    if len(all_rows) != expected_total:
        raise ValueError(f"source v2 record count differs: {len(all_rows)} != {expected_total}")
    samples = [
        row for row in manifest["samples"]
        if row["ordinal"] % args.num_shards == args.shard_index
    ]
    if args.pair_prefix:
        requested = set(args.pair_prefix)
        samples = [row for row in samples if row["pair_prefix"] in requested]
        if {row["pair_prefix"] for row in samples} != requested:
            raise ValueError("requested prefix is not in selected shard")
    selected = {row["pair_prefix"] for row in samples}
    source_rows = [row for row in all_rows if row["pair_prefix"] in selected]
    output = Path(args.output)
    assert_not_protected_write(output)
    records = _read_jsonl_any(output) if args.resume and output.is_file() else []
    existing = set()
    for row in records:
        if row.get("schema_version") != RECORD_SCHEMA:
            raise ValueError("resume v3 record schema differs")
        key = (row["pair_prefix"], row["role"], row["pass_index"])
        if row["pair_prefix"] not in selected or key in existing:
            raise ValueError(f"invalid resume v3 record {key}")
        existing.add(key)
    role_order = {role: index for index, role in enumerate(ROLES)}
    ordinal = {row["pair_prefix"]: row["ordinal"] for row in samples}
    source_rows.sort(key=lambda row: (ordinal[row["pair_prefix"]], role_order[row["role"]], row["pass_index"]))
    judge = NativeVideoQwenJudge(Path(args.model).resolve(strict=True), args.max_new_tokens, 32)
    for row in source_rows:
        key = (row["pair_prefix"], row["role"], row["pass_index"])
        if key in existing:
            print(json.dumps({"pair_prefix": key[0], "role": key[1], "pass": key[2], "resumed": True}), flush=True)
            continue
        prefix = row["pair_prefix"]
        trace = validate_trace(row["neutral_trace"])
        prompt = make_source_aware_prompt(
            target_contracts[prefix], source_contracts[prefix], trace,
            reverse_order=bool(row["pass_index"]),
        )
        raw = judge.generate_text(prompt, system_prompt=SOURCE_AWARE_SYSTEM_PROMPT)
        try:
            source_only = parse_source_only_observation(raw, source_contracts[prefix])
            target_only = row.get("observation")
            if target_only is None:
                raise ValueError("frozen v2 target-only observation is missing")
            observation = dict(target_only)
            observation["schema_version"] = OBSERVATION_SCHEMA
            observation["source_action_residue"] = source_only["source_action_residue"]
            observation["summary"] = (
                f"Target-only: {target_only['summary']} Source-only: {source_only['summary']}"
            )
            validate_observation(observation, target_contracts[prefix], source_contracts[prefix])
            parse_error = None
        except Exception as error:
            source_only = None
            observation = None
            parse_error = f"{type(error).__name__}: {error}"
        updated = dict(row)
        updated["v2_trace_judgment"] = {
            "schema_version": row["schema_version"],
            "prompt_sha256": row["prompt_sha256"],
            "raw_output_sha256": hashlib.sha256(row["raw_output"].encode("utf-8")).hexdigest(),
            "parse_error": row["parse_error"],
            "observation": row["observation"],
        }
        updated.update({
            "schema_version": RECORD_SCHEMA,
            "source_action_caption": source_contracts[prefix]["source_action_caption"],
            "source_residue_contract": source_contracts[prefix]["source_residue"],
            "source_residue_contracts_sha256": file_sha256(source_path),
            "target_contracts_sha256": file_sha256(target_path),
            "manifest_sha256": file_sha256(manifest_path),
            "source_video_or_layout_shown_to_model": False,
            "source_action_caption_disclosed_only_after_neutral_trace": True,
            "neutral_trace_reused_without_video_inference": True,
            "target_and_source_judged_in_separate_prompts": True,
            "target_only_observation_reused_from_v2": True,
            "trace_only_system_prompt_sha256": hashlib.sha256(
                SOURCE_AWARE_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output": raw,
            "parse_error": parse_error,
            "source_only_observation": source_only,
            "observation": observation,
            "rescored_at": utc_now(),
        })
        records.append(updated)
        existing.add(key)
        write_jsonl(output, records)
        print(json.dumps({
            "pair_prefix": prefix, "role": row["role"], "pass": row["pass_index"],
            "parsed": observation is not None,
        }), flush=True)
    write_jsonl(output, records)
    return 0


def qwen_summarize(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    target_contracts = load_contracts(args.target_contracts)
    validate_inputs(manifest, target_contracts)
    source_contracts = load_source_contracts(args.source_contracts, manifest)
    rows: list[Mapping[str, Any]] = []
    for path in sorted(Path(args.records_dir).glob("qwen-v3-shard-*.jsonl")):
        rows.extend(_read_jsonl_any(path))
    grouped = {prefix: {role: [] for role in ROLES} for prefix in target_contracts}
    for row in rows:
        if row.get("schema_version") != RECORD_SCHEMA:
            raise ValueError("v3 record schema differs")
        grouped[row["pair_prefix"]][row["role"]].append(row)
    pairs = []
    expected = {role: (2 if role in {"anchor", "frozen_base"} else 1) for role in ROLES}
    for prefix, target_contract in target_contracts.items():
        role_rows = grouped[prefix]
        if any(len(role_rows[role]) != count for role, count in expected.items()):
            raise ValueError(f"incomplete Qwen v3 records for {prefix}")
        gates: dict[str, list[int | None]] = {}
        coverage: dict[str, list[float | None]] = {}
        target_only_gates: dict[str, list[int | None]] = {}
        residue_results: dict[str, list[str | None]] = {}
        for role, role_values in role_rows.items():
            role_values.sort(key=lambda row: row["pass_index"])
            gates[role] = [observation_gate(row["observation"]) if row["observation"] else None for row in role_values]
            coverage[role] = [observation_coverage(row["observation"]) if row["observation"] else None for row in role_values]
            target_only_gates[role] = [
                min(observation_components(row["observation"])[:-1]) if row["observation"] else None
                for row in role_values
            ]
            residue_results[role] = [
                row["observation"]["source_action_residue"]["result"] if row["observation"] else None
                for row in role_values
            ]
        pass_winners = []
        for index in range(2):
            av = gates["anchor"][index]
            bv = gates["frozen_base"][index]
            ac = coverage["anchor"][index]
            bc = coverage["frozen_base"][index]
            pass_winners.append(
                "abstain" if None in {av, bv, ac, bc}
                else _winner((int(av), float(ac)), (int(bv), float(bc)))
            )
        winner = pass_winners[0] if pass_winners[0] == pass_winners[1] else "abstain"
        manual = target_contract["manual_winner"]
        pairs.append({
            "pair_prefix": prefix,
            "manual_winner": manual,
            "qwen_winner": winner,
            "agrees_with_manual": winner == manual,
            "pass_winners": pass_winners,
            "source_aware_gate_scores": gates,
            "target_only_gate_scores": target_only_gates,
            "coverage_scores": coverage,
            "source_residue_results": residue_results,
            "source_action_caption": source_contracts[prefix]["source_action_caption"],
            "source_residue_contract": source_contracts[prefix]["source_residue"],
            "target_action": target_contract["target_action"],
            "human_note": target_contract["human_note"],
            "roles": role_rows,
        })
    counts = Counter(row["qwen_winner"] for row in pairs)
    agreement = sum(row["agrees_with_manual"] for row in pairs)
    control = {
        "target_forward_gate_mean": mean(row["source_aware_gate_scores"]["target_forward"][0] for row in pairs),
        "source_noop_gate_mean": mean(row["source_aware_gate_scores"]["source_noop"][0] for row in pairs),
        "target_reverse_gate_mean": mean(row["source_aware_gate_scores"]["target_reverse"][0] for row in pairs),
        "target_shuffle_gate_mean": mean(row["source_aware_gate_scores"]["target_shuffle"][0] for row in pairs),
        "target_forward_strict_pass_count": sum(row["source_aware_gate_scores"]["target_forward"][0] == 4 for row in pairs),
        "source_noop_strict_pass_count": sum(row["source_aware_gate_scores"]["source_noop"][0] == 4 for row in pairs),
        "reverse_below_forward_count": sum(row["source_aware_gate_scores"]["target_reverse"][0] < row["source_aware_gate_scores"]["target_forward"][0] for row in pairs),
        "shuffle_below_forward_count": sum(row["source_aware_gate_scores"]["target_shuffle"][0] < row["source_aware_gate_scores"]["target_forward"][0] for row in pairs),
        "source_noop_residue_yes_count": sum(row["source_residue_results"]["source_noop"][0] == "yes" for row in pairs),
        "target_forward_residue_no_count": sum(row["source_residue_results"]["target_forward"][0] == "no" for row in pairs),
    }
    candidate_residue = {
        role: dict(Counter(
            result
            for row in pairs
            for result in row["source_residue_results"][role]
        ))
        for role in ("anchor", "frozen_base")
    }
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "manifest_sha256": file_sha256(args.manifest),
        "target_contracts_sha256": file_sha256(args.target_contracts),
        "source_residue_contracts_sha256": file_sha256(args.source_contracts),
        "evaluation_role": "calibration_only_not_independent_test",
        "decision_rule": "lexicographic noncompensatory minimum gate over an independently frozen v2 target-only judgment and a v3 source-only negative gate, then coverage only on equal gates; same winner under candidate-slot-swap trace pass",
        "candidate_context": "instruction-free native-video trace and target-only judgment reused from v2; separate source-only trace judgment sees no target instruction, source video, target video, role, identity, or layout comparison",
        "pair_count": len(pairs),
        "winner_counts": dict(counts),
        "manual_agreement_count": agreement,
        "manual_agreement_rate": agreement / len(pairs),
        "control_calibration": control,
        "candidate_source_residue_pass_counts": candidate_residue,
        "pairs": pairs,
    }
    write_json(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rescore = sub.add_parser("qwen-rescore")
    rescore.add_argument("--manifest", required=True)
    rescore.add_argument("--target-contracts", required=True)
    rescore.add_argument("--source-contracts", required=True)
    rescore.add_argument("--records", action="append", required=True)
    rescore.add_argument("--model", required=True)
    rescore.add_argument("--shard-index", type=int, required=True)
    rescore.add_argument("--num-shards", type=int, required=True)
    rescore.add_argument("--pair-prefix", action="append")
    rescore.add_argument("--max-new-tokens", type=int, default=1792)
    rescore.add_argument("--resume", action="store_true")
    rescore.add_argument("--output", required=True)
    rescore.set_defaults(function=qwen_rescore)
    summary = sub.add_parser("qwen-summarize")
    summary.add_argument("--manifest", required=True)
    summary.add_argument("--target-contracts", required=True)
    summary.add_argument("--source-contracts", required=True)
    summary.add_argument("--records-dir", required=True)
    summary.add_argument("--output", required=True)
    summary.set_defaults(function=qwen_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
