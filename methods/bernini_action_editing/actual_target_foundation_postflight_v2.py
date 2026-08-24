#!/usr/bin/env python3
"""External CPU postflight that alone may seal a foundation canary candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, NoReturn, Optional, Sequence

import actual_target_foundation_canary_v1 as authority
import actual_target_foundation_runtime_v1 as runtime


class PostflightV2Error(RuntimeError): pass
def fail(message: str) -> NoReturn: raise PostflightV2Error(message)


def load_candidate(path: Path) -> Mapping[str,Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True)!=path or not path.is_file(): fail("candidate path is not canonical absolute plain file")
    try: value=json.loads(path.read_text(encoding="ascii"))
    except Exception as error: raise PostflightV2Error("candidate is not one parseable JSON object") from error
    if not isinstance(value,dict): fail("candidate is not one JSON object")
    digest=value.get("digest"); body=dict(value); body.pop("digest",None)
    if digest!=authority.object_sha256(body): fail("candidate top-level digest differs")
    return value


def seal_candidate(candidate_path: Path, seal_path: Path, expected_contract_digest: str, slurm_exit_code: int, *, asset_verifier: Any=None) -> Mapping[str,Any]:
    if slurm_exit_code!=0: fail("Slurm/torchrun/rank-wrapper exit was nonzero")
    candidate=load_candidate(candidate_path); contract=runtime.launch_contract()
    if expected_contract_digest!=contract["digest"]: fail("reviewed launch contract digest differs")
    if candidate.get("launch_contract_digest")!=contract["digest"]: fail("candidate launch contract digest differs")
    def nested_digest(row: Any, label: str) -> None:
        if not isinstance(row,Mapping): fail(f"{label} is not an object")
        body=dict(row); claim=body.pop("digest",None)
        if claim!=authority.object_sha256(body): fail(f"{label} digest differs")
    nested_digest(candidate.get("aggregate"),"aggregate")
    for row in candidate.get("cases",()): nested_digest(row,"case")
    nested_digest(candidate.get("raw_ownership"),"raw ownership")
    nested_digest(candidate.get("model_device_closure"),"model/device closure")
    nested_digest(candidate.get("decoded_media_closure"),"decoded media closure")
    expected_counts={"media_decode":8,"sam2":96,"dinov2":96,"cotracker":20,"vjepa2":20}
    if candidate.get("logical_forward_counts")!=expected_counts: fail("forward closure differs")
    if candidate.get("runtime_source_closure")!=contract["source_closure"]: fail("runtime source closure differs")
    live_assets=(asset_verifier or authority.verify_remote_assets)()
    if candidate.get("asset_closure",{}).get("verified") is not True or candidate.get("asset_closure",{}).get("digest")!=live_assets["digest"] or candidate.get("model_device_closure",{}).get("mode")=="fake_cpu_contract": fail("asset/model/device closure differs")
    media=candidate.get("decoded_media_closure",{})
    expected_media={(row["r1b_ordinal"],row["role"]):(row["compressed_sha256"],row["frame_count"],[720,1280,3],"uint8",row["decoded_rgb_sha256"]) for row in authority.load_decode_receipt()["rows"]}
    observed_media={(row.get("r1b_ordinal"),row.get("role")):(row.get("compressed_sha256"),row.get("frame_count"),row.get("shape_hwc"),row.get("dtype"),row.get("decoded_rgb_sha256")) for row in media.get("rows",()) if isinstance(row,Mapping)}
    if media.get("verified") is not True or observed_media!=expected_media or media.get("decode_receipt_file_sha256")!=authority.file_sha256(authority.DECODE_RECEIPT_PATH) or media.get("decode_receipt_self_sha256")!=authority.load_decode_receipt()["decode_receipt_self_sha256"]: fail("decoded media closure differs")
    if candidate.get("raw_ownership",{}).get("verified") is not True: fail("raw ownership was not completely zeroized")
    if candidate.get("aggregate",{}).get("diagnostic_canary_pass") is not True or candidate.get("aggregate",{}).get("passed_case_count")!=4: fail("development diagnostic did not pass exact 4/4")
    if any((candidate.get("training_performed") is not False,candidate.get("parameter_updates")!=0,candidate.get("generator_loaded") is not False,candidate.get("generator_forward_calls")!=0,candidate.get("representation_admission_hard_false") is not True)): fail("claim boundary differs")
    completion=candidate.get("completion_authority",{})
    if completion.get("candidate_file_presence_is_completion_authority") is not False or completion.get("external_completion_seal_written_by_probe") is not False: fail("candidate completion authority differs")
    value={"schema_version":"actual-target-foundation-external-completion-seal-v2","candidate_path":str(candidate_path),"candidate_file_sha256":authority.file_sha256(candidate_path),"candidate_digest":candidate["digest"],"reviewed_launch_contract_digest":contract["digest"],"runtime_source_closure_digest":contract["source_closure"]["digest"],"slurm_exit_code":0,"external_postflight_pass":True,"gpu_used_by_postflight":False}
    seal={**value,"digest":authority.object_sha256(value)}; runtime._create_only_json(seal_path,seal); return seal


def main(argv:Optional[Sequence[str]]=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--candidate",type=Path,required=True); parser.add_argument("--seal",type=Path,required=True); parser.add_argument("--expected-contract-digest",required=True); parser.add_argument("--slurm-exit-code",type=int,required=True)
    args=parser.parse_args(argv); seal_candidate(args.candidate,args.seal,args.expected_contract_digest,args.slurm_exit_code); return 0


if __name__=="__main__": raise SystemExit(main())
