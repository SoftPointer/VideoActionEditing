#!/usr/bin/env python3
"""Measure the exact Bernini RV2V positive-prompt token lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.method_root))
    import infer_native_identity_generation_canary as native
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer

    authority = json.loads(args.authority.read_text(encoding="ascii"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint),
        subfolder="tokenizer",
        **native.legacy.tokenizer_load_kwargs(),
    )
    rows = []
    for arm in ("P0", "P1", "P2"):
        action = authority["prompts"][arm]["full_prompt_utf8"]
        full = native.build_task_prompt("rv2v", action, prompt_cleaner=prompt_clean)
        encoded = tokenizer(full, **native.legacy.training_prompt_tokenizer_kwargs())
        length = int(encoded.input_ids.shape[1])
        if length >= 512:
            raise RuntimeError(f"{arm} would be sliced at the 512-token renderer boundary")
        rows.append(
            {
                "arm": arm,
                "action_prompt_utf8_sha256": hashlib.sha256(action.encode("utf-8")).hexdigest(),
                "renderer_full_prompt_utf8_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest(),
                "untruncated_token_count": length,
                "renderer_limit": 512,
                "sliced_or_truncated": False,
            }
        )
    result = {
        "schema": "mev840-native-rv2v-prompt-token-length-audit-v1",
        "complete": True,
        "tokenizer_fix_mistral_regex": tokenizer.init_kwargs.get("fix_mistral_regex") is True,
        "padding_side": tokenizer.padding_side,
        "rows": rows,
    }
    if result["tokenizer_fix_mistral_regex"] is not True or result["padding_side"] != "right":
        raise RuntimeError("tokenizer contract differs")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
