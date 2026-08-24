#!/usr/bin/env python3
"""Fresh-checkpoint Level-B Bernini action-edit product inference.

This module closes the renderer gap left intentionally open by
``infer_action_edit_product_abi_0817_v1``.  Its production entry point accepts
one authenticated source MP4, one edit instruction, and one explicit inference
seed.  It never accepts an edited target, action anchor, teacher tensor,
annotation, callback, or caller-supplied denoiser.

The implementation is pinned to the audited Bernini ``v2v_apg`` call graph:

* every UniPC step performs the native negative ``shared_step`` followed by the
  native action ``shared_step`` on the same complete pre-SP
  ``[clean-source prefix | evolving noisy target]`` embedding;
* the persisted 0817 conditioner is prepared from that live clean-source
  prefix and the positive branch's complete, actual-length contextual T5
  tokens;
* the same 30 block-indexed residual heads are invoked exactly once, and only
  during the action forward, at every one of the 40 native denoise steps; and
* the untouched official APG result reaches the original live UniPC
  ``scheduler.step`` exactly once per step.

The negative forward is observed by the installed hooks but returned by object
identity without calling the conditioner.  Source and sequence-padding rows in
the action forward are selected from the native block output byte-for-byte.

This remains ``PRE_D0_ENGINEERING_ONLY``.  A real MP4 proves that the loaded
checkpoint is consumable through the full product path; it is not training
parity, target quality, a D0 result, or promotion authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import stat
import subprocess
import sys
import threading
import types
from types import CodeType, ModuleType
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence, get_args, get_origin


METHOD = "bernini-action-edit-level-b-renderer-0817-v1"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
RECEIPT_SCHEMA = "bernini-action-edit-level-b-offline-product-receipt-v1"
RUNTIME_RELEASE_SCHEMA = "bernini-action-edit-level-b-runtime-release-v1"
BRIDGE_TRACE_SCHEMA = "bernini-action-edit-level-b-native-renderer-trace-v1"
VIDEO_VALIDATION_SCHEMA = "bernini-action-edit-level-b-full-mp4-validation-v1"
OUTPUT_TRANSACTION_SCHEMA = "bernini-action-edit-level-b-output-transaction-v1"
OUTPUT_COMMIT_MARKER_SCHEMA = (
    "bernini-action-edit-level-b-atomic-commit-marker-v1"
)
PRECOMMIT_GATE_SCHEMA = "bernini-action-edit-level-b-world8-precommit-gate-v1"
WORLD8_RANK0_PHASE_SCHEMA = "bernini-action-edit-level-b-world8-rank0-phase-v1"
CPU_STATIC_PREFLIGHT_SCHEMA = (
    "bernini-action-edit-level-b-cpu-static-runtime-preflight-v1"
)

FRAME_COUNT = 81
FPS = 25.0
PHASES = 21
LATENT_CHANNELS = 16
PATCH_VALUES = 64
TRANSFORMER_BLOCKS = 30
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FLOW_SHIFT = 5.0
NUM_INFERENCE_STEPS = 40
FFMPEG_WALL_TIMEOUT_SECONDS = 120
GUIDANCE_MODE = "v2v_apg"
PINNED_UNIPC_TIMESTEPS = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)
PINNED_UNIPC_SCHEDULE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)
FORBIDDEN_PUBLIC_ARGUMENT_FRAGMENTS = (
    "target",
    "anchor",
    "teacher",
    "callback",
    "denoiser",
    "track",
    "pose",
    "mask",
    "flow",
    "trajectory",
    "annotation",
)
OFFICIAL_RENDERER_SAMPLE_KEYWORDS = (
    "input_ids",
    "attention_mask",
    "uncond_input_ids",
    "uncond_attention_mask",
    "image_vae_latents",
    "multi_video_vae_latents",
    "multi_image_vae_latents",
    "width",
    "height",
    "device",
    "num_frames",
    "num_inference_steps",
    "guidance_mode",
    "omega_vid",
    "omega_img",
    "omega_txt",
    "omega_scale",
    "flow_shift",
    "seed",
    "eta",
    "norm_threshold",
    "momentum",
)
INTERNAL_DIFFUSION_SAMPLE_PARAMETERS = (
    "prompt_embeds",
    "prompt_embeds_t2",
    "uncond_prompt_embeds",
    "uncond_embeds_t2",
    "num_frames",
    "width",
    "height",
    "image_vae_latents",
    "multi_video_vae_latents",
    "multi_image_vae_latents",
    "num_inference_steps",
    "guidance_mode",
    "omega_vid",
    "omega_img",
    "omega_txt",
    "omega_scale",
    "flow_shift",
    "seed",
    "device",
    "eta",
    "norm_threshold",
    "momentum",
)
RENDERER_SAMPLE_PARAMETERS = (
    "input_ids",
    "attention_mask",
    "uncond_input_ids",
    "uncond_attention_mask",
    "image_vae_latents",
    "multi_video_vae_latents",
    "multi_image_vae_latents",
    "num_frames",
    "width",
    "height",
    "num_inference_steps",
    "guidance_mode",
    "omega_vid",
    "omega_img",
    "omega_txt",
    "omega_scale",
    "flow_shift",
    "seed",
    "device",
    "eta",
    "norm_threshold",
    "momentum",
)
RENDERER_SAMPLE_DEFAULTS = (
    None, None, None, 1, 832, 480, 50, "rv2v", 3.0, 3.0, 4.0,
    0.75, 5.0, 42, "cuda", 0.5, (50.0, 50.0), -0.5,
)
RENDERER_SAMPLE_ANNOTATIONS = {
    "num_frames": int,
    "width": int,
    "height": int,
    "num_inference_steps": int,
    "guidance_mode": str,
    "omega_vid": float,
    "omega_img": float,
    "omega_txt": float,
    "omega_scale": float,
    "flow_shift": float,
    "seed": int,
    "eta": float,
    "momentum": float,
}
INTERNAL_DIFFUSION_SAMPLE_DEFAULTS = (
    None, None, None, None, 1, 832, 480, None, None, None, 50, "rv2v",
    3.0, 3.0, 4.0, 0.75, 5.0, 42, "cuda", 1.0, (50.0, 50.0), 0.0,
)

# These are infrastructure authority, not product inputs.  The Level-B
# launcher may choose the release-manifest path, but its literal expected
# manifest SHA must be sealed outside this module.  Once that manifest is
# authenticated, the only source files that may be compiled are the exact
# closed members below.  Hashes for all transitive source members other than
# this module are compiled into the consumer as an independent authority.
LEVEL_B_RELEASE_MEMBER_PATHS = (
    "action_preservation_decoded_eval_model_authority_v2.py",
    "infer_action_edit_level_b_renderer_0817_v1.py",
    "infer_lora.py",
    "tools/build_renderer_dataset.py",
    "tools/materialize_vae.py",
)
LEVEL_B_STATIC_SOURCE_MEMBER_PINS = {
    "action_preservation_decoded_eval_model_authority_v2.py": {
        "sha256": "413508d42551fd1ab0ff83d9af7b29f144b1f6bcdccf29ed7c590417e3384ecb",
        "size": 114872,
    },
    "infer_lora.py": {
        "sha256": "c2e55a4ea41a21d0761e660ab630002b1bc569705e8c0bcafa1bc8c6c38ccc06",
        "size": 151393,
    },
    "tools/build_renderer_dataset.py": {
        "sha256": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        "size": 31012,
    },
    "tools/materialize_vae.py": {
        "sha256": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        "size": 32195,
    },
}
PINNED_BERNINI_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
)
PINNED_VEOMNI_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
)
PINNED_BASE_CHECKPOINT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
PINNED_BASE_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/"
    "runtime/methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
PINNED_FFMPEG_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
PINNED_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
PINNED_FFPROBE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
    "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
PINNED_FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
PINNED_PYTHON_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
PINNED_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
PINNED_STDLIB_SOCKET_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/socket.py"
)
PINNED_STDLIB_SOCKET_SHA256 = (
    "7d4d4c66e6f4bcc961ab462c4f08002ca97def8713a4be1c7373bdbd970a5274"
)
PINNED_SITE_PACKAGES = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
)
PINNED_DIFFUSERS_VERSION = "0.38.0"
PINNED_TRANSFORMERS_VERSION = "5.5.4"
PINNED_TORCH_VERSION = "2.7.1+rocm6.3"
# Exact module/source closure observed in two independent fresh, CPU-only,
# no-write imports of the pinned Bernini and VeOmni trees.  The tuple is
# captured by the public preflight wrapper at module initialization; no
# process-specific repr, object address, inode, or timestamp enters it.
PINNED_CPU_STATIC_SCOPED_MODULE_CLOSURE = (
    ("bernini", "bernini", "bernini/__init__.py", "2204c78bacfbbf0c8d3ee9a20679de588a9cd3c04995f00bab4af5182d688bf2", 1079),
    ("bernini.attention", "bernini", "bernini/attention.py", "e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055", 3418),
    ("bernini.models", "bernini", "bernini/models/__init__.py", "0e204886ec485ab3ae64a8eecdb168b978d0d182417b49cd4b9e2c3bfc831529", 908),
    ("bernini.models.bernini", "bernini", "bernini/models/bernini.py", "1c820200191cdb1dd36ca014f9a3a2ceae041a32bb19f425759308ab3d7129ef", 23567),
    ("bernini.models.diffloss_fm", "bernini", "bernini/models/diffloss_fm.py", "1374654c5bbf38215de099a6f4f2e29aee056632e2538b7aa64eaf2e5e697d2f", 16751),
    ("bernini.models.modeling_qwen2_5_vl", "bernini", "bernini/models/modeling_qwen2_5_vl.py", "59a8225ae29faa664eedac1bab3b5f2958163add4d51833085bad68313153510", 125736),
    ("bernini.models.renderer", "bernini", "bernini/models/renderer.py", "fec319f3ede3482b28873dc55622208f1242ecba0caedea8e710093748dc7159", 15980),
    ("bernini.models.scheduler", "bernini", "bernini/models/scheduler.py", "b6d729187fd784bf66831d5260a5c9482d89c452881d2f700c8887278f52ef97", 3529),
    ("bernini.models.transformer_wan", "bernini", "bernini/models/transformer_wan.py", "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223", 25309),
    ("bernini.models.wan_diffusion", "bernini", "bernini/models/wan_diffusion.py", "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512", 49226),
    ("bernini.parallel", "bernini", "bernini/parallel/__init__.py", "ef16834c0af0e4e2201db37fbbd3a13be6622ac8e09d076a6e6bf68543c9bc29", 1485),
    ("bernini.parallel.ops", "bernini", "bernini/parallel/ops.py", "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30", 5040),
    ("bernini.parallel.state", "bernini", "bernini/parallel/state.py", "32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa", 3913),
    ("veomni", "veomni", "veomni/__init__.py", "fd965658a108911ad73893510106be94e24f7ace9b1e101ecf540cd4cd8d42b5", 919),
    ("veomni._version", "veomni", "veomni/_version.py", "9e5943ac7d23c8284cbae62b2b4082e7989304a50d4d48de8ed730c72505d844", 23),
    ("veomni.distributed", "veomni", "veomni/distributed/__init__.py", "ced14be479d27e864cf3557c37d8abe00d2204ba6c030620bf48ea85e25a5921", 600),
    ("veomni.distributed.parallel_state", "veomni", "veomni/distributed/parallel_state.py", "43024419fe9cd33f5932615ce8261b746d5d68681cbd6edee73c266f2672ff2f", 21155),
    ("veomni.distributed.sequence_parallel", "veomni", "veomni/distributed/sequence_parallel/__init__.py", "1541a1816f64c33abc594e4147e1d75b384874c95785b7679b7ba8bc59c04ad8", 2810),
    ("veomni.distributed.sequence_parallel.async_ulysses", "veomni", "veomni/distributed/sequence_parallel/async_ulysses.py", "37398458f42c6926861cdf5cbf894ed3a33b6d0ef8cde76ecde5ee6748a5e036", 18592),
    ("veomni.distributed.sequence_parallel.comm", "veomni", "veomni/distributed/sequence_parallel/comm.py", "aaee38fb444c6c4c9ae0e3f1e09e01da10808e299299e2d8be30fcf393ef1fd4", 12099),
    ("veomni.distributed.sequence_parallel.data", "veomni", "veomni/distributed/sequence_parallel/data.py", "b635d5272a6dadfa9f6b2501345179db8b5f76a15e575c7a1bf46ffe01550a60", 4578),
    ("veomni.distributed.sequence_parallel.loss", "veomni", "veomni/distributed/sequence_parallel/loss.py", "80f377ab1f7cbe972186c4194c162cf94f8c329225a108f6930c46f0ba459c69", 2520),
    ("veomni.distributed.sequence_parallel.ulysses", "veomni", "veomni/distributed/sequence_parallel/ulysses.py", "98230d5219d5f327fafc436ec5d2df99c97a341676c2e964f7b8ec5470fd48e2", 14396),
    ("veomni.distributed.sequence_parallel.utils", "veomni", "veomni/distributed/sequence_parallel/utils.py", "17abb6e969097bc6bae35be6498d1b7edd9d4f8d78f836f35fff137606c84361", 5761),
    ("veomni.ops", "veomni", "veomni/ops/__init__.py", "f0ed09b6c93769c1be7a69aa63c2fd302ad4945296bdd0246e60290fb0107a2c", 4923),
    ("veomni.ops.config", "veomni", "veomni/ops/config/__init__.py", "7999b440dafbd3733972a39ffc2ef4e5b26a68f97a32107954312f2786e1f813", 1171),
    ("veomni.ops.config.registry", "veomni", "veomni/ops/config/registry.py", "4fd297a8b32d69329455164a8c879dfb88a31ca0c49b76ddb91c2f32cb7b16b1", 13467),
    ("veomni.ops.config.singleton", "veomni", "veomni/ops/config/singleton.py", "ff70e158f392cd5ed74e4d2cebf98ee1ba8dbf5cf977ebf605b14162f9646b68", 1647),
    ("veomni.ops.dispatch", "veomni", "veomni/ops/dispatch.py", "bed2f448781c17cab280645251139b73fbe3d9f9d6e2c3b38561a99ed2021633", 4234),
    ("veomni.ops.kernel_registry", "veomni", "veomni/ops/kernel_registry.py", "46ff907f6b162a6f15044d7a9c5e7edc1b51dbbea6cb40657f01c97955a42e0a", 7146),
    ("veomni.ops.kernels", "veomni", "veomni/ops/kernels/__init__.py", "885ab227f54a66c5080e2b05272d680310ba34d5723cffb6311b7a1daa270515", 1037),
    ("veomni.ops.kernels.attention", "veomni", "veomni/ops/kernels/attention/__init__.py", "81b7c8936dd4f627d1e71e01dc2500f4b2602ec830b21a73a597e7acc7b07a22", 16132),
    ("veomni.ops.kernels.cross_entropy", "veomni", "veomni/ops/kernels/cross_entropy/__init__.py", "39105e746892e9783475f0dd25e1ed1f7615bc0c958be6b67290f5e7f7dfb706", 27786),
    ("veomni.ops.kernels.cross_entropy.chunk_logprobs", "veomni", "veomni/ops/kernels/cross_entropy/chunk_logprobs.py", "b4c1caade0c444eadb920c6f6ae17422e59ccf7403b5499419a530aaa9dfdab3", 16413),
    ("veomni.ops.kernels.cross_entropy.chunk_loss", "veomni", "veomni/ops/kernels/cross_entropy/chunk_loss.py", "3cd9c740e32d5dc1fdb8497e02a87db081fc68f4844314af1d89a05dbe43e4be", 5815),
    ("veomni.ops.kernels.cross_entropy.chunk_topk_distill", "veomni", "veomni/ops/kernels/cross_entropy/chunk_topk_distill.py", "0a3d84654fb09f02658fe54bbf2562968a80be203b5b6d36353e29947119a13b", 21016),
    ("veomni.ops.kernels.cross_entropy.eager", "veomni", "veomni/ops/kernels/cross_entropy/eager.py", "061d6ca924b7abe283c3062252f478d39819922e85479f2a964ff6bd7fea1d20", 1332),
    ("veomni.ops.kernels.gated_delta_rule", "veomni", "veomni/ops/kernels/gated_delta_rule/__init__.py", "f754ae5cef2aaeafe2f571a3903243aa69a2ed107139fcece041aabef98daca9", 6804),
    ("veomni.ops.kernels.load_balancing_loss", "veomni", "veomni/ops/kernels/load_balancing_loss/__init__.py", "a1af8a2d1ac9afadf1a1ff0b929d2d92275384bd3031c792e4431e3e0fff5981", 3933),
    ("veomni.ops.kernels.load_balancing_loss.eager", "veomni", "veomni/ops/kernels/load_balancing_loss/eager.py", "3a9f8b376c5349c1b26ad221b129c7be20d578d6b0a2aab6f04fbfb7fd43f11f", 5003),
    ("veomni.ops.kernels.moe", "veomni", "veomni/ops/kernels/moe/__init__.py", "6523dc1cdaac884f801ef3a99db8f7d6d1e63c636c70014ce6ab523d5182a542", 7139),
    ("veomni.ops.kernels.rms_norm", "veomni", "veomni/ops/kernels/rms_norm/__init__.py", "64de5a27cff2f81118403badfec0b699b576b57b294eac0c8a141810c6bfb3d1", 2690),
    ("veomni.ops.kernels.rotary", "veomni", "veomni/ops/kernels/rotary/__init__.py", "e2bb7e36d8ddd19fc817d3bda29253ef8aae6d04a5ee7e3cd9072b5d0dea4611", 3204),
    ("veomni.ops.kernels.swiglu", "veomni", "veomni/ops/kernels/swiglu/__init__.py", "fab95f7671fc8713269d0ed6b4fe817bdc61112e6354892c6bfe33977f5a15b3", 1224),
    ("veomni.ops.liger", "veomni", "veomni/ops/liger/__init__.py", "fbb5a798b3821e759e4c44bebbe94020e9125f7c6baa854313c6f62fdc4b00d3", 4854),
    ("veomni.utils", "veomni", "veomni/utils/__init__.py", "ced14be479d27e864cf3557c37d8abe00d2204ba6c030620bf48ea85e25a5921", 600),
    ("veomni.utils.constants", "veomni", "veomni/utils/constants.py", "cae11eeed9b4ba380f5496904876c96d24620d9b3ecd5c76adf72dc87cb42eb1", 1158),
    ("veomni.utils.device", "veomni", "veomni/utils/device.py", "d1c02d0aa6e084b980880d59229614944b353bcb2f3e1f43934e8d19e2a35ccc", 4483),
    ("veomni.utils.env", "veomni", "veomni/utils/env.py", "acb3fb3edce021ced8c775f2bb54b97c81c972df8a3c44c31ec667528d3facc1", 1402),
    ("veomni.utils.import_utils", "veomni", "veomni/utils/import_utils.py", "40b1cb4f7a39661f71cdc0d256d27b84f1e95382683e8d779bc1aeb009377148", 3756),
    ("veomni.utils.logging", "veomni", "veomni/utils/logging.py", "91a613a68a5a32b239900bd72cfdf5d172996fec37bf67a69b0cefa699c9fc5a", 5246),
    ("veomni.utils.model_outputs", "veomni", "veomni/utils/model_outputs.py", "e50dd19a93ff44cf84dacea8b4056dbddf454c8e770b68163c7e60b62fbeb94b", 8875),
)
PINNED_BERNINI_RUNTIME_FILE_HASHES = {
    "bernini/cli.py": "26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf",
    "bernini/io_utils.py": "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a",
    "bernini/pipeline.py": "c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40",
    **{
        relative: sha256
        for _module, prefix, relative, sha256, _size
        in PINNED_CPU_STATIC_SCOPED_MODULE_CLOSURE
        if prefix == "bernini"
    },
}
PINNED_VEOMNI_RUNTIME_FILE_HASHES = {
    relative: sha256
    for _module, prefix, relative, sha256, _size
    in PINNED_CPU_STATIC_SCOPED_MODULE_CLOSURE
    if prefix == "veomni"
}
# Source hashes are the exact files shipped by the pinned vace environment's
# diffusers 0.38.0 and transformers 5.5.4 installations, including the live
# huggingface_hub decorator code executed by their inherited loaders.  These
# are executable authority: matching package version strings alone are not
# sufficient.
PINNED_SITE_PACKAGE_SOURCE_HASHES = {
    "botocore/vendored/six.py": "4ce39f422ee71467ccac8bed76beb05f8c321c7f0ceda9279ae2dfa3670106b3",
    "numpy/core/__init__.py": "08db0ef806f8cb03365b3dc06ea58e1f78a0d6ae419e8f4fb1432b0aff87352e",
    "diffusers/configuration_utils.py": "4a7af9be48913edfa77f3d32c375c997fa15b53c3f04efd5c371b36c5c6c1960",
    "diffusers/models/modeling_utils.py": "a2a3a115d2b61ea396e196bb3b0fb545230b2cc42d6f64cf76d0a64a4dde0fbe",
    "diffusers/models/modeling_outputs.py": "5c7dec24edf83115ba52e5aaa8aa34e6656ac464811516b3a5aa7ff982f03b62",
    "diffusers/models/autoencoders/autoencoder_kl_wan.py": "836820d112a9310ece586ba9fa51d51daef04cbe866e59a673843476a4d7e087",
    "diffusers/models/autoencoders/vae.py": "90f6db6ed05b3a6bd61ab1abefc0414ebacf730f89135e6c4b2155b52c001d72",
    "diffusers/pipelines/wan/pipeline_wan.py": "ba1647444b5bcdf0c06f0991daa1fd1bc4f188ad757d226246120709fd66dfd3",
    "diffusers/schedulers/scheduling_unipc_multistep.py": "5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872",
    "diffusers/utils/accelerate_utils.py": "664a2938adbdffa42badd9083e27479ced3bf01f01b73cc54adb37ba5d9c3fc4",
    "diffusers/utils/torch_utils.py": "8af046c42c841cd1d5cf5a2d7879dec4d50d0de5e7d026d7370ab2212624f36a",
    "huggingface_hub/utils/_validators.py": "b6e0b5538cb107ee172886a364c9ca65448abd9eaa3e9c9d63c2c8b5f6b4838d",
    "six.py": "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df",
    "torch/autograd/grad_mode.py": "a67ddb0da569646f5d3806e248c2093b6e2f75f0a6b4959ab966e119c1b28d6d",
    "torch/distributed/nn/jit/instantiator.py": "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615",
    "torch/distributed/nn/api/remote_module.py": "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1",
    "torch/utils/_contextlib.py": "cf7aa5b08f44974ba8c1d08cd71ef70ffd13d1c48a4931576eb235306bfa46b5",
    "transformers/models/auto/tokenization_auto.py": "bbfe4c497c7fa006fb50321adff83e07770a661b676401dc009da6b1d6757539",
    "transformers/models/t5/tokenization_t5.py": "6fa6696aa2bf6bf40bcd7c7aea81b5b581e365ada45c6d7768d53609174496d5",
    "transformers/tokenization_utils_base.py": "6c60e0607c9b298a891293d02af4527ea2444341a644bd218f5d4fc6ccc615b8",
    "urllib3/util/connection.py": "2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CUDA_DEVICE = re.compile(r"cuda(?::([0-9]+))?\Z")
_NO_BOUND_OWNER = object()

# A separately sealed launcher must pre-seed this one module-global before it
# compile/execs the authenticated Level-B source bytes.  Pop it immediately so
# the public authenticator closes over an import-time literal and exposes only
# ``authenticate_level_b_runtime_release(manifest_path)``.  A runtime product
# caller cannot submit a path and a freshly computed expected SHA together.
_SEALED_LAUNCHER_MANIFEST_SHA_AT_IMPORT = globals().pop(
    "_LEVEL_B_SEALED_LAUNCHER_EXPECTED_MANIFEST_SHA256", None
)


class LevelBRendererError(RuntimeError):
    """Raised before an unauthenticated or ambiguous renderer action runs."""


def fail(message: str) -> NoReturn:
    raise LevelBRendererError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise LevelBRendererError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase full SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    try:
        resolved = requested.resolve(strict=True)
        info = requested.lstat()
    except OSError as error:
        raise LevelBRendererError(f"{label} is unavailable") from error
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or resolved != requested
        or not stat.S_ISREG(info.st_mode)
    ):
        fail(f"{label} must be one canonical plain absolute file")
    return requested


def _plain_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    try:
        resolved = requested.resolve(strict=True)
        info = requested.lstat()
    except OSError as error:
        raise LevelBRendererError(f"{label} is unavailable") from error
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or resolved != requested
        or not stat.S_ISDIR(info.st_mode)
    ):
        fail(f"{label} must be one canonical plain absolute directory")
    return requested


def stable_file_sha256(path: Path, *, label: str) -> tuple[str, Mapping[str, int]]:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        named = path.lstat()
    except OSError as error:
        raise LevelBRendererError(f"cannot hash {label}") from error
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"{label} changed during hashing")
    return digest.hexdigest(), {
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mode": int(stat.S_IMODE(before.st_mode)),
        "size": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
    }


def tensor_sha256(value: Any, *, torch_module: Any) -> str:
    torch = torch_module
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        fail("tensor digest requires one materialized tensor")
    # Preserve the historical digest byte stream while bounding host memory.
    # Calling ``.cpu()`` on the complete contiguous tensor used to retain a
    # second, potentially multi-hundred-megabyte host allocation until this
    # function returned.  Keep the canonical contiguous view on its original
    # device and transfer at most one raw-byte chunk at a time instead.
    item = value.detach().contiguous()
    metadata = canonical_json_bytes(
        {"shape": [int(x) for x in item.shape], "dtype": str(item.dtype)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = item.reshape(-1).view(torch.uint8)
    # Avoid a NumPy ABI dependency in the checkpoint consumer.  Bounded chunks
    # also avoid the multi-gigabyte Python-list peak of one whole large prefix.
    # ``tolist`` materializes Python integers; 64 KiB keeps that transient
    # object graph near ~2 MiB even under the tight formal cgroup margin.
    raw_chunk_bytes = 64 * 1024
    for start in range(0, int(raw.numel()), raw_chunk_bytes):
        chunk = raw[start : start + raw_chunk_bytes].cpu()
        digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


def _bits_equal(left: Any, right: Any, *, torch_module: Any) -> bool:
    torch = torch_module
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _output_tensor(output: Any, *, torch_module: Any) -> tuple[Any, Any]:
    torch = torch_module
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value, *output[1:])
    fail("Bernini block output must be Tensor or tensor-first tuple")


def _bind_call(function: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if not callable(function):
        fail("pinned Bernini callable is not callable")
    try:
        bound = inspect.signature(function).bind(*args, **dict(kwargs))
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise LevelBRendererError("call differs from the pinned Bernini signature") from error
    return dict(bound.arguments)


def _pop_exact_phase_owned_value(
    owner: Any, key: str, expected: Any, *, label: str
) -> Any:
    """Remove one phase-local strong reference only when identity is exact."""

    if type(owner) is not dict or type(key) is not str or not key or key not in owner:
        fail(f"{label} phase-owned value is absent")
    value = owner.pop(key)
    if value is not expected:
        fail(f"{label} phase-owned object identity differs")
    return value


def _same_exact_runtime_literal(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(observed) == len(expected) and all(
            _same_exact_runtime_literal(left, right)
            for left, right in zip(observed, expected)
        )
    try:
        return bool(observed == expected)
    except Exception:
        return False


_STATIC_PREFLIGHT_MUTATION_AUDIT_EVENTS = frozenset(
    {
        "os.chmod",
        "os.chown",
        "os.link",
        "os.mkdir",
        "os.putenv",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.unlink",
        "os.unsetenv",
        "os.utime",
        "shutil.copyfile",
        "shutil.copymode",
        "shutil.copystat",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)
_STATIC_PREFLIGHT_PROCESS_AUDIT_EVENTS = frozenset(
    {"os.fork", "os.forkpty", "os.posix_spawn", "os.system", "pty.spawn", "subprocess.Popen"}
)
_STATIC_PREFLIGHT_NETWORK_AUDIT_EVENTS = frozenset(
    {"socket.__new__", "socket.bind", "socket.connect", "socket.getaddrinfo"}
)
_STATIC_PREFLIGHT_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
)


class _CPUStaticPreflightAuditGuard:
    __slots__ = (
        "base_checkpoint",
        "devnull_path",
        "devnull_rdev",
        "devnull_mode",
        "devnull_open_counts",
        "forbidden_violation_count",
        "installed",
        "sealed",
        "network_probe_configured",
        "blocked_network_probe_count",
        "socket_source_path",
        "socket_source_sha256",
        "urllib3_connection_path",
        "urllib3_connection_sha256",
    )

    def __init__(self, *, base_checkpoint: Path) -> None:
        self.base_checkpoint = str(base_checkpoint)
        devnull = Path(os.devnull).resolve(strict=True)
        info = devnull.stat()
        if str(devnull) != "/dev/null" or not stat.S_ISCHR(info.st_mode):
            fail("CPU static preflight /dev/null authority differs")
        self.devnull_path = str(devnull)
        self.devnull_rdev = int(info.st_rdev)
        self.devnull_mode = int(stat.S_IMODE(info.st_mode))
        devnull_identity = (
            self.devnull_rdev,
            int(os.major(self.devnull_rdev)),
            int(os.minor(self.devnull_rdev)),
        )
        expected_devnull_identity = {
            # Formal vace/Linux authority: st_rdev=259, major=1, minor=3.
            "linux": (259, 1, 3),
            # Exact Darwin identity keeps the real child-process guard testable.
            "darwin": (50331650, 3, 2),
        }.get(sys.platform)
        if (
            self.devnull_mode != 0o666
            or expected_devnull_identity is None
            or devnull_identity != expected_devnull_identity
        ):
            fail("CPU static preflight /dev/null device identity differs")
        self.devnull_open_counts = {
            "python-open-r+": 0,
            "python-open-w": 0,
            "os-open-O_RDWR": 0,
            "os-open-O_WRONLY|O_CREAT|O_TRUNC": 0,
        }
        self.forbidden_violation_count = 0
        self.installed = False
        self.sealed = False
        self.network_probe_configured = False
        self.blocked_network_probe_count = 0
        self.socket_source_path = ""
        self.socket_source_sha256 = ""
        self.urllib3_connection_path = ""
        self.urllib3_connection_sha256 = ""

    def configure_blocked_network_probe(
        self,
        *,
        socket_source_path: Path,
        socket_source_sha256: str,
        urllib3_connection_path: Path,
        urllib3_connection_sha256: str,
    ) -> None:
        if (
            not self.installed
            or self.sealed
            or self.network_probe_configured
            or self.blocked_network_probe_count != 0
            or self.forbidden_violation_count != 0
        ):
            fail("CPU static preflight network-probe guard state differs")
        self.socket_source_path = str(
            _plain_file(
                socket_source_path,
                label="CPU static preflight stdlib socket source",
            )
        )
        self.socket_source_sha256 = _require_sha(
            socket_source_sha256,
            label="CPU static preflight stdlib socket source SHA",
        )
        self.urllib3_connection_path = str(
            _plain_file(
                urllib3_connection_path,
                label="CPU static preflight urllib3 connection source",
            )
        )
        self.urllib3_connection_sha256 = _require_sha(
            urllib3_connection_sha256,
            label="CPU static preflight urllib3 connection source SHA",
        )
        self.network_probe_configured = True

    def _reject(self, message: str) -> NoReturn:
        self.forbidden_violation_count += 1
        fail(message)

    def _is_expected_blocked_socket_probe(
        self, arguments: tuple[Any, ...]
    ) -> bool:
        if (
            not self.network_probe_configured
            or self.blocked_network_probe_count != 0
            or len(arguments) != 4
            or arguments[1:] != (10, 1, 0)
        ):
            return False
        socket_module = sys.modules.get("socket")
        urllib3_module = sys.modules.get("urllib3.util.connection")
        socket_value = arguments[0]
        socket_class = getattr(socket_module, "socket", None)
        socket_init = getattr(socket_class, "__init__", None)
        has_ipv6 = getattr(urllib3_module, "_has_ipv6", None)
        try:
            socket_frame = sys._getframe(2)
            urllib3_frame = socket_frame.f_back
            module_frame = urllib3_frame.f_back if urllib3_frame is not None else None
            socket_state = (
                socket_value.fileno(),
                socket_value.family,
                socket_value.type,
                socket_value.proto,
            )
        except Exception:
            return False
        return bool(
            socket_module is not None
            and urllib3_module is not None
            and type(socket_value) is socket_class
            and socket_state == (-1, 0, 0, 0)
            and inspect.isfunction(socket_init)
            and socket_frame.f_code is socket_init.__code__
            and socket_frame.f_lineno == 233
            and socket_init.__module__ == "socket"
            and socket_init.__qualname__ == "socket.__init__"
            and socket_init.__code__.co_firstlineno == 221
            and tuple(socket_init.__code__.co_freevars) == ()
            and socket_frame.f_globals is vars(socket_module)
            and socket_frame.f_code.co_filename == self.socket_source_path
            and inspect.isfunction(has_ipv6)
            and urllib3_frame is not None
            and urllib3_frame.f_code is has_ipv6.__code__
            and urllib3_frame.f_lineno == 126
            and has_ipv6.__module__ == "urllib3.util.connection"
            and has_ipv6.__qualname__ == "_has_ipv6"
            and has_ipv6.__code__.co_firstlineno == 114
            and tuple(has_ipv6.__code__.co_freevars) == ()
            and urllib3_frame.f_globals is vars(urllib3_module)
            and urllib3_frame.f_locals.get("host") == "::1"
            and urllib3_frame.f_code.co_filename == self.urllib3_connection_path
            and module_frame is not None
            and module_frame.f_globals is vars(urllib3_module)
            and module_frame.f_code.co_name == "<module>"
            and getattr(module_frame.f_code, "co_qualname", "<module>")
            == "<module>"
            and module_frame.f_code.co_firstlineno == 1
            and tuple(module_frame.f_code.co_freevars) == ()
            and module_frame.f_lineno == 137
            and module_frame.f_code.co_filename == self.urllib3_connection_path
        )

    def audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event == "open":
            path = arguments[0] if arguments else None
            mode = arguments[1] if len(arguments) > 1 else None
            flags = arguments[2] if len(arguments) > 2 else 0
            try:
                raw_path = os.fsdecode(os.fspath(path))
            except (TypeError, ValueError):
                raw_path = ""
            text_mode = mode if isinstance(mode, str) else ""
            numeric_flags = flags if type(flags) is int else 0
            write_mask = (
                os.O_WRONLY
                | os.O_RDWR
                | os.O_CREAT
                | os.O_TRUNC
                | os.O_APPEND
                | os.O_EXCL
            )
            write_capable = any(value in text_mode for value in "wax+") or bool(
                numeric_flags & write_mask
            )
            absolute = os.path.abspath(raw_path) if raw_path else ""
            in_checkpoint = absolute == self.base_checkpoint or absolute.startswith(
                self.base_checkpoint + os.sep
            )
            suffix = Path(raw_path).suffix.lower() if raw_path else ""
            if in_checkpoint or suffix in _STATIC_PREFLIGHT_WEIGHT_SUFFIXES:
                self._reject("CPU static preflight attempted to read model weights")
            if write_capable:
                if self.sealed:
                    self._reject(
                        "CPU static preflight attempted a write-capable open after audit seal"
                    )
                close_on_exec = getattr(os, "O_CLOEXEC", 0)
                allowed_devnull_open = {
                    ("r+", os.O_RDWR | close_on_exec): "python-open-r+",
                    (
                        "w",
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | close_on_exec,
                    ): "python-open-w",
                    (None, os.O_RDWR | close_on_exec): "os-open-O_RDWR",
                    (
                        None,
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | close_on_exec,
                    ): "os-open-O_WRONLY|O_CREAT|O_TRUNC",
                }
                event_mode = mode if isinstance(mode, str) or mode is None else object()
                count_key = allowed_devnull_open.get((event_mode, numeric_flags))
                if (
                    absolute != self.devnull_path
                    or count_key is None
                ):
                    self._reject(
                        "CPU static preflight attempted a persistent filesystem write"
                    )
                self.devnull_open_counts[count_key] += 1
            return
        if event in _STATIC_PREFLIGHT_MUTATION_AUDIT_EVENTS:
            self._reject("CPU static preflight attempted filesystem mutation")
        if (
            event in _STATIC_PREFLIGHT_PROCESS_AUDIT_EVENTS
            or event.startswith("os.spawn")
        ):
            self._reject("CPU static preflight attempted process creation")
        if event == "socket.__new__" and self._is_expected_blocked_socket_probe(
            arguments
        ):
            self.blocked_network_probe_count += 1
            fail("CPU static preflight blocked the pinned IPv6 capability probe")
        if event in _STATIC_PREFLIGHT_NETWORK_AUDIT_EVENTS:
            self._reject("CPU static preflight attempted network access")

    def receipt(
        self, *, require_complete_static_import: bool = False
    ) -> Mapping[str, Any]:
        expected_devnull_open_counts = {
            "python-open-r+": 1,
            "python-open-w": 1,
            "os-open-O_RDWR": 1,
            "os-open-O_WRONLY|O_CREAT|O_TRUNC": 1,
        }
        if (
            not self.installed
            or self.sealed
            or self.forbidden_violation_count != 0
            or self.devnull_open_counts != expected_devnull_open_counts
            or (
                require_complete_static_import
                and (
                    not self.network_probe_configured
                    or self.blocked_network_probe_count != 1
                )
            )
        ):
            fail("CPU static preflight audit guard state differs")
        self.sealed = True
        rules = {
            "filesystem_mutation_events": sorted(
                _STATIC_PREFLIGHT_MUTATION_AUDIT_EVENTS
            ),
            "process_events": sorted(_STATIC_PREFLIGHT_PROCESS_AUDIT_EVENTS),
            "network_events": sorted(_STATIC_PREFLIGHT_NETWORK_AUDIT_EVENTS),
            "blocked_network_probe": {
                "socket_source_path": self.socket_source_path,
                "socket_source_sha256": self.socket_source_sha256,
                "socket_init_line": 233,
                "urllib3_connection_path": self.urllib3_connection_path,
                "urllib3_connection_sha256": self.urllib3_connection_sha256,
                "urllib3_has_ipv6_line": 126,
                "arguments_tail": [10, 1, 0],
            },
            "weight_suffixes": sorted(_STATIC_PREFLIGHT_WEIGHT_SUFFIXES),
            "weight_root": self.base_checkpoint,
            "only_write_capable_open_exceptions": [
                "/dev/null r+/O_RDWR",
                "/dev/null w/O_WRONLY|O_CREAT|O_TRUNC",
                "/dev/null mode=None/O_RDWR",
                "/dev/null mode=None/O_WRONLY|O_CREAT|O_TRUNC",
            ],
        }
        return {
            "installed_before_vendor_imports": True,
            "irreversible_for_short_lived_preflight_process": True,
            "rules_digest": object_sha256(rules),
            "forbidden_violation_count": self.forbidden_violation_count,
            "devnull_path": self.devnull_path,
            "devnull_rdev": self.devnull_rdev,
            "devnull_mode": self.devnull_mode,
            "devnull_major": int(os.major(self.devnull_rdev)),
            "devnull_minor": int(os.minor(self.devnull_rdev)),
            "devnull_open_counts": dict(self.devnull_open_counts),
            "write_capable_open_exceptions_sealed": self.sealed,
            "persistent_filesystem_writes": False,
            "model_weight_reads": False,
            "subprocesses_spawned": False,
            "blocked_network_probe_count": self.blocked_network_probe_count,
            "socket_objects_created": False,
            "network_accessed": False,
        }


def _install_cpu_static_preflight_audit_guard(
    *, base_checkpoint: str | Path
) -> _CPUStaticPreflightAuditGuard:
    guard = _CPUStaticPreflightAuditGuard(
        base_checkpoint=_plain_directory(
            base_checkpoint, label="CPU static preflight base checkpoint"
        )
    )
    try:
        sys.addaudithook(guard.audit)
    except Exception as error:
        raise LevelBRendererError(
            "cannot install CPU static preflight process audit guard"
        ) from error
    guard.installed = True
    return guard


def _same_object(left: Any, right: Any, *, label: str) -> None:
    if left is not right:
        fail(f"negative/action {label} must be the exact same object")


def _equal_metadata(left: Any, right: Any, *, label: str) -> None:
    try:
        equal = left == right
        if hasattr(equal, "all"):
            equal = equal.all()
        if hasattr(equal, "item"):
            equal = equal.item()
        equal = bool(equal)
    except Exception as error:
        raise LevelBRendererError(f"cannot compare {label}") from error
    if not equal:
        fail(f"negative/action {label} differ")


def _scalar_int(value: Any, *, label: str) -> int:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            fail(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        numeric = float(candidate)
    except LevelBRendererError:
        raise
    except Exception as error:
        raise LevelBRendererError(f"{label} must be numeric") from error
    result = int(numeric)
    if not math.isfinite(numeric) or numeric != float(result):
        fail(f"{label} must be one finite integer")
    return result


def _single_length(value: Any, *, maximum: int, label: str) -> int:
    candidate = value
    if isinstance(candidate, (list, tuple)):
        if len(candidate) != 1:
            fail(f"{label} must contain one row")
        candidate = candidate[0]
    elif hasattr(candidate, "reshape"):
        flattened = candidate.reshape(-1)
        if int(flattened.numel()) != 1:
            fail(f"{label} must contain one row")
        candidate = flattened[0]
    result = _scalar_int(candidate, label=label)
    if not 0 < result <= maximum:
        fail(f"{label} is outside the contextual embedding length")
    return result


def _canonical_signature_receipt(function: Any, *, label: str) -> Mapping[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        raise LevelBRendererError(f"cannot inspect {label} signature") from error
    parameters = []
    for parameter in signature.parameters.values():
        kind = parameter.kind
        parameters.append(
            {
                "name": parameter.name,
                "kind": kind.name,
                "required": (
                    parameter.default is inspect.Parameter.empty
                    and kind
                    not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                ),
                "has_default": parameter.default is not inspect.Parameter.empty,
                "has_annotation": (
                    parameter.annotation is not inspect.Parameter.empty
                ),
            }
        )
    return {
        "parameters": parameters,
        "has_var_positional": any(
            row["kind"] == inspect.Parameter.VAR_POSITIONAL.name
            for row in parameters
        ),
        "has_var_keyword": any(
            row["kind"] == inspect.Parameter.VAR_KEYWORD.name
            for row in parameters
        ),
        "has_return_annotation": (
            signature.return_annotation is not inspect.Signature.empty
        ),
    }


def _canonical_callable_receipt(function: Any, *, label: str) -> Mapping[str, Any]:
    if not callable(function):
        fail(f"{label} is not callable")
    module = inspect.getmodule(function)
    source_path_raw = inspect.getsourcefile(function)
    if module is None or source_path_raw is None:
        fail(f"{label} has no Python source provenance")
    source_path = _plain_file(Path(source_path_raw).resolve(strict=True), label=f"{label} source")
    source_file_sha, _ = stable_file_sha256(source_path, label=f"{label} source")
    try:
        callable_source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError) as error:
        raise LevelBRendererError(f"cannot inspect {label} source") from error
    return {
        "module": str(getattr(module, "__name__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "source_file": str(source_path),
        "source_file_sha256": source_file_sha,
        "callable_source_sha256": hashlib.sha256(callable_source).hexdigest(),
        "runtime_signature": _canonical_signature_receipt(function, label=label),
    }


def _audit_callable_against_authenticated_source(
    function: Any,
    *,
    label: str,
    expected_module: str,
    expected_qualname: str,
    expected_path: str | Path,
    expected_sha256: str,
    expected_bound_owner: Any = _NO_BOUND_OWNER,
) -> Mapping[str, Any]:
    """Bind one live Python callable to an exact authenticated source file."""

    receipt = dict(_canonical_callable_receipt(function, label=label))
    path = _plain_file(expected_path, label=f"pinned {label} source")
    observed_sha, identity = stable_file_sha256(path, label=f"pinned {label} source")
    module = inspect.getmodule(function)
    code_function = getattr(function, "__func__", function)
    code = getattr(code_function, "__code__", None)
    try:
        code_path = (
            Path(code.co_filename).resolve(strict=True) if code is not None else None
        )
        module_path = (
            Path(getattr(module, "__file__", "")).resolve(strict=True)
            if module is not None
            else None
        )
    except OSError as error:
        raise LevelBRendererError(f"cannot resolve {label} code source") from error
    declared: Any = module
    if module is not None:
        try:
            for component in expected_qualname.split("."):
                declared = getattr(declared, component)
        except (AttributeError, TypeError):
            declared = None
    declared_function = getattr(declared, "__func__", declared)
    if (
        observed_sha != _require_sha(expected_sha256, label=f"pinned {label} SHA")
        or receipt.get("module") != expected_module
        or receipt.get("qualname") != expected_qualname
        or receipt.get("source_file") != str(path)
        or receipt.get("source_file_sha256") != observed_sha
        or module is None
        or sys.modules.get(expected_module) is not module
        or module_path != path
        or code_path != path
        or declared_function is not code_function
        or (
            expected_bound_owner is not _NO_BOUND_OWNER
            and getattr(function, "__self__", None) is not expected_bound_owner
        )
    ):
        fail(f"live {label} callable is not owned by its pinned source bytes")
    receipt["source_file_identity"] = identity
    receipt["module_object_identity_verified"] = True
    receipt["module_declared_callable_identity_verified"] = True
    receipt["code_filename_identity_verified"] = True
    if expected_bound_owner is not _NO_BOUND_OWNER:
        receipt["exact_bound_owner_identity_verified"] = True
        receipt["bound_owner_class"] = (
            f"{type(expected_bound_owner).__module__}."
            f"{type(expected_bound_owner).__qualname__}"
            if not inspect.isclass(expected_bound_owner)
            else (
                f"{expected_bound_owner.__module__}."
                f"{expected_bound_owner.__qualname__}"
            )
        )
    return receipt


def _audit_inherited_wrapped_classmethod_against_authenticated_sources(
    function: Any,
    *,
    label: str,
    expected_bound_owner: Any,
    method_name: str,
    expected_definition_module: str,
    expected_definition_owner_qualname: str,
    expected_definition_qualname: str,
    expected_definition_path: str | Path,
    expected_definition_sha256: str,
    expected_wrapper_module: str,
    expected_wrapper_factory_qualname: str,
    expected_wrapper_path: str | Path,
    expected_wrapper_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate both executable layers of an inherited decorated classmethod.

    ``validate_hf_hub_args`` uses ``functools.wraps``.  Consequently, the live
    bound method advertises the Diffusers definition's module and qualname even
    though its executing ``__code__`` belongs to ``huggingface_hub``.  Treating
    either advertised metadata or ``inspect.getsourcefile`` as the whole owner
    would leave the other executable layer unauthenticated.  This gate proves:

    * the exact class in the expected MRO owns the classmethod descriptor;
    * the descriptor's live wrapper is bound to the exact requested subclass;
    * that wrapper's code object is the nested wrapper code owned by the exact
      authenticated decorator factory bytes;
    * its only closure capabilities are the exact original function and its
      canonical ``inspect.Signature``; and
    * the one-hop ``__wrapped__`` target is the exact original definition in
      the separately authenticated Diffusers source file.
    """

    if not inspect.isclass(expected_bound_owner) or not method_name:
        fail(f"live {label} classmethod authority differs")
    definition_path = _plain_file(
        expected_definition_path, label=f"pinned {label} definition source"
    )
    definition_sha, definition_identity = stable_file_sha256(
        definition_path, label=f"pinned {label} definition source"
    )
    wrapper_path = _plain_file(
        expected_wrapper_path, label=f"pinned {label} wrapper source"
    )
    wrapper_sha, wrapper_identity = stable_file_sha256(
        wrapper_path, label=f"pinned {label} wrapper source"
    )
    if (
        definition_sha
        != _require_sha(
            expected_definition_sha256,
            label=f"pinned {label} definition SHA",
        )
        or wrapper_sha
        != _require_sha(
            expected_wrapper_sha256,
            label=f"pinned {label} wrapper SHA",
        )
    ):
        fail(f"live {label} classmethod source bytes differ")

    definition_module = sys.modules.get(expected_definition_module)
    wrapper_module = sys.modules.get(expected_wrapper_module)
    try:
        definition_module_path = (
            _plain_file(
                getattr(definition_module, "__file__", ""),
                label=f"live {label} definition module",
            )
            if definition_module is not None
            else None
        )
        wrapper_module_path = (
            _plain_file(
                getattr(wrapper_module, "__file__", ""),
                label=f"live {label} wrapper module",
            )
            if wrapper_module is not None
            else None
        )
    except (OSError, RuntimeError) as error:
        raise LevelBRendererError(
            f"cannot resolve live {label} classmethod modules"
        ) from error
    declared_owner: Any = definition_module
    if definition_module is not None:
        try:
            for component in expected_definition_owner_qualname.split("."):
                declared_owner = getattr(declared_owner, component)
        except (AttributeError, TypeError):
            declared_owner = None
    if (
        definition_module is None
        or wrapper_module is None
        or definition_module_path != definition_path
        or wrapper_module_path != wrapper_path
        or not inspect.isclass(declared_owner)
        or getattr(declared_owner, "__module__", None)
        != expected_definition_module
        or getattr(declared_owner, "__qualname__", None)
        != expected_definition_owner_qualname
    ):
        fail(f"live {label} classmethod module ownership differs")

    try:
        mro_definers = tuple(
            owner
            for owner in expected_bound_owner.__mro__
            if method_name in vars(owner)
        )
        descriptor = vars(declared_owner).get(method_name)
        rebound = getattr(expected_bound_owner, method_name)
    except (AttributeError, TypeError) as error:
        raise LevelBRendererError(
            f"cannot inspect live {label} classmethod ownership"
        ) from error
    if (
        not mro_definers
        or mro_definers[0] is not declared_owner
        or not isinstance(descriptor, classmethod)
        or not inspect.ismethod(function)
        or not inspect.ismethod(rebound)
        or getattr(function, "__self__", None) is not expected_bound_owner
        or getattr(rebound, "__self__", None) is not expected_bound_owner
        or getattr(function, "__func__", None) is not descriptor.__func__
        or getattr(rebound, "__func__", None) is not descriptor.__func__
    ):
        fail(f"live {label} inherited classmethod owner differs")

    wrapper = descriptor.__func__
    original = getattr(wrapper, "__wrapped__", None)
    wrapper_code = getattr(wrapper, "__code__", None)
    original_code = getattr(original, "__code__", None)
    try:
        wrapper_code_path = (
            Path(wrapper_code.co_filename).resolve(strict=True)
            if isinstance(wrapper_code, CodeType)
            else None
        )
        wrapper_source_path = (
            Path(inspect.getsourcefile(wrapper) or "").resolve(strict=True)
            if callable(wrapper)
            else None
        )
        original_code_path = (
            Path(original_code.co_filename).resolve(strict=True)
            if isinstance(original_code, CodeType)
            else None
        )
    except OSError as error:
        raise LevelBRendererError(
            f"cannot resolve live {label} executable source"
        ) from error
    if (
        not callable(original)
        or getattr(original, "__wrapped__", _NO_BOUND_OWNER)
        is not _NO_BOUND_OWNER
        or inspect.unwrap(wrapper) is not original
        or getattr(wrapper, "__module__", None) != expected_definition_module
        or getattr(wrapper, "__qualname__", None) != expected_definition_qualname
        or getattr(original, "__module__", None) != expected_definition_module
        or getattr(original, "__qualname__", None) != expected_definition_qualname
        or inspect.getmodule(wrapper) is not definition_module
        or inspect.getmodule(original) is not definition_module
        or wrapper_code_path != wrapper_path
        or wrapper_source_path != wrapper_path
        or original_code_path != definition_path
    ):
        fail(f"live {label} wrapper/definition ownership differs")

    factory: Any = wrapper_module
    try:
        for component in expected_wrapper_factory_qualname.split("."):
            factory = getattr(factory, component)
    except (AttributeError, TypeError):
        factory = None
    factory_receipt = _audit_callable_against_authenticated_source(
        factory,
        label=f"{label} wrapper factory",
        expected_module=expected_wrapper_module,
        expected_qualname=expected_wrapper_factory_qualname,
        expected_path=wrapper_path,
        expected_sha256=wrapper_sha,
    )
    factory_code = getattr(factory, "__code__", None)
    pending = [factory_code] if isinstance(factory_code, CodeType) else []
    seen_codes: set[int] = set()
    wrapper_code_owner_identity_count = 0
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen_codes:
            continue
        seen_codes.add(id(candidate))
        for constant in candidate.co_consts:
            if constant is wrapper_code:
                wrapper_code_owner_identity_count += 1
            if isinstance(constant, CodeType):
                pending.append(constant)
    if wrapper_code_owner_identity_count != 1:
        fail(f"live {label} wrapper code is not owned by its decorator factory")

    closure = getattr(wrapper, "__closure__", None)
    freevars = tuple(getattr(wrapper_code, "co_freevars", ()))
    try:
        closure_values = tuple(cell.cell_contents for cell in closure or ())
    except ValueError as error:
        raise LevelBRendererError(f"live {label} wrapper closure is empty") from error
    closure_by_name = dict(zip(freevars, closure_values))
    canonical_original_signature = inspect.signature(
        original, follow_wrapped=False
    )
    if (
        freevars != ("fn", "signature")
        or len(closure_values) != 2
        or closure_by_name.get("fn") is not original
        or type(closure_by_name.get("signature")) is not inspect.Signature
        or closure_by_name.get("signature") != canonical_original_signature
    ):
        fail(f"live {label} wrapper closure capabilities differ")

    original_receipt = dict(_canonical_callable_receipt(original, label=label))
    if (
        original_receipt.get("module") != expected_definition_module
        or original_receipt.get("qualname") != expected_definition_qualname
        or original_receipt.get("source_file") != str(definition_path)
        or original_receipt.get("source_file_sha256") != definition_sha
    ):
        fail(f"live {label} original definition source differs")
    try:
        wrapper_source = inspect.getsource(wrapper).encode("utf-8")
    except (OSError, TypeError) as error:
        raise LevelBRendererError(f"cannot inspect live {label} wrapper source") from error
    return {
        "definition_owner": (
            f"{declared_owner.__module__}.{declared_owner.__qualname__}"
        ),
        "bound_owner": (
            f"{expected_bound_owner.__module__}."
            f"{expected_bound_owner.__qualname__}"
        ),
        "method_name": method_name,
        "exact_mro_definition_owner_identity_verified": True,
        "exact_bound_owner_identity_verified": True,
        "descriptor_is_exact_declared_classmethod": True,
        "wrapper": {
            "advertised_module": str(getattr(wrapper, "__module__", "")),
            "advertised_qualname": str(getattr(wrapper, "__qualname__", "")),
            "executable_source_file": str(wrapper_path),
            "executable_source_file_sha256": wrapper_sha,
            "executable_source_file_identity": wrapper_identity,
            "callable_source_sha256": hashlib.sha256(wrapper_source).hexdigest(),
            "code_firstlineno": int(wrapper_code.co_firstlineno),
            "runtime_signature": _canonical_signature_receipt(
                wrapper, label=f"{label} wrapper"
            ),
            "exact_decorator_factory_code_identity_verified": True,
            "decorator_factory_nested_code_identity_count": (
                wrapper_code_owner_identity_count
            ),
            "exact_two_cell_closure_verified": True,
            "closure_freevars": list(freevars),
            "factory": factory_receipt,
        },
        "unwrapped_definition": {
            **original_receipt,
            "source_file_identity": definition_identity,
            "one_hop_wrapped_target_identity_verified": True,
        },
        "all_executable_layers_owned_by_authenticated_source_bytes": True,
    }


def _audit_native_unipc_scheduler_callable(
    scheduler: Any,
    *,
    expected_path: str = str(
        Path(PINNED_SITE_PACKAGES)
        / "diffusers/schedulers/scheduling_unipc_multistep.py"
    ),
    expected_sha256: str = PINNED_SITE_PACKAGE_SOURCE_HASHES[
        "diffusers/schedulers/scheduling_unipc_multistep.py"
    ],
) -> Mapping[str, Any]:
    """Reject scheduler lookalikes before any formal bridge can be installed."""

    step = getattr(scheduler, "step", None)
    owner = type(scheduler)
    module = sys.modules.get("diffusers.schedulers.scheduling_unipc_multistep")
    declared = getattr(owner, "step", None)
    if (
        owner.__module__ != "diffusers.schedulers.scheduling_unipc_multistep"
        or owner.__name__ != "UniPCMultistepScheduler"
        or module is None
        or getattr(module, "UniPCMultistepScheduler", None) is not owner
        or not inspect.ismethod(step)
        or getattr(step, "__self__", None) is not scheduler
        or getattr(step, "__func__", None) is not declared
        or getattr(getattr(scheduler, "config", None), "_class_name", None)
        != "UniPCMultistepScheduler"
    ):
        fail("live scheduler is not the exact pinned UniPC vendor class")
    return _audit_callable_against_authenticated_source(
        step,
        label="native UniPC scheduler.step",
        expected_module="diffusers.schedulers.scheduling_unipc_multistep",
        expected_qualname="UniPCMultistepScheduler.step",
        expected_path=expected_path,
        expected_sha256=expected_sha256,
    )


def _contains_callable(value: Any, *, seen: Optional[set[int]] = None) -> bool:
    if callable(value):
        return True
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        return any(
            _contains_callable(key, seen=seen)
            or _contains_callable(item, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        return any(_contains_callable(item, seen=seen) for item in value)
    return False


def audit_official_renderer_sample_call(
    *, renderer: Any, sample_kwargs: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Bind the exact source-only call against the live renderer method.

    This is deliberately a runtime check: the ``FreshWorld8ModelBundle`` owns
    the actual loaded Bernini renderer, so a stale local assumption about
    ``model.sample`` cannot pass merely because the orchestration source is
    hash-pinned.  No user callable is permitted anywhere in the argument tree.
    """

    sample = getattr(renderer, "sample", None)
    if not callable(sample) or getattr(sample, "__self__", None) is not renderer:
        fail("official renderer sample is not the live bound model method")
    keyword_names = tuple(sample_kwargs)
    if keyword_names != OFFICIAL_RENDERER_SAMPLE_KEYWORDS:
        fail("official renderer sample keyword set/order differs")
    # Exact tuple equality above is the stronger closed-world condition.  Do
    # not substring-filter the approved names: ``attention_mask`` and
    # ``flow_shift`` are native sampler controls, not external mask/flow
    # conditions.
    if _contains_callable(sample_kwargs):
        fail("renderer sample kwargs contain a hidden callable")
    try:
        signature = inspect.signature(sample)
        undeclared = sorted(
            name for name in keyword_names if name not in signature.parameters
        )
        if undeclared:
            fail(
                "live renderer sample does not explicitly declare source-only "
                f"kwargs: {undeclared}"
            )
        bound = signature.bind(**dict(sample_kwargs))
    except LevelBRendererError:
        raise
    except (TypeError, ValueError) as error:
        raise LevelBRendererError(
            "exact source-only kwargs do not bind the live renderer sample signature"
        ) from error
    return {
        "live_bound_method": True,
        "exact_keyword_names_in_call_order": list(keyword_names),
        "exact_keyword_names_sorted": sorted(keyword_names),
        "bound_runtime_parameter_names": list(bound.arguments),
        "runtime_signature": _canonical_signature_receipt(
            sample, label="official renderer sample"
        ),
        "callable_provenance": _canonical_callable_receipt(
            sample, label="official renderer sample"
        ),
        "source_instruction_seed_only": True,
        "target_anchor_teacher_external_annotation_kwargs_present": False,
        "caller_callback_or_custom_denoiser_present": False,
    }


def _mode_and_links(
    path: Path, *, mode: int, label: str, nlink: Optional[int] = 1
) -> Mapping[str, int]:
    try:
        info = path.lstat()
    except OSError as error:
        raise LevelBRendererError(f"{label} is unavailable") from error
    if stat.S_IMODE(info.st_mode) != mode or (
        nlink is not None and int(info.st_nlink) != nlink
    ):
        fail(f"{label} physical mode/link contract differs")
    return {
        "mode": int(stat.S_IMODE(info.st_mode)),
        "nlink": int(info.st_nlink),
    }


def _validate_release_member_path(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "\\" in relative
        or "//" in relative
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        fail("Level-B release member path differs")
    candidate = root.joinpath(*relative.split("/"))
    path = _plain_file(candidate, label=f"Level-B release member {relative}")
    if path != candidate:
        fail("Level-B release member escaped its root")
    return path


def _validate_level_b_runtime_release_manifest(
    manifest_path: str | Path,
    *,
    sealed_launcher_expected_manifest_sha256: str,
    authority_snapshot: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Authenticate the exact Level-B source closure.

    The expected manifest digest is deliberately named as launcher authority:
    it must be a literal in the separately sealed executor and must never be
    accepted by the high-level product call.  Compile-time pins below prevent a
    caller from manufacturing a manifest for alternate executable source.
    """

    if authority_snapshot is None:
        runtime_schema = RUNTIME_RELEASE_SCHEMA
        authority_name = AUTHORITY
        member_paths = tuple(LEVEL_B_RELEASE_MEMBER_PATHS)
        static_source_pins = {
            key: dict(value) for key, value in LEVEL_B_STATIC_SOURCE_MEMBER_PINS.items()
        }
        executed_self_path = str(Path(__file__).resolve(strict=True))
    else:
        runtime_schema = authority_snapshot.get("runtime_release_schema")
        authority_name = authority_snapshot.get("authority")
        member_paths = tuple(authority_snapshot.get("release_member_paths", ()))
        static_source_pins = authority_snapshot.get("static_source_member_pins")
        executed_self_path = authority_snapshot.get("executed_self_path")
        if (
            runtime_schema != RUNTIME_RELEASE_SCHEMA
            or authority_name != AUTHORITY
            or member_paths != tuple(LEVEL_B_RELEASE_MEMBER_PATHS)
            or not isinstance(static_source_pins, Mapping)
            or not isinstance(executed_self_path, str)
        ):
            fail("captured Level-B release authority differs")
    manifest = _plain_file(manifest_path, label="sealed Level-B release manifest")
    expected_manifest_sha = _require_sha(
        sealed_launcher_expected_manifest_sha256,
        label="sealed-launcher Level-B release manifest SHA",
    )
    manifest_sha, manifest_identity = stable_file_sha256(
        manifest, label="sealed Level-B release manifest"
    )
    if manifest_sha != expected_manifest_sha:
        fail("sealed Level-B release manifest bytes differ")
    root = _plain_directory(manifest.parent, label="sealed Level-B release root")
    if manifest != root / "RELEASE_MANIFEST.json":
        fail("sealed Level-B release manifest name/root differs")
    _mode_and_links(
        root, mode=0o555, label="sealed Level-B release root", nlink=None
    )
    _mode_and_links(manifest, mode=0o444, label="sealed Level-B release manifest")
    payload = _loads_strict_json(
        manifest.read_bytes(), label="sealed Level-B release manifest"
    )
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "authority",
        "member_count",
        "members",
        "release_digest",
    }:
        fail("sealed Level-B release manifest field closure differs")
    rows = payload.get("members")
    if (
        payload.get("schema_version") != runtime_schema
        or payload.get("authority") != authority_name
        or type(payload.get("member_count")) is not int
        or payload.get("member_count") != len(member_paths)
        or not isinstance(rows, list)
        or len(rows) != len(member_paths)
        or payload.get("release_digest") != object_sha256(rows)
    ):
        fail("sealed Level-B release manifest contract differs")
    expected_paths = list(member_paths)
    observed_paths = [row.get("path") if isinstance(row, Mapping) else None for row in rows]
    if observed_paths != expected_paths:
        fail("sealed Level-B release exact member order/set differs")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            fail("sealed Level-B release contains a link or special entry")
        if stat.S_ISREG(mode):
            actual_files.add(relative)
        else:
            actual_directories.add(relative)
    if actual_files != set(expected_paths) | {"RELEASE_MANIFEST.json"}:
        fail("sealed Level-B release physical file closure differs")
    if actual_directories != {"tools"}:
        fail("sealed Level-B release physical directory closure differs")
    _mode_and_links(
        root / "tools", mode=0o555, label="Level-B tools directory", nlink=None
    )

    member_receipts: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size",
            "mode",
        }:
            fail("sealed Level-B release member row differs")
        relative = row["path"]
        expected_sha = _require_sha(
            row.get("sha256"), label=f"Level-B release member {relative} SHA"
        )
        if (
            type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("mode") != 0o444
        ):
            fail("sealed Level-B release member metadata differs")
        path = _validate_release_member_path(root, relative)
        physical = _mode_and_links(
            path, mode=0o444, label=f"Level-B release member {relative}"
        )
        observed_sha, identity = stable_file_sha256(
            path, label=f"Level-B release member {relative}"
        )
        if observed_sha != expected_sha or identity["size"] != row["size"]:
            fail("sealed Level-B release member bytes/size differ")
        static_pin = static_source_pins.get(relative)
        if static_pin is not None and (
            expected_sha != static_pin["sha256"] or row["size"] != static_pin["size"]
        ):
            fail("Level-B transitive source differs from its compile-time pin")
        member_receipts[relative] = {
            "path": str(path),
            "sha256": observed_sha,
            "size": row["size"],
            **dict(physical),
            "file_identity": identity,
        }
    self_path = _plain_file(executed_self_path, label="Level-B source")
    if self_path != root / "infer_action_edit_level_b_renderer_0817_v1.py":
        fail("executed Level-B source is not the sealed release member")
    self_row = member_receipts["infer_action_edit_level_b_renderer_0817_v1.py"]
    self_sha, _ = stable_file_sha256(self_path, label="executed Level-B source")
    if self_sha != self_row["sha256"]:
        fail("executed Level-B source bytes differ from sealed release")
    return {
        "schema_version": runtime_schema,
        "authority": authority_name,
        "release_root": str(root),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_file_identity": manifest_identity,
        "release_digest": payload["release_digest"],
        "exact_member_count": len(member_receipts),
        "members": member_receipts,
        "executed_level_b_member_verified": True,
        "compile_time_transitive_source_pins_verified": True,
    }


class VerifiedLevelBRuntime:
    """Single-use opaque capability owned by the sealed-launcher closure."""

    __slots__ = ()

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        fail("VerifiedLevelBRuntime may only be issued by release authentication")

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        fail("VerifiedLevelBRuntime is final and cannot be subclassed")

    def validate_at_use(self) -> Mapping[str, Any]:
        fail("opaque Level-B runtime capability was not issued by this module")


def _make_level_b_runtime_authenticator(
    sealed_launcher_manifest_sha256: Any,
) -> Any:
    # Everything below is captured once while authenticated module bytes are
    # executing.  Mutating visible module constants later can only make the
    # checks fail; it cannot create a new authority or alter this snapshot.
    authority_snapshot = {
        "runtime_release_schema": str(RUNTIME_RELEASE_SCHEMA),
        "authority": str(AUTHORITY),
        "release_member_paths": tuple(LEVEL_B_RELEASE_MEMBER_PATHS),
        "static_source_member_pins": {
            key: dict(value)
            for key, value in LEVEL_B_STATIC_SOURCE_MEMBER_PINS.items()
        },
        "executed_self_path": str(Path(__file__).resolve(strict=True)),
        "bernini_root": str(PINNED_BERNINI_ROOT),
        "veomni_root": str(PINNED_VEOMNI_ROOT),
        "base_checkpoint": str(PINNED_BASE_CHECKPOINT),
        "base_checkpoint_tree_sha256": str(PINNED_BASE_CHECKPOINT_TREE_SHA256),
        "pinned_python_path": str(PINNED_PYTHON_PATH),
        "pinned_python_sha256": str(PINNED_PYTHON_SHA256),
        "fixed_files": (
            (
                "checkpoint_content_manifest",
                str(PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST),
                str(PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256),
                False,
            ),
            ("ffmpeg", str(PINNED_FFMPEG_PATH), str(PINNED_FFMPEG_SHA256), True),
            ("ffprobe", str(PINNED_FFPROBE_PATH), str(PINNED_FFPROBE_SHA256), True),
            ("python", str(PINNED_PYTHON_PATH), str(PINNED_PYTHON_SHA256), True),
            (
                "stdlib_socket_source",
                str(PINNED_STDLIB_SOCKET_PATH),
                str(PINNED_STDLIB_SOCKET_SHA256),
                False,
            ),
        ),
        "vendor_source_files": tuple(
            (
                f"bernini:{relative}",
                str(Path(PINNED_BERNINI_ROOT) / relative),
                str(expected_sha),
            )
            for relative, expected_sha in PINNED_BERNINI_RUNTIME_FILE_HASHES.items()
        )
        + tuple(
            (
                f"veomni:{relative}",
                str(Path(PINNED_VEOMNI_ROOT) / relative),
                str(expected_sha),
            )
            for relative, expected_sha in PINNED_VEOMNI_RUNTIME_FILE_HASHES.items()
        )
        + tuple(
            (
                f"site-packages:{relative}",
                str(Path(PINNED_SITE_PACKAGES) / relative),
                str(expected_sha),
            )
            for relative, expected_sha in PINNED_SITE_PACKAGE_SOURCE_HASHES.items()
        ),
        "diffusers_version": str(PINNED_DIFFUSERS_VERSION),
        "transformers_version": str(PINNED_TRANSFORMERS_VERSION),
        "torch_version": str(PINNED_TORCH_VERSION),
    }
    release_validator = _validate_level_b_runtime_release_manifest
    registry: dict[int, tuple[Any, Mapping[str, Any]]] = {}
    issued = False

    def authenticate_file(
        *, name: str, raw_path: str, expected_sha: str, executable: bool
    ) -> Mapping[str, Any]:
        path = _plain_file(raw_path, label=f"pinned {name}")
        observed, identity = stable_file_sha256(path, label=f"pinned {name}")
        if observed != _require_sha(expected_sha, label=f"pinned {name} SHA"):
            fail(f"pinned {name} bytes differ")
        if executable and not os.access(path, os.X_OK):
            fail(f"pinned {name} is not executable")
        return {"path": str(path), "sha256": observed, "file_identity": identity}

    def validate_registered_capability_at_use(self: Any) -> Mapping[str, Any]:
        entry = registry.pop(id(self), None)
        if entry is None or entry[0] is not self:
            fail("opaque Level-B runtime capability is forged, stale, or already consumed")
        receipt = _loads_strict_json(
            canonical_json_bytes(entry[1]), label="verified Level-B runtime"
        )
        expected_keys = {
            "schema_version",
            "authority",
            "release",
            "sealed_launcher_manifest_sha256",
            "sealed_launcher_pin_captured_before_level_b_source_exec",
            "bernini_root",
            "veomni_root",
            "base_checkpoint",
            "base_checkpoint_tree_sha256",
            "checkpoint_content_manifest",
            "ffmpeg",
            "ffprobe",
            "python",
            "stdlib_socket_source",
            "vendor_source_files",
            "diffusers_version",
            "transformers_version",
            "torch_version",
            "fixed_paths_and_hashes_are_not_product_inputs",
            "single_use_opaque_capability",
            "runtime_digest",
        }
        runtime_digest = receipt.get("runtime_digest")
        if (
            set(receipt) != expected_keys
            or receipt.get("schema_version")
            != authority_snapshot["runtime_release_schema"]
            or receipt.get("authority") != authority_snapshot["authority"]
            or runtime_digest
            != object_sha256(
                {key: value for key, value in receipt.items() if key != "runtime_digest"}
            )
            or receipt.get("base_checkpoint_tree_sha256")
            != authority_snapshot["base_checkpoint_tree_sha256"]
            or receipt.get("fixed_paths_and_hashes_are_not_product_inputs") is not True
            or receipt.get("single_use_opaque_capability") is not True
            or receipt.get("sealed_launcher_pin_captured_before_level_b_source_exec")
            is not True
            or not isinstance(receipt.get("release"), Mapping)
            or receipt.get("sealed_launcher_manifest_sha256")
            != receipt["release"].get("manifest_sha256")
        ):
            fail("opaque verified Level-B runtime receipt digest/contract differs")
        live_release = release_validator(
            receipt["release"]["manifest_path"],
            sealed_launcher_expected_manifest_sha256=(
                sealed_launcher_manifest_sha256
            ),
            authority_snapshot=authority_snapshot,
        )
        if canonical_json_bytes(live_release) != canonical_json_bytes(receipt["release"]):
            fail("live exact-five Level-B release closure changed after authentication")
        if set(live_release["members"]) != set(
            authority_snapshot["release_member_paths"]
        ):
            fail("live Level-B release is not the captured exact-five closure")
        for name, raw_path, expected_sha, executable in authority_snapshot[
            "fixed_files"
        ]:
            live = authenticate_file(
                name=name,
                raw_path=raw_path,
                expected_sha=expected_sha,
                executable=executable,
            )
            if canonical_json_bytes(live) != canonical_json_bytes(receipt[name]):
                fail(f"pinned {name} changed after runtime authentication")
        vendor_rows = receipt.get("vendor_source_files")
        if not isinstance(vendor_rows, Mapping) or set(vendor_rows) != {
            name for name, _path, _sha in authority_snapshot["vendor_source_files"]
        }:
            fail("pinned vendor source closure differs at use")
        for name, raw_path, expected_sha in authority_snapshot["vendor_source_files"]:
            live = authenticate_file(
                name=name,
                raw_path=raw_path,
                expected_sha=expected_sha,
                executable=False,
            )
            if canonical_json_bytes(live) != canonical_json_bytes(vendor_rows[name]):
                fail(f"pinned vendor source {name} changed after authentication")
        if (
            _plain_directory(
                authority_snapshot["bernini_root"], label="Bernini root at use"
            )
            != Path(authority_snapshot["bernini_root"])
            or _plain_directory(
                authority_snapshot["base_checkpoint"], label="base checkpoint at use"
            )
            != Path(authority_snapshot["base_checkpoint"])
            or receipt.get("bernini_root") != authority_snapshot["bernini_root"]
            or _plain_directory(
                authority_snapshot["veomni_root"], label="VeOmni root at use"
            )
            != Path(authority_snapshot["veomni_root"])
            or receipt.get("veomni_root") != authority_snapshot["veomni_root"]
            or receipt.get("base_checkpoint")
            != authority_snapshot["base_checkpoint"]
            or receipt.get("diffusers_version")
            != authority_snapshot["diffusers_version"]
            or receipt.get("transformers_version")
            != authority_snapshot["transformers_version"]
            or receipt.get("torch_version")
            != authority_snapshot["torch_version"]
            or _plain_file(
                sys.executable, label="running Python executable at use"
            )
            != Path(authority_snapshot["pinned_python_path"])
        ):
            fail("captured Level-B roots/vendor versions differ at use")
        return receipt

    # The method closes over the private registry.  There is no module-global
    # seal and no issuer method on the capability class.
    VerifiedLevelBRuntime.validate_at_use = validate_registered_capability_at_use

    def authenticate_level_b_runtime_release(
        manifest_path: str | Path,
    ) -> VerifiedLevelBRuntime:
        """Issue an opaque runtime using the launcher's import-time SHA pin."""

        nonlocal issued
        if issued:
            fail("sealed Level-B runtime capability was already issued")
        expected_manifest_sha = _require_sha(
            sealed_launcher_manifest_sha256,
            label="import-time sealed-launcher Level-B manifest SHA",
        )
        release = release_validator(
            manifest_path,
            sealed_launcher_expected_manifest_sha256=expected_manifest_sha,
            authority_snapshot=authority_snapshot,
        )
        bernini_root = _plain_directory(
            authority_snapshot["bernini_root"], label="pinned Bernini root"
        )
        veomni_root = _plain_directory(
            authority_snapshot["veomni_root"], label="pinned VeOmni root"
        )
        base_checkpoint = _plain_directory(
            authority_snapshot["base_checkpoint"], label="pinned base checkpoint"
        )
        receipt: dict[str, Any] = {
            "schema_version": authority_snapshot["runtime_release_schema"],
            "authority": authority_snapshot["authority"],
            "release": dict(release),
            "sealed_launcher_manifest_sha256": expected_manifest_sha,
            "sealed_launcher_pin_captured_before_level_b_source_exec": True,
            "bernini_root": str(bernini_root),
            "veomni_root": str(veomni_root),
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_tree_sha256": authority_snapshot[
                "base_checkpoint_tree_sha256"
            ],
            "diffusers_version": authority_snapshot["diffusers_version"],
            "transformers_version": authority_snapshot["transformers_version"],
            "torch_version": authority_snapshot["torch_version"],
        }
        for name, raw_path, expected_sha, executable in authority_snapshot[
            "fixed_files"
        ]:
            receipt[name] = authenticate_file(
                name=name,
                raw_path=raw_path,
                expected_sha=expected_sha,
                executable=executable,
            )
        if Path(sys.executable).resolve(strict=True) != Path(
            authority_snapshot["pinned_python_path"]
        ):
            fail("running Python executable is not the pinned vace interpreter")
        receipt["vendor_source_files"] = {
            name: authenticate_file(
                name=name,
                raw_path=raw_path,
                expected_sha=expected_sha,
                executable=False,
            )
            for name, raw_path, expected_sha in authority_snapshot[
                "vendor_source_files"
            ]
        }
        receipt["fixed_paths_and_hashes_are_not_product_inputs"] = True
        receipt["single_use_opaque_capability"] = True
        receipt["runtime_digest"] = object_sha256(receipt)
        detached = _loads_strict_json(
            canonical_json_bytes(receipt), label="verified Level-B runtime receipt"
        )
        value = object.__new__(VerifiedLevelBRuntime)
        registry[id(value)] = (value, detached)
        issued = True
        return value

    return authenticate_level_b_runtime_release


authenticate_level_b_runtime_release = _make_level_b_runtime_authenticator(
    _SEALED_LAUNCHER_MANIFEST_SHA_AT_IMPORT
)
del _make_level_b_runtime_authenticator
del _SEALED_LAUNCHER_MANIFEST_SHA_AT_IMPORT


def _audit_live_vendor_callable_closure(
    *,
    runtime_receipt: Mapping[str, Any],
    diffusers_version: str,
    transformers_version: str,
    vae_class: Any,
    tokenizer_factory_class: Any,
    vae_load_config: Any,
    vae_from_pretrained: Any,
    tokenizer_from_pretrained: Any,
    prompt_cleaner: Any,
    noise_factory: Any,
    vae_encode: Any,
    vae_decode: Any,
    video_save: Any,
    scheduler: Any,
) -> Mapping[str, Any]:
    """Prove that every orchestration callable is from frozen vendor bytes."""

    rows = runtime_receipt.get("vendor_source_files")
    if (
        diffusers_version != PINNED_DIFFUSERS_VERSION
        or transformers_version != PINNED_TRANSFORMERS_VERSION
        or not isinstance(rows, Mapping)
    ):
        fail("live diffusers/transformers vendor authority differs")
    vae_module = sys.modules.get(
        "diffusers.models.autoencoders.autoencoder_kl_wan"
    )
    tokenizer_factory_module = sys.modules.get(
        "transformers.models.auto.tokenization_auto"
    )
    if (
        not inspect.isclass(vae_class)
        or vae_class.__module__
        != "diffusers.models.autoencoders.autoencoder_kl_wan"
        or vae_module is None
        or getattr(vae_module, "AutoencoderKLWan", None) is not vae_class
        or not inspect.isclass(tokenizer_factory_class)
        or tokenizer_factory_class.__module__
        != "transformers.models.auto.tokenization_auto"
        or tokenizer_factory_module is None
        or getattr(tokenizer_factory_module, "AutoTokenizer", None)
        is not tokenizer_factory_class
    ):
        fail("live VAE/tokenizer factory class identity differs")

    def row(logical_name: str) -> Mapping[str, Any]:
        value = rows.get(logical_name)
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "file_identity",
        }:
            fail(f"authenticated vendor source row differs: {logical_name}")
        return value

    specs = (
        (
            "Bernini VAE encode",
            vae_encode,
            "bernini.pipeline",
            "_vae_encode",
            "bernini:bernini/pipeline.py",
            _NO_BOUND_OWNER,
        ),
        (
            "Bernini VAE decode",
            vae_decode,
            "bernini.pipeline",
            "_vae_decode",
            "bernini:bernini/pipeline.py",
            _NO_BOUND_OWNER,
        ),
        (
            "Bernini video save",
            video_save,
            "bernini.io_utils",
            "save_output",
            "bernini:bernini/io_utils.py",
            _NO_BOUND_OWNER,
        ),
        (
            "Wan prompt cleaner",
            prompt_cleaner,
            "diffusers.pipelines.wan.pipeline_wan",
            "prompt_clean",
            "site-packages:diffusers/pipelines/wan/pipeline_wan.py",
            _NO_BOUND_OWNER,
        ),
        (
            "Diffusers Gaussian factory",
            noise_factory,
            "diffusers.utils.torch_utils",
            "randn_tensor",
            "site-packages:diffusers/utils/torch_utils.py",
            _NO_BOUND_OWNER,
        ),
        (
            "AutoTokenizer from_pretrained",
            tokenizer_from_pretrained,
            "transformers.models.auto.tokenization_auto",
            "AutoTokenizer.from_pretrained",
            "site-packages:transformers/models/auto/tokenization_auto.py",
            tokenizer_factory_class,
        ),
    )
    receipts: dict[str, Any] = {}
    for label, function, module_name, qualname, logical_name, bound_owner in specs:
        authority = row(logical_name)
        audited = _audit_callable_against_authenticated_source(
            function,
            label=label,
            expected_module=module_name,
            expected_qualname=qualname,
            expected_path=authority["path"],
            expected_sha256=authority["sha256"],
            expected_bound_owner=bound_owner,
        )
        if audited["source_file_identity"] != authority["file_identity"]:
            fail(f"live {label} source identity changed after runtime admission")
        receipts[label] = audited

    wrapper_authority = row(
        "site-packages:huggingface_hub/utils/_validators.py"
    )
    for label, function, method_name, module_name, owner_qualname, qualname, logical_name in (
        (
            "Wan VAE load_config",
            vae_load_config,
            "load_config",
            "diffusers.configuration_utils",
            "ConfigMixin",
            "ConfigMixin.load_config",
            "site-packages:diffusers/configuration_utils.py",
        ),
        (
            "Wan VAE from_pretrained",
            vae_from_pretrained,
            "from_pretrained",
            "diffusers.models.modeling_utils",
            "ModelMixin",
            "ModelMixin.from_pretrained",
            "site-packages:diffusers/models/modeling_utils.py",
        ),
    ):
        definition_authority = row(logical_name)
        audited = _audit_inherited_wrapped_classmethod_against_authenticated_sources(
            function,
            label=label,
            expected_bound_owner=vae_class,
            method_name=method_name,
            expected_definition_module=module_name,
            expected_definition_owner_qualname=owner_qualname,
            expected_definition_qualname=qualname,
            expected_definition_path=definition_authority["path"],
            expected_definition_sha256=definition_authority["sha256"],
            expected_wrapper_module="huggingface_hub.utils._validators",
            expected_wrapper_factory_qualname="validate_hf_hub_args",
            expected_wrapper_path=wrapper_authority["path"],
            expected_wrapper_sha256=wrapper_authority["sha256"],
        )
        if (
            audited["wrapper"]["executable_source_file_identity"]
            != wrapper_authority["file_identity"]
            or audited["unwrapped_definition"]["source_file_identity"]
            != definition_authority["file_identity"]
        ):
            fail(f"live {label} source identity changed after runtime admission")
        receipts[label] = audited
    scheduler_row = row(
        "site-packages:diffusers/schedulers/scheduling_unipc_multistep.py"
    )
    scheduler_receipt = _audit_native_unipc_scheduler_callable(
        scheduler,
        expected_path=scheduler_row["path"],
        expected_sha256=scheduler_row["sha256"],
    )
    if scheduler_receipt["source_file_identity"] != scheduler_row["file_identity"]:
        fail("native UniPC source identity changed after runtime admission")
    receipts["native UniPC scheduler.step"] = scheduler_receipt
    return {
        "diffusers_version": diffusers_version,
        "transformers_version": transformers_version,
        "exact_callable_count": len(receipts),
        "callables": receipts,
        "fake_or_substitute_scheduler_accepted": False,
        "all_live_callables_owned_by_authenticated_source_bytes": True,
    }


def _audit_wrapped_instance_method_against_authenticated_sources(
    function: Any,
    *,
    label: str,
    expected_instance: Any,
    expected_class: Any,
    method_name: str,
    expected_value_parameter: str,
    expected_definition_module: str,
    expected_definition_class_qualname: str,
    expected_definition_qualname: str,
    expected_definition_path: str | Path,
    expected_definition_sha256: str,
    expected_original_firstlineno: int,
    expected_value_annotation: Any,
    expected_return_union_origin: Any,
    expected_return_primary: Any,
    expected_return_secondary: Any,
    expected_return_secondary_container: Any,
    expected_wrapper_module: str,
    expected_wrapper_factory_qualname: str,
    expected_wrapper_qualname: str,
    expected_wrapper_path: str | Path,
    expected_wrapper_sha256: str,
    expected_wrapper_factory_firstlineno: int,
    expected_wrapper_firstlineno: int,
) -> Mapping[str, Any]:
    """Authenticate an ``apply_forward_hook`` instance-method wrapper.

    Diffusers replaces ``AutoencoderKLWan.encode`` and ``decode`` at class
    construction time.  Their class descriptors are functions executing from
    ``accelerate_utils.py``; each closes over the original function from
    ``autoencoder_kl_wan.py`` without exposing ``__wrapped__``.  Authenticate
    the descriptor, bound instance, decorator code and sole closure capability
    before treating either live method as part of the product call graph.
    """

    if (
        not inspect.isclass(expected_class)
        or type(expected_instance) is not expected_class
        or not method_name
        or not expected_value_parameter
    ):
        fail(f"live {label} wrapped-method authority differs")
    definition_path = _plain_file(
        expected_definition_path, label=f"pinned {label} definition source"
    )
    definition_sha, definition_identity = stable_file_sha256(
        definition_path, label=f"pinned {label} definition source"
    )
    wrapper_path = _plain_file(
        expected_wrapper_path, label=f"pinned {label} wrapper source"
    )
    wrapper_sha, wrapper_identity = stable_file_sha256(
        wrapper_path, label=f"pinned {label} wrapper source"
    )
    if (
        definition_sha
        != _require_sha(
            expected_definition_sha256,
            label=f"pinned {label} definition SHA",
        )
        or wrapper_sha
        != _require_sha(
            expected_wrapper_sha256,
            label=f"pinned {label} wrapper SHA",
        )
    ):
        fail(f"live {label} wrapped-method source bytes differ")

    definition_module = sys.modules.get(expected_definition_module)
    wrapper_module = sys.modules.get(expected_wrapper_module)
    try:
        definition_module_path = (
            _plain_file(
                getattr(definition_module, "__file__", ""),
                label=f"live {label} definition module",
            )
            if definition_module is not None
            else None
        )
        wrapper_module_path = (
            _plain_file(
                getattr(wrapper_module, "__file__", ""),
                label=f"live {label} wrapper module",
            )
            if wrapper_module is not None
            else None
        )
    except RuntimeError as error:
        raise LevelBRendererError(
            f"cannot resolve live {label} wrapped-method modules"
        ) from error
    declared_class: Any = definition_module
    if definition_module is not None:
        try:
            for component in expected_definition_class_qualname.split("."):
                declared_class = getattr(declared_class, component)
        except (AttributeError, TypeError):
            declared_class = None
    if (
        definition_module is None
        or wrapper_module is None
        or definition_module_path != definition_path
        or wrapper_module_path != wrapper_path
        or declared_class is not expected_class
        or getattr(expected_class, "__module__", None)
        != expected_definition_module
        or getattr(expected_class, "__qualname__", None)
        != expected_definition_class_qualname
    ):
        fail(f"live {label} wrapped-method module/class ownership differs")

    try:
        mro_definers = tuple(
            owner for owner in expected_class.__mro__ if method_name in vars(owner)
        )
        descriptor = vars(expected_class).get(method_name)
        rebound = getattr(expected_instance, method_name)
    except (AttributeError, TypeError) as error:
        raise LevelBRendererError(
            f"cannot inspect live {label} wrapped-method ownership"
        ) from error
    if (
        not mro_definers
        or mro_definers[0] is not expected_class
        or not inspect.isfunction(descriptor)
        or not inspect.ismethod(function)
        or not inspect.ismethod(rebound)
        or getattr(function, "__self__", None) is not expected_instance
        or getattr(rebound, "__self__", None) is not expected_instance
        or getattr(function, "__func__", None) is not descriptor
        or getattr(rebound, "__func__", None) is not descriptor
    ):
        fail(f"live {label} exact descriptor/bound-instance ownership differs")

    wrapper = descriptor
    wrapper_code = getattr(wrapper, "__code__", None)
    try:
        wrapper_code_path = (
            Path(wrapper_code.co_filename).resolve(strict=True)
            if isinstance(wrapper_code, CodeType)
            else None
        )
        wrapper_source_path = Path(
            inspect.getsourcefile(wrapper) or ""
        ).resolve(strict=True)
    except OSError as error:
        raise LevelBRendererError(
            f"cannot resolve live {label} wrapper source"
        ) from error
    if (
        getattr(wrapper, "__wrapped__", _NO_BOUND_OWNER)
        is not _NO_BOUND_OWNER
        or getattr(wrapper, "__module__", None) != expected_wrapper_module
        or getattr(wrapper, "__qualname__", None) != expected_wrapper_qualname
        or inspect.getmodule(wrapper) is not wrapper_module
        or wrapper_code_path != wrapper_path
        or wrapper_source_path != wrapper_path
        or not isinstance(wrapper_code, CodeType)
        or wrapper_code.co_firstlineno != expected_wrapper_firstlineno
        or tuple(wrapper_code.co_freevars) != ("method",)
    ):
        fail(f"live {label} wrapper executable ownership differs")
    wrapper_signature = inspect.signature(wrapper, follow_wrapped=False)
    wrapper_parameters = tuple(wrapper_signature.parameters.values())
    if (
        len(wrapper_parameters) != 3
        or wrapper_parameters[0].name != "self"
        or wrapper_parameters[0].kind
        is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or wrapper_parameters[0].default is not inspect.Parameter.empty
        or wrapper_parameters[1].name != "args"
        or wrapper_parameters[1].kind is not inspect.Parameter.VAR_POSITIONAL
        or wrapper_parameters[2].name != "kwargs"
        or wrapper_parameters[2].kind is not inspect.Parameter.VAR_KEYWORD
        or wrapper_signature.return_annotation is not inspect.Signature.empty
    ):
        fail(f"live {label} wrapper signature differs")

    factory: Any = wrapper_module
    try:
        for component in expected_wrapper_factory_qualname.split("."):
            factory = getattr(factory, component)
    except (AttributeError, TypeError):
        factory = None
    factory_receipt = _audit_callable_against_authenticated_source(
        factory,
        label=f"{label} wrapper factory",
        expected_module=expected_wrapper_module,
        expected_qualname=expected_wrapper_factory_qualname,
        expected_path=wrapper_path,
        expected_sha256=wrapper_sha,
    )
    factory_code = getattr(factory, "__code__", None)
    if (
        not isinstance(factory_code, CodeType)
        or factory_code.co_firstlineno != expected_wrapper_factory_firstlineno
        or tuple(factory_code.co_freevars) != ()
    ):
        fail(f"live {label} wrapper factory code identity differs")
    pending = [factory_code]
    seen_codes: set[int] = set()
    wrapper_code_owner_identity_count = 0
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen_codes:
            continue
        seen_codes.add(id(candidate))
        for constant in candidate.co_consts:
            if constant is wrapper_code:
                wrapper_code_owner_identity_count += 1
            if isinstance(constant, CodeType):
                pending.append(constant)
    if wrapper_code_owner_identity_count != 1:
        fail(f"live {label} wrapper code is not uniquely owned by its factory")

    closure = getattr(wrapper, "__closure__", None)
    try:
        closure_values = tuple(cell.cell_contents for cell in closure or ())
    except ValueError as error:
        raise LevelBRendererError(f"live {label} wrapper closure is empty") from error
    if len(closure_values) != 1 or not inspect.isfunction(closure_values[0]):
        fail(f"live {label} wrapper closure capabilities differ")
    original = closure_values[0]
    original_code = getattr(original, "__code__", None)
    try:
        original_code_path = (
            Path(original_code.co_filename).resolve(strict=True)
            if isinstance(original_code, CodeType)
            else None
        )
    except OSError as error:
        raise LevelBRendererError(
            f"cannot resolve live {label} original definition"
        ) from error
    if (
        getattr(original, "__wrapped__", _NO_BOUND_OWNER)
        is not _NO_BOUND_OWNER
        or getattr(original, "__module__", None) != expected_definition_module
        or getattr(original, "__qualname__", None) != expected_definition_qualname
        or inspect.getmodule(original) is not definition_module
        or not isinstance(original_code, CodeType)
        or original_code_path != definition_path
        or original_code.co_firstlineno != expected_original_firstlineno
        or tuple(original_code.co_freevars) != ()
    ):
        fail(f"live {label} closure-original ownership differs")
    original_signature = inspect.signature(original, follow_wrapped=False)
    original_parameters = tuple(original_signature.parameters.values())
    original_annotations = getattr(original, "__annotations__", None)
    return_annotation = original_signature.return_annotation
    return_args = get_args(return_annotation)
    strict_pep604_union = (
        getattr(expected_return_union_origin, "__module__", None) == "types"
        and getattr(expected_return_union_origin, "__qualname__", None)
        == "UnionType"
    )
    secondary_annotation = return_args[1] if len(return_args) == 2 else None
    secondary_args = get_args(secondary_annotation)
    secondary_origin = get_origin(secondary_annotation)
    if (
        len(original_parameters) != 3
        or tuple(parameter.name for parameter in original_parameters)
        != ("self", expected_value_parameter, "return_dict")
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in original_parameters
        )
        or original_parameters[0].default is not inspect.Parameter.empty
        or original_parameters[1].default is not inspect.Parameter.empty
        or original_parameters[2].default is not True
        or original_parameters[1].annotation is not expected_value_annotation
        or original_parameters[2].annotation is not bool
        or type(original_annotations) is not dict
        or set(original_annotations)
        != {expected_value_parameter, "return_dict", "return"}
        or get_origin(return_annotation) is not expected_return_union_origin
        or (strict_pep604_union and type(return_annotation) is not types.UnionType)
        or len(return_args) != 2
        or return_args[0] is not expected_return_primary
        or (
            expected_return_secondary_container is None
            and secondary_annotation is not expected_return_secondary
        )
        or (
            expected_return_secondary_container is not None
            and (
                secondary_origin is not expected_return_secondary_container
                or secondary_args != (expected_return_secondary,)
                or (
                    strict_pep604_union
                    and (
                        type(secondary_annotation).__module__ != "types"
                        or type(secondary_annotation).__qualname__ != "GenericAlias"
                    )
                )
            )
        )
    ):
        fail(f"live {label} closure-original signature/annotations differ")
    original_receipt = dict(_canonical_callable_receipt(original, label=label))
    if (
        original_receipt.get("module") != expected_definition_module
        or original_receipt.get("qualname") != expected_definition_qualname
        or original_receipt.get("source_file") != str(definition_path)
        or original_receipt.get("source_file_sha256") != definition_sha
    ):
        fail(f"live {label} closure-original source differs")
    try:
        wrapper_source = inspect.getsource(wrapper).encode("utf-8")
    except (OSError, TypeError) as error:
        raise LevelBRendererError(f"cannot inspect live {label} wrapper source") from error
    return {
        "definition_class": (
            f"{expected_class.__module__}.{expected_class.__qualname__}"
        ),
        "method_name": method_name,
        "exact_mro_definition_owner_identity_verified": True,
        "exact_bound_instance_identity_verified": True,
        "exact_class_descriptor_identity_verified": True,
        "wrapper": {
            "module": str(getattr(wrapper, "__module__", "")),
            "qualname": str(getattr(wrapper, "__qualname__", "")),
            "executable_source_file": str(wrapper_path),
            "executable_source_file_sha256": wrapper_sha,
            "executable_source_file_identity": wrapper_identity,
            "callable_source_sha256": hashlib.sha256(wrapper_source).hexdigest(),
            "code_firstlineno": int(wrapper_code.co_firstlineno),
            "runtime_signature": _canonical_signature_receipt(
                wrapper, label=f"{label} wrapper"
            ),
            "factory": factory_receipt,
            "factory_code_firstlineno": int(factory_code.co_firstlineno),
            "factory_freevars": list(factory_code.co_freevars),
            "decorator_factory_nested_code_identity_count": (
                wrapper_code_owner_identity_count
            ),
            "closure_freevars": list(wrapper_code.co_freevars),
            "exact_single_original_closure_capability_verified": True,
            "no_dunder_wrapped_escape": True,
        },
        "closure_original": {
            **original_receipt,
            "source_file_identity": definition_identity,
            "code_firstlineno": int(original_code.co_firstlineno),
            "code_freevars": list(original_code.co_freevars),
            "exact_closure_identity_verified": True,
            "exact_parameter_annotation_identities_verified": True,
            "exact_union_annotation_component_identities_verified": True,
        },
        "all_executable_layers_owned_by_authenticated_source_bytes": True,
    }


def _audit_loaded_vae_callables(
    vae: Any,
    *,
    expected_vae_class: Any,
    runtime_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = runtime_receipt.get("vendor_source_files")
    logical = (
        "site-packages:diffusers/models/autoencoders/autoencoder_kl_wan.py"
    )
    authority = rows.get(logical) if isinstance(rows, Mapping) else None
    wrapper_logical = "site-packages:diffusers/utils/accelerate_utils.py"
    wrapper_authority = (
        rows.get(wrapper_logical) if isinstance(rows, Mapping) else None
    )
    module = sys.modules.get(
        "diffusers.models.autoencoders.autoencoder_kl_wan"
    )
    if (
        not isinstance(authority, Mapping)
        or not isinstance(wrapper_authority, Mapping)
        or not inspect.isclass(expected_vae_class)
        or type(vae) is not expected_vae_class
        or module is None
        or getattr(module, "AutoencoderKLWan", None) is not expected_vae_class
    ):
        fail("loaded VAE exact class authority differs")
    annotation_specs = (
        (
            "AutoencoderKLOutput",
            "site-packages:diffusers/models/modeling_outputs.py",
            "diffusers.models.modeling_outputs",
        ),
        (
            "DecoderOutput",
            "site-packages:diffusers/models/autoencoders/vae.py",
            "diffusers.models.autoencoders.vae",
        ),
        (
            "DiagonalGaussianDistribution",
            "site-packages:diffusers/models/autoencoders/vae.py",
            "diffusers.models.autoencoders.vae",
        ),
    )
    annotation_classes: dict[str, Any] = {}
    annotation_source_receipts: dict[str, Mapping[str, Any]] = {}
    for class_name, source_logical, module_name in annotation_specs:
        source_authority = (
            rows.get(source_logical) if isinstance(rows, Mapping) else None
        )
        source_module = sys.modules.get(module_name)
        source_class = getattr(source_module, class_name, None)
        try:
            module_path = (
                _plain_file(
                    getattr(source_module, "__file__", ""),
                    label=f"loaded VAE annotation module {module_name}",
                )
                if source_module is not None
                else None
            )
            class_source = (
                _plain_file(
                    inspect.getsourcefile(source_class) or "",
                    label=f"loaded VAE annotation class {class_name}",
                )
                if inspect.isclass(source_class)
                else None
            )
        except (OSError, RuntimeError) as error:
            raise LevelBRendererError(
                f"cannot resolve loaded VAE annotation class {class_name}"
            ) from error
        if (
            not isinstance(source_authority, Mapping)
            or set(source_authority) != {"path", "sha256", "file_identity"}
            or source_module is None
            or not inspect.isclass(source_class)
            or getattr(source_class, "__module__", None) != module_name
            or getattr(source_class, "__qualname__", None) != class_name
            or module_path != Path(source_authority["path"])
            or class_source != module_path
        ):
            fail(f"loaded VAE annotation class {class_name} authority differs")
        source_sha, source_identity = stable_file_sha256(
            module_path, label=f"loaded VAE annotation source {class_name}"
        )
        if (
            source_sha != source_authority["sha256"]
            or source_identity != source_authority["file_identity"]
        ):
            fail(f"loaded VAE annotation class {class_name} source differs")
        annotation_classes[class_name] = source_class
        annotation_source_receipts[class_name] = {
            "module": module_name,
            "qualname": class_name,
            "source_file": str(module_path),
            "source_file_sha256": source_sha,
            "source_file_identity": source_identity,
            "exact_module_export_identity_verified": True,
        }
    torch_module = sys.modules.get("torch")
    tensor_annotation = getattr(torch_module, "Tensor", None)
    union_origin = getattr(types, "UnionType", None)
    if not inspect.isclass(tensor_annotation) or union_origin is None:
        fail("loaded VAE torch/PEP604 annotation authority differs")
    receipts: dict[str, Any] = {}
    for name, value_parameter, original_firstlineno in (
        ("encode", "x", 1160),
        ("decode", "z", 1218),
    ):
        call = getattr(vae, name, None)
        if not inspect.ismethod(call) or getattr(call, "__self__", None) is not vae:
            fail(f"loaded VAE {name} is not one exact bound method")
        receipt = _audit_wrapped_instance_method_against_authenticated_sources(
            call,
            label=f"loaded VAE {name}",
            expected_instance=vae,
            expected_class=expected_vae_class,
            method_name=name,
            expected_value_parameter=value_parameter,
            expected_definition_module=(
                "diffusers.models.autoencoders.autoencoder_kl_wan"
            ),
            expected_definition_class_qualname="AutoencoderKLWan",
            expected_definition_qualname=f"AutoencoderKLWan.{name}",
            expected_definition_path=authority["path"],
            expected_definition_sha256=authority["sha256"],
            expected_original_firstlineno=original_firstlineno,
            expected_value_annotation=tensor_annotation,
            expected_return_union_origin=union_origin,
            expected_return_primary=(
                annotation_classes["AutoencoderKLOutput"]
                if name == "encode"
                else annotation_classes["DecoderOutput"]
            ),
            expected_return_secondary=(
                annotation_classes["DiagonalGaussianDistribution"]
                if name == "encode"
                else tensor_annotation
            ),
            expected_return_secondary_container=(tuple if name == "encode" else None),
            expected_wrapper_module="diffusers.utils.accelerate_utils",
            expected_wrapper_factory_qualname="apply_forward_hook",
            expected_wrapper_qualname="apply_forward_hook.<locals>.wrapper",
            expected_wrapper_path=wrapper_authority["path"],
            expected_wrapper_sha256=wrapper_authority["sha256"],
            expected_wrapper_factory_firstlineno=27,
            expected_wrapper_firstlineno=43,
        )
        if (
            receipt["closure_original"]["source_file_identity"]
            != authority["file_identity"]
            or receipt["wrapper"]["executable_source_file_identity"]
            != wrapper_authority["file_identity"]
        ):
            fail(f"loaded VAE {name} source identity changed after runtime admission")
        receipts[name] = receipt
    return {
        "exact_loaded_class": (
            f"{expected_vae_class.__module__}.{expected_vae_class.__qualname__}"
        ),
        "exact_loaded_class_identity_verified": True,
        "bound_callables": receipts,
        "annotation_class_authority": annotation_source_receipts,
        "pep604_union_annotation_structure_verified": True,
    }


def _audit_torch_no_grad_wrapped_instance_method(
    function: Any,
    *,
    label: str,
    expected_instance: Any,
    expected_class: Any,
    method_name: str,
    expected_definition_module: str,
    expected_definition_class_qualname: str,
    expected_definition_qualname: str,
    expected_definition_path: str | Path,
    expected_definition_sha256: str,
    expected_original_firstlineno: int,
    expected_original_parameter_names: Sequence[str],
    expected_original_defaults: tuple[Any, ...],
    expected_original_annotations: Mapping[str, Any],
    expected_contextlib_path: str | Path,
    expected_contextlib_sha256: str,
    expected_grad_mode_path: str | Path,
    expected_grad_mode_sha256: str,
    torch_module: Any,
    expected_torch_version: str = PINNED_TORCH_VERSION,
) -> Mapping[str, Any]:
    """Authenticate a bound ``@torch.no_grad()`` method, end to end.

    ``functools.wraps`` makes the executable context wrapper advertise the
    Bernini original's module and qualname.  This audit therefore binds both
    layers: exact Bernini descriptor/original bytes and exact Torch decorator,
    clone context and ``no_grad`` lifecycle bytes.  Closure capabilities and
    every descriptor/bound-owner relationship are identity checked.
    """

    if (
        not inspect.isclass(expected_class)
        or type(expected_instance) is not expected_class
        or not method_name
        or type(expected_torch_version) is not str
        or getattr(torch_module, "__version__", None) != expected_torch_version
        or sys.modules.get("torch") is not torch_module
    ):
        fail(f"live {label} torch no-grad authority differs")

    definition_path = _plain_file(
        expected_definition_path, label=f"pinned {label} definition source"
    )
    contextlib_path = _plain_file(
        expected_contextlib_path, label=f"pinned {label} torch context source"
    )
    grad_mode_path = _plain_file(
        expected_grad_mode_path, label=f"pinned {label} torch no-grad source"
    )
    definition_sha, definition_identity = stable_file_sha256(
        definition_path, label=f"pinned {label} definition source"
    )
    contextlib_sha, contextlib_identity = stable_file_sha256(
        contextlib_path, label=f"pinned {label} torch context source"
    )
    grad_mode_sha, grad_mode_identity = stable_file_sha256(
        grad_mode_path, label=f"pinned {label} torch no-grad source"
    )
    if (
        definition_sha
        != _require_sha(
            expected_definition_sha256, label=f"pinned {label} definition SHA"
        )
        or contextlib_sha
        != _require_sha(
            expected_contextlib_sha256, label=f"pinned {label} context SHA"
        )
        or grad_mode_sha
        != _require_sha(
            expected_grad_mode_sha256, label=f"pinned {label} no-grad SHA"
        )
    ):
        fail(f"live {label} torch no-grad source bytes differ")

    definition_module = sys.modules.get(expected_definition_module)
    contextlib_module = sys.modules.get("torch.utils._contextlib")
    grad_mode_module = sys.modules.get("torch.autograd.grad_mode")
    try:
        definition_module_path = _plain_file(
            getattr(definition_module, "__file__", ""),
            label=f"live {label} definition module",
        )
        contextlib_module_path = _plain_file(
            getattr(contextlib_module, "__file__", ""),
            label=f"live {label} torch context module",
        )
        grad_mode_module_path = _plain_file(
            getattr(grad_mode_module, "__file__", ""),
            label=f"live {label} torch no-grad module",
        )
    except (OSError, RuntimeError) as error:
        raise LevelBRendererError(
            f"cannot resolve live {label} torch no-grad modules"
        ) from error
    declared_class: Any = definition_module
    if definition_module is not None:
        try:
            for component in expected_definition_class_qualname.split("."):
                declared_class = getattr(declared_class, component)
        except (AttributeError, TypeError):
            declared_class = None
    decorator_base = getattr(contextlib_module, "_DecoratorContextManager", None)
    no_param_base = getattr(
        contextlib_module, "_NoParamDecoratorContextManager", None
    )
    no_grad_class = getattr(grad_mode_module, "no_grad", None)
    if (
        definition_module is None
        or contextlib_module is None
        or grad_mode_module is None
        or definition_module_path != definition_path
        or contextlib_module_path != contextlib_path
        or grad_mode_module_path != grad_mode_path
        or declared_class is not expected_class
        or getattr(expected_class, "__module__", None)
        != expected_definition_module
        or getattr(expected_class, "__qualname__", None)
        != expected_definition_class_qualname
        or not inspect.isclass(decorator_base)
        or not inspect.isclass(no_param_base)
        or not inspect.isclass(no_grad_class)
        or getattr(no_grad_class, "__module__", None)
        != "torch.autograd.grad_mode"
        or getattr(no_grad_class, "__qualname__", None) != "no_grad"
        or getattr(torch_module, "no_grad", None) is not no_grad_class
        or tuple(no_grad_class.__mro__[:4])
        != (no_grad_class, no_param_base, decorator_base, object)
    ):
        fail(f"live {label} module/class ownership differs")

    try:
        class_definers = tuple(
            owner for owner in expected_class.__mro__ if method_name in vars(owner)
        )
        descriptor = vars(expected_class).get(method_name)
        rebound = getattr(expected_instance, method_name)
    except (AttributeError, TypeError) as error:
        raise LevelBRendererError(
            f"cannot inspect live {label} descriptor ownership"
        ) from error
    if (
        class_definers != (expected_class,)
        or not inspect.isfunction(descriptor)
        or not inspect.ismethod(function)
        or not inspect.ismethod(rebound)
        or getattr(function, "__self__", None) is not expected_instance
        or getattr(rebound, "__self__", None) is not expected_instance
        or getattr(function, "__func__", None) is not descriptor
        or getattr(rebound, "__func__", None) is not descriptor
    ):
        fail(f"live {label} exact descriptor/bound-instance ownership differs")

    wrapper = descriptor
    wrapper_code = getattr(wrapper, "__code__", None)
    original = getattr(wrapper, "__wrapped__", None)
    original_code = getattr(original, "__code__", None)
    try:
        wrapper_code_path = (
            Path(wrapper_code.co_filename).resolve(strict=True)
            if isinstance(wrapper_code, CodeType)
            else None
        )
        original_code_path = (
            Path(original_code.co_filename).resolve(strict=True)
            if isinstance(original_code, CodeType)
            else None
        )
    except OSError as error:
        raise LevelBRendererError(
            f"cannot resolve live {label} wrapper/original source"
        ) from error
    if (
        not callable(original)
        or getattr(original, "__wrapped__", _NO_BOUND_OWNER)
        is not _NO_BOUND_OWNER
        or inspect.unwrap(wrapper) is not original
        or getattr(wrapper, "__module__", None) != expected_definition_module
        or getattr(wrapper, "__qualname__", None) != expected_definition_qualname
        or getattr(original, "__module__", None) != expected_definition_module
        or getattr(original, "__qualname__", None) != expected_definition_qualname
        or inspect.getmodule(wrapper) is not definition_module
        or inspect.getmodule(original) is not definition_module
        or not isinstance(wrapper_code, CodeType)
        or wrapper_code_path != contextlib_path
        or wrapper_code.co_firstlineno != 113
        or tuple(wrapper_code.co_freevars) != ("ctx_factory", "func")
        or not isinstance(original_code, CodeType)
        or original_code_path != definition_path
        or original_code.co_firstlineno != expected_original_firstlineno
        or tuple(original_code.co_freevars) != ()
    ):
        fail(f"live {label} wrapper/original executable ownership differs")
    wrapper_signature = inspect.signature(wrapper, follow_wrapped=False)
    wrapper_parameters = tuple(wrapper_signature.parameters.values())
    if (
        len(wrapper_parameters) != 2
        or wrapper_parameters[0].name != "args"
        or wrapper_parameters[0].kind is not inspect.Parameter.VAR_POSITIONAL
        or wrapper_parameters[1].name != "kwargs"
        or wrapper_parameters[1].kind is not inspect.Parameter.VAR_KEYWORD
        or wrapper_signature.return_annotation is not inspect.Signature.empty
        or wrapper.__defaults__ is not None
        or wrapper.__kwdefaults__ is not None
    ):
        fail(f"live {label} torch wrapper signature differs")

    closure = getattr(wrapper, "__closure__", None)
    try:
        closure_values = tuple(cell.cell_contents for cell in closure or ())
    except ValueError as error:
        raise LevelBRendererError(f"live {label} wrapper closure is empty") from error
    closure_by_name = dict(zip(wrapper_code.co_freevars, closure_values))
    if (
        len(closure_values) != 2
        or set(closure_by_name) != {"ctx_factory", "func"}
        or closure_by_name["func"] is not original
    ):
        fail(f"live {label} torch wrapper closure capabilities differ")

    context_factory = getattr(contextlib_module, "context_decorator", None)
    factory_receipt = _audit_callable_against_authenticated_source(
        context_factory,
        label=f"{label} torch context decorator",
        expected_module="torch.utils._contextlib",
        expected_qualname="context_decorator",
        expected_path=contextlib_path,
        expected_sha256=contextlib_sha,
    )
    factory_code = getattr(context_factory, "__code__", None)
    direct_nested_codes = tuple(
        constant
        for constant in getattr(factory_code, "co_consts", ())
        if isinstance(constant, CodeType)
    )
    nested_contract = tuple(
        (code.co_name, code.co_firstlineno, tuple(code.co_freevars))
        for code in direct_nested_codes
    )
    factory_signature = inspect.signature(context_factory, follow_wrapped=False)
    factory_parameters = tuple(factory_signature.parameters.values())
    if (
        not isinstance(factory_code, CodeType)
        or factory_code.co_firstlineno != 70
        or tuple(factory_code.co_freevars) != ()
        or tuple(parameter.name for parameter in factory_parameters)
        != ("ctx", "func")
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameter.default is not inspect.Parameter.empty
            for parameter in factory_parameters
        )
        or factory_signature.return_annotation is not inspect.Signature.empty
        or nested_contract
        != (
            ("ctx_factory", 95, ("ctx",)),
            ("decorate_context", 113, ("ctx_factory", "func")),
        )
        or sum(code is wrapper_code for code in direct_nested_codes) != 1
    ):
        fail(f"live {label} torch context decorator identity differs")

    ctx_factory = closure_by_name["ctx_factory"]
    context_owner = getattr(ctx_factory, "__self__", None)
    clone_descriptor = vars(decorator_base).get("clone")
    clone_definers = tuple(
        owner for owner in no_grad_class.__mro__ if "clone" in vars(owner)
    )
    if (
        not inspect.ismethod(ctx_factory)
        or type(context_owner) is not no_grad_class
        or type(getattr(context_owner, "__dict__", None)) is not dict
        or context_owner.__dict__ != {"prev": False}
        or clone_definers != (decorator_base,)
        or not inspect.isfunction(clone_descriptor)
        or getattr(ctx_factory, "__func__", None) is not clone_descriptor
    ):
        fail(f"live {label} no-grad context-factory owner differs")
    clone_receipt = _audit_callable_against_authenticated_source(
        ctx_factory,
        label=f"{label} no-grad clone",
        expected_module="torch.utils._contextlib",
        expected_qualname="_DecoratorContextManager.clone",
        expected_path=contextlib_path,
        expected_sha256=contextlib_sha,
        expected_bound_owner=context_owner,
    )
    clone_code = getattr(clone_descriptor, "__code__", None)
    clone_signature = inspect.signature(ctx_factory, follow_wrapped=False)
    if (
        not isinstance(clone_code, CodeType)
        or clone_code.co_firstlineno != 146
        or tuple(clone_code.co_freevars) != ()
        or tuple(clone_signature.parameters) != ()
        or clone_signature.return_annotation is not inspect.Signature.empty
        or getattr(clone_descriptor, "__wrapped__", _NO_BOUND_OWNER)
        is not _NO_BOUND_OWNER
    ):
        fail(f"live {label} no-grad clone executable identity differs")

    lifecycle_receipts: dict[str, Any] = {}
    lifecycle_specs = (
        ("__init__", 75, ("__class__",), ("self",), {"return": None}),
        ("__enter__", 80, (), ("self",), {"return": None}),
        (
            "__exit__",
            84,
            (),
            ("self", "exc_type", "exc_value", "traceback"),
            {
                "exc_type": Any,
                "exc_value": Any,
                "traceback": Any,
                "return": None,
            },
        ),
    )
    for name, firstlineno, freevars, parameter_names, annotations in lifecycle_specs:
        lifecycle = vars(no_grad_class).get(name)
        receipt = _audit_callable_against_authenticated_source(
            lifecycle,
            label=f"{label} no_grad.{name}",
            expected_module="torch.autograd.grad_mode",
            expected_qualname=f"no_grad.{name}",
            expected_path=grad_mode_path,
            expected_sha256=grad_mode_sha,
        )
        lifecycle_code = getattr(lifecycle, "__code__", None)
        lifecycle_signature = inspect.signature(lifecycle, follow_wrapped=False)
        parameters = tuple(lifecycle_signature.parameters.values())
        observed_annotations = getattr(lifecycle, "__annotations__", None)
        if (
            not isinstance(lifecycle_code, CodeType)
            or lifecycle_code.co_firstlineno != firstlineno
            or tuple(lifecycle_code.co_freevars) != freevars
            or getattr(lifecycle, "__wrapped__", _NO_BOUND_OWNER)
            is not _NO_BOUND_OWNER
            or tuple(parameter.name for parameter in parameters) != parameter_names
            or any(
                parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
                or parameter.default is not inspect.Parameter.empty
                for parameter in parameters
            )
            or type(observed_annotations) is not dict
            or set(observed_annotations) != set(annotations)
            or any(
                observed_annotations[key] is not expected
                for key, expected in annotations.items()
            )
        ):
            fail(f"live {label} no_grad.{name} lifecycle identity differs")
        lifecycle_receipts[name] = {
            **receipt,
            "code_firstlineno": firstlineno,
            "code_freevars": list(freevars),
        }

    original_signature = inspect.signature(original, follow_wrapped=False)
    original_parameters = tuple(original_signature.parameters.values())
    observed_annotations = getattr(original, "__annotations__", None)
    if (
        tuple(parameter.name for parameter in original_parameters)
        != tuple(expected_original_parameter_names)
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in original_parameters
        )
        or original_signature.return_annotation is not inspect.Signature.empty
        or not _same_exact_runtime_literal(
            getattr(original, "__defaults__", None), expected_original_defaults
        )
        or getattr(original, "__kwdefaults__", None) is not None
        or type(observed_annotations) is not dict
        or set(observed_annotations) != set(expected_original_annotations)
        or any(
            observed_annotations[key] is not expected
            for key, expected in expected_original_annotations.items()
        )
    ):
        fail(f"live {label} closure-original signature/defaults differ")
    original_receipt = dict(_canonical_callable_receipt(original, label=label))
    if (
        original_receipt.get("module") != expected_definition_module
        or original_receipt.get("qualname") != expected_definition_qualname
        or original_receipt.get("source_file") != str(definition_path)
        or original_receipt.get("source_file_sha256") != definition_sha
    ):
        fail(f"live {label} closure-original source differs")

    return {
        **original_receipt,
        "source_file_identity": definition_identity,
        "code_firstlineno": int(original_code.co_firstlineno),
        "code_freevars": list(original_code.co_freevars),
        "exact_mro_definition_owner_identity_verified": True,
        "exact_bound_instance_identity_verified": True,
        "exact_class_descriptor_identity_verified": True,
        "wrapper": {
            "advertised_module": str(getattr(wrapper, "__module__", "")),
            "advertised_qualname": str(getattr(wrapper, "__qualname__", "")),
            "executable_source_file": str(contextlib_path),
            "executable_source_file_sha256": contextlib_sha,
            "executable_source_file_identity": contextlib_identity,
            "code_firstlineno": int(wrapper_code.co_firstlineno),
            "code_freevars": list(wrapper_code.co_freevars),
            "runtime_signature": _canonical_signature_receipt(
                wrapper, label=f"{label} raw torch wrapper"
            ),
            "context_decorator": factory_receipt,
            "context_decorator_direct_nested_codes": [
                [name, line, list(freevars)]
                for name, line, freevars in nested_contract
            ],
            "decorator_factory_nested_code_identity_count": 1,
            "one_hop_wrapped_target_identity_verified": True,
            "exact_closure_capabilities_verified": True,
        },
        "no_grad_context": {
            "class": "torch.autograd.grad_mode.no_grad",
            "source_file": str(grad_mode_path),
            "source_file_sha256": grad_mode_sha,
            "source_file_identity": grad_mode_identity,
            "exact_torch_module_export_identity_verified": True,
            "exact_context_owner_type_verified": True,
            "fresh_context_state": {"prev": False},
            "clone": clone_receipt,
            "lifecycle": lifecycle_receipts,
        },
        "torch_version": expected_torch_version,
        "all_executable_layers_owned_by_authenticated_source_bytes": True,
    }


def _audit_renderer_sample_pair_against_runtime(
    *,
    renderer: Any,
    diffusion: Any,
    runtime_receipt: Mapping[str, Any],
    torch_module: Any,
) -> Mapping[str, Any]:
    rows = runtime_receipt.get("vendor_source_files")
    if not isinstance(rows, Mapping):
        fail("authenticated vendor source rows are absent before sample audit")

    def authority(logical_name: str) -> Mapping[str, Any]:
        value = rows.get(logical_name)
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "file_identity",
        }:
            fail(f"authenticated sample source row differs: {logical_name}")
        return value

    renderer_authority = authority("bernini:bernini/models/renderer.py")
    wan_authority = authority("bernini:bernini/models/wan_diffusion.py")
    context_authority = authority("site-packages:torch/utils/_contextlib.py")
    grad_authority = authority("site-packages:torch/autograd/grad_mode.py")
    renderer_receipt = _audit_torch_no_grad_wrapped_instance_method(
        renderer.sample,
        label="official renderer sample",
        expected_instance=renderer,
        expected_class=type(renderer),
        method_name="sample",
        expected_definition_module="bernini.models.renderer",
        expected_definition_class_qualname="BerniniRendererModel",
        expected_definition_qualname="BerniniRendererModel.sample",
        expected_definition_path=renderer_authority["path"],
        expected_definition_sha256=renderer_authority["sha256"],
        expected_original_firstlineno=319,
        expected_original_parameter_names=("self", *RENDERER_SAMPLE_PARAMETERS),
        expected_original_defaults=RENDERER_SAMPLE_DEFAULTS,
        expected_original_annotations=RENDERER_SAMPLE_ANNOTATIONS,
        expected_contextlib_path=context_authority["path"],
        expected_contextlib_sha256=context_authority["sha256"],
        expected_grad_mode_path=grad_authority["path"],
        expected_grad_mode_sha256=grad_authority["sha256"],
        torch_module=torch_module,
    )
    diffusion_receipt = _audit_torch_no_grad_wrapped_instance_method(
        diffusion.sample,
        label="native diffusion sample",
        expected_instance=diffusion,
        expected_class=type(diffusion),
        method_name="sample",
        expected_definition_module="bernini.models.wan_diffusion",
        expected_definition_class_qualname="GEN_Wanx22",
        expected_definition_qualname="GEN_Wanx22.sample",
        expected_definition_path=wan_authority["path"],
        expected_definition_sha256=wan_authority["sha256"],
        expected_original_firstlineno=274,
        expected_original_parameter_names=(
            "self",
            *INTERNAL_DIFFUSION_SAMPLE_PARAMETERS,
        ),
        expected_original_defaults=INTERNAL_DIFFUSION_SAMPLE_DEFAULTS,
        expected_original_annotations={},
        expected_contextlib_path=context_authority["path"],
        expected_contextlib_sha256=context_authority["sha256"],
        expected_grad_mode_path=grad_authority["path"],
        expected_grad_mode_sha256=grad_authority["sha256"],
        torch_module=torch_module,
    )
    for label, receipt, definition_authority in (
        ("official renderer sample", renderer_receipt, renderer_authority),
        ("native diffusion sample", diffusion_receipt, wan_authority),
    ):
        if (
            receipt.get("source_file_identity")
            != definition_authority["file_identity"]
            or receipt.get("wrapper", {}).get("executable_source_file_identity")
            != context_authority["file_identity"]
            or receipt.get("no_grad_context", {}).get("source_file_identity")
            != grad_authority["file_identity"]
        ):
            fail(f"{label} source identity changed after runtime admission")
    return {
        "renderer_sample": renderer_receipt,
        "native_diffusion_sample": diffusion_receipt,
        "torch_version": str(torch_module.__version__),
        "exact_no_grad_sample_count": 2,
        "all_sample_executable_layers_authenticated": True,
    }


def _audit_loaded_tokenizer_callable(
    tokenizer: Any,
    *,
    expected_tokenizer_class: Any,
    runtime_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = runtime_receipt.get("vendor_source_files")
    logical = "site-packages:transformers/tokenization_utils_base.py"
    authority = rows.get(logical) if isinstance(rows, Mapping) else None
    class_logical = "site-packages:transformers/models/t5/tokenization_t5.py"
    class_authority = (
        rows.get(class_logical) if isinstance(rows, Mapping) else None
    )
    call = getattr(tokenizer, "__call__", None)
    class_module = sys.modules.get("transformers.models.t5.tokenization_t5")
    if (
        not isinstance(authority, Mapping)
        or not isinstance(class_authority, Mapping)
        or not inspect.isclass(expected_tokenizer_class)
        or type(tokenizer) is not expected_tokenizer_class
        or class_module is None
        or getattr(class_module, "T5Tokenizer", None) is not expected_tokenizer_class
        or not inspect.ismethod(call)
        or getattr(call, "__self__", None) is not tokenizer
    ):
        fail("loaded tokenizer callable authority is absent")
    receipt = _audit_callable_against_authenticated_source(
        call,
        label="loaded tokenizer __call__",
        expected_module="transformers.tokenization_utils_base",
        expected_qualname="PreTrainedTokenizerBase.__call__",
        expected_path=authority["path"],
        expected_sha256=authority["sha256"],
        expected_bound_owner=tokenizer,
    )
    if receipt["source_file_identity"] != authority["file_identity"]:
        fail("loaded tokenizer source identity changed after runtime admission")
    class_source_raw = inspect.getsourcefile(expected_tokenizer_class)
    if not isinstance(class_source_raw, str) or not class_source_raw:
        fail("loaded tokenizer exact T5 class has no source provenance")
    class_path = _plain_file(
        class_authority["path"], label="loaded T5 tokenizer class source"
    )
    class_source = _plain_file(
        Path(class_source_raw).resolve(strict=True),
        label="loaded T5 tokenizer live class source",
    )
    class_sha, class_identity = stable_file_sha256(
        class_path, label="loaded T5 tokenizer class source"
    )
    if (
        expected_tokenizer_class.__module__
        != "transformers.models.t5.tokenization_t5"
        or expected_tokenizer_class.__qualname__ != "T5Tokenizer"
        or class_source != class_path
        or class_sha != class_authority["sha256"]
        or class_identity != class_authority["file_identity"]
    ):
        fail("loaded tokenizer exact T5 class source identity differs")
    return {
        **dict(receipt),
        "exact_loaded_class": (
            f"{expected_tokenizer_class.__module__}."
            f"{expected_tokenizer_class.__qualname__}"
        ),
        "exact_loaded_class_identity_verified": True,
        "exact_loaded_class_source_sha256": class_sha,
    }


def validate_public_product_signature() -> Mapping[str, Any]:
    names = tuple(inspect.signature(run_level_b_pre_d0_offline_inference).parameters)
    bad = sorted(
        name
        for name in names
        if any(fragment in name.lower() for fragment in FORBIDDEN_PUBLIC_ARGUMENT_FRAGMENTS)
    )
    if bad:
        fail(f"Level-B public API exposes forbidden inputs: {bad}")
    if names != (
        "fresh_bundle",
        "verified_runtime",
        "source_video_path",
        "expected_source_video_sha256",
        "edit_instruction",
        "inference_seed",
        "output_mp4_path",
    ):
        fail("Level-B public product signature field set/order differs")
    return {
        "function": "run_level_b_pre_d0_offline_inference",
        "parameters": list(names),
        "source_instruction_seed_are_only_semantic_inputs": True,
        "target_anchor_teacher_callback_accepted": False,
    }


@dataclass
class _BranchContext:
    kind: str
    route: Any = None
    calls: list[int] = field(default_factory=list)


_ACTIVE_BRANCH: ContextVar[Optional[_BranchContext]] = ContextVar(
    "bernini_action_edit_level_b_renderer_branch_v1", default=None
)


@dataclass
class _ActiveSample:
    action_prompt: Any
    negative_prompt: Any
    expected_seed: int
    pending_negative: Optional[Mapping[str, Any]] = None
    pending_action_receipt: Optional[Mapping[str, Any]] = None
    integrated_steps: int = 0


@dataclass
class InstalledNativeActionRendererBridge:
    """One-shot, branch-aware owner of native sample/shared-step/UniPC calls."""

    bundle: Any
    product_module: Any
    inference_policy: Any
    patch_grid: tuple[int, int, int]
    row_identity: str
    torch_module: Any
    noise_observer: Any
    source_condition_list: list[Any]
    expected_width: int
    expected_height: int
    expected_device: Any
    expected_internal_sampling: Mapping[str, Any]
    expected_instruction_token_count: int
    handles: tuple[Any, ...] = field(default_factory=tuple)
    route_receipts: list[Mapping[str, Any]] = field(default_factory=list)
    target_state_sha256: list[str] = field(default_factory=list)
    negative_passthrough_calls: int = 0
    action_injection_calls: int = 0
    _patches: list[tuple[Any, str, bool, Any, Any]] = field(default_factory=list)
    _installed_wrappers: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    _active: Optional[_ActiveSample] = None
    _source_prefix_reference: Any = None
    _source_prefix_sha256: Optional[str] = None
    _instruction_object: Any = None
    _instruction_reference: Any = None
    _instruction_sha256: Optional[str] = None
    _instruction_length: Optional[int] = None
    _instruction_padded_length: Optional[int] = None
    _scheduler_receipt: Optional[Mapping[str, Any]] = None
    _scheduler_callable_authority: Optional[Mapping[str, Any]] = None
    _level_a_hook_rows: tuple[tuple[Any, int, Any], ...] = field(
        default_factory=tuple
    )
    _installed: bool = False

    def __post_init__(self) -> None:
        torch = self.torch_module
        if (
            len(self.patch_grid) != 3
            or tuple(int(x) for x in self.patch_grid)[0] != PHASES
            or any(int(x) <= 0 for x in self.patch_grid)
            or type(self.inference_policy.seed) is not int
            or self.inference_policy.seed < 0
            or not self.row_identity
            or type(self.expected_width) is not int
            or type(self.expected_height) is not int
            or self.expected_width <= 0
            or self.expected_height <= 0
            or not isinstance(self.source_condition_list, list)
            or len(self.source_condition_list) != 1
            or not isinstance(self.expected_internal_sampling, Mapping)
            or type(self.expected_instruction_token_count) is not int
            or not 0 < self.expected_instruction_token_count <= 512
        ):
            fail("Level-B renderer bridge geometry/policy differs")
        self.patch_grid = tuple(int(x) for x in self.patch_grid)
        self.inference_policy.validate()
        if (
            self.inference_policy.num_inference_steps != NUM_INFERENCE_STEPS
            or self.inference_policy.flow_shift != FLOW_SHIFT
            or self.inference_policy.schedule_sha256 != PINNED_UNIPC_SCHEDULE_SHA256
        ):
            fail("Level-B exact40 inference policy differs")
        transformer = self.bundle.transformer
        conditioner = self.bundle.conditioner
        offline = self.bundle.offline_hooks
        diffusion = self.bundle.renderer.diff_dec
        scheduler = getattr(diffusion, "scheduler", None)
        self._original_sample = getattr(diffusion, "sample", None)
        self._original_shared_step = getattr(diffusion, "shared_step", None)
        self._original_scheduler_step = getattr(scheduler, "step", None)
        if any(
            not callable(value)
            for value in (
                self._original_sample,
                self._original_shared_step,
                self._original_scheduler_step,
            )
        ):
            fail("fresh Bernini renderer lacks native sample/shared_step/UniPC")
        self._scheduler_callable_authority = _audit_native_unipc_scheduler_callable(
            scheduler
        )
        if tuple(inspect.signature(self._original_sample).parameters) != (
            INTERNAL_DIFFUSION_SAMPLE_PARAMETERS
        ):
            fail("native Bernini internal sample signature differs")
        shared_step_signature = inspect.signature(self._original_shared_step)
        shared_step_parameters = tuple(shared_step_signature.parameters.values())
        if tuple(parameter.name for parameter in shared_step_parameters) != (
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
            "kwargs",
        ) or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in shared_step_parameters[:-1]
        ) or shared_step_parameters[-1].kind is not inspect.Parameter.VAR_KEYWORD:
            fail("native Bernini shared_step signature differs")
        if tuple(inspect.signature(self._original_scheduler_step).parameters) != (
            "model_output",
            "timestep",
            "sample",
            "return_dict",
        ):
            fail("native UniPC step signature differs")
        if getattr(diffusion, "use_unipc", None) is not True:
            fail("fresh Bernini renderer is not configured for UniPC")
        for owner, name in (
            (diffusion, "sample"),
            (diffusion, "shared_step"),
            (scheduler, "step"),
        ):
            try:
                if name in vars(owner):
                    fail(f"refusing to stack on an existing {name} instance wrapper")
            except TypeError as error:
                raise LevelBRendererError(f"cannot inspect native {name} owner") from error
        if (
            getattr(offline, "transformer", None) is not transformer
            or getattr(offline, "conditioner", None) is not conditioner
            or getattr(offline, "restored", None) is not False
            or len(tuple(getattr(offline, "handles", ()))) != TRANSFORMER_BLOCKS
        ):
            fail("fresh bundle Level-A hook ownership differs")
        blocks = tuple(getattr(transformer, "blocks", ()))
        if len(blocks) != TRANSFORMER_BLOCKS:
            fail("fresh Bernini transformer is not exact30")
        if getattr(transformer, "action_plan_conditioner_v1", None) is not conditioner:
            fail("fresh Level-A conditioner registration differs")
        self.handles = tuple(offline.handles)
        factory = getattr(
            self.product_module, "install_offline_action_plan_hooks", None
        )
        nested_callbacks = (
            [
                value
                for value in getattr(getattr(factory, "__code__", None), "co_consts", ())
                if isinstance(value, CodeType) and value.co_name == "callback"
            ]
            if callable(factory)
            else []
        )
        if len(nested_callbacks) != 1:
            fail("authenticated Level-A hook factory code ownership differs")
        expected_callback_code = nested_callbacks[0]
        rows: list[tuple[Any, int, Any]] = []
        for block_index, (block, handle) in enumerate(zip(blocks, self.handles)):
            registry = getattr(block, "_forward_hooks", None)
            hook_id = getattr(handle, "id", None)
            if (
                not isinstance(registry, Mapping)
                or type(hook_id) is not int
                or set(registry) != {hook_id}
                or not callable(registry.get(hook_id))
            ):
                fail(f"fresh Level-A exact hook ownership differs at block {block_index}")
            callback = registry[hook_id]
            callback_code = getattr(callback, "__code__", None)
            closure = {
                name: cell.cell_contents
                for name, cell in zip(
                    getattr(callback_code, "co_freevars", ()),
                    getattr(callback, "__closure__", ()) or (),
                )
            }
            if (
                getattr(callback, "__module__", None)
                != self.product_module.__name__
                or callback_code is not expected_callback_code
                or getattr(callback, "__kwdefaults__", None)
                != {"bound_index": block_index}
                or set(closure) != {"conditioner", "torch"}
                or closure.get("conditioner") is not conditioner
                or closure.get("torch") is not torch
            ):
                fail(f"fresh Level-A exact callback provenance differs at block {block_index}")
            rows.append((block, hook_id, callback))
        self._level_a_hook_rows = tuple(rows)


    @property
    def diffusion(self) -> Any:
        return self.bundle.renderer.diff_dec

    @property
    def scheduler(self) -> Any:
        return self.diffusion.scheduler

    @property
    def source_tokens(self) -> int:
        return math.prod(self.patch_grid)

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had, previous, value))
        self._installed_wrappers[name] = (owner, value)

    def _assert_installed_wrapper_identities(self) -> None:
        if not self._installed or set(self._installed_wrappers) != {
            "sample",
            "shared_step",
            "step",
        }:
            fail("Level-B installed wrapper registry differs")
        for name, (owner, expected) in self._installed_wrappers.items():
            if getattr(owner, name, None) is not expected:
                fail(f"installed Level-B {name} wrapper identity changed")

    @contextmanager
    def _negative_native_passthrough(
        self, context: _BranchContext
    ) -> Iterator[None]:
        """Swap only the exact Level-A hook slots for closed negative guards.

        The original hook callable objects are restored before the action
        forward.  A concurrent or re-entrant block call sees a guard without
        this task's ContextVar and fails; it cannot exploit a hook-free window.
        """

        torch = self.torch_module
        guards: list[tuple[Any, int, Any, Any]] = []
        try:
            for block_index, (block, hook_id, original) in enumerate(
                self._level_a_hook_rows
            ):
                registry = block._forward_hooks
                if registry.get(hook_id) is not original:
                    fail("Level-A hook callable changed before negative branch")

                def guard(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    bound_index: int = block_index,
                ) -> Any:
                    current = _ACTIVE_BRANCH.get()
                    native, _ = _output_tensor(output, torch_module=torch)
                    expected_local = math.ceil(
                        (2 * self.source_tokens)
                        / int(self.bundle.distributed.topology.sp_size)
                    )
                    if (
                        current is not context
                        or self._active is None
                        or context.kind != "negative"
                        or bound_index != len(context.calls)
                        or native.ndim != 3
                        or int(native.shape[0]) != 1
                        or int(native.shape[1]) != expected_local
                        or int(native.shape[2])
                        != int(self.bundle.conditioner.renderer_hidden_width)
                        or not native.is_floating_point()
                        or not bool(torch.isfinite(native).all().item())
                    ):
                        fail("negative native pass-through guard differs")
                    context.calls.append(bound_index)
                    self.negative_passthrough_calls += 1
                    return None

                registry[hook_id] = guard
                guards.append((block, hook_id, original, guard))
            yield
        finally:
            errors = []
            for block, hook_id, original, guard in guards:
                registry = block._forward_hooks
                if registry.get(hook_id) is not guard:
                    errors.append(hook_id)
                else:
                    registry[hook_id] = original
            if errors:
                fail("negative guard hook slot changed before restoration")

    @contextmanager
    def _branch(self, context: _BranchContext) -> Iterator[None]:
        if _ACTIVE_BRANCH.get() is not None:
            fail("nested Level-B renderer branches are forbidden")
        token: Token[Optional[_BranchContext]] = _ACTIVE_BRANCH.set(context)
        try:
            yield
        finally:
            _ACTIVE_BRANCH.reset(token)

    def _validate_shared_identity(
        self, negative: Mapping[str, Any], action: Mapping[str, Any]
    ) -> None:
        expected_fields = (
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
        )
        if tuple(negative) != expected_fields or tuple(action) != expected_fields:
            fail("negative/action shared_step complete call field order differs")
        if str(negative.get("model_id")) != "transformer_1" or str(
            action.get("model_id")
        ) != "transformer_1":
            fail("Level-B requires the single Bernini transformer_1 expert")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            _same_object(negative.get(name), action.get(name), label=name)
        _equal_metadata(
            negative.get("batch_vae_seqlen"),
            action.get("batch_vae_seqlen"),
            label="batch_vae_seqlen",
        )
        vae_lengths = action.get("batch_vae_seqlen")
        if (
            type(vae_lengths) is not list
            or len(vae_lengths) != 1
            or type(vae_lengths[0]) is not int
            or vae_lengths[0] != 2 * self.source_tokens
        ):
            fail("native Bernini VAE sequence-length metadata differs")
        _equal_metadata(
            negative.get("batch_text_seqlen"),
            action.get("batch_text_seqlen"),
            label="batch_text_seqlen",
        )

    def _prepare_route(self, bound: Mapping[str, Any]) -> Any:
        torch = self.torch_module
        embedded = bound.get("noisy_latents")
        prompt = bound.get("cond_embeds")
        if (
            not isinstance(embedded, torch.Tensor)
            or embedded.ndim != 3
            or tuple(int(x) for x in embedded.shape[:2])
            != (1, 2 * self.source_tokens)
            or not isinstance(prompt, torch.Tensor)
            or prompt.ndim != 3
            or int(prompt.shape[0]) != 1
            or int(prompt.shape[-1])
            != int(self.bundle.conditioner.config.instruction_token_width)
            or prompt.device != embedded.device
            or not bool(torch.isfinite(embedded).all().item())
            or not bool(torch.isfinite(prompt).all().item())
        ):
            fail("live Bernini pre-SP embedding/T5 geometry differs")
        source_prefix = embedded[:, : self.source_tokens, :]
        if self._source_prefix_reference is None:
            self._source_prefix_reference = source_prefix.detach().clone()
            self._source_prefix_sha256 = tensor_sha256(
                source_prefix, torch_module=torch
            )
        elif not _bits_equal(
            source_prefix, self._source_prefix_reference, torch_module=torch
        ):
            fail("clean-source pre-SP prefix changed across denoise steps")
        padded_metadata = bound.get("batch_text_seqlen")
        padded_length = int(prompt.shape[1])
        if (
            type(padded_metadata) is not list
            or len(padded_metadata) != 1
            or type(padded_metadata[0]) is not int
            or padded_metadata[0] != padded_length
            or padded_length != 512
            or self.expected_instruction_token_count > padded_length
        ):
            fail("native Bernini contextual T5 padded-length metadata differs")
        length = self.expected_instruction_token_count
        padding = prompt[:, length:, :].detach().contiguous()
        if not _bits_equal(
            padding, torch.zeros_like(padding), torch_module=torch
        ):
            fail("native Bernini contextual T5 padding is not bit-exact zero")
        instruction = prompt[:, :length, :].detach().contiguous()
        if instruction.requires_grad:
            fail("actual-length contextual T5 tokens require gradients")
        if self._instruction_object is None:
            self._instruction_object = prompt
            self._instruction_reference = instruction.clone()
            self._instruction_length = length
            self._instruction_padded_length = padded_length
            self._instruction_sha256 = tensor_sha256(
                instruction, torch_module=torch
            )
        elif (
            prompt is not self._instruction_object
            or length != self._instruction_length
            or padded_length != self._instruction_padded_length
            or not _bits_equal(
                instruction, self._instruction_reference, torch_module=torch
            )
        ):
            fail("contextual T5 instruction tokens changed across denoise steps")
        distributed = self.bundle.distributed
        route = self.product_module.prepare_product_route_from_packed_embeddings(
            conditioner=self.bundle.conditioner,
            predictor_module=self.bundle.checkpoint.predictor_module,
            packed_embeddings=embedded,
            instruction_tokens=instruction,
            source_token_count=self.source_tokens,
            patch_grid=self.patch_grid,
            sequence_parallel_rank=int(distributed.sp_rank),
            sequence_parallel_size=int(distributed.topology.sp_size),
            row_identity=self.row_identity,
            torch_module=torch,
        )
        return route

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        self._assert_installed_wrapper_identities()
        state = self._active
        if state is None:
            fail("shared_step ran outside one authenticated Level-B sample")
        bound = _bind_call(self._original_shared_step, args, kwargs)
        if tuple(bound) != (
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
            "kwargs",
        ):
            fail("native Bernini shared_step complete call contract differs")
        extra_kwargs = bound.get("kwargs")
        if type(extra_kwargs) is not dict or extra_kwargs:
            fail("native Bernini shared_step variadic kwargs are not exactly empty")
        del bound["kwargs"]
        prompt = bound.get("cond_embeds")
        if state.pending_negative is None:
            if prompt is not state.negative_prompt:
                fail("first native shared_step is not the exact negative prompt")
            context = _BranchContext(kind="negative")
            with self._branch(context):
                with self._negative_native_passthrough(context):
                    result = self._original_shared_step(*args, **kwargs)
            self._assert_installed_wrapper_identities()
            if context.calls != list(range(TRANSFORMER_BLOCKS)):
                fail("negative native pass-through did not traverse exact30")
            state.pending_negative = dict(bound)
            return result
        if state.pending_action_receipt is not None:
            fail("more than negative/action shared_step calls occurred before UniPC")
        if prompt is not state.action_prompt:
            fail("second native shared_step is not the exact action prompt")
        self._validate_shared_identity(state.pending_negative, bound)
        route = self._prepare_route(bound)
        with self.product_module.activate_offline_route(route):
            result = self._original_shared_step(*args, **kwargs)
        self._assert_installed_wrapper_identities()
        receipt = route.finish()
        if (
            receipt.get("exact_block_indices") != list(range(TRANSFORMER_BLOCKS))
            or receipt.get("source_and_padding_bit_exact_under_injection") is not True
        ):
            fail("action branch did not invoke the exact30 target-only heads once")
        self.action_injection_calls += TRANSFORMER_BLOCKS
        state.pending_action_receipt = dict(receipt)
        return result

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        self._assert_installed_wrapper_identities()
        state = self._active
        if (
            state is None
            or state.pending_negative is None
            or state.pending_action_receipt is None
        ):
            fail("UniPC arrived before one complete negative/action branch pair")
        if len(self.route_receipts) >= NUM_INFERENCE_STEPS:
            fail("native UniPC executed more than exact40 steps")
        bound = _bind_call(self._original_scheduler_step, args, kwargs)
        if tuple(bound) != ("model_output", "timestep", "sample", "return_dict"):
            fail("native UniPC complete call contract differs")
        timestep = _scalar_int(bound.get("timestep"), label="runtime UniPC timestep")
        step_index = len(self.route_receipts)
        if timestep != PINNED_UNIPC_TIMESTEPS[step_index]:
            fail("runtime UniPC timestep differs from the pinned exact40 grid")
        scheduler_receipt = self.product_module.audit_live_inference_scheduler(
            scheduler=self.scheduler,
            inference_policy=self.inference_policy,
            sigma_contract_module=self.bundle.sigma_contract_module,
            initialize=False,
        )
        if scheduler_receipt.get("schedule_sha256") != PINNED_UNIPC_SCHEDULE_SHA256:
            fail("live UniPC schedule changed before scheduler.step")
        if self._scheduler_receipt is None:
            self._scheduler_receipt = scheduler_receipt
        elif scheduler_receipt != self._scheduler_receipt:
            fail("live UniPC schedule receipt changed across denoise steps")
        sample = bound.get("sample")
        model_output = bound.get("model_output")
        torch = self.torch_module
        if (
            not isinstance(sample, torch.Tensor)
            or tuple(int(x) for x in sample.shape)
            != (1, self.source_tokens, PATCH_VALUES)
            or not bool(torch.isfinite(sample).all().item())
            or not isinstance(model_output, torch.Tensor)
            or tuple(int(x) for x in model_output.shape)
            != (1, self.source_tokens, PATCH_VALUES)
            or model_output.device != sample.device
            or not model_output.is_floating_point()
            or not bool(torch.isfinite(model_output).all().item())
            or bound.get("return_dict") is not False
        ):
            fail("native UniPC complete tensor/call contract differs")
        state_sha = tensor_sha256(sample, torch_module=torch)
        if state_sha in self.target_state_sha256:
            fail("native noisy target state did not evolve at one exact40 step")
        if step_index == 0:
            expected_initial = self.noise_observer.packed_for(sample.device)
            if not _bits_equal(sample, expected_initial, torch_module=torch):
                fail("first native UniPC state differs from observed official Gaussian")
        result = self._original_scheduler_step(*args, **kwargs)
        self._assert_installed_wrapper_identities()
        row = {
            **dict(state.pending_action_receipt),
            "inference_step_index": step_index,
            "runtime_timestep": timestep,
            "scheduler_schedule_sha256": PINNED_UNIPC_SCHEDULE_SHA256,
            "negative_native_passthrough_blocks": list(range(TRANSFORMER_BLOCKS)),
            "action_residual_head_calls": TRANSFORMER_BLOCKS,
            "original_unipc_scheduler_step_calls": 1,
            "evolving_target_state_sha256": state_sha,
            "clean_source_prefix_bit_exact_across_steps": True,
            "actual_length_contextual_t5_tokens": self._instruction_length,
            "native_padded_contextual_t5_tokens": self._instruction_padded_length,
            "native_batch_vae_seqlen": [2 * self.source_tokens],
        }
        self.route_receipts.append(row)
        self.target_state_sha256.append(state_sha)
        state.integrated_steps += 1
        state.pending_negative = None
        state.pending_action_receipt = None
        return result

    def _validate_internal_sample_contract(self, values: Mapping[str, Any]) -> None:
        if tuple(values) != INTERNAL_DIFFUSION_SAMPLE_PARAMETERS:
            fail("native Bernini internal sampler complete field order differs")
        expected_sampling = dict(self.expected_internal_sampling)
        expected_sampling_fields = {
            "num_frames",
            "num_inference_steps",
            "guidance_mode",
            "omega_vid",
            "omega_img",
            "omega_txt",
            "omega_scale",
            "flow_shift",
            "seed",
            "eta",
            "norm_threshold",
            "momentum",
        }
        if set(expected_sampling) != expected_sampling_fields:
            fail("expected internal sampling authority field closure differs")
        for name in expected_sampling_fields:
            observed = values.get(name)
            expected = expected_sampling[name]
            if name == "norm_threshold":
                try:
                    observed = tuple(observed)
                    expected = tuple(expected)
                except TypeError as error:
                    raise LevelBRendererError(
                        "native norm-threshold contract differs"
                    ) from error
            if observed != expected:
                fail(f"native Bernini internal sampler {name} differs")
        conditions = values.get("multi_video_vae_latents")
        if (
            values.get("prompt_embeds_t2") is not None
            or values.get("uncond_embeds_t2") is not None
            or values.get("image_vae_latents") is not None
            or values.get("multi_image_vae_latents") is not None
            or conditions is not self.source_condition_list
            or len(conditions) != 1
            or conditions[0] is not self.source_condition_list[0]
            or values.get("width") != self.expected_width
            or values.get("height") != self.expected_height
            or str(self.torch_module.device(values.get("device")))
            != str(self.torch_module.device(self.expected_device))
        ):
            fail("native Bernini internal source/geometry/device contract differs")

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        self._assert_installed_wrapper_identities()
        if self._active is not None:
            fail("nested/concurrent native Bernini sample calls are forbidden")
        if self.route_receipts or self.target_state_sha256:
            fail("Level-B renderer bridge permits exactly one native sample call")
        values = _bind_call(self._original_sample, args, kwargs)
        self._validate_internal_sample_contract(values)
        if getattr(self.diffusion, "transformer_2", None) is not None:
            fail("native Bernini sample left the single-expert contract")
        action = values.get("prompt_embeds")
        negative = values.get("uncond_prompt_embeds")
        if action is None or negative is None or action is negative:
            fail("native sample requires distinct action and negative prompt objects")
        state = _ActiveSample(
            action_prompt=action,
            negative_prompt=negative,
            expected_seed=int(self.inference_policy.seed),
        )
        self._active = state
        try:
            result = self._original_sample(*args, **kwargs)
            self._assert_installed_wrapper_identities()
            if (
                state.pending_negative is not None
                or state.pending_action_receipt is not None
                or state.integrated_steps != NUM_INFERENCE_STEPS
                or len(self.route_receipts) != NUM_INFERENCE_STEPS
                or self.negative_passthrough_calls
                != NUM_INFERENCE_STEPS * TRANSFORMER_BLOCKS
                or self.action_injection_calls
                != NUM_INFERENCE_STEPS * TRANSFORMER_BLOCKS
            ):
                fail("native Bernini sample returned without exact40 branch closure")
            return result
        finally:
            self._active = None

    def install(self) -> None:
        if self._installed:
            fail("Level-B native renderer bridge is already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_action_edit_level_b_renderer_v1", self)
        try:
            self._set_patch(self.diffusion, "sample", sample_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
        except Exception:
            self.restore()
            raise
        self._installed = True

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had, previous, installed = self._patches.pop()
            try:
                if getattr(owner, name, None) is not installed:
                    errors.append(
                        LevelBRendererError(
                            f"installed Level-B {name} wrapper identity changed before exit"
                        )
                    )
                if had:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:
                errors.append(error)
        for block, hook_id, original in self._level_a_hook_rows:
            if getattr(block, "_forward_hooks", {}).get(hook_id) is not original:
                errors.append(
                    LevelBRendererError("Level-A action hook was not restored exactly")
                )
        if (
            getattr(self.bundle.offline_hooks, "restored", None) is not False
            or getattr(
                self.bundle.transformer, "action_plan_conditioner_v1", None
            )
            is not self.bundle.conditioner
        ):
            errors.append(
                LevelBRendererError("Level-A hook owner changed during Level-B")
            )
        self._installed = False
        self._active = None
        self._installed_wrappers.clear()
        if errors:
            raise LevelBRendererError(
                f"failed to restore {len(errors)} Level-B renderer wrapper(s): "
                f"{errors[0]}"
            ) from errors[0]

    def receipt(self) -> Mapping[str, Any]:
        if self._installed or self._active is not None:
            fail("Level-B bridge receipt requires restored native method wrappers")
        if (
            len(self.route_receipts) != NUM_INFERENCE_STEPS
            or self._scheduler_receipt is None
            or self._source_prefix_sha256 is None
            or self._instruction_sha256 is None
            or self._instruction_length is None
            or self._instruction_padded_length is None
        ):
            fail("Level-B renderer trace is incomplete")
        return {
            "schema_version": BRIDGE_TRACE_SCHEMA,
            "official_call_graph": "negative_shared_step_then_action_shared_step_then_unipc",
            "native_sample_calls": 1,
            "native_shared_step_calls": 2 * NUM_INFERENCE_STEPS,
            "negative_native_forward_count": NUM_INFERENCE_STEPS,
            "action_conditioned_forward_count": NUM_INFERENCE_STEPS,
            "negative_residual_head_calls": 0,
            "action_residual_head_calls": self.action_injection_calls,
            "same_level_a_exact30_hook_objects_reused_all40": True,
            "level_a_hooks_still_installed_fail_closed_after_render": True,
            "original_unipc_scheduler_step_calls": NUM_INFERENCE_STEPS,
            "clean_source_prefix_sha256": self._source_prefix_sha256,
            "clean_source_prefix_bit_exact_all40": True,
            "actual_contextual_instruction_length": self._instruction_length,
            "native_contextual_instruction_padded_length": (
                self._instruction_padded_length
            ),
            "actual_contextual_instruction_sha256": self._instruction_sha256,
            "evolving_target_state_sha256": list(self.target_state_sha256),
            "forty_distinct_evolving_target_states": (
                len(set(self.target_state_sha256)) == NUM_INFERENCE_STEPS
            ),
            "scheduler": dict(self._scheduler_receipt),
            "native_unipc_callable_authority": dict(
                self._scheduler_callable_authority or {}
            ),
            "route_receipts": [dict(row) for row in self.route_receipts],
            "source_and_padding_bits_preserved": True,
            "clean_target_anchor_teacher_consumed": False,
            "caller_callback_or_denoiser_consumed": False,
        }


@contextmanager
def native_action_renderer_bridge(
    *,
    fresh_bundle: Any,
    product_module: Any,
    inference_policy: Any,
    patch_grid: Sequence[int],
    row_identity: str,
    torch_module: Any,
    noise_observer: Any,
    source_condition_list: list[Any],
    expected_width: int,
    expected_height: int,
    expected_device: Any,
    expected_internal_sampling: Mapping[str, Any],
    expected_instruction_token_count: int,
) -> Iterator[InstalledNativeActionRendererBridge]:
    bridge = InstalledNativeActionRendererBridge(
        bundle=fresh_bundle,
        product_module=product_module,
        inference_policy=inference_policy,
        patch_grid=tuple(int(x) for x in patch_grid),
        row_identity=row_identity,
        torch_module=torch_module,
        noise_observer=noise_observer,
        source_condition_list=source_condition_list,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_device=expected_device,
        expected_internal_sampling=expected_internal_sampling,
        expected_instruction_token_count=expected_instruction_token_count,
    )
    bridge.install()
    try:
        yield bridge
    finally:
        bridge.restore()


def _pack_wan_latent(value: Any, *, torch_module: Any) -> Any:
    """Pack ``[B,16,21,H,W]`` in the vendor's native ``1x2x2`` order."""

    torch = torch_module
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or tuple(int(x) for x in value.shape[:3]) != (1, LATENT_CHANNELS, PHASES)
        or int(value.shape[3]) % 2
        or int(value.shape[4]) % 2
        or not bool(torch.isfinite(value).all().item())
    ):
        fail("native Gaussian requires finite [1,16,21,evenH,evenW]")
    batch, channels, phases, height, width = (int(x) for x in value.shape)
    return (
        value.reshape(
            batch,
            channels,
            phases,
            height // 2,
            2,
            width // 2,
            2,
        )
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(batch, phases * (height // 2) * (width // 2), PATCH_VALUES)
        .detach()
        .contiguous()
    )


@dataclass
class NativeInitialNoiseObserver:
    """Internal observer of the one official CPU-generator Gaussian draw.

    It returns the exact object created by Diffusers and never accepts a tensor
    from the caller, so this cannot become an external initial-noise injection.
    """

    wan_diffusion_module: Any
    canonical_randn_tensor: Any
    expected_shape: tuple[int, ...]
    expected_device: Any
    expected_seed: int
    torch_module: Any
    capture_cpu: Any = None
    call_receipt: Optional[Mapping[str, Any]] = None
    _original: Any = None
    _installed: bool = False

    def __post_init__(self) -> None:
        if (
            tuple(int(x) for x in self.expected_shape[:3])
            != (1, LATENT_CHANNELS, PHASES)
            or len(self.expected_shape) != 5
            or any(int(x) <= 0 for x in self.expected_shape)
            or type(self.expected_seed) is not int
            or not 0 <= self.expected_seed < 2**63
            or not callable(self.canonical_randn_tensor)
        ):
            fail("official native Gaussian observer contract differs")
        self.expected_shape = tuple(int(x) for x in self.expected_shape)
        self._original = getattr(self.wan_diffusion_module, "randn_tensor", None)
        if self._original is not self.canonical_randn_tensor:
            fail("wan_diffusion.randn_tensor is replaced or noncanonical")

    def install(self) -> None:
        if self._installed or self.capture_cpu is not None:
            fail("native Gaussian observer is single-use")
        torch = self.torch_module

        def observed(*args: Any, **kwargs: Any) -> Any:
            if self.capture_cpu is not None:
                fail("official sampler requested more than one initial Gaussian")
            shape_raw = args[0] if args else kwargs.get("shape")
            try:
                shape = tuple(int(x) for x in shape_raw)
            except Exception as error:
                raise LevelBRendererError("official randn_tensor shape differs") from error
            generator = kwargs.get("generator")
            device = kwargs.get("device")
            dtype = kwargs.get("dtype")
            if (
                shape != self.expected_shape
                or not isinstance(generator, torch.Generator)
                or str(generator.device) != "cpu"
                or int(generator.initial_seed()) != self.expected_seed
                or str(torch.device(device)) != str(torch.device(self.expected_device))
                or dtype != torch.float32
            ):
                fail("official sampler initial Gaussian call contract differs")
            returned = self._original(*args, **kwargs)
            if (
                not isinstance(returned, torch.Tensor)
                or tuple(int(x) for x in returned.shape) != self.expected_shape
                or returned.device != torch.device(self.expected_device)
                or returned.dtype != torch.float32
                or not bool(torch.isfinite(returned).all().item())
            ):
                fail("official randn_tensor result differs")
            self.capture_cpu = returned.detach().to(device="cpu").contiguous().clone()
            self.call_receipt = {
                "call_count": 1,
                "requested_shape": list(shape),
                "requested_device": str(torch.device(device)),
                "requested_dtype": str(dtype),
                "generator_device": str(generator.device),
                "generator_initial_seed": int(generator.initial_seed()),
                "returned_object_forwarded_by_identity": True,
                "external_initial_noise_injection": False,
            }
            return returned

        setattr(observed, "_bernini_action_edit_level_b_noise_observer_v1", self)
        setattr(self.wan_diffusion_module, "randn_tensor", observed)
        self._observed = observed
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        if getattr(self.wan_diffusion_module, "randn_tensor", None) is not self._observed:
            fail("wan_diffusion.randn_tensor changed while observed")
        setattr(self.wan_diffusion_module, "randn_tensor", self._original)
        self._installed = False
        if getattr(self.wan_diffusion_module, "randn_tensor", None) is not self._original:
            fail("wan_diffusion.randn_tensor restoration failed")

    def packed_for(self, device: Any) -> Any:
        if self.capture_cpu is None or self.call_receipt is None:
            fail("official initial Gaussian has not been observed")
        return _pack_wan_latent(
            self.capture_cpu.to(device=device), torch_module=self.torch_module
        )

    def receipt(self) -> Mapping[str, Any]:
        if self._installed or self.capture_cpu is None or self.call_receipt is None:
            fail("native Gaussian observer receipt requires one restored call")
        return {
            **dict(self.call_receipt),
            "spatial_tensor_sha256": tensor_sha256(
                self.capture_cpu, torch_module=self.torch_module
            ),
            "packed_tensor_sha256": tensor_sha256(
                _pack_wan_latent(self.capture_cpu, torch_module=self.torch_module),
                torch_module=self.torch_module,
            ),
            "noise_factory": "diffusers.utils.torch_utils.randn_tensor",
            "counter_based_cpu_torch_generator": True,
            "global_rng_used": False,
        }


@contextmanager
def observe_native_initial_noise(observer: NativeInitialNoiseObserver) -> Iterator[None]:
    observer.install()
    try:
        yield
    finally:
        observer.restore()


def _rng_snapshot(torch_module: Any) -> Mapping[str, Any]:
    torch = torch_module
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except Exception:
        numpy_state = None
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": (
            [item.detach().cpu().clone() for item in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any], *, torch_module: Any) -> bool:
    torch = torch_module
    if left.get("python") != right.get("python"):
        return False
    left_numpy, right_numpy = left.get("numpy"), right.get("numpy")
    if (left_numpy is None) != (right_numpy is None):
        return False
    if left_numpy is not None:
        if (
            left_numpy[0] != right_numpy[0]
            or left_numpy[2:] != right_numpy[2:]
            or not bool((left_numpy[1] == right_numpy[1]).all())
        ):
            return False
    if not bool(torch.equal(left["torch_cpu"], right["torch_cpu"])):
        return False
    if len(left["torch_cuda"]) != len(right["torch_cuda"]):
        return False
    return all(
        bool(torch.equal(a, b))
        for a, b in zip(left["torch_cuda"], right["torch_cuda"])
    )


def _loads_strict_json(raw: bytes, *, label: str) -> Any:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise LevelBRendererError(f"{label} is not strict JSON") from error


def _parse_fraction(value: Any, *, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        fail(f"{label} is not an exact rational value")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise LevelBRendererError(f"{label} is invalid") from error
    if result.denominator <= 0:
        fail(f"{label} has an invalid denominator")
    return result


def _parse_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        fail(f"{label} is not one exact integer")
    try:
        text = str(value)
        if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text) is None:
            fail(f"{label} is not one exact integer")
        return int(text)
    except (TypeError, ValueError) as error:
        raise LevelBRendererError(f"{label} is invalid") from error


def _decoded_frame_duration_field_mode(frames: Any) -> str:
    if (
        not isinstance(frames, list)
        or len(frames) != FRAME_COUNT
        or any(not isinstance(row, Mapping) for row in frames)
    ):
        fail("ffprobe show_frames row/count differs")
    presence = tuple(
        ("pkt_duration" in row, "pkt_duration_time" in row) for row in frames
    )
    if all(value == (True, True) for value in presence):
        return "all-present-and-exact"
    if all(value == (False, False) for value in presence):
        return "all-absent-packet-authoritative"
    fail("ffprobe show_frames duration-field presence is mixed or partial")


def _run_ffprobe_json(
    command: Sequence[str], *, label: str, timeout_seconds: int = 120
) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise LevelBRendererError(f"{label} exceeded its wall timeout") from error
    if completed.returncode != 0 or completed.stderr:
        fail(f"{label} rejected the encoded output")
    payload = _loads_strict_json(completed.stdout, label=f"{label} output")
    if not isinstance(payload, Mapping):
        fail(f"{label} output root differs")
    return payload


def _validate_full_mp4_bytes_with_authenticated_tools(
    *,
    mp4_path: str | Path,
    expected_height: int,
    expected_width: int,
    ffmpeg_path: str | Path,
    ffprobe_path: str | Path,
    expected_ffmpeg_sha256: str,
    expected_ffprobe_sha256: str,
) -> Mapping[str, Any]:
    """Verify exact-CFR timing and fully decode every RGB byte of one MP4."""

    path = _plain_file(mp4_path, label="encoded output MP4")
    ffmpeg = _plain_file(ffmpeg_path, label="ffmpeg executable")
    ffprobe = _plain_file(ffprobe_path, label="ffprobe executable")
    ffmpeg_sha, ffmpeg_identity = stable_file_sha256(
        ffmpeg, label="ffmpeg executable"
    )
    ffprobe_sha, ffprobe_identity = stable_file_sha256(
        ffprobe, label="ffprobe executable"
    )
    if (
        ffmpeg_sha
        != _require_sha(expected_ffmpeg_sha256, label="pinned ffmpeg SHA")
        or ffprobe_sha
        != _require_sha(expected_ffprobe_sha256, label="pinned ffprobe SHA")
        or not os.access(ffmpeg, os.X_OK)
        or not os.access(ffprobe, os.X_OK)
    ):
        fail("ffmpeg/ffprobe executable authority differs")
    if path.suffix.lower() != ".mp4" or path.stat().st_size <= 128:
        fail("encoded output is not a nonempty MP4 candidate")
    if (
        type(expected_height) is not int
        or type(expected_width) is not int
        or expected_height <= 0
        or expected_width <= 0
    ):
        fail("expected output geometry differs")
    media_sha_before, media_identity_before = stable_file_sha256(
        path, label="encoded output MP4 before probing"
    )
    metadata_command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-count_packets",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(path),
    ]
    payload = _run_ffprobe_json(metadata_command, label="ffprobe metadata")
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    video_streams = [
        row for row in streams or ()
        if isinstance(row, Mapping) and row.get("codec_type") == "video"
    ]
    if len(video_streams) != 1 or len(streams or ()) != 1:
        fail("encoded output must expose exactly one stream and it must be video")
    stream = video_streams[0]
    frame_count_raw = stream.get("nb_frames")
    frame_count = int(frame_count_raw) if str(frame_count_raw).isdigit() else None
    read_frame_count_raw = stream.get("nb_read_frames")
    read_frame_count = (
        int(read_frame_count_raw) if str(read_frame_count_raw).isdigit() else None
    )
    average_rate = _parse_fraction(
        stream.get("avg_frame_rate"), label="average frame rate"
    )
    real_rate = _parse_fraction(stream.get("r_frame_rate"), label="real frame rate")
    time_base = _parse_fraction(stream.get("time_base"), label="stream time base")
    expected_duration = Fraction(FRAME_COUNT, int(FPS))
    expected_tick = Fraction(1, int(FPS)) / time_base
    if expected_tick.denominator != 1:
        fail("stream time base cannot represent exact 25-fps timestamps")
    expected_tick_count = int(expected_tick)
    format_row = payload.get("format")
    if (
        stream.get("index") != 0
        or stream.get("codec_name") != "h264"
        or stream.get("pix_fmt") != "yuv420p"
        or _parse_int(stream.get("height"), label="stream height")
        != expected_height
        or _parse_int(stream.get("width"), label="stream width")
        != expected_width
        or average_rate != Fraction(int(FPS), 1)
        or real_rate != Fraction(int(FPS), 1)
        or _parse_int(stream.get("start_pts"), label="stream start PTS") != 0
        or _parse_fraction(stream.get("start_time"), label="stream start time") != 0
        or _parse_int(stream.get("duration_ts"), label="stream duration ticks")
        != FRAME_COUNT * expected_tick_count
        or _parse_fraction(stream.get("duration"), label="stream duration")
        != expected_duration
        or frame_count != FRAME_COUNT
        or read_frame_count != FRAME_COUNT
        or _parse_int(
            stream.get("nb_read_packets"), label="stream packet count"
        )
        != FRAME_COUNT
        or not isinstance(format_row, Mapping)
        or "mp4" not in str(format_row.get("format_name", "")).split(",")
        or _parse_int(format_row.get("nb_streams"), label="container stream count")
        != 1
        or _parse_fraction(
            format_row.get("start_time"), label="container start time"
        )
        != 0
        or _parse_fraction(format_row.get("duration"), label="container duration")
        != expected_duration
    ):
        fail("ffprobe stream CFR/duration/frame/geometry contract differs")

    frame_command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        (
            "frame=media_type,stream_index,pts,pts_time,best_effort_timestamp,"
            "best_effort_timestamp_time,pkt_duration,pkt_duration_time"
        ),
        "-print_format",
        "json",
        str(path),
    ]
    frame_payload = _run_ffprobe_json(frame_command, label="ffprobe show_frames")
    frames = frame_payload.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        fail("ffprobe show_frames count differs")
    # The pinned ffprobe omits both decoded-frame duration aliases for exact
    # H264.  All-absent remains closed-world because stream duration, exact
    # frame count and every packet's duration/PTS are mandatory below.
    frame_duration_mode = _decoded_frame_duration_field_mode(frames)
    frame_timing_rows = []
    for index, row in enumerate(frames):
        # Older ffprobe builds omit the decoded-frame ``pts`` aliases while
        # still exposing the normative best-effort presentation timestamp.
        # Require the latter unconditionally, and require ``pts`` too whenever
        # that build emits it.
        timestamp_raw = row.get("pts", row.get("best_effort_timestamp"))
        timestamp_time_raw = row.get(
            "pts_time", row.get("best_effort_timestamp_time")
        )
        if (
            row.get("media_type") != "video"
            or _parse_int(row.get("stream_index"), label="frame stream index")
            != 0
            or _parse_int(timestamp_raw, label="decoded frame PTS")
            != index * expected_tick_count
            or _parse_fraction(
                timestamp_time_raw, label="decoded frame presentation time"
            )
            != Fraction(index, int(FPS))
            or _parse_int(
                row.get("best_effort_timestamp"),
                label="decoded frame best-effort PTS",
            )
            != index * expected_tick_count
            or _parse_fraction(
                row.get("best_effort_timestamp_time"),
                label="decoded frame best-effort presentation time",
            )
            != Fraction(index, int(FPS))
        ):
            fail("ffprobe show_frames exact81 CFR PTS/duration contract differs")
        if frame_duration_mode == "all-present-and-exact" and (
            _parse_int(row.get("pkt_duration"), label="decoded frame duration")
            != expected_tick_count
            or _parse_fraction(
                row.get("pkt_duration_time"), label="decoded frame duration"
            )
            != Fraction(1, int(FPS))
        ):
            fail("ffprobe show_frames exact81 CFR PTS/duration contract differs")
        frame_timing_rows.append(
            {
                "index": index,
                "pts": _parse_int(timestamp_raw, label="decoded frame PTS"),
                "pts_time": str(timestamp_time_raw),
                "duration": (
                    _parse_int(row["pkt_duration"], label="decoded frame duration")
                    if frame_duration_mode == "all-present-and-exact"
                    else None
                ),
                "duration_time": (
                    str(row["pkt_duration_time"])
                    if frame_duration_mode == "all-present-and-exact"
                    else None
                ),
            }
        )

    packet_command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=codec_type,stream_index,pts,pts_time,duration,duration_time",
        "-print_format",
        "json",
        str(path),
    ]
    packet_payload = _run_ffprobe_json(packet_command, label="ffprobe show_packets")
    packets = packet_payload.get("packets")
    if not isinstance(packets, list) or len(packets) != FRAME_COUNT:
        fail("ffprobe show_packets count differs")
    packet_timing_rows = []
    observed_packet_indices = []
    for decode_order, row in enumerate(packets):
        pts = (
            _parse_int(row.get("pts"), label="packet PTS")
            if isinstance(row, Mapping)
            else -1
        )
        presentation_index = (
            pts // expected_tick_count
            if expected_tick_count > 0 and pts % expected_tick_count == 0
            else -1
        )
        if (
            not isinstance(row, Mapping)
            or row.get("codec_type") != "video"
            or _parse_int(row.get("stream_index"), label="packet stream index")
            != 0
            or presentation_index not in range(FRAME_COUNT)
            or _parse_fraction(row.get("pts_time"), label="packet PTS time")
            != Fraction(presentation_index, int(FPS))
            or _parse_int(row.get("duration"), label="packet duration")
            != expected_tick_count
            or _parse_fraction(
                row.get("duration_time"), label="packet duration time"
            )
            != Fraction(1, int(FPS))
        ):
            fail("ffprobe show_packets exact81 CFR PTS/duration contract differs")
        observed_packet_indices.append(presentation_index)
        packet_timing_rows.append(
            {
                "decode_order": decode_order,
                "presentation_index": presentation_index,
                "pts": _parse_int(row["pts"], label="packet PTS"),
                "pts_time": str(row["pts_time"]),
                "duration": _parse_int(row["duration"], label="packet duration"),
                "duration_time": str(row["duration_time"]),
            }
        )
    if sorted(observed_packet_indices) != list(range(FRAME_COUNT)):
        fail("ffprobe show_packets presentation-index closure differs")
    decode_command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(
        decode_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        fail("ffmpeg full-decode pipes are unavailable")
    expected_bytes = FRAME_COUNT * expected_height * expected_width * 3
    try:
        decoded_bytes, decode_stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise LevelBRendererError("ffmpeg full decode exceeded wall timeout") from error
    count = len(decoded_bytes)
    if process.returncode != 0 or decode_stderr or count != expected_bytes:
        fail("ffmpeg did not completely decode exact81 RGB frames")
    digest = hashlib.sha256(decoded_bytes).hexdigest()
    output_sha, identity = stable_file_sha256(
        path, label="encoded output MP4 after full decode"
    )
    ffmpeg_sha_after, ffmpeg_identity_after = stable_file_sha256(
        ffmpeg, label="ffmpeg executable after full decode"
    )
    ffprobe_sha_after, ffprobe_identity_after = stable_file_sha256(
        ffprobe, label="ffprobe executable after probes"
    )
    if (
        output_sha != media_sha_before
        or identity != media_identity_before
        or ffmpeg_sha_after != ffmpeg_sha
        or ffmpeg_identity_after != ffmpeg_identity
        or ffprobe_sha_after != ffprobe_sha
        or ffprobe_identity_after != ffprobe_identity
    ):
        fail("media or pinned executable changed during exact81 validation")
    return {
        "schema_version": VIDEO_VALIDATION_SCHEMA,
        "mp4_path": str(path),
        "mp4_sha256": output_sha,
        "mp4_file_identity": identity,
        "ffmpeg_executable_sha256": ffmpeg_sha,
        "ffprobe_executable_sha256": ffprobe_sha,
        "ffprobe_metadata_command_argv": metadata_command,
        "ffprobe_show_frames_command_argv": frame_command,
        "ffprobe_show_packets_command_argv": packet_command,
        "ffprobe_video_stream": dict(stream),
        "ffprobe_codec_name": "h264",
        "ffprobe_pixel_format": "yuv420p",
        "ffprobe_exact81": (
            frame_count == FRAME_COUNT and read_frame_count == FRAME_COUNT
        ),
        "ffprobe_fps": float(average_rate),
        "ffprobe_time_base": str(time_base),
        "ffprobe_exact_duration": str(expected_duration),
        "ffprobe_geometry_hw": [expected_height, expected_width],
        "show_frames_count": len(frames),
        "show_frames_timing_sha256": object_sha256(frame_timing_rows),
        "show_frames_exact_pts_n_over_25": True,
        "show_frames_packet_duration_field_mode": frame_duration_mode,
        "show_frames_duration_all_present_or_all_absent": True,
        "show_packets_count": len(packets),
        "show_packets_timing_sha256": object_sha256(packet_timing_rows),
        "show_packets_exact_pts_n_over_25": True,
        "constant_frame_rate_verified": True,
        "variable_frame_rate_rejected": True,
        "full_decode_command_argv": decode_command,
        "full_decode_frame_count": FRAME_COUNT,
        "full_decode_rgb_byte_count": count,
        "full_decode_rgb_sha256": digest,
        "complete_decode_verified": True,
    }


def _audit_decoded_video_array(
    value: Any,
    *,
    expected_height: int,
    expected_width: int,
    numpy_module: Any,
) -> Mapping[str, Any]:
    """Validate decoded float frames without one full-size temporary mask."""

    np = numpy_module
    expected_shape = (FRAME_COUNT, expected_height, expected_width, 3)
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or tuple(int(x) for x in value.shape) != expected_shape
    ):
        fail("frozen VAE decoded output ndarray/dtype/geometry differs")
    minimum = math.inf
    maximum = -math.inf
    for frame_index in range(FRAME_COUNT):
        frame = value[frame_index]
        if not bool(np.isfinite(frame).all()):
            fail("frozen VAE decoded output contains non-finite values")
        frame_minimum = float(frame.min())
        frame_maximum = float(frame.max())
        if frame_minimum < 0.0 or frame_maximum > 1.0:
            fail("frozen VAE decoded output is outside [0,1]")
        minimum = min(minimum, frame_minimum)
        maximum = max(maximum, frame_maximum)
    return {
        "exact_type": "numpy.ndarray",
        "dtype": "float32",
        "shape": list(expected_shape),
        "finite": True,
        "closed_unit_interval": True,
        "minimum": minimum,
        "maximum": maximum,
        "framewise_bounded_validation": True,
    }


def _authenticated_source_module(
    *, path: Path, expected_sha256: str, module_name: str
) -> ModuleType:
    source = _plain_file(path, label=f"{module_name} source")
    expected = _require_sha(expected_sha256, label=f"{module_name} expected SHA")
    observed, _ = stable_file_sha256(source, label=f"{module_name} source")
    if observed != expected or module_name in sys.modules:
        fail(f"{module_name} source/import ownership differs")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        fail(f"{module_name} compiled source bytes differ")
    post_observed, _ = stable_file_sha256(
        source, label=f"{module_name} source after read"
    )
    if post_observed != expected:
        fail(f"{module_name} source changed before compilation")
    module = ModuleType(module_name)
    module.__file__ = str(source)
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    try:
        code = compile(raw, str(source), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _install_authenticated_named_module(
    *, name: str, path: Path, expected_sha256: str
) -> ModuleType:
    if not name or name in sys.modules:
        fail(f"preimported authenticated module is forbidden: {name}")
    return _authenticated_source_module(
        path=path,
        expected_sha256=expected_sha256,
        module_name=name,
    )


def _install_authenticated_materialize_module(
    *,
    path: Path,
    expected_sha256: str,
    builder_path: Path,
    expected_builder_sha256: str,
) -> ModuleType:
    name = "tools.materialize_vae"
    if "tools" in sys.modules or name in sys.modules:
        fail("preimported tools package/materialize module is forbidden")
    package_root = _plain_directory(path.parent, label="authenticated tools package root")
    if (
        package_root.name != "tools"
        or builder_path.parent != package_root
        or builder_path.name != "build_renderer_dataset.py"
    ):
        fail("authenticated materialize source is not under the sealed tools root")
    tools_package = ModuleType("tools")
    tools_package.__package__ = "tools"
    tools_package.__path__ = [str(package_root)]
    sys.modules["tools"] = tools_package
    try:
        builder = _install_authenticated_named_module(
            name="tools.build_renderer_dataset",
            path=builder_path,
            expected_sha256=expected_builder_sha256,
        )
        setattr(tools_package, "build_renderer_dataset", builder)
        module = _install_authenticated_named_module(
            name=name, path=path, expected_sha256=expected_sha256
        )
        setattr(tools_package, "materialize_vae", module)
    except Exception as error:
        sys.modules.pop(name, None)
        sys.modules.pop("tools.build_renderer_dataset", None)
        sys.modules.pop("tools", None)
        raise LevelBRendererError("cannot bind authenticated tools.materialize_vae") from error
    return module


def _resolve_authenticated_product_module(fresh_bundle: Any) -> ModuleType:
    module_name = str(getattr(fresh_bundle.offline_hooks.__class__, "__module__", ""))
    module = sys.modules.get(module_name)
    receipt = getattr(fresh_bundle, "consumer_receipt", None)
    expected = receipt.get("product_bridge_source_sha256") if isinstance(receipt, Mapping) else None
    if module is None or not isinstance(module, ModuleType):
        fail("authenticated product ABI module is unavailable")
    path = _plain_file(
        Path(getattr(module, "__file__", "")).resolve(strict=True),
        label="authenticated product ABI source",
    )
    observed, _ = stable_file_sha256(path, label="authenticated product ABI source")
    if observed != _require_sha(expected, label="product ABI source SHA"):
        fail("authenticated product ABI module source differs")
    required = (
        "OfflineInferencePolicyV1",
        "ProductRequestV1",
        "activate_offline_route",
        "audit_live_inference_scheduler",
        "prepare_product_route_from_packed_embeddings",
    )
    if any(not hasattr(module, name) for name in required):
        fail("authenticated product ABI interface differs")
    return module


def _validate_fresh_bundle(fresh_bundle: Any) -> Mapping[str, Any]:
    receipt = getattr(fresh_bundle, "consumer_receipt", None)
    checkpoint = getattr(fresh_bundle, "checkpoint", None)
    distributed = getattr(fresh_bundle, "distributed", None)
    topology = getattr(distributed, "topology", None)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("authority") != AUTHORITY
        or receipt.get("complete") is not True
        or receipt.get("promotable") is not False
        or receipt.get("world8_consumer_complete") is not True
        or receipt.get("fresh_world8_process_forward_exact_consensus_verified") is not True
        or receipt.get("full_bernini_renderer_forward_executed") is not False
        or receipt.get("loaded_parameter_sha256")
        != receipt.get("checkpoint_parameter_sha256")
        or checkpoint is None
        or distributed is None
        or int(distributed.world_size) != WORLD_SIZE
        or int(topology.sp_size) != SP_SIZE
        or int(topology.dp_size) != DP_SIZE
        or int(distributed.sp_rank) not in range(SP_SIZE)
    ):
        fail("fresh WORLD8 checkpoint bundle is not admissible for Level-B")
    if (
        getattr(fresh_bundle, "renderer", None) is None
        or getattr(fresh_bundle, "transformer", None) is None
        or getattr(fresh_bundle, "conditioner", None) is None
        or getattr(fresh_bundle.renderer, "diff_dec", None) is None
    ):
        fail("fresh WORLD8 model objects are incomplete")
    return dict(receipt)


def _validate_live_base_checkpoint_binding(
    *, consumer_receipt: Mapping[str, Any], fresh_bundle: Any, runtime: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Re-hash the live base tree and bind it to the consumer's prior audit."""

    runner = getattr(getattr(fresh_bundle, "checkpoint", None), "runner", None)
    content = consumer_receipt.get("base_checkpoint_content")
    if not isinstance(content, Mapping) or not isinstance(runner, ModuleType):
        fail("fresh consumer base-checkpoint authority is absent")
    if (
        consumer_receipt.get("base_checkpoint_tree_sha256")
        != PINNED_BASE_CHECKPOINT_TREE_SHA256
        or getattr(runner, "CHECKPOINT_TREE_SHA256", None)
        != PINNED_BASE_CHECKPOINT_TREE_SHA256
        or getattr(runner, "CHECKPOINT_CONTENT_MANIFEST_SHA256", None)
        != PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or runtime.get("base_checkpoint") != PINNED_BASE_CHECKPOINT
        or runtime.get("base_checkpoint_tree_sha256")
        != PINNED_BASE_CHECKPOINT_TREE_SHA256
    ):
        fail("base-checkpoint path/tree compile-time authority differs")
    checkpoint = _plain_directory(
        PINNED_BASE_CHECKPOINT, label="live pinned base checkpoint"
    )
    manifest = _plain_file(
        PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST,
        label="live pinned checkpoint content manifest",
    )
    manifest_sha, manifest_identity = stable_file_sha256(
        manifest, label="live pinned checkpoint content manifest"
    )
    if (
        manifest_sha != PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or runtime["checkpoint_content_manifest"]["path"] != str(manifest)
        or runtime["checkpoint_content_manifest"]["sha256"] != manifest_sha
        or runtime["checkpoint_content_manifest"]["file_identity"]
        != manifest_identity
    ):
        fail("live checkpoint content manifest differs from verified runtime")
    expected_content_fields = {
        "checkpoint_root": str(checkpoint),
        "tree_sha256": PINNED_BASE_CHECKPOINT_TREE_SHA256,
        "manifest_path": str(manifest),
        "manifest_sha256": PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "every_non_cache_file_sha256_verified": True,
    }
    if any(content.get(name) != value for name, value in expected_content_fields.items()):
        fail("consumer receipt base-checkpoint content binding differs")
    content_unsigned = {key: value for key, value in content.items() if key != "digest"}
    if content.get("digest") != object_sha256(content_unsigned):
        fail("consumer receipt base-checkpoint content digest differs")

    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise LevelBRendererError("cannot read live checkpoint manifest") from error
    expression = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)\Z")
    expected: dict[str, str] = {}
    for line in lines:
        match = expression.fullmatch(line)
        if match is None:
            fail("live checkpoint manifest row differs")
        digest, raw_path = match.groups()
        relative = raw_path[2:]
        parts = relative.split("/")
        if (
            not relative
            or any(part in ("", ".", "..") for part in parts)
            or relative in expected
        ):
            fail("live checkpoint manifest path differs")
        expected[relative] = digest
    if (
        type(content.get("verified_file_count")) is not int
        or content["verified_file_count"] != len(expected)
        or not expected
    ):
        fail("live checkpoint manifest count differs from consumer receipt")
    actual: set[str] = set()
    for path in checkpoint.rglob("*"):
        relative_path = path.relative_to(checkpoint)
        if ".cache" in relative_path.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            fail("live base checkpoint contains a non-cache symlink")
        if stat.S_ISREG(mode):
            actual.add(relative_path.as_posix())
        elif not stat.S_ISDIR(mode):
            fail("live base checkpoint contains a special entry")
    if actual != set(expected):
        fail("live base-checkpoint file closure differs")
    rows = []
    for relative in sorted(expected):
        path = _validate_release_member_path(checkpoint, relative)
        observed, _ = stable_file_sha256(
            path, label=f"live base checkpoint member {relative}"
        )
        if observed != expected[relative]:
            fail("live base-checkpoint member bytes differ")
        rows.append({"path": relative, "sha256": observed})
    entries_digest = object_sha256(rows)
    if entries_digest != content.get("verified_entries_digest"):
        fail("live base-checkpoint entries differ from consumer receipt")
    manifest_sha_after, manifest_identity_after = stable_file_sha256(
        manifest, label="live pinned checkpoint content manifest after tree audit"
    )
    if (
        manifest_sha_after != manifest_sha
        or manifest_identity_after != manifest_identity
    ):
        fail("live checkpoint manifest changed during base-tree audit")
    return {
        "checkpoint_root": str(checkpoint),
        "tree_sha256": PINNED_BASE_CHECKPOINT_TREE_SHA256,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "manifest_file_identity": manifest_identity,
        "verified_file_count": len(rows),
        "verified_entries_digest": entries_digest,
        "consumer_base_checkpoint_content_digest": content["digest"],
        "every_non_cache_file_rehashed_at_level_b_use": True,
    }


def _world8_string_consensus(
    value: str, *, distributed_module: Any, group: Any
) -> Mapping[str, Any]:
    if not isinstance(value, str) or not value:
        fail("WORLD8 string consensus value differs")
    gathered: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(gathered, value, group=group)
    if gathered != [value] * WORLD_SIZE:
        fail("WORLD8 product execution bytes differ")
    return {
        "world_size": WORLD_SIZE,
        "all8_exact_consensus": True,
        "ordered_sha256": list(gathered),
    }


def _run_world8_rank0_collective_phase(
    *,
    phase: str,
    rank: int,
    operation: Any,
    distributed_module: Any,
    group: Any,
    rollback_on_failure: Any = None,
) -> Any:
    """Run one rank-0-only phase without stranding the other seven ranks.

    Rank 0 always converts a captured operation failure into a small broadcast
    status.  A side-effecting phase may also supply one idempotent rollback
    callback.  Rank 0 keeps that callback armed until the success status has
    crossed the collective; an operation or collective failure runs it before
    rank 0 exits the phase.  Heavy tensors stay in rank-local holders and are
    transferred by the caller only after this status gate succeeds.
    """

    if (
        not isinstance(phase, str)
        or not phase
        or type(rank) is not int
        or rank not in range(WORLD_SIZE)
        or not callable(operation)
        or (rollback_on_failure is not None and not callable(rollback_on_failure))
    ):
        fail("WORLD8 rank-0 phase invocation contract differs")
    rollback_attempted = False

    def rollback_rank0() -> Optional[str]:
        nonlocal rollback_attempted
        if rank != 0 or rollback_on_failure is None or rollback_attempted:
            return None
        rollback_attempted = True
        try:
            rollback_on_failure()
        except BaseException as rollback_error:
            return f"{type(rollback_error).__name__}: {rollback_error}"
        return None

    box: list[Any] = [None]
    if rank == 0:
        try:
            box[0] = {
                "schema_version": WORLD8_RANK0_PHASE_SCHEMA,
                "phase": phase,
                "ok": True,
                "value": operation(),
            }
        except BaseException as error:
            rollback_error = rollback_rank0()
            message = str(error)
            if rollback_error is not None:
                message += f"; rank-0 rollback failed: {rollback_error}"
            box[0] = {
                "schema_version": WORLD8_RANK0_PHASE_SCHEMA,
                "phase": phase,
                "ok": False,
                "error_type": type(error).__name__,
                "error": message,
            }
    try:
        distributed_module.broadcast_object_list(box, src=0, group=group)
    except BaseException as collective_error:
        rollback_error = rollback_rank0()
        message = (
            f"WORLD8 rank-0 phase {phase} collective broadcast failed: "
            f"{type(collective_error).__name__}: {collective_error}"
        )
        if rollback_error is not None:
            message += f"; rank-0 rollback failed: {rollback_error}"
        raise LevelBRendererError(message) from collective_error
    try:
        status = box[0]
        if not isinstance(status, Mapping):
            fail(f"WORLD8 rank-0 phase {phase} status differs")
        ok = status.get("ok")
        expected_keys = (
            {"schema_version", "phase", "ok", "value"}
            if ok is True
            else {"schema_version", "phase", "ok", "error_type", "error"}
        )
        if (
            set(status) != expected_keys
            or status.get("schema_version") != WORLD8_RANK0_PHASE_SCHEMA
            or status.get("phase") != phase
            or type(ok) is not bool
        ):
            fail(f"WORLD8 rank-0 phase {phase} status closure differs")
        if ok is not True:
            error_type = status.get("error_type")
            error_message = status.get("error")
            if not isinstance(error_type, str) or not isinstance(error_message, str):
                fail(f"WORLD8 rank-0 phase {phase} failure status differs")
            fail(
                f"WORLD8 rank-0 phase {phase} failed: "
                f"{error_type}: {error_message}"
            )
        return status["value"]
    except BaseException as status_error:
        rollback_error = rollback_rank0()
        if rollback_error is not None:
            raise LevelBRendererError(
                f"{status_error}; rank-0 rollback failed: {rollback_error}"
            ) from status_error
        raise


def _normalize_rank_local_cuda_indices(value: Any, *, local_cuda_index: int) -> Any:
    """Remove the expected per-rank CUDA ordinal from canonical receipts.

    Only an exact device token is normalized.  A receipt that names another
    visible CUDA ordinal is rejected rather than silently erased.
    """

    if type(local_cuda_index) is not int or local_cuda_index not in range(WORLD_SIZE):
        fail("rank-local CUDA index differs")
    if isinstance(value, str):
        match = _CUDA_DEVICE.fullmatch(value)
        if match is None:
            return value
        ordinal = match.group(1)
        if ordinal is not None and int(ordinal) != local_cuda_index:
            fail("receipt contains a non-local CUDA ordinal")
        return "cuda:<rank-local>"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            fail("WORLD8 receipt contains a non-string JSON object key")
        return {
            key: _normalize_rank_local_cuda_indices(
                item, local_cuda_index=local_cuda_index
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_rank_local_cuda_indices(
                item, local_cuda_index=local_cuda_index
            )
            for item in value
        ]
    if value is None or type(value) in (bool, int, float):
        return value
    fail("WORLD8 receipt contains a non-JSON value before canonicalization")


def _world8_rank0_canonical_consensus(
    value: Mapping[str, Any],
    *,
    local_cuda_index: int,
    distributed_module: Any,
    group: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    normalized = _normalize_rank_local_cuda_indices(
        value, local_cuda_index=local_cuda_index
    )
    encoded = canonical_json_bytes(normalized)
    gathered: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(gathered, encoded, group=group)
    if any(not isinstance(item, bytes) for item in gathered):
        fail("WORLD8 canonical receipt gather type differs")
    rank0_bytes = gathered[0]
    if any(item != rank0_bytes for item in gathered):
        fail("WORLD8 normalized product receipts differ")
    rank0 = _loads_strict_json(rank0_bytes, label="rank-0 canonical WORLD8 receipt")
    if not isinstance(rank0, Mapping) or canonical_json_bytes(rank0) != rank0_bytes:
        fail("rank-0 canonical WORLD8 receipt bytes differ")
    digest = hashlib.sha256(rank0_bytes).hexdigest()
    return rank0, {
        "world_size": WORLD_SIZE,
        "rank_local_cuda_index_normalized": True,
        "all8_exact_canonical_bytes_consensus": True,
        "rank0_canonical_sha256": digest,
        "ordered_canonical_sha256": [
            hashlib.sha256(item).hexdigest() for item in gathered
        ],
    }


def _fsync_directory(path: Path, *, label: str) -> None:
    directory = _plain_directory(path, label=label)
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise LevelBRendererError(f"cannot fsync {label}") from error


def _write_new_staged_json(path: Path, value: Mapping[str, Any]) -> Mapping[str, Any]:
    parent = _plain_directory(path.parent, label="staged receipt parent")
    if not path.is_absolute() or path.parent != parent or path.exists() or path.is_symlink():
        fail("staged Level-B receipt path must be absent")
    payload = canonical_json_bytes(value) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except BaseException:
        try:
            info = path.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                fail("refusing to clean a replaced staged Level-B receipt")
            path.unlink()
            _fsync_directory(parent, label="failed staged receipt directory")
        raise
    digest, identity = stable_file_sha256(path, label="staged Level-B receipt")
    if digest != hashlib.sha256(payload).hexdigest():
        fail("staged Level-B receipt bytes differ")
    _fsync_directory(parent, label="staged receipt directory")
    return {"sha256": digest, "file_identity": identity, "size": len(payload)}


def _prepare_absent_output(value: str | Path) -> tuple[Path, Path, Path]:
    output = Path(value).expanduser()
    if (
        not output.is_absolute()
        or output.name in ("", ".", "..")
        or output.suffix.lower() != ".mp4"
    ):
        fail("Level-B output must be one absolute .mp4 path")
    parent = _plain_directory(output.parent, label="Level-B output parent")
    if output.parent != parent or output.exists() or output.is_symlink():
        fail("Level-B output path must be absent")
    receipt = output.with_name(output.name + ".receipt.json")
    marker = output.with_name(output.name + ".COMMITTED.json")
    if (
        receipt.exists()
        or receipt.is_symlink()
        or marker.exists()
        or marker.is_symlink()
    ):
        fail("Level-B output receipt/commit-marker paths must be absent")
    return output, receipt, marker


def _world8_precommit_gate_contract(
    *,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    mp4_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": PRECOMMIT_GATE_SCHEMA,
        "world_size": WORLD_SIZE,
        "ordered_ranks": list(range(WORLD_SIZE)),
        "output_path": str(output_path),
        "receipt_path": str(receipt_path),
        "commit_marker_path": str(commit_marker_path),
        "mp4_sha256": _require_sha(
            mp4_sha256, label="precommit gate MP4 SHA"
        ),
        "final_receipt_exact_bytes_world8_consensus_required": True,
        "all8_public_precommit_reopen_required": True,
        "exact_ordered_rank_rows_required": True,
        "rank0_status_broadcast_before_barrier_required": True,
        "final_world8_barrier_before_marker_required": True,
        "marker_is_receipt_inode_alias": True,
        "parent_directory_fsynced_before_terminal_marker_link": True,
        "post_marker_collective_or_business_hook_forbidden": True,
    }


def _commit_marker_envelope(
    *,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    mp4_sha256: str,
) -> Mapping[str, Any]:
    gate = _world8_precommit_gate_contract(
        output_path=output_path,
        receipt_path=receipt_path,
        commit_marker_path=commit_marker_path,
        mp4_sha256=mp4_sha256,
    )
    unsigned = {
        "schema_version": OUTPUT_COMMIT_MARKER_SCHEMA,
        "authority": AUTHORITY,
        "complete": True,
        "output_basename": output_path.name,
        "receipt_basename": receipt_path.name,
        "commit_marker_basename": commit_marker_path.name,
        "mp4_sha256": mp4_sha256,
        "consumer_acceptance_requires_receipt_inode_alias_marker": True,
        "bare_mp4_or_receipt_pair_is_never_complete": True,
        "receipt_and_marker_must_be_same_inode": True,
        "world8_precommit_gate_contract": gate,
    }
    return {**unsigned, "envelope_digest": object_sha256(unsigned)}


def _rollback_exact_level_b_publication(
    *,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    mp4_identity: Mapping[str, Any],
    receipt_identity: Mapping[str, Any],
    commit_marker_identity: Optional[Mapping[str, Any]],
) -> None:
    """Idempotently revoke marker first, then its exact receipt and MP4."""

    expected_rows: list[tuple[Path, Mapping[str, Any]]] = []
    if commit_marker_identity is not None:
        expected_rows.append((commit_marker_path, commit_marker_identity))
    expected_rows.extend(
        ((receipt_path, receipt_identity), (output_path, mp4_identity))
    )
    expected = tuple(expected_rows)
    last_cleanup_errors: list[BaseException] = []
    # Retry once so a single transient NFS unlink/lstat/fsync failure cannot
    # preserve completion authority.  Exact inode checks make the retry safe.
    for _attempt in range(2):
        cleanup_errors: list[BaseException] = []
        for path, identity in expected:
            if not isinstance(identity, Mapping):
                cleanup_errors.append(
                    LevelBRendererError(
                        "cannot safely identify a published Level-B output for rollback"
                    )
                )
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            except BaseException as error:
                cleanup_errors.append(error)
                continue
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or int(info.st_dev) != int(identity.get("device", -1))
                or int(info.st_ino) != int(identity.get("inode", -1))
            ):
                cleanup_errors.append(
                    LevelBRendererError(
                        "refusing to roll back a foreign late Level-B output"
                    )
                )
                continue
            try:
                path.unlink()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            _fsync_directory(
                output_path.parent, label="Level-B late-failure rollback parent"
            )
        except BaseException as error:
            cleanup_errors.append(error)
        if not cleanup_errors:
            return
        last_cleanup_errors = cleanup_errors
    raise LevelBRendererError(
        "cannot durably roll back every committed Level-B artifact"
    ) from last_cleanup_errors[0]


@dataclass
class _StagedProductPair:
    directory: Path
    mp4_path: Path
    receipt_path: Path
    commit_marker_path: Path
    final_mp4_path: Path
    final_receipt_path: Path
    final_commit_marker_path: Path
    mp4_sha256: str
    mp4_identity: Mapping[str, int]
    publication_identities: Optional[Mapping[str, Mapping[str, Any]]] = None
    precommitted: bool = False
    committed: bool = False

    def register_precommit_publication_identity(
        self,
        *,
        receipt_identity: Mapping[str, Any],
    ) -> None:
        if (
            self.publication_identities is not None
            or self.precommitted
            or self.committed
        ):
            fail("Level-B precommit identity ledger was already sealed")
        identities = {
            "mp4": dict(self.mp4_identity),
            "receipt": dict(receipt_identity),
        }
        if any(
            type(identity.get("device")) is not int
            or type(identity.get("inode")) is not int
            for identity in identities.values()
        ):
            fail("Level-B precommit identity ledger differs")
        self.publication_identities = identities

    def cleanup_publication(self) -> None:
        if self.publication_identities is None:
            return
        _rollback_exact_level_b_publication(
            output_path=self.final_mp4_path,
            receipt_path=self.final_receipt_path,
            commit_marker_path=self.final_commit_marker_path,
            mp4_identity=self.publication_identities["mp4"],
            receipt_identity=self.publication_identities["receipt"],
            commit_marker_identity=self.publication_identities.get(
                "commit_marker"
            ),
        )

    def cleanup_stage(self) -> None:
        last_error: Optional[BaseException] = None
        # As with public unlink, retry one transient private-stage filesystem
        # failure.  A second pass is idempotent because absent names are valid.
        for _attempt in range(2):
            try:
                self._cleanup_stage_once()
                return
            except BaseException as error:
                last_error = error
        raise LevelBRendererError(
            "cannot durably clean the Level-B staging transaction"
        ) from last_error

    def _cleanup_stage_once(self) -> None:
        parent = _plain_directory(
            self.directory.parent, label="Level-B staging rollback parent"
        )
        try:
            directory_info = self.directory.lstat()
        except FileNotFoundError:
            # Idempotent retries still persist the already-completed removal.
            _fsync_directory(parent, label="Level-B staging rollback parent")
            return
        except BaseException as error:
            raise LevelBRendererError(
                "cannot inspect Level-B staging directory for cleanup"
            ) from error
        if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(
            directory_info.st_mode
        ):
            fail("refusing to clean a replaced Level-B staging directory")
        cleanup_errors: list[BaseException] = []
        for path in (self.commit_marker_path, self.receipt_path, self.mp4_path):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            except BaseException as error:
                cleanup_errors.append(error)
                continue
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                cleanup_errors.append(
                    LevelBRendererError(
                        "refusing to clean a replaced Level-B staging file"
                    )
                )
                continue
            try:
                path.unlink()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            _fsync_directory(
                self.directory, label="Level-B staging directory after cleanup"
            )
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            self.directory.rmdir()
        except FileNotFoundError:
            pass
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            _fsync_directory(parent, label="Level-B staging rollback parent")
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise LevelBRendererError(
                "cannot durably clean every Level-B staging artifact"
            ) from cleanup_errors[0]

def validate_committed_level_b_product(
    *, output_mp4_path: str | Path
) -> Mapping[str, Any]:
    """Open a product only through its atomically published commit marker.

    A naked MP4, or even an MP4+receipt pair, is never a completed product.
    The marker is linked last and binds the exact two payload hashes.
    """

    output = _plain_file(output_mp4_path, label="committed Level-B MP4")
    if output.suffix.lower() != ".mp4":
        fail("committed Level-B product path is not MP4")
    receipt = _plain_file(
        output.with_name(output.name + ".receipt.json"),
        label="committed Level-B receipt",
    )
    marker = _plain_file(
        output.with_name(output.name + ".COMMITTED.json"),
        label="committed Level-B atomic marker",
    )
    output_sha, output_identity = stable_file_sha256(
        output, label="committed Level-B MP4"
    )
    receipt_sha, receipt_identity = stable_file_sha256(
        receipt, label="committed Level-B receipt"
    )
    marker_sha, marker_identity = stable_file_sha256(
        marker, label="committed Level-B atomic marker"
    )
    payload = _loads_strict_json(
        receipt.read_bytes(), label="committed Level-B receipt"
    )
    marker_payload = _loads_strict_json(
        marker.read_bytes(), label="committed Level-B atomic marker"
    )
    output_sha_after_parse, output_identity_after_parse = stable_file_sha256(
        output, label="committed Level-B MP4 after marker parse"
    )
    receipt_sha_after_read, receipt_identity_after_read = stable_file_sha256(
        receipt, label="committed Level-B receipt after parse"
    )
    marker_sha_after_read, marker_identity_after_read = stable_file_sha256(
        marker, label="committed Level-B marker after parse"
    )
    transaction = payload.get("output_transaction") if isinstance(payload, Mapping) else None
    envelope = payload.get("commit_marker_envelope") if isinstance(payload, Mapping) else None
    envelope_unsigned = (
        {key: value for key, value in envelope.items() if key != "envelope_digest"}
        if isinstance(envelope, Mapping)
        else {}
    )
    expected_envelope = _commit_marker_envelope(
        output_path=output,
        receipt_path=receipt,
        commit_marker_path=marker,
        mp4_sha256=output_sha,
    )
    if (
        output_sha_after_parse != output_sha
        or output_identity_after_parse != output_identity
        or receipt_sha_after_read != receipt_sha
        or receipt_identity_after_read != receipt_identity
        or marker_sha_after_read != marker_sha
        or marker_identity_after_read != marker_identity
        or receipt_sha != marker_sha
        or receipt_identity != marker_identity
        or not isinstance(payload, Mapping)
        or not isinstance(marker_payload, Mapping)
        or canonical_json_bytes(marker_payload) != canonical_json_bytes(payload)
        or payload.get("schema_version") != RECEIPT_SCHEMA
        or payload.get("authority") != AUTHORITY
        or payload.get("complete") is not True
        or payload.get("receipt_digest")
        != object_sha256({key: value for key, value in payload.items() if key != "receipt_digest"})
        or not isinstance(transaction, Mapping)
        or transaction.get("atomic_commit_marker_schema")
        != OUTPUT_COMMIT_MARKER_SCHEMA
        or transaction.get("commit_marker_path") != str(marker)
        or transaction.get("commit_marker_is_only_consumer_completion_authority")
        is not True
        or transaction.get("bare_mp4_or_receipt_pair_is_never_complete")
        is not True
        or transaction.get("world8_canonical_gate_required_before_commit")
        is not True
        or transaction.get("all8_precommit_reopen_required_before_marker") is not True
        or transaction.get("marker_link_is_final_business_action") is not True
        or transaction.get("marker_is_receipt_inode_alias") is not True
        or not isinstance(envelope, Mapping)
        or dict(envelope) != dict(expected_envelope)
        or envelope.get("envelope_digest") != object_sha256(envelope_unsigned)
        or output_identity["mode"] != 0o444
        or receipt_identity["mode"] != 0o444
        or marker_identity["mode"] != 0o444
        or output.lstat().st_nlink != 1
        or receipt.lstat().st_nlink != 2
        or marker.lstat().st_nlink != 2
    ):
        fail("Level-B atomic commit-marker product closure differs")
    return {
        "mp4_sha256": output_sha,
        "receipt_sha256": receipt_sha,
        "commit_marker_sha256": marker_sha,
        "mp4_file_identity": output_identity,
        "receipt_file_identity": receipt_identity,
        "commit_marker_file_identity": marker_identity,
        "commit_marker": dict(envelope),
        "receipt_inode_alias_marker_verified": True,
        "bare_mp4_is_never_completion_authority": True,
        "atomic_commit_marker_is_only_completion_authority": True,
    }


def _validate_precommit_level_b_product_pair(
    *,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
) -> Mapping[str, Any]:
    """Open only the non-authoritative MP4+receipt prepared for WORLD8.

    This validator requires the public COMMITTED name to be absent.  It is an
    internal gate only; the public validator above deliberately rejects the
    same pair until the final marker link exists.
    """

    if (
        receipt_path != output_path.with_name(output_path.name + ".receipt.json")
        or commit_marker_path
        != output_path.with_name(output_path.name + ".COMMITTED.json")
    ):
        fail("Level-B precommit product names differ")
    try:
        commit_marker_path.lstat()
    except FileNotFoundError:
        pass
    except BaseException as error:
        raise LevelBRendererError(
            "cannot prove the Level-B COMMITTED marker is absent"
        ) from error
    else:
        fail("Level-B precommit pair unexpectedly has a COMMITTED marker")
    output = _plain_file(output_path, label="precommit Level-B MP4")
    receipt = _plain_file(receipt_path, label="precommit Level-B receipt")
    output_sha, output_identity = stable_file_sha256(
        output, label="precommit Level-B MP4"
    )
    receipt_sha, receipt_identity = stable_file_sha256(
        receipt, label="precommit Level-B receipt"
    )
    payload = _loads_strict_json(
        receipt.read_bytes(), label="precommit Level-B receipt"
    )
    output_sha_after, output_identity_after = stable_file_sha256(
        output, label="precommit Level-B MP4 after receipt parse"
    )
    receipt_sha_after, receipt_identity_after = stable_file_sha256(
        receipt, label="precommit Level-B receipt after parse"
    )
    transaction = payload.get("output_transaction") if isinstance(payload, Mapping) else None
    envelope = payload.get("commit_marker_envelope") if isinstance(payload, Mapping) else None
    expected_envelope = _commit_marker_envelope(
        output_path=output,
        receipt_path=receipt,
        commit_marker_path=commit_marker_path,
        mp4_sha256=output_sha,
    )
    if (
        output_sha_after != output_sha
        or output_identity_after != output_identity
        or receipt_sha_after != receipt_sha
        or receipt_identity_after != receipt_identity
        or not isinstance(payload, Mapping)
        or payload.get("schema_version") != RECEIPT_SCHEMA
        or payload.get("authority") != AUTHORITY
        or payload.get("complete") is not True
        or payload.get("receipt_digest")
        != object_sha256(
            {key: value for key, value in payload.items() if key != "receipt_digest"}
        )
        or not isinstance(transaction, Mapping)
        or transaction.get("atomic_commit_marker_schema")
        != OUTPUT_COMMIT_MARKER_SCHEMA
        or transaction.get("commit_marker_path") != str(commit_marker_path)
        or transaction.get("commit_marker_is_only_consumer_completion_authority")
        is not True
        or transaction.get("bare_mp4_or_receipt_pair_is_never_complete") is not True
        or transaction.get("all8_precommit_reopen_required_before_marker") is not True
        or transaction.get("marker_link_is_final_business_action") is not True
        or transaction.get("marker_is_receipt_inode_alias") is not True
        or not isinstance(envelope, Mapping)
        or dict(envelope) != dict(expected_envelope)
        or output_identity["mode"] != 0o444
        or receipt_identity["mode"] != 0o444
        or output.lstat().st_nlink != 1
        or receipt.lstat().st_nlink != 1
    ):
        fail("Level-B non-authoritative precommit pair closure differs")
    return {
        "mp4_sha256": output_sha,
        "receipt_sha256": receipt_sha,
        "mp4_file_identity": output_identity,
        "receipt_file_identity": receipt_identity,
        "commit_marker_absent": True,
        "public_validator_must_reject": True,
        "bare_mp4_or_receipt_pair_is_never_complete": True,
    }


def _nfs_reopen_precommit_product_pair(
    *,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    expected_mp4_sha256: str,
    expected_receipt_sha256: str,
) -> Mapping[str, Any]:
    reopened = dict(
        _validate_precommit_level_b_product_pair(
            output_path=output_path,
            receipt_path=receipt_path,
            commit_marker_path=commit_marker_path,
        )
    )
    if (
        reopened["mp4_sha256"] != expected_mp4_sha256
        or reopened["receipt_sha256"] != expected_receipt_sha256
    ):
        fail("NFS-reopened Level-B precommit bytes differ")
    return {
        **reopened,
        "directory_fsync_completed_before_reopen": True,
        "nfs_precommit_reopen_completed": True,
    }


def _nfs_reopen_product_pair(
    *,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    expected_mp4_sha256: str,
    expected_receipt_sha256: str,
    expected_commit_marker_sha256: str,
) -> Mapping[str, Any]:
    if (
        receipt_path != output_path.with_name(output_path.name + ".receipt.json")
        or commit_marker_path
        != output_path.with_name(output_path.name + ".COMMITTED.json")
    ):
        fail("Level-B NFS reopen product names differ")
    reopened = dict(
        validate_committed_level_b_product(output_mp4_path=output_path)
    )
    if (
        reopened["mp4_sha256"] != expected_mp4_sha256
        or reopened["receipt_sha256"] != expected_receipt_sha256
        or reopened["commit_marker_sha256"] != expected_commit_marker_sha256
    ):
        fail("NFS-reopened Level-B committed product bytes differ")
    return {
        **reopened,
        "directory_fsync_completed": True,
        "nfs_reopen_completed": True,
    }


def _publish_precommit_product_pair(
    *, transaction: _StagedProductPair, receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    if transaction.precommitted or transaction.committed:
        fail("Level-B output transaction was already precommitted")
    output = transaction.final_mp4_path
    receipt_output = transaction.final_receipt_path
    marker_output = transaction.final_commit_marker_path
    parent = _plain_directory(output.parent, label="Level-B output transaction parent")
    try:
        if (
            output.exists()
            or output.is_symlink()
            or receipt_output.exists()
            or receipt_output.is_symlink()
            or marker_output.exists()
            or marker_output.is_symlink()
        ):
            fail("Level-B create-only precommit product is no longer absent")
        receipt_stage = _write_new_staged_json(transaction.receipt_path, receipt)
        transaction.register_precommit_publication_identity(
            receipt_identity=receipt_stage["file_identity"],
        )
        # Publish only the non-authoritative pair.  No COMMITTED name exists
        # anywhere in this function, so even persistent cleanup failure cannot
        # leave a consumer-acceptable product.
        os.link(transaction.mp4_path, output, follow_symlinks=False)
        output_info = output.lstat()
        if (
            int(output_info.st_dev) != int(transaction.mp4_identity["device"])
            or int(output_info.st_ino) != int(transaction.mp4_identity["inode"])
        ):
            fail("create-only Level-B MP4 did not retain the staged inode")
        os.link(transaction.receipt_path, receipt_output, follow_symlinks=False)
        receipt_info = receipt_output.lstat()
        if (
            int(receipt_info.st_dev)
            != int(receipt_stage["file_identity"]["device"])
            or int(receipt_info.st_ino)
            != int(receipt_stage["file_identity"]["inode"])
        ):
            fail("create-only Level-B receipt did not retain the staged inode")
        _fsync_directory(parent, label="Level-B output parent precommit")
        transaction.cleanup_stage()
        _fsync_directory(
            parent, label="Level-B output parent after precommit stage removal"
        )
        reopened = _nfs_reopen_precommit_product_pair(
            output_path=output,
            receipt_path=receipt_output,
            commit_marker_path=marker_output,
            expected_mp4_sha256=transaction.mp4_sha256,
            expected_receipt_sha256=receipt_stage["sha256"],
        )
        if (
            reopened["mp4_file_identity"]["mode"] != 0o444
            or reopened["receipt_file_identity"]["mode"] != 0o444
            or output.lstat().st_nlink != 1
            or receipt_output.lstat().st_nlink != 1
        ):
            fail("precommit Level-B pair mode/link closure differs")
        transaction.precommitted = True
        return {
            "schema_version": OUTPUT_TRANSACTION_SCHEMA,
            **dict(reopened),
            "create_only_precommit_pair_publication": True,
            "commit_marker_absent_throughout_precommit": True,
            "bare_mp4_or_receipt_pair_never_completion_authority": True,
            "persistent_cleanup_failure_cannot_create_completion_authority": True,
        }
    except BaseException as precommit_error:
        cleanup_errors: list[BaseException] = []
        try:
            transaction.cleanup_publication()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            try:
                transaction.cleanup_stage()
            except BaseException as error:
                cleanup_errors.append(error)
        finally:
            try:
                _fsync_directory(
                    parent, label="Level-B output parent failed precommit"
                )
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise LevelBRendererError(
                "cannot remove every artifact after failed Level-B precommit"
            ) from cleanup_errors[0]
        raise precommit_error


@dataclass
class _FinalReceiptAliasMarkerPublisher:
    receipt_basename: str
    marker_basename: str
    parent_directory_fd: int
    link_function: Any
    armed: bool = True

    def publish(self) -> None:
        if not self.armed:
            fail("final Level-B marker publisher was already consumed")
        # Disarm before the irreversible syscall.  After a successful link the
        # method performs no lstat/hash/validator/collective/fsync/cleanup hook;
        # control only unwinds to the caller's return.
        self.armed = False
        self.link_function(
            self.receipt_basename,
            self.marker_basename,
            src_dir_fd=self.parent_directory_fd,
            dst_dir_fd=self.parent_directory_fd,
            follow_symlinks=False,
        )


def _prepare_final_receipt_alias_marker_publisher(
    *,
    transaction: _StagedProductPair,
    precommit_receipt: Mapping[str, Any],
    world8_gate_evidence: Mapping[str, Any],
) -> tuple[_FinalReceiptAliasMarkerPublisher, Mapping[str, Any]]:
    if not transaction.precommitted or transaction.committed:
        fail("final Level-B marker publisher requires one precommit pair")
    output = transaction.final_mp4_path
    receipt = transaction.final_receipt_path
    marker = transaction.final_commit_marker_path
    reopened = _nfs_reopen_precommit_product_pair(
        output_path=output,
        receipt_path=receipt,
        commit_marker_path=marker,
        expected_mp4_sha256=transaction.mp4_sha256,
        expected_receipt_sha256=str(precommit_receipt.get("receipt_sha256", "")),
    )
    receipt_payload = _loads_strict_json(
        receipt.read_bytes(), label="final marker source receipt"
    )
    envelope = (
        receipt_payload.get("commit_marker_envelope")
        if isinstance(receipt_payload, Mapping)
        else None
    )
    gate_contract = (
        envelope.get("world8_precommit_gate_contract")
        if isinstance(envelope, Mapping)
        else None
    )
    if (
        not isinstance(world8_gate_evidence, Mapping)
        or world8_gate_evidence.get("schema_version") != PRECOMMIT_GATE_SCHEMA
        or world8_gate_evidence.get("mp4_sha256") != reopened["mp4_sha256"]
        or world8_gate_evidence.get("receipt_sha256")
        != reopened["receipt_sha256"]
        or not isinstance(gate_contract, Mapping)
        or world8_gate_evidence.get("gate_contract_sha256")
        != object_sha256(gate_contract)
    ):
        fail("final marker WORLD8 precommit evidence differs")
    parent = _plain_directory(output.parent, label="final marker output parent")
    directory_fd = os.open(
        parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    os.fsync(directory_fd)
    with receipt.open("rb") as handle:
        os.fsync(handle.fileno())
    readiness = {
        "schema_version": PRECOMMIT_GATE_SCHEMA,
        "mp4_sha256": reopened["mp4_sha256"],
        "receipt_sha256": reopened["receipt_sha256"],
        "receipt_file_identity": reopened["receipt_file_identity"],
        "world8_gate_evidence_sha256": object_sha256(world8_gate_evidence),
        "gate_contract_sha256": object_sha256(gate_contract),
        "marker_is_receipt_inode_alias": True,
        "parent_directory_fd_opened_and_fsynced_before_marker": True,
        "receipt_fsynced_before_marker": True,
        "marker_link_is_terminal_filesystem_action": True,
    }
    return (
        _FinalReceiptAliasMarkerPublisher(
            receipt_basename=receipt.name,
            marker_basename=marker.name,
            parent_directory_fd=directory_fd,
            link_function=os.link,
        ),
        readiness,
    )


def _rollback_committed_product_pair(
    *,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    commit_receipt: Mapping[str, Any],
) -> None:
    _rollback_exact_level_b_publication(
        output_path=output_path,
        receipt_path=receipt_path,
        commit_marker_path=commit_marker_path,
        mp4_identity=commit_receipt.get("mp4_file_identity"),
        receipt_identity=commit_receipt.get("receipt_file_identity"),
        commit_marker_identity=commit_receipt.get(
            "commit_marker_file_identity"
        ),
    )


@dataclass
class _LevelBOutputRollbackGuard:
    """Own every private/public output until the full WORLD8 path returns."""

    output_path: Optional[Path] = None
    receipt_path: Optional[Path] = None
    commit_marker_path: Optional[Path] = None
    transaction: Optional[_StagedProductPair] = None
    armed: bool = True

    def bind_output_paths(
        self, *, output_path: Path, receipt_path: Path, commit_marker_path: Path
    ) -> None:
        if (
            not self.armed
            or self.output_path is not None
            or receipt_path != output_path.with_name(output_path.name + ".receipt.json")
            or commit_marker_path
            != output_path.with_name(output_path.name + ".COMMITTED.json")
        ):
            fail("Level-B output rollback guard path binding differs")
        self.output_path = output_path
        self.receipt_path = receipt_path
        self.commit_marker_path = commit_marker_path

    def register_stage(self, transaction: _StagedProductPair) -> None:
        if (
            not self.armed
            or self.output_path is None
            or self.transaction is not None
            or transaction.final_mp4_path != self.output_path
            or transaction.final_receipt_path != self.receipt_path
            or transaction.final_commit_marker_path != self.commit_marker_path
        ):
            fail("Level-B output rollback guard stage binding differs")
        self.transaction = transaction

    def rollback(self) -> None:
        if not self.armed:
            return
        cleanup_errors: list[BaseException] = []
        if self.transaction is not None:
            try:
                # The transaction seals this inode ledger before its first
                # os.link, so it also covers an exception in the narrow gap
                # after commit returns but before register_commit completes.
                self.transaction.cleanup_publication()
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                self.transaction.cleanup_stage()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise LevelBRendererError(
                "Level-B output rollback guard could not remove every artifact"
            ) from cleanup_errors[0]

    def disarm(self) -> None:
        if not self.armed:
            fail("Level-B output rollback guard was already disarmed")
        self.armed = False


def _run_with_level_b_output_rollback(operation: Any) -> Any:
    """Outer finally: output ownership is released only after a full return."""

    if not callable(operation):
        fail("Level-B guarded output operation differs")
    guard = _LevelBOutputRollbackGuard()
    completed = False
    try:
        result = operation(guard)
        completed = True
        return result
    finally:
        if guard.armed:
            if completed:
                guard.disarm()
            else:
                guard.rollback()


def _gather_world8_precommit_reopen_consensus(
    *,
    rank: int,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    expected_mp4_sha256: str,
    expected_receipt_sha256: str,
    expected_gate_contract: Mapping[str, Any],
    distributed_module: Any,
    group: Any,
) -> Mapping[str, Any]:
    """Gather exact all8 evidence while the COMMITTED name is still absent."""

    if type(rank) is not int or rank not in range(WORLD_SIZE):
        fail("WORLD8 precommit reopen rank differs")
    expected_gate_sha = object_sha256(expected_gate_contract)
    local_reopen: Mapping[str, Any]
    try:
        local_reopen = {
            "ok": True,
            "rank": rank,
            "receipt": _nfs_reopen_precommit_product_pair(
                output_path=output_path,
                receipt_path=receipt_path,
                commit_marker_path=commit_marker_path,
                expected_mp4_sha256=expected_mp4_sha256,
                expected_receipt_sha256=expected_receipt_sha256,
            ),
            "gate_contract_sha256": expected_gate_sha,
        }
    except BaseException as error:
        local_reopen = {
            "ok": False,
            "rank": rank,
            "error": f"{type(error).__name__}: {error}",
        }
    reopen_rows: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(reopen_rows, local_reopen, group=group)
    gate_failure = any(
        not isinstance(row, Mapping)
        or set(row) != {"ok", "rank", "receipt", "gate_contract_sha256"}
        or row.get("ok") is not True
        or type(row.get("rank")) is not int
        or row.get("rank") != expected_rank
        or not isinstance(row.get("receipt"), Mapping)
        or row.get("gate_contract_sha256") != expected_gate_sha
        for expected_rank, row in enumerate(reopen_rows)
    )
    if not gate_failure:
        try:
            canonical_reopen = canonical_json_bytes(reopen_rows[0]["receipt"])
            gate_failure = any(
                canonical_json_bytes(row["receipt"]) != canonical_reopen
                for row in reopen_rows
            )
        except BaseException:
            gate_failure = True
    if gate_failure:
        fail("WORLD8 precommit reopen/rank/hash consensus differs")
    canonical_reopen_sha = hashlib.sha256(
        canonical_json_bytes(reopen_rows[0]["receipt"])
    ).hexdigest()
    return {
        "schema_version": PRECOMMIT_GATE_SCHEMA,
        "world_size": WORLD_SIZE,
        "all8_precommit_nfs_reopen_exact_consensus": True,
        "ordered_reopen_ranks": list(range(WORLD_SIZE)),
        "canonical_reopen_receipt_sha256": canonical_reopen_sha,
        "gate_contract_sha256": expected_gate_sha,
        "mp4_sha256": expected_mp4_sha256,
        "receipt_sha256": expected_receipt_sha256,
        "commit_marker_absent_on_all8_reopens": True,
    }


def _rank_pattern_consensus(
    value: Mapping[str, Any],
    *,
    sp_rank: int,
    dp_rank: int,
    local_cuda_index: int,
    distributed_module: Any,
    group: Any,
) -> Mapping[str, Any]:
    if sp_rank not in range(SP_SIZE) or dp_rank not in range(DP_SIZE):
        fail("DP2xSP4 rank coordinate differs")
    normalized = _normalize_rank_local_cuda_indices(
        value, local_cuda_index=local_cuda_index
    )
    digest = object_sha256(normalized)
    local = {"dp_rank": dp_rank, "sp_rank": sp_rank, "sha256": digest}
    gathered: list[Any] = [None] * WORLD_SIZE
    distributed_module.all_gather_object(gathered, local, group=group)
    if any(
        not isinstance(row, Mapping)
        or set(row) != {"dp_rank", "sp_rank", "sha256"}
        for row in gathered
    ):
        fail("WORLD8 rank-local trace gather differs")
    ordered = sorted(
        (dict(row) for row in gathered),
        key=lambda row: (row["dp_rank"], row["sp_rank"]),
    )
    if [(row["dp_rank"], row["sp_rank"]) for row in ordered] != [
        (dp, sp) for dp in range(DP_SIZE) for sp in range(SP_SIZE)
    ]:
        fail("WORLD8 DP2xSP4 trace coordinate closure differs")
    by_sp = {
        sp: [row["sha256"] for row in ordered if row["sp_rank"] == sp]
        for sp in range(SP_SIZE)
    }
    if any(len(values) != DP_SIZE or len(set(values)) != 1 for values in by_sp.values()):
        fail("DP2 replicas disagree on rank-local Level-B renderer traces")
    return {
        "world_size": WORLD_SIZE,
        "dp2_rank_local_trace_pairs_exact": True,
        "rank_local_cuda_index_normalized": True,
        "ordered_rank_coordinates": ordered,
        "sp_rank_trace_sha256": [by_sp[sp][0] for sp in range(SP_SIZE)],
    }


def _make_authenticated_save_output_encoder(
    *, save_output: Any, runtime_receipt: Mapping[str, Any]
) -> tuple[Any, Mapping[str, Any]]:
    """Close official ``save_output`` over the one authenticated ffmpeg.

    Bernini's writer resolves imageio-ffmpeg through ``IMAGEIO_FFMPEG_EXE``.
    Set that selector only for the bounded call, verify imageio's live resolver
    chose the exact authenticated file, and restore the process environment in
    all outcomes.  The public product API never accepts an encoder or tool path.
    """

    ffmpeg_row = runtime_receipt.get("ffmpeg")
    if not callable(save_output) or not isinstance(ffmpeg_row, Mapping):
        fail("authenticated Bernini encoder authority is absent")
    ffmpeg = _plain_file(ffmpeg_row.get("path", ""), label="encoder ffmpeg")
    expected_sha = _require_sha(
        ffmpeg_row.get("sha256"), label="encoder ffmpeg SHA"
    )
    observed_sha, identity = stable_file_sha256(ffmpeg, label="encoder ffmpeg")
    if observed_sha != expected_sha or not os.access(ffmpeg, os.X_OK):
        fail("encoder ffmpeg bytes/executable authority differs")
    save_receipt = _canonical_callable_receipt(
        save_output, label="authenticated Bernini save_output encoder"
    )
    sentinel = object()
    encoder_environment_lock = threading.Lock()

    def encode(decoded_frames: Any, output_path: str, *, fps: int) -> Mapping[str, Any]:
        before_sha, before_identity = stable_file_sha256(
            ffmpeg, label="encoder ffmpeg immediately before encode"
        )
        if before_sha != expected_sha or before_identity != identity:
            fail("encoder ffmpeg changed before encode")
        use_signal_timeout = threading.current_thread() is threading.main_thread()

        def encoder_timeout_handler(_signum: int, _frame: Any) -> NoReturn:
            raise LevelBRendererError("authenticated ffmpeg encode exceeded wall timeout")

        with encoder_environment_lock:
            # Capture ambient state and enter its restoration scope before the
            # first signal/getitimer/setitimer call.  Even a hostile failure in
            # signal-state inspection cannot leak the authenticated tool path.
            previous = os.environ.get("IMAGEIO_FFMPEG_EXE", sentinel)
            previous_handler: Any = sentinel
            restore_handler = False
            cancel_timer = False
            signal_cleanup_errors: list[BaseException] = []
            try:
                os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg)
                import imageio_ffmpeg

                selected = _plain_file(
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    label="imageio selected encoder ffmpeg",
                )
                if selected != ffmpeg:
                    fail("imageio did not select the authenticated encoder ffmpeg")
                if use_signal_timeout:
                    previous_handler = signal.getsignal(signal.SIGALRM)
                    previous_timer = signal.getitimer(signal.ITIMER_REAL)
                    if previous_timer != (0.0, 0.0):
                        fail(
                            "refusing to replace an active process wall timer for ffmpeg"
                        )
                    # Arm restoration before each mutating call so a wrapper
                    # that performs the real operation and then raises is safe.
                    restore_handler = True
                    signal.signal(signal.SIGALRM, encoder_timeout_handler)
                    cancel_timer = True
                    signal.setitimer(
                        signal.ITIMER_REAL,
                        float(FFMPEG_WALL_TIMEOUT_SECONDS),
                    )
                save_output(decoded_frames, output_path, fps=fps)
            finally:
                try:
                    if use_signal_timeout:
                        if cancel_timer:
                            try:
                                signal.setitimer(signal.ITIMER_REAL, 0.0)
                            except BaseException as error:
                                signal_cleanup_errors.append(error)
                        if restore_handler and previous_handler is not sentinel:
                            try:
                                signal.signal(signal.SIGALRM, previous_handler)
                            except BaseException as error:
                                signal_cleanup_errors.append(error)
                finally:
                    if previous is sentinel:
                        os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
                    else:
                        os.environ["IMAGEIO_FFMPEG_EXE"] = str(previous)
                if signal_cleanup_errors:
                    raise LevelBRendererError(
                        "cannot restore authenticated encoder signal state"
                    ) from signal_cleanup_errors[0]
        after_sha, after_identity = stable_file_sha256(
            ffmpeg, label="encoder ffmpeg immediately after encode"
        )
        if after_sha != expected_sha or after_identity != identity:
            fail("encoder ffmpeg changed during encode")
        return {
            "authenticated_ffmpeg_path": str(ffmpeg),
            "authenticated_ffmpeg_sha256": expected_sha,
            "imageio_ffmpeg_exe_explicitly_bound": True,
            "ambient_encoder_path_ignored_and_restored": True,
            "encoder_wall_timeout_seconds": FFMPEG_WALL_TIMEOUT_SECONDS,
            "encoder_timeout_mode": (
                "sigalrm-main-thread"
                if use_signal_timeout
                else "no-signal-nonmain-thread"
            ),
            "nonmain_thread_signal_api_avoided": not use_signal_timeout,
            "encoder_timeout_failure_enters_world8_phase_envelope": True,
        }

    return encode, {
        "authenticated_ffmpeg_path": str(ffmpeg),
        "authenticated_ffmpeg_sha256": expected_sha,
        "authenticated_ffmpeg_file_identity": identity,
        "official_save_output_callable": save_receipt,
        "sealed_encoder_closure": True,
        "encoder_wall_timeout_seconds": FFMPEG_WALL_TIMEOUT_SECONDS,
        "nonmain_thread_signal_api_forbidden": True,
        "ambient_environment_restored_on_signal_api_failure": True,
        "encoder_is_not_a_product_input": True,
    }


def _stage_real_mp4(
    *,
    decoded_frames: Any,
    output_path: Path,
    receipt_path: Path,
    commit_marker_path: Path,
    authenticated_encoder: Any,
    encoder_authority: Mapping[str, Any],
    expected_height: int,
    expected_width: int,
    runtime_receipt: Mapping[str, Any],
    output_rollback_guard: _LevelBOutputRollbackGuard,
) -> tuple[_StagedProductPair, Mapping[str, Any]]:
    parent = _plain_directory(output_path.parent, label="Level-B staging parent")
    staging: Optional[Path] = None
    temporary: Optional[Path] = None
    staged_receipt: Optional[Path] = None
    staged_marker: Optional[Path] = None
    try:
        # Pre-bind the candidate path before the create syscall.  Unlike an
        # opaque mkdtemp return, this lets cleanup find the directory even if a
        # hostile wrapper performs the real mkdir and then raises.
        for _attempt in range(128):
            staging = parent / (
                f".{output_path.name}.stage-{os.urandom(8).hex()}"
            )
            try:
                os.mkdir(staging, 0o700)
            except FileExistsError:
                staging = None
                continue
            break
        else:
            fail("cannot allocate a unique private Level-B staging directory")
        if staging is None:
            fail("private Level-B staging directory allocation differs")
        temporary = staging / "payload.mp4"
        staged_receipt = staging / "receipt.json"
        staged_marker = staging / "COMMITTED.json"
        # Arm the outer owner before chmod/encode.  MP4 hash/identity are filled
        # only after validation, but private-stage cleanup needs paths alone.
        transaction = _StagedProductPair(
            directory=staging,
            mp4_path=temporary,
            receipt_path=staged_receipt,
            commit_marker_path=staged_marker,
            final_mp4_path=output_path,
            final_receipt_path=receipt_path,
            final_commit_marker_path=commit_marker_path,
            mp4_sha256="",
            mp4_identity={},
        )
        output_rollback_guard.register_stage(transaction)
        # The private directory is cleanup-owned from the instant mkdir
        # succeeds; chmod and every later action are inside this BaseException
        # envelope and the outer transaction guard.
        os.chmod(staging, 0o700)
        encoder_execution = authenticated_encoder(
            decoded_frames, str(temporary), fps=int(FPS)
        )
        temporary = _plain_file(temporary, label="staged Level-B MP4")
        os.chmod(temporary, 0o444)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        validation = dict(
            _validate_full_mp4_bytes_with_authenticated_tools(
                mp4_path=temporary,
                expected_height=expected_height,
                expected_width=expected_width,
                ffmpeg_path=runtime_receipt["ffmpeg"]["path"],
                ffprobe_path=runtime_receipt["ffprobe"]["path"],
                expected_ffmpeg_sha256=runtime_receipt["ffmpeg"]["sha256"],
                expected_ffprobe_sha256=runtime_receipt["ffprobe"]["sha256"],
            )
        )
        if output_path.exists() or output_path.is_symlink():
            fail("Level-B output appeared before transaction staging completed")
        if receipt_path.exists() or receipt_path.is_symlink():
            fail("Level-B receipt appeared before transaction staging completed")
        if commit_marker_path.exists() or commit_marker_path.is_symlink():
            fail("Level-B commit marker appeared before staging completed")
        _fsync_directory(staging, label="Level-B product staging directory")
        transaction.mp4_sha256 = validation["mp4_sha256"]
        transaction.mp4_identity = validation["mp4_file_identity"]
        validation.update(
            {
                "mp4_path": str(output_path),
                "validated_staging_name": "payload.mp4",
                "mp4_and_receipt_staged_before_publication": True,
                "publication_deferred_until_world8_canonical_gate": True,
                "encoder_authority": dict(encoder_authority),
                "encoder_execution": dict(encoder_execution),
            }
        )
        return transaction, validation
    except BaseException as stage_error:
        cleanup_errors: list[BaseException] = []
        for path in (staged_marker, staged_receipt, temporary):
            if path is None:
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            except BaseException as error:
                cleanup_errors.append(error)
                continue
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                cleanup_errors.append(
                    LevelBRendererError(
                        "refusing to clean a replaced private Level-B stage"
                    )
                )
                continue
            try:
                path.unlink()
            except BaseException as error:
                cleanup_errors.append(error)
        if staging is not None:
            try:
                _fsync_directory(
                    staging, label="failed private Level-B staging directory"
                )
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                staging.rmdir()
            except FileNotFoundError:
                pass
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            _fsync_directory(parent, label="failed Level-B staging parent")
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise LevelBRendererError(
                "cannot durably remove every failed private Level-B stage"
            ) from cleanup_errors[0]
        raise stage_error


def _run_level_b_pre_d0_offline_inference_authenticated_core(
    *,
    fresh_bundle: Any,
    runtime_receipt: Mapping[str, Any],
    source_video_path: str,
    expected_source_video_sha256: str,
    edit_instruction: str,
    inference_seed: int,
    output_mp4_path: str,
    output_rollback_guard: _LevelBOutputRollbackGuard,
) -> Mapping[str, Any]:
    """Run one real WORLD8 Bernini render from source+instruction+seed only."""

    validate_public_product_signature()
    consumer_receipt = _validate_fresh_bundle(fresh_bundle)
    product = _resolve_authenticated_product_module(fresh_bundle)
    runtime = runtime_receipt
    output_path, receipt_path, commit_marker_path = _prepare_absent_output(
        output_mp4_path
    )
    output_rollback_guard.bind_output_paths(
        output_path=output_path,
        receipt_path=receipt_path,
        commit_marker_path=commit_marker_path,
    )

    policy = product.OfflineInferencePolicyV1(seed=inference_seed)
    policy.validate()
    request = product.ProductRequestV1(
        source_video_path=source_video_path,
        expected_source_video_sha256=expected_source_video_sha256,
        instruction=edit_instruction,
        inference_policy=policy,
    )
    request_receipt = request.validate()
    source_path = Path(source_video_path)

    # ``tools.materialize_vae`` is installed under the exact name imported by
    # the authenticated official inference source.  This prevents a same-name
    # module from changing decode/bucket semantics after the source SHA check.
    release_members = runtime["release"]["members"]
    materialize_row = release_members["tools/materialize_vae.py"]
    builder_row = release_members["tools/build_renderer_dataset.py"]
    materialize_path = Path(materialize_row["path"])
    _install_authenticated_materialize_module(
        path=materialize_path,
        expected_sha256=materialize_row["sha256"],
        builder_path=Path(builder_row["path"]),
        expected_builder_sha256=builder_row["sha256"],
    )
    authority_row = release_members[
        "action_preservation_decoded_eval_model_authority_v2.py"
    ]
    model_authority = _install_authenticated_named_module(
        name="action_preservation_decoded_eval_model_authority_v2",
        path=Path(authority_row["path"]),
        expected_sha256=authority_row["sha256"],
    )
    official_row = release_members["infer_lora.py"]
    official_name = (
        "_bernini_action_edit_level_b_official_inference_"
        + official_row["sha256"][:16]
    )
    official = _authenticated_source_module(
        path=Path(official_row["path"]),
        expected_sha256=official_row["sha256"],
        module_name=official_name,
    )
    bernini_root = Path(runtime["bernini_root"])
    base_checkpoint = Path(runtime["base_checkpoint"])
    if consumer_receipt.get("official_bernini_commit") != getattr(
        fresh_bundle.checkpoint.runner, "BERNINI_COMMIT", None
    ):
        fail("fresh consumer and training runner Bernini revisions differ")
    legacy_trainer = sys.modules.get("train_lora")
    release_closure = getattr(
        getattr(fresh_bundle.checkpoint, "checkpoint", None), "metadata", {}
    ).get("release_closure")
    release_rows = (
        release_closure.get("members")
        if isinstance(release_closure, Mapping)
        else None
    )
    legacy_row = next(
        (
            row
            for row in release_rows or ()
            if isinstance(row, Mapping) and row.get("path") == "train_lora.py"
        ),
        None,
    )
    if (
        legacy_trainer is None
        or getattr(official, "trainer", None) is not legacy_trainer
        or getattr(official, "model_authority", None) is not model_authority
        or not isinstance(legacy_row, Mapping)
    ):
        fail("official inference orchestration transitive module ownership differs")
    legacy_path = _plain_file(
        Path(getattr(legacy_trainer, "__file__", "")).resolve(strict=True),
        label="authenticated train_lora source",
    )
    legacy_sha, _ = stable_file_sha256(
        legacy_path, label="authenticated train_lora source"
    )
    if (
        legacy_sha != legacy_row.get("sha256")
        or legacy_path.name != "train_lora.py"
    ):
        fail("official inference trainer release member differs")
    if consumer_receipt.get("base_checkpoint_tree_sha256") != getattr(
        fresh_bundle.checkpoint.runner, "CHECKPOINT_TREE_SHA256", None
    ):
        fail("fresh consumer and Level-B base checkpoint tree differ")
    try:
        official_source_hashes = official.validate_inference_source_files(bernini_root)
    except Exception as error:
        raise LevelBRendererError("official Bernini inference source audit failed") from error
    expected_inference_hashes = {
        name: PINNED_BERNINI_RUNTIME_FILE_HASHES[name]
        for name in ("bernini/cli.py", "bernini/io_utils.py", "bernini/pipeline.py")
    }
    if official_source_hashes != expected_inference_hashes:
        fail("official Bernini inference file closure differs from compile-time pins")

    import gc
    import numpy as np
    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        AutoencoderKLWan as NativeAutoencoderKLWan,
    )
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from diffusers.utils.torch_utils import randn_tensor as canonical_randn_tensor
    from transformers import AutoTokenizer, __version__ as transformers_version
    from transformers.models.t5.tokenization_t5 import T5Tokenizer
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.pipeline import _vae_decode, _vae_encode
    import bernini.models.wan_diffusion as wan_diffusion

    # Freeze the exact audited bound descriptors before any source/model work;
    # later calls use these objects rather than performing a fresh mutable
    # class-attribute lookup after the provenance gate.
    vae_load_config = AutoencoderKLWan.load_config
    vae_from_pretrained = AutoencoderKLWan.from_pretrained
    tokenizer_from_pretrained = AutoTokenizer.from_pretrained
    vendor_callable_closure = _audit_live_vendor_callable_closure(
        runtime_receipt=runtime,
        diffusers_version=diffusers_version,
        transformers_version=transformers_version,
        vae_class=AutoencoderKLWan,
        tokenizer_factory_class=AutoTokenizer,
        vae_load_config=vae_load_config,
        vae_from_pretrained=vae_from_pretrained,
        tokenizer_from_pretrained=tokenizer_from_pretrained,
        prompt_cleaner=prompt_clean,
        noise_factory=canonical_randn_tensor,
        vae_encode=_vae_encode,
        vae_decode=_vae_decode,
        video_save=save_output,
        scheduler=fresh_bundle.renderer.diff_dec.scheduler,
    )
    if AutoencoderKLWan is not NativeAutoencoderKLWan:
        fail("public/native AutoencoderKLWan class identity differs")
    authenticated_encoder, encoder_authority = (
        _make_authenticated_save_output_encoder(
            save_output=save_output, runtime_receipt=runtime
        )
    )

    distributed = fresh_bundle.distributed
    parallel = fresh_bundle.parallel
    rank = int(distributed.rank)
    device = fresh_bundle.device
    device_object = torch.device(device)
    local_cuda_index = device_object.index
    local_rank = int(getattr(distributed, "local_rank", -1))
    sp_rank = int(distributed.sp_rank)
    dp_rank = int(getattr(distributed, "dp_rank", rank // SP_SIZE))
    if (
        not dist.is_initialized()
        or dist.get_world_size(group=parallel.world_group) != WORLD_SIZE
        or device_object.type != "cuda"
        or type(local_cuda_index) is not int
        or local_cuda_index != local_rank
        or local_rank not in range(WORLD_SIZE)
        or sp_rank not in range(SP_SIZE)
        or dp_rank not in range(DP_SIZE)
        or (dp_rank, sp_rank) != (rank // SP_SIZE, rank % SP_SIZE)
    ):
        fail("Level-B product render requires the loaded WORLD8 accelerator group")
    base_checkpoint_binding = _run_world8_rank0_collective_phase(
        phase="base_checkpoint_binding",
        rank=rank,
        operation=lambda: _validate_live_base_checkpoint_binding(
            consumer_receipt=consumer_receipt,
            fresh_bundle=fresh_bundle,
            runtime=runtime,
        ),
        distributed_module=dist,
        group=parallel.world_group,
    )
    if (
        not isinstance(base_checkpoint_binding, Mapping)
        or base_checkpoint_binding.get(
            "every_non_cache_file_rehashed_at_level_b_use"
        )
        is not True
    ):
        fail("rank-0 live base-checkpoint binding broadcast differs")
    consumer_box: list[Any] = [dict(consumer_receipt) if rank == 0 else None]
    dist.broadcast_object_list(consumer_box, src=0, group=parallel.world_group)
    rank0_consumer_receipt = consumer_box[0]
    if not isinstance(rank0_consumer_receipt, Mapping):
        fail("rank-0 checkpoint consumer receipt broadcast differs")
    if DEFAULT_NEG_PROMPT != official.DEFAULT_NEGATIVE_PROMPT:
        fail("official Bernini negative prompt differs")
    wan_path = _plain_file(
        bernini_root / "bernini/models/wan_diffusion.py",
        label="official wan_diffusion source",
    )
    wan_sha, _ = stable_file_sha256(wan_path, label="official wan_diffusion source")
    pinned_bernini_files = getattr(
        legacy_trainer, "BERNINI_PINNED_FILE_HASHES", None
    )
    if (
        Path(getattr(wan_diffusion, "__file__", "")).resolve(strict=True)
        != wan_path
        or not isinstance(pinned_bernini_files, Mapping)
        or wan_sha
        != pinned_bernini_files.get("bernini/models/wan_diffusion.py")
    ):
        fail("imported wan_diffusion module path/bytes differ")

    source_phase_local: dict[str, Any] = {}

    def prepare_source_on_rank0() -> Mapping[str, Any]:
        try:
            tensor, metadata = official.prepare_exact_source(source_path)
        except Exception as error:
            raise LevelBRendererError("exact81 source preprocessing failed") from error
        digest = tensor_sha256(tensor, torch_module=torch)
        source_phase_local["tensor"] = tensor
        return {"metadata": metadata, "tensor_sha256": digest}

    source_phase = _run_world8_rank0_collective_phase(
        phase="source_preprocessing",
        rank=rank,
        operation=prepare_source_on_rank0,
        distributed_module=dist,
        group=parallel.world_group,
    )
    source_metadata = (
        source_phase.get("metadata") if isinstance(source_phase, Mapping) else None
    )
    source_tensor_sha = (
        source_phase.get("tensor_sha256")
        if isinstance(source_phase, Mapping)
        else None
    )
    source_tensor = source_phase_local.get("tensor") if rank == 0 else None
    if (
        not isinstance(source_metadata, Mapping)
        or source_metadata.get("frame_count") != FRAME_COUNT
        or source_metadata.get("source_derived_bucket_hw") is None
        or not isinstance(source_tensor_sha, str)
    ):
        fail("broadcast exact81 source preprocessing receipt differs")
    bucket_h, bucket_w = (
        int(source_metadata["source_derived_bucket_hw"][0]),
        int(source_metadata["source_derived_bucket_hw"][1]),
    )

    vae_config = vae_load_config(
        str(base_checkpoint), subfolder="vae", local_files_only=True
    )
    expected_latent_shape = (
        1,
        int(vae_config["z_dim"]),
        PHASES,
        bucket_h // 8,
        bucket_w // 8,
    )
    vae_phase_local: dict[str, Any] = {}

    def encode_source_with_vae_on_rank0() -> Mapping[str, Any]:
        vae_value = vae_from_pretrained(
            str(base_checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        callable_before = _audit_loaded_vae_callables(
            vae_value,
            expected_vae_class=AutoencoderKLWan,
            runtime_receipt=runtime,
        )
        vae_value.eval().requires_grad_(False).to(device)
        if any(parameter.requires_grad for parameter in vae_value.parameters()):
            fail("source VAE is not frozen")
        with torch.no_grad():
            latent = _vae_encode(
                vae_value, source_tensor.to(device=device, dtype=torch.float32)
            ).contiguous()
        if (
            tuple(int(x) for x in latent.shape) != expected_latent_shape
            or latent.dtype != torch.float32
            or latent.requires_grad
            or not bool(torch.isfinite(latent).all().item())
        ):
            fail("frozen source VAE latent contract differs")
        callable_after = _audit_loaded_vae_callables(
            vae_value,
            expected_vae_class=AutoencoderKLWan,
            runtime_receipt=runtime,
        )
        if canonical_json_bytes(callable_after) != canonical_json_bytes(callable_before):
            fail("loaded VAE callable identities changed during encode")
        vae_phase_local["vae"] = vae_value
        vae_phase_local["source_latent"] = latent
        return {"callable_identity": callable_after}

    vae_phase = _run_world8_rank0_collective_phase(
        phase="vae_load_and_source_encode",
        rank=rank,
        operation=encode_source_with_vae_on_rank0,
        distributed_module=dist,
        group=parallel.world_group,
    )
    loaded_vae_callables = (
        vae_phase.get("callable_identity")
        if isinstance(vae_phase, Mapping)
        else None
    )
    if not isinstance(loaded_vae_callables, Mapping):
        fail("rank-0 loaded VAE callable receipt differs")
    vae = vae_phase_local.get("vae") if rank == 0 else None
    if rank == 0:
        source_latent = vae_phase_local.get("source_latent")
        retained_source_tensor = _pop_exact_phase_owned_value(
            source_phase_local,
            "tensor",
            source_tensor,
            label="rank-0 preprocessed source tensor",
        )
        _pop_exact_phase_owned_value(
            vae_phase_local,
            "vae",
            vae,
            label="rank-0 loaded VAE",
        )
        _pop_exact_phase_owned_value(
            vae_phase_local,
            "source_latent",
            source_latent,
            label="rank-0 source latent",
        )
        if source_phase_local or vae_phase_local:
            fail("rank-0 phase-local tensor ownership did not close")
        del retained_source_tensor
        del source_tensor
    else:
        source_latent = torch.empty(
            expected_latent_shape, device=device, dtype=torch.float32
        )
    dist.broadcast(source_latent, src=0, group=parallel.world_group)
    source_latent_sha = tensor_sha256(source_latent, torch_module=torch)
    source_latent_consensus = _world8_string_consensus(
        source_latent_sha,
        distributed_module=dist,
        group=parallel.world_group,
    )
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = tokenizer_from_pretrained(
        str(base_checkpoint), subfolder="tokenizer", **official.tokenizer_load_kwargs()
    )
    loaded_tokenizer_callable = _audit_loaded_tokenizer_callable(
        tokenizer,
        expected_tokenizer_class=T5Tokenizer,
        runtime_receipt=runtime,
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        fail("frozen T5 tokenizer runtime contract differs")
    full_prompt = official.build_training_prompt(
        edit_instruction, prompt_cleaner=prompt_clean
    )
    input_ids, attention_mask = official._tokenize_training_prompt(
        tokenizer, full_prompt
    )
    negative_ids, negative_mask = official._tokenize_renderer_negative(
        tokenizer, DEFAULT_NEG_PROMPT
    )
    actual_input_token_count = int(attention_mask.sum().item())
    if not 0 < actual_input_token_count <= 512:
        fail("actual frozen-T5 input token count differs")
    # Token IDs/masks and the callable receipt are the complete downstream
    # authority.  Do not retain the tokenizer object through denoise/decode.
    del tokenizer
    gc.collect()
    t5 = getattr(fresh_bundle.renderer, "t5_text_encoder", None)
    if (
        t5 is None
        or bool(getattr(t5, "training", True))
        or any(parameter.requires_grad for parameter in t5.parameters())
    ):
        fail("fresh renderer T5 encoder is not frozen/eval")

    sampling = official.sampler_contract(
        steps=NUM_INFERENCE_STEPS, seed=inference_seed
    )
    if (
        sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or sampling.get("guidance_mode") != GUIDANCE_MODE
        or sampling.get("flow_shift") != FLOW_SHIFT
        or sampling.get("seed") != inference_seed
    ):
        fail("official renderer sampler kwargs differ")
    sample_kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "uncond_input_ids": negative_ids.to(device),
        "uncond_attention_mask": negative_mask.to(device),
        "image_vae_latents": None,
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": None,
        "width": bucket_w,
        "height": bucket_h,
        "device": device,
        **sampling,
    }
    if (
        sample_kwargs["image_vae_latents"] is not None
        or sample_kwargs["multi_image_vae_latents"] is not None
        or not isinstance(sample_kwargs["multi_video_vae_latents"], list)
        or len(sample_kwargs["multi_video_vae_latents"]) != 1
        or sample_kwargs["multi_video_vae_latents"][0] is not source_latent
    ):
        fail("renderer received a condition other than the single clean source video")

    # Freeze the actual loaded call boundary before any denoise or output side
    # effect.  ``Signature.bind`` below is against the live bound
    # ``BerniniRendererModel.sample`` method, not a locally copied signature.
    renderer_sample_call_audit = audit_official_renderer_sample_call(
        renderer=fresh_bundle.renderer,
        sample_kwargs=sample_kwargs,
    )
    diffusion = fresh_bundle.renderer.diff_dec
    sample_pair_provenance = _audit_renderer_sample_pair_against_runtime(
        renderer=fresh_bundle.renderer,
        diffusion=diffusion,
        runtime_receipt=runtime,
        torch_module=torch,
    )
    renderer_sample_provenance = sample_pair_provenance["renderer_sample"]
    native_diffusion_sample_provenance = sample_pair_provenance[
        "native_diffusion_sample"
    ]
    renderer_sample_call_audit = {
        **dict(renderer_sample_call_audit),
        "callable_provenance": renderer_sample_provenance,
    }
    renderer_condition_tensor_sha256_before = {
        name: tensor_sha256(sample_kwargs[name], torch_module=torch)
        for name in (
            "input_ids",
            "attention_mask",
            "uncond_input_ids",
            "uncond_attention_mask",
        )
    }
    renderer_condition_tensor_sha256_before[
        "single_multi_video_vae_latent"
    ] = source_latent_sha
    sample_kwargs_receipt = {
        "exact_keyword_names": sorted(sample_kwargs),
        "exact_keyword_names_in_call_order": list(sample_kwargs),
        "input_ids_shape": [int(x) for x in input_ids.shape],
        "input_ids_sha256": tensor_sha256(input_ids.contiguous(), torch_module=torch),
        "attention_mask_sha256": tensor_sha256(
            attention_mask.contiguous(), torch_module=torch
        ),
        "actual_input_token_count": actual_input_token_count,
        "negative_ids_sha256": tensor_sha256(
            negative_ids.contiguous(), torch_module=torch
        ),
        "negative_attention_mask_sha256": tensor_sha256(
            negative_mask.contiguous(), torch_module=torch
        ),
        "single_multi_video_vae_latent_sha256": source_latent_sha,
        "image_vae_latents": None,
        "multi_image_vae_latents": None,
        "width": bucket_w,
        "height": bucket_h,
        "device": str(torch.device(device)),
        "sampling": dict(sampling),
        "live_condition_tensor_sha256_before": dict(
            renderer_condition_tensor_sha256_before
        ),
        "target_anchor_teacher_external_annotation_kwargs_present": False,
        "caller_callback_or_custom_denoiser_present": False,
    }
    callable_provenance = {
        "level_b_entrypoint": _canonical_callable_receipt(
            run_level_b_pre_d0_offline_inference, label="Level-B entrypoint"
        ),
        "official_source_preprocessor": _canonical_callable_receipt(
            official.prepare_exact_source, label="official source preprocessor"
        ),
        "official_prompt_builder": _canonical_callable_receipt(
            official.build_training_prompt, label="official prompt builder"
        ),
        "official_training_prompt_tokenizer": _canonical_callable_receipt(
            official._tokenize_training_prompt,
            label="official training prompt tokenizer",
        ),
        "official_vae_encode": _canonical_callable_receipt(
            _vae_encode, label="official VAE encode"
        ),
        "official_vae_decode": _canonical_callable_receipt(
            _vae_decode, label="official VAE decode"
        ),
        "official_video_save": _canonical_callable_receipt(
            save_output, label="official video save"
        ),
        "official_prompt_cleaner": _canonical_callable_receipt(
            prompt_clean, label="official prompt cleaner"
        ),
        "native_noise_factory": _canonical_callable_receipt(
            canonical_randn_tensor, label="native noise factory"
        ),
        "product_route_builder": _canonical_callable_receipt(
            product.prepare_product_route_from_packed_embeddings,
            label="authenticated product route builder",
        ),
        "product_route_activation": _canonical_callable_receipt(
            product.activate_offline_route,
            label="authenticated product route activation",
        ),
        "live_scheduler_audit": _canonical_callable_receipt(
            product.audit_live_inference_scheduler,
            label="authenticated live scheduler audit",
        ),
        "renderer_sample": renderer_sample_provenance,
        "native_diffusion_sample": native_diffusion_sample_provenance,
        "native_shared_step": _canonical_callable_receipt(
            fresh_bundle.renderer.diff_dec.shared_step,
            label="native shared step",
        ),
        "native_unipc_step": _canonical_callable_receipt(
            fresh_bundle.renderer.diff_dec.scheduler.step,
            label="native UniPC step",
        ),
        "persisted_action_residual": _canonical_callable_receipt(
            fresh_bundle.conditioner.injection.residual,
            label="persisted action residual",
        ),
    }
    renderer_source = _plain_file(
        bernini_root / "bernini/models/renderer.py",
        label="official renderer source",
    )
    predictor_source = _plain_file(
        Path(getattr(fresh_bundle.checkpoint.predictor_module, "__file__", ""))
        .resolve(strict=True),
        label="authenticated predictor source",
    )
    expected_callable_files = {
        "renderer_sample": (
            renderer_source,
            pinned_bernini_files.get("bernini/models/renderer.py"),
        ),
        "native_diffusion_sample": (wan_path, wan_sha),
        "native_shared_step": (wan_path, wan_sha),
        "persisted_action_residual": (
            predictor_source,
            consumer_receipt.get("predictor_source_sha256"),
        ),
    }
    for callable_name, (expected_path, expected_sha) in expected_callable_files.items():
        row = callable_provenance[callable_name]
        if (
            row.get("source_file") != str(expected_path)
            or row.get("source_file_sha256") != expected_sha
        ):
            fail(f"{callable_name} is not owned by its authenticated source file")
    native_signature_requirements = {
        "native_diffusion_sample": (
            "prompt_embeds",
            "uncond_prompt_embeds",
            "num_inference_steps",
            "guidance_mode",
            "flow_shift",
            "seed",
        ),
        "native_shared_step": (
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
        ),
        "native_unipc_step": ("model_output", "timestep", "sample"),
    }
    native_signature_audit = {}
    for callable_name, required_names in native_signature_requirements.items():
        signature_rows = callable_provenance[callable_name]["runtime_signature"][
            "parameters"
        ]
        declared_names = tuple(row["name"] for row in signature_rows)
        missing = sorted(set(required_names) - set(declared_names))
        if missing:
            fail(f"{callable_name} signature lacks pinned branch fields: {missing}")
        native_signature_audit[callable_name] = {
            "required_explicit_parameter_names": list(required_names),
            "live_declared_parameter_names": list(declared_names),
            "all_required_explicitly_declared": True,
        }

    observer = NativeInitialNoiseObserver(
        wan_diffusion_module=wan_diffusion,
        canonical_randn_tensor=canonical_randn_tensor,
        expected_shape=expected_latent_shape,
        expected_device=device,
        expected_seed=inference_seed,
        torch_module=torch,
    )
    patch_grid = (PHASES, expected_latent_shape[3] // 2, expected_latent_shape[4] // 2)
    row_identity = object_sha256(
        {
            "source_video_sha256": expected_source_video_sha256,
            "instruction_utf8_sha256": request_receipt["instruction_utf8_sha256"],
            "seed": inference_seed,
        }
    )
    rng_before = _rng_snapshot(torch)
    bridge: InstalledNativeActionRendererBridge
    loaded_parameter_after: Optional[str] = None
    with observe_native_initial_noise(observer):
        with native_action_renderer_bridge(
            fresh_bundle=fresh_bundle,
            product_module=product,
            inference_policy=policy,
            patch_grid=patch_grid,
            row_identity=row_identity,
            torch_module=torch,
            noise_observer=observer,
            source_condition_list=sample_kwargs["multi_video_vae_latents"],
            expected_width=bucket_w,
            expected_height=bucket_h,
            expected_device=device,
            expected_internal_sampling=sampling,
            expected_instruction_token_count=actual_input_token_count,
        ) as bridge:
            with torch.no_grad():
                generated_latent = fresh_bundle.renderer.sample(**sample_kwargs)
            trainable_named = (
                fresh_bundle.checkpoint.runner.exact_trainable_named_parameters(
                    fresh_bundle.model, fresh_bundle.conditioner
                )
            )
            loaded_parameter_after = fresh_bundle.checkpoint.runner.tensor_digest(
                trainable_named
            )
    rng_after = _rng_snapshot(torch)
    if not _rng_equal(rng_before, rng_after, torch_module=torch):
        fail("official counter-based inference changed process-global RNG state")
    if (
        tuple(int(x) for x in generated_latent.shape) != expected_latent_shape
        or generated_latent.dtype != torch.float32
        or not bool(torch.isfinite(generated_latent).all().item())
    ):
        fail("full Bernini exact40 result latent differs")
    renderer_condition_tensor_sha256_after = {
        name: tensor_sha256(sample_kwargs[name], torch_module=torch)
        for name in (
            "input_ids",
            "attention_mask",
            "uncond_input_ids",
            "uncond_attention_mask",
        )
    }
    renderer_condition_tensor_sha256_after[
        "single_multi_video_vae_latent"
    ] = tensor_sha256(source_latent, torch_module=torch)
    if (
        renderer_condition_tensor_sha256_after
        != renderer_condition_tensor_sha256_before
    ):
        fail("official renderer mutated a source or instruction condition tensor")
    sample_kwargs_receipt["live_condition_tensor_sha256_after"] = dict(
        renderer_condition_tensor_sha256_after
    )
    sample_kwargs_receipt["condition_tensors_unchanged_by_renderer"] = True
    bridge_trace = bridge.receipt()
    noise_receipt = observer.receipt()
    source_prefix_consensus = _world8_string_consensus(
        str(bridge_trace["clean_source_prefix_sha256"]),
        distributed_module=dist,
        group=parallel.world_group,
    )
    instruction_context_consensus = _world8_string_consensus(
        str(bridge_trace["actual_contextual_instruction_sha256"]),
        distributed_module=dist,
        group=parallel.world_group,
    )
    target_trajectory_consensus = _world8_string_consensus(
        object_sha256(bridge_trace["evolving_target_state_sha256"]),
        distributed_module=dist,
        group=parallel.world_group,
    )
    bridge_trace_consensus = _rank_pattern_consensus(
        bridge_trace,
        sp_rank=sp_rank,
        dp_rank=dp_rank,
        local_cuda_index=local_cuda_index,
        distributed_module=dist,
        group=parallel.world_group,
    )
    trace_box: list[Any] = [bridge_trace if rank == 0 else None]
    dist.broadcast_object_list(trace_box, src=0, group=parallel.world_group)
    rank0_bridge_trace = trace_box[0]
    if (
        not isinstance(rank0_bridge_trace, Mapping)
        or rank0_bridge_trace.get("actual_contextual_instruction_length")
        != actual_input_token_count
    ):
        fail("rank-0 Level-B renderer trace broadcast differs")
    generated_latent_sha = tensor_sha256(generated_latent, torch_module=torch)
    generated_consensus = _world8_string_consensus(
        generated_latent_sha,
        distributed_module=dist,
        group=parallel.world_group,
    )
    noise_consensus = _world8_string_consensus(
        noise_receipt["spatial_tensor_sha256"],
        distributed_module=dist,
        group=parallel.world_group,
    )

    post_source_sha, post_source_identity = stable_file_sha256(
        source_path, label="source video after Level-B render"
    )
    if (
        post_source_sha != expected_source_video_sha256
        or post_source_identity != request_receipt["source_file_identity"]
    ):
        fail("source video bytes changed during Level-B render")
    if loaded_parameter_after != consumer_receipt["loaded_parameter_sha256"]:
        fail("full renderer inference mutated persisted trainable bytes")

    output_phase_local: dict[str, Any] = {}

    def decode_output_with_vae_on_rank0() -> Mapping[str, Any]:
        if vae is None:
            fail("rank 0 did not retain the frozen VAE for output decode")
        callable_before_decode = _audit_loaded_vae_callables(
            vae,
            expected_vae_class=AutoencoderKLWan,
            runtime_receipt=runtime,
        )
        if canonical_json_bytes(callable_before_decode) != canonical_json_bytes(
            loaded_vae_callables
        ):
            fail("loaded VAE callable identities changed before decode")
        vae.to(device)
        with torch.no_grad():
            decoded = _vae_decode(vae, generated_latent)
        callable_after_decode = _audit_loaded_vae_callables(
            vae,
            expected_vae_class=AutoencoderKLWan,
            runtime_receipt=runtime,
        )
        if canonical_json_bytes(callable_after_decode) != canonical_json_bytes(
            loaded_vae_callables
        ):
            fail("loaded VAE callable identities changed during decode")
        decoded_array_audit = _audit_decoded_video_array(
            decoded,
            expected_height=bucket_h,
            expected_width=bucket_w,
            numpy_module=np,
        )
        output_phase_local["decoded"] = decoded
        return {
            "decoded_shape": [int(x) for x in decoded.shape],
            "decoded_array_audit": decoded_array_audit,
            "vae_callable_identity": callable_after_decode,
        }

    decode_phase = _run_world8_rank0_collective_phase(
        phase="vae_output_decode",
        rank=rank,
        operation=decode_output_with_vae_on_rank0,
        distributed_module=dist,
        group=parallel.world_group,
    )
    if (
        not isinstance(decode_phase, Mapping)
        or decode_phase.get("decoded_shape")
        != [FRAME_COUNT, bucket_h, bucket_w, 3]
        or not isinstance(decode_phase.get("decoded_array_audit"), Mapping)
        or decode_phase["decoded_array_audit"].get("finite") is not True
        or decode_phase["decoded_array_audit"].get("closed_unit_interval") is not True
        or canonical_json_bytes(decode_phase.get("vae_callable_identity"))
        != canonical_json_bytes(loaded_vae_callables)
    ):
        fail("rank-0 frozen VAE decode receipt differs")
    # The post-decode callable audit above is the final VAE consumer.  Keep the
    # model on GPU through that audit (avoiding a large CPU weight restoration),
    # then release its sole outer reference before the decoded-frame encoder
    # allocates host-side uint8 buffers.
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    def validate_and_stage_output_on_rank0() -> Mapping[str, Any]:
        decoded = output_phase_local.get("decoded")
        if decoded is None:
            fail("rank-0 decoded frame tensor is absent before staging")
        transaction, validation = _stage_real_mp4(
            decoded_frames=decoded,
            output_path=output_path,
            receipt_path=receipt_path,
            commit_marker_path=commit_marker_path,
            authenticated_encoder=authenticated_encoder,
            encoder_authority=encoder_authority,
            expected_height=bucket_h,
            expected_width=bucket_w,
            runtime_receipt=runtime,
            output_rollback_guard=output_rollback_guard,
        )
        output_phase_local["transaction"] = transaction
        output_phase_local.pop("decoded", None)
        return validation

    video_validation = _run_world8_rank0_collective_phase(
        phase="ffmpeg_validate_and_output_staging",
        rank=rank,
        operation=validate_and_stage_output_on_rank0,
        distributed_module=dist,
        group=parallel.world_group,
        rollback_on_failure=output_rollback_guard.rollback,
    )
    output_transaction = (
        output_phase_local.get("transaction") if rank == 0 else None
    )
    if (
        not isinstance(video_validation, Mapping)
        or video_validation.get("complete_decode_verified") is not True
        or video_validation.get("full_decode_frame_count") != FRAME_COUNT
    ):
        if rank == 0 and output_transaction is not None:
            output_transaction.cleanup_stage()
        fail("rank-0 real-MP4 validation receipt differs")

    marker_envelope = _commit_marker_envelope(
        output_path=output_path,
        receipt_path=receipt_path,
        commit_marker_path=commit_marker_path,
        mp4_sha256=video_validation["mp4_sha256"],
    )
    unsigned = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "complete": True,
        "promotable": False,
        "formal_training_started": False,
        "counts_as_d0": False,
        "scientific_claim_authorized": False,
        "action_quality_claim_authorized": False,
        "training_to_fresh_forward_parity_verified": False,
        "full_bernini_renderer_training_to_fresh_forward_parity_verified": False,
        "public_product_signature": validate_public_product_signature(),
        "request": request_receipt,
        "verified_level_b_runtime": runtime,
        "checkpoint_binding": {
            "checkpoint_parameter_sha256": consumer_receipt[
                "checkpoint_parameter_sha256"
            ],
            "loaded_parameter_sha256_before": consumer_receipt[
                "loaded_parameter_sha256"
            ],
            "loaded_parameter_sha256_after": loaded_parameter_after,
            "checkpoint_metadata_sha256": consumer_receipt[
                "checkpoint_metadata_sha256"
            ],
            "campaign_receipt_sha256": consumer_receipt[
                "campaign_receipt_sha256"
            ],
            "training_release_manifest_sha256": consumer_receipt[
                "release_manifest_sha256"
            ],
            "checkpoint_consumer_receipt_sha256": object_sha256(
                dict(rank0_consumer_receipt)
            ),
            "consumer_source_sha256": consumer_receipt[
                "consumer_source_sha256"
            ],
            "product_abi_source_sha256": consumer_receipt[
                "product_bridge_source_sha256"
            ],
            "base_checkpoint_tree_sha256": consumer_receipt[
                "base_checkpoint_tree_sha256"
            ],
            "live_base_checkpoint_content": base_checkpoint_binding,
            "trainable_bytes_unchanged_by_full_render": True,
        },
        "official_runtime": {
            "bernini_commit": consumer_receipt["official_bernini_commit"],
            "veomni_commit": consumer_receipt["veomni_commit"],
            "official_inference_source_sha256": official_row["sha256"],
            "legacy_train_lora_source_sha256": legacy_sha,
            "model_authority_source_sha256": authority_row["sha256"],
            "official_bernini_inference_file_hashes": official_source_hashes,
            "wan_diffusion_source_sha256": wan_sha,
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "diffusers": diffusers_version,
            "transformers": transformers_version,
            "vendor_callable_closure": vendor_callable_closure,
            "loaded_vae_callables": loaded_vae_callables,
            "loaded_tokenizer_callable": loaded_tokenizer_callable,
            "authenticated_encoder_authority": encoder_authority,
            "tokenizer_released_before_denoise": True,
            "vae_retained_on_gpu_through_post_decode_callable_audit": True,
            "vae_never_restored_to_host_after_source_encode": True,
            "vae_reference_released_before_output_staging": True,
            "decoded_video_array": dict(decode_phase["decoded_array_audit"]),
        },
        "source_preprocessing": {
            **dict(source_metadata),
            "source_preprocessed_tensor_sha256": source_tensor_sha,
            "source_clean_vae_latent_sha256": source_latent_sha,
            "source_latent_world8_consensus": source_latent_consensus,
            "frozen_vae": True,
            "rank0_source_tensor_phase_reference_released_after_encode": True,
            "rank0_vae_phase_dictionary_references_released": True,
            "bounded_host_memory_tensor_hashing": True,
        },
        "instruction_encoding": {
            "full_training_prompt_utf8_sha256": hashlib.sha256(
                full_prompt.encode("utf-8")
            ).hexdigest(),
            "actual_input_token_count": actual_input_token_count,
            "actual_contextual_t5_token_count": rank0_bridge_trace[
                "actual_contextual_instruction_length"
            ],
            "actual_contextual_t5_sha256": rank0_bridge_trace[
                "actual_contextual_instruction_sha256"
            ],
            "frozen_t5": True,
            "complete_unpadded_contextual_tokens_used_by_conditioner": True,
        },
        "inference_policy": dict(policy.receipt()),
        "native_initial_noise": noise_receipt,
        "native_initial_noise_world8_consensus": noise_consensus,
        "process_global_rng_unchanged": True,
        "renderer_sample_kwargs": sample_kwargs_receipt,
        "renderer_sample_live_signature_audit": renderer_sample_call_audit,
        "native_callable_live_signature_audit": native_signature_audit,
        "renderer_trace_rank0": dict(rank0_bridge_trace),
        "renderer_trace_world8": bridge_trace_consensus,
        "clean_source_prefix_world8_consensus": source_prefix_consensus,
        "actual_contextual_instruction_world8_consensus": (
            instruction_context_consensus
        ),
        "evolving_target_trajectory_world8_consensus": (
            target_trajectory_consensus
        ),
        "generated_latent_sha256": generated_latent_sha,
        "generated_latent_world8_consensus": generated_consensus,
        "callable_provenance": callable_provenance,
        "output_mp4": dict(video_validation),
        "commit_marker_envelope": marker_envelope,
        "output_transaction": {
            "schema_version": OUTPUT_TRANSACTION_SCHEMA,
            "mp4_and_receipt_staged_before_commit": True,
            "atomic_commit_marker_schema": OUTPUT_COMMIT_MARKER_SCHEMA,
            "commit_marker_path": str(commit_marker_path),
            "commit_marker_is_only_consumer_completion_authority": True,
            "bare_mp4_or_receipt_pair_is_never_complete": True,
            "create_only_marker_product_commit_required": True,
            "world8_canonical_gate_required_before_commit": True,
            "all8_precommit_reopen_required_before_marker": True,
            "marker_link_is_final_business_action": True,
            "marker_is_receipt_inode_alias": True,
            "post_marker_collective_or_business_hook_forbidden": True,
            "directory_fsync_and_nfs_reopen_required": True,
            "late_failure_must_leave_no_acceptable_product": True,
        },
        "full40_denoise_executed": True,
        "full_bernini_renderer_denoise_verified": True,
        "offline_product_inference_completed": True,
        "mp4_emitted": True,
        "ffprobe_and_full_decode_verified": True,
        "clean_source_prefix_plus_evolving_noisy_target_verified": True,
        "exact30_target_only_action_hooks_once_per_denoise_step": True,
        "clean_target_or_anchor_consumed": False,
        "teacher_or_external_annotation_consumed": False,
        "hidden_user_callback_or_custom_denoiser_consumed": False,
        "completion_scope": (
            "PRE_D0 full native Bernini product-path engineering inference only"
        ),
        "promotion_authorized": False,
    }
    try:
        rank0_canonical_unsigned, canonical_consensus = (
            _world8_rank0_canonical_consensus(
                unsigned,
                local_cuda_index=local_cuda_index,
                distributed_module=dist,
                group=parallel.world_group,
            )
        )
        pre_consensus_digest = object_sha256(rank0_canonical_unsigned)
        receipt_digest_consensus = _world8_string_consensus(
            pre_consensus_digest,
            distributed_module=dist,
            group=parallel.world_group,
        )
        final_unsigned = {
            **rank0_canonical_unsigned,
            "pre_consensus_receipt_digest": pre_consensus_digest,
            "pre_consensus_receipt_digest_world8": receipt_digest_consensus,
            "normalized_rank0_canonical_world8_receipt": canonical_consensus,
            "final_receipt_exact_bytes_world8_precommit_gate": True,
        }
        local_receipt = {
            **final_unsigned,
            "receipt_digest": object_sha256(final_unsigned),
        }
        # This second gather covers the exact receipt that will be staged,
        # including the first consensus evidence and its final digest.  No
        # public MP4/receipt name exists until these bytes agree on all ranks.
        receipt, _final_receipt_consensus = _world8_rank0_canonical_consensus(
            local_receipt,
            local_cuda_index=local_cuda_index,
            distributed_module=dist,
            group=parallel.world_group,
        )
        if canonical_json_bytes(receipt) != canonical_json_bytes(local_receipt):
            fail("final rank-0 Level-B receipt differs after WORLD8 consensus")
        receipt_file_sha = hashlib.sha256(
            canonical_json_bytes(receipt) + b"\n"
        ).hexdigest()
    except Exception:
        if rank == 0 and output_transaction is not None:
            output_transaction.cleanup_stage()
        raise

    def precommit_on_rank0() -> Mapping[str, Any]:
        if output_transaction is None:
            fail("rank-0 staging transaction is absent")
        return _publish_precommit_product_pair(
            transaction=output_transaction, receipt=receipt
        )

    precommit_receipt = _run_world8_rank0_collective_phase(
        phase="create_only_output_precommit",
        rank=rank,
        operation=precommit_on_rank0,
        distributed_module=dist,
        group=parallel.world_group,
        rollback_on_failure=output_rollback_guard.rollback,
    )
    if (
        not isinstance(precommit_receipt, Mapping)
        or precommit_receipt.get("commit_marker_absent") is not True
        or precommit_receipt.get("receipt_sha256") != receipt_file_sha
    ):
        fail("rank-0 Level-B precommit transaction receipt differs")

    gate_contract = marker_envelope["world8_precommit_gate_contract"]
    world8_precommit_gate = _gather_world8_precommit_reopen_consensus(
        rank=rank,
        output_path=output_path,
        receipt_path=receipt_path,
        commit_marker_path=commit_marker_path,
        expected_mp4_sha256=video_validation["mp4_sha256"],
        expected_receipt_sha256=receipt_file_sha,
        expected_gate_contract=gate_contract,
        distributed_module=dist,
        group=parallel.world_group,
    )
    final_marker_local: dict[str, Any] = {}

    def prepare_final_marker_on_rank0() -> Mapping[str, Any]:
        if output_transaction is None:
            fail("rank-0 precommit transaction is absent before marker readiness")
        publisher, readiness = _prepare_final_receipt_alias_marker_publisher(
            transaction=output_transaction,
            precommit_receipt=precommit_receipt,
            world8_gate_evidence=world8_precommit_gate,
        )
        final_marker_local["publisher"] = publisher
        return readiness

    marker_readiness = _run_world8_rank0_collective_phase(
        phase="final_marker_readiness_status",
        rank=rank,
        operation=prepare_final_marker_on_rank0,
        distributed_module=dist,
        group=parallel.world_group,
        rollback_on_failure=output_rollback_guard.rollback,
    )
    if (
        not isinstance(marker_readiness, Mapping)
        or marker_readiness.get("mp4_sha256")
        != video_validation["mp4_sha256"]
        or marker_readiness.get("receipt_sha256") != receipt_file_sha
        or marker_readiness.get("world8_gate_evidence_sha256")
        != object_sha256(world8_precommit_gate)
        or marker_readiness.get("marker_link_is_terminal_filesystem_action")
        is not True
    ):
        fail("rank-0 final marker readiness status differs")
    dist.barrier(group=parallel.world_group)
    if rank == 0:
        final_publisher = final_marker_local.get("publisher")
        if type(final_publisher) is not _FinalReceiptAliasMarkerPublisher:
            fail("rank-0 final marker publisher is absent after WORLD8 barrier")
        # The guard relinquishes rollback responsibility while no marker exists.
        # From the create-only link onward, no collective, validation, cleanup,
        # lstat/hash, fsync, or other patchable business hook is invoked.
        output_rollback_guard.disarm()
        final_publisher.publish()
    return receipt


def _run_level_b_pre_d0_offline_inference_authenticated(
    *,
    fresh_bundle: Any,
    runtime_receipt: Mapping[str, Any],
    source_video_path: str,
    expected_source_video_sha256: str,
    edit_instruction: str,
    inference_seed: int,
    output_mp4_path: str,
) -> Mapping[str, Any]:
    # This owner rolls back private/public *precommit* state on every failure.
    # The core disarms it immediately before the one irreversible terminal
    # receipt-to-COMMITTED link; no fallible business action follows that link.
    return _run_with_level_b_output_rollback(
        lambda output_rollback_guard: (
            _run_level_b_pre_d0_offline_inference_authenticated_core(
                fresh_bundle=fresh_bundle,
                runtime_receipt=runtime_receipt,
                source_video_path=source_video_path,
                expected_source_video_sha256=expected_source_video_sha256,
                edit_instruction=edit_instruction,
                inference_seed=inference_seed,
                output_mp4_path=output_mp4_path,
                output_rollback_guard=output_rollback_guard,
            )
        )
    )


def _make_level_b_public_product_entrypoint(
    implementation: Any, runtime_validator: Any
) -> Any:
    """Capture the real capability validator outside mutable class dispatch."""

    def run_level_b_pre_d0_offline_inference(
        *,
        fresh_bundle: Any,
        verified_runtime: VerifiedLevelBRuntime,
        source_video_path: str,
        expected_source_video_sha256: str,
        edit_instruction: str,
        inference_seed: int,
        output_mp4_path: str,
    ) -> Mapping[str, Any]:
        if type(verified_runtime) is not VerifiedLevelBRuntime:
            fail("Level-B product requires one exact opaque verified runtime")
        # Call the captured closure directly.  Replacing the visible class
        # method cannot turn a caller-created object into runtime authority.
        runtime = runtime_validator(verified_runtime)
        return implementation(
            fresh_bundle=fresh_bundle,
            runtime_receipt=runtime,
            source_video_path=source_video_path,
            expected_source_video_sha256=expected_source_video_sha256,
            edit_instruction=edit_instruction,
            inference_seed=inference_seed,
            output_mp4_path=output_mp4_path,
        )

    return run_level_b_pre_d0_offline_inference


def _run_level_b_cpu_static_runtime_preflight_authenticated(
    runtime_receipt: Mapping[str, Any],
    *,
    expected_scoped_module_closure: tuple[
        tuple[str, str, str, str, int], ...
    ] | None = None,
) -> Mapping[str, Any]:
    """Read/import/inspect only; never construct a model or initialize CUDA."""

    previous_dont_write_bytecode = sys.dont_write_bytecode
    path_snapshot = tuple(sys.path)
    meta_path_snapshot = tuple(sys.meta_path)
    importer_cache_snapshot = dict(sys.path_importer_cache)

    def meta_path_identity_is(expected: tuple[Any, ...]) -> bool:
        live = tuple(sys.meta_path)
        return len(live) == len(expected) and all(
            observed is required for observed, required in zip(live, expected)
        )
    bernini_root_text = str(
        _plain_directory(
            runtime_receipt.get("bernini_root"),
            label="CPU static preflight Bernini root",
        )
    )
    veomni_root_text = str(
        _plain_directory(
            runtime_receipt.get("veomni_root"),
            label="CPU static preflight VeOmni root",
        )
    )
    preexisting_scoped_modules = tuple(
        name
        for name in sys.modules
        if name == "bernini"
        or name.startswith("bernini.")
        or name == "veomni"
        or name.startswith("veomni.")
    )
    importer_cache_sentinel = object()
    prior_importer_cache = sys.path_importer_cache.get(
        bernini_root_text, importer_cache_sentinel
    )
    prior_veomni_importer_cache = sys.path_importer_cache.get(
        veomni_root_text, importer_cache_sentinel
    )
    scoped_root_installed = False
    scope_tampered = False
    tempfile_scope_installed = False
    tempfile_scope_tampered = False
    tempfile_scope_restored = False
    tempfile_override = "/nonexistent"
    tempfile_previous = _NO_BOUND_OWNER
    tempfile_module: Any = None
    temporary_directory_previous = _NO_BOUND_OWNER
    atexit_register_previous = _NO_BOUND_OWNER
    temporary_directory_scope_installed = False
    temporary_directory_scope_restored = False
    temporary_directory_scope_tampered = False
    atexit_scope_installed = False
    atexit_scope_restored = False
    atexit_scope_tampered = False
    jit_tempdir_constructor_calls = 0
    jit_tempdir_registration_calls = 0
    jit_tempdir_cleanup_calls = 0
    jit_sys_path_append_removed = False
    jit_finder_calls = 0
    jit_loader_create_module_calls = 0
    jit_loader_exec_module_calls = 0
    jit_finder_scope_installed = False
    jit_finder_scope_restored = False
    jit_finder_scope_tampered = False
    six_importer_scope_installed = False
    six_importer_scope_restored = False
    six_importer_scope_tampered = False
    botocore_six_importer: Any = None
    six_importer: Any = None
    remote_finder_calls = 0
    remote_loader_create_module_calls = 0
    remote_loader_exec_module_calls = 0
    remote_template_factory_calls = 0
    remote_template_factory_scope_installed = False
    remote_template_factory_scope_restored = False
    remote_template_factory_scope_tampered = False
    remote_template_factory_previous = _NO_BOUND_OWNER
    remote_template_value: Any = None
    importer_cache_restored = False
    importer_cache_preexisting_tampered = False
    importer_cache_jit_parent_added = False
    importer_cache_remote_parent_added = False
    jit_tempdir_name = (
        "/nonexistent/bernini-level-b-static-preflight-torch-jit"
    )
    jit_tempdir_value: Any = None
    atexit_module: Any = None
    visibility_environment = {
        name: os.environ.get(name, "")
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
    }
    if any(value != "" for value in visibility_environment.values()):
        fail("CPU static preflight accelerator visibility is not empty")
    blas_import_environment = {
        name: os.environ.get(name)
        for name in ("OPENBLAS_MAIN_FREE", "GOTOBLAS_MAIN_FREE")
    }
    if blas_import_environment != {
        "OPENBLAS_MAIN_FREE": "1",
        "GOTOBLAS_MAIN_FREE": "1",
    }:
        fail("CPU static preflight BLAS import environment differs")
    veomni_logging_environment = {
        "VEOMNI_VERBOSITY": os.environ.get("VEOMNI_VERBOSITY")
    }
    if veomni_logging_environment != {"VEOMNI_VERBOSITY": "ERROR"}:
        fail("CPU static preflight VeOmni logging environment differs")
    audit_guard = _install_cpu_static_preflight_audit_guard(
        base_checkpoint=runtime_receipt.get("base_checkpoint")
    )
    sys.dont_write_bytecode = True
    try:
        # torch -> dill asks tempfile for an output directory while importing.
        # Seed only its in-memory cache so that no random O_CREAT probe occurs.
        # The process audit hook is already active and the cached value is
        # restored exactly before this function returns.
        import atexit as atexit_module
        import tempfile as tempfile_module

        if tempfile_module.tempdir is not None:
            fail("CPU static preflight tempfile cache is not fresh")
        rows = runtime_receipt.get("vendor_source_files")
        if not isinstance(rows, Mapping):
            fail("CPU static preflight vendor source closure is absent")
        if (
            type(expected_scoped_module_closure) is not tuple
            or len(expected_scoped_module_closure) != 52
            or any(
                type(row) is not tuple
                or len(row) != 5
                or any(type(value) is not str for value in row[:4])
                or row[1] not in ("bernini", "veomni")
                or not _SHA256.fullmatch(row[3])
                or type(row[4]) is not int
                or row[4] <= 0
                for row in expected_scoped_module_closure
            )
        ):
            fail("CPU static preflight scoped module closure authority differs")
        if (
            tuple(sorted(row[0] for row in expected_scoped_module_closure))
            != tuple(row[0] for row in expected_scoped_module_closure)
            or len({row[0] for row in expected_scoped_module_closure}) != 52
            or len({(row[1], row[2]) for row in expected_scoped_module_closure})
            != 52
            or sum(row[1] == "bernini" for row in expected_scoped_module_closure)
            != 13
            or sum(row[1] == "veomni" for row in expected_scoped_module_closure)
            != 39
        ):
            fail("CPU static preflight scoped module closure set differs")

        def authority(logical_name: str) -> Mapping[str, Any]:
            value = rows.get(logical_name)
            if not isinstance(value, Mapping) or set(value) != {
                "path",
                "sha256",
                "file_identity",
            }:
                fail(f"CPU static preflight source row differs: {logical_name}")
            return value

        numpy_core_authority = authority(
            "site-packages:numpy/core/__init__.py"
        )
        numpy_core_path = _plain_file(
            numpy_core_authority["path"],
            label="CPU static preflight NumPy core initialization source",
        )
        numpy_core_sha, numpy_core_identity = stable_file_sha256(
            numpy_core_path,
            label="CPU static preflight NumPy core initialization source",
        )
        if (
            numpy_core_sha != numpy_core_authority["sha256"]
            or numpy_core_sha
            != "08db0ef806f8cb03365b3dc06ea58e1f78a0d6ae419e8f4fb1432b0aff87352e"
            or numpy_core_identity != numpy_core_authority["file_identity"]
            or numpy_core_identity.get("size") != 5780
        ):
            fail("CPU static preflight NumPy import authority differs")

        veomni_logging_authority = authority(
            "veomni:veomni/utils/logging.py"
        )
        veomni_logging_path = _plain_file(
            veomni_logging_authority["path"],
            label="CPU static preflight VeOmni logging source",
        )
        veomni_logging_sha, veomni_logging_identity = stable_file_sha256(
            veomni_logging_path,
            label="CPU static preflight VeOmni logging source",
        )
        if (
            veomni_logging_sha != veomni_logging_authority["sha256"]
            or veomni_logging_sha
            != "91a613a68a5a32b239900bd72cfdf5d172996fec37bf67a69b0cefa699c9fc5a"
            or veomni_logging_identity
            != veomni_logging_authority["file_identity"]
            or veomni_logging_identity.get("size") != 5246
        ):
            fail("CPU static preflight VeOmni logging authority differs")

        botocore_six_authority = authority(
            "site-packages:botocore/vendored/six.py"
        )
        botocore_six_path = _plain_file(
            botocore_six_authority["path"],
            label="CPU static preflight botocore vendored six source",
        )
        botocore_six_sha, botocore_six_identity = stable_file_sha256(
            botocore_six_path,
            label="CPU static preflight botocore vendored six source",
        )
        if (
            botocore_six_sha != botocore_six_authority["sha256"]
            or botocore_six_sha
            != "4ce39f422ee71467ccac8bed76beb05f8c321c7f0ceda9279ae2dfa3670106b3"
            or botocore_six_identity != botocore_six_authority["file_identity"]
            or botocore_six_identity.get("size") != 34549
        ):
            fail("CPU static preflight botocore six authority differs")
        six_authority = authority("site-packages:six.py")
        six_path = _plain_file(
            six_authority["path"], label="CPU static preflight six source"
        )
        six_sha, six_identity = stable_file_sha256(
            six_path, label="CPU static preflight six source"
        )
        if (
            six_sha != six_authority["sha256"]
            or six_sha
            != "c51c91f703d3d4b3696c923cb5fec213e05e75d9215393befac7f2fa6a3904df"
            or six_identity != six_authority["file_identity"]
            or six_identity.get("size") != 34703
        ):
            fail("CPU static preflight six authority differs")

        def audit_six_meta_path_importer(
            importer: Any,
            *,
            module_name: str,
            source_path: Path,
            source_sha256: str,
            source_identity: Mapping[str, int],
        ) -> Mapping[str, Any]:
            module = sys.modules.get(module_name)
            spec = getattr(module, "__spec__", None)
            importer_type = type(importer)
            module_file = getattr(module, "__file__", None)
            spec_origin = getattr(spec, "origin", None)
            if (
                module is None
                or getattr(module, "__name__", None) != module_name
                or getattr(importer, "__module__", None) != module_name
                or getattr(importer, "__qualname__", None) is not None
                or importer_type.__module__ != module_name
                or importer_type.__qualname__ != "_SixMetaPathImporter"
                or getattr(module, "_SixMetaPathImporter", None)
                is not importer_type
                or type(module_file) is not str
                or type(spec_origin) is not str
                or module_file != str(source_path)
                or spec_origin != str(source_path)
                or Path(module_file).resolve(strict=True) != source_path
                or Path(spec_origin).resolve(strict=True) != source_path
                or source_identity.get("size") not in (34549, 34703)
            ):
                fail("CPU static preflight six meta-path importer differs")
            observed_sha, observed_identity = stable_file_sha256(
                source_path,
                label=f"CPU static preflight {module_name} importer source",
            )
            if (
                observed_sha != source_sha256
                or observed_identity != source_identity
            ):
                fail("CPU static preflight six importer source changed")
            return {
                "module": module_name,
                "type_module": importer_type.__module__,
                "type_qualname": importer_type.__qualname__,
                "module_export_identity_verified": True,
                "module_file_and_spec_origin_exact": True,
                "source_path": str(source_path),
                "source_sha256": observed_sha,
                "source_size": observed_identity["size"],
            }

        def audit_scoped_module_source_closure() -> tuple[Mapping[str, Any], ...]:
            live_names = tuple(
                sorted(
                    name
                    for name in sys.modules
                    if name == "bernini"
                    or name.startswith("bernini.")
                    or name == "veomni"
                    or name.startswith("veomni.")
                )
            )
            expected_names = tuple(
                row[0] for row in expected_scoped_module_closure
            )
            if live_names != expected_names:
                fail("CPU static preflight scoped module set differs")
            closure_receipt = []
            for module_name, prefix, relative, expected_sha, expected_size in (
                expected_scoped_module_closure
            ):
                module = sys.modules.get(module_name)
                spec = getattr(module, "__spec__", None)
                source_root = (
                    Path(bernini_root_text)
                    if prefix == "bernini"
                    else Path(veomni_root_text)
                )
                source_path = _plain_file(
                    source_root / relative,
                    label=f"CPU static preflight scoped module {module_name}",
                )
                logical_name = f"{prefix}:{relative}"
                source_authority = authority(logical_name)
                observed_sha, observed_identity = stable_file_sha256(
                    source_path,
                    label=f"CPU static preflight scoped module {module_name}",
                )
                if (
                    module is None
                    or getattr(module, "__name__", None) != module_name
                    or getattr(module, "__file__", None) != str(source_path)
                    or getattr(spec, "origin", None) != str(source_path)
                    or getattr(spec, "name", None) != module_name
                    or source_authority.get("path") != str(source_path)
                    or source_authority.get("sha256") != expected_sha
                    or source_authority.get("file_identity") != observed_identity
                    or observed_sha != expected_sha
                    or observed_identity.get("size") != expected_size
                ):
                    fail(
                        f"CPU static preflight scoped module ownership differs: {module_name}"
                    )
                closure_receipt.append(
                    {
                        "module": module_name,
                        "prefix": prefix,
                        "relative_path": relative,
                        "source_path": str(source_path),
                        "sha256": observed_sha,
                        "size": observed_identity["size"],
                    }
                )
            return tuple(closure_receipt)

        jit_instantiator_authority = authority(
            "site-packages:torch/distributed/nn/jit/instantiator.py"
        )
        jit_instantiator_path = _plain_file(
            jit_instantiator_authority["path"],
            label="CPU static preflight Torch JIT instantiator source",
        )
        jit_instantiator_sha, jit_instantiator_identity = stable_file_sha256(
            jit_instantiator_path,
            label="CPU static preflight Torch JIT instantiator source",
        )
        if (
            jit_instantiator_sha != jit_instantiator_authority["sha256"]
            or jit_instantiator_sha
            != "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
            or jit_instantiator_identity
            != jit_instantiator_authority["file_identity"]
            or jit_instantiator_identity.get("size") != 5510
        ):
            fail("CPU static preflight Torch JIT instantiator authority differs")
        remote_module_authority = authority(
            "site-packages:torch/distributed/nn/api/remote_module.py"
        )
        remote_module_path = _plain_file(
            remote_module_authority["path"],
            label="CPU static preflight Torch remote-module source",
        )
        remote_module_sha, remote_module_identity = stable_file_sha256(
            remote_module_path,
            label="CPU static preflight Torch remote-module source",
        )
        if (
            remote_module_sha != remote_module_authority["sha256"]
            or remote_module_sha
            != "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1"
            or remote_module_identity != remote_module_authority["file_identity"]
            or remote_module_identity.get("size") != 31251
        ):
            fail("CPU static preflight Torch remote-module authority differs")
        jit_importer_parent = str(jit_instantiator_path.parent)
        remote_importer_parent = str(remote_module_path.parent)
        jit_importer_parent_prior = importer_cache_snapshot.get(
            jit_importer_parent, importer_cache_sentinel
        )
        remote_importer_parent_prior = importer_cache_snapshot.get(
            remote_importer_parent, importer_cache_sentinel
        )
        if jit_importer_parent_prior is not importer_cache_sentinel:
            fail("CPU static preflight Torch JIT importer cache is not fresh")
        socket_authority = runtime_receipt.get("stdlib_socket_source")
        if not isinstance(socket_authority, Mapping) or set(socket_authority) != {
            "path",
            "sha256",
            "file_identity",
        }:
            fail("CPU static preflight stdlib socket authority differs")
        socket_source_path = _plain_file(
            socket_authority["path"],
            label="CPU static preflight stdlib socket source",
        )
        socket_source_sha, socket_source_identity = stable_file_sha256(
            socket_source_path,
            label="CPU static preflight stdlib socket source",
        )
        if (
            socket_source_sha != socket_authority["sha256"]
            or socket_source_sha != PINNED_STDLIB_SOCKET_SHA256
            or socket_source_identity != socket_authority["file_identity"]
            or socket_source_identity.get("size") != 37815
        ):
            fail("CPU static preflight stdlib socket source differs")
        urllib3_authority = authority("site-packages:urllib3/util/connection.py")
        urllib3_connection_path = _plain_file(
            urllib3_authority["path"],
            label="CPU static preflight urllib3 connection source",
        )
        urllib3_connection_sha, urllib3_connection_identity = stable_file_sha256(
            urllib3_connection_path,
            label="CPU static preflight urllib3 connection source",
        )
        if (
            urllib3_connection_sha != urllib3_authority["sha256"]
            or urllib3_connection_sha
            != "2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74"
            or urllib3_connection_identity != urllib3_authority["file_identity"]
            or urllib3_connection_identity.get("size") != 4444
        ):
            fail("CPU static preflight urllib3 connection source differs")
        audit_guard.configure_blocked_network_probe(
            socket_source_path=socket_source_path,
            socket_source_sha256=socket_source_sha,
            urllib3_connection_path=urllib3_connection_path,
            urllib3_connection_sha256=urllib3_connection_sha,
        )
        try:
            os.lstat(tempfile_override)
        except FileNotFoundError:
            pass
        else:
            fail("CPU static preflight sealed tempfile sentinel exists")
        tempfile_previous = tempfile_module.tempdir
        tempfile_module.tempdir = tempfile_override
        if tempfile_module.tempdir is not tempfile_override:
            fail("CPU static preflight tempfile scope did not install exactly")
        tempfile_scope_installed = True
        try:
            os.lstat(jit_tempdir_name)
        except FileNotFoundError:
            pass
        else:
            fail("CPU static preflight Torch JIT tempfile sentinel exists")

        class NoWriteTemporaryDirectory:
            __slots__ = ("name",)

            def __init__(self, name: str) -> None:
                self.name = name

            def cleanup(self) -> None:
                nonlocal jit_tempdir_cleanup_calls
                jit_tempdir_cleanup_calls += 1
                return None

        jit_tempdir_value = NoWriteTemporaryDirectory(jit_tempdir_name)

        def require_instantiator_caller(expected_line: int, *, label: str) -> None:
            caller = sys._getframe(2)
            module = sys.modules.get(
                "torch.distributed.nn.jit.instantiator"
            )
            code = caller.f_code
            if (
                type(expected_line) is not int
                or caller.f_lineno != expected_line
                or module is None
                or vars(module) is not caller.f_globals
                or caller.f_globals.get("__name__")
                != "torch.distributed.nn.jit.instantiator"
                or code.co_name != "<module>"
                or getattr(code, "co_qualname", code.co_name) != "<module>"
                or code.co_firstlineno != 1
                or tuple(code.co_freevars) != ()
                or Path(code.co_filename).resolve(strict=True)
                != jit_instantiator_path
            ):
                fail(f"CPU static preflight Torch JIT {label} caller differs")

        def no_write_temporary_directory(*args: Any, **kwargs: Any) -> Any:
            nonlocal jit_tempdir_constructor_calls
            require_instantiator_caller(21, label="TemporaryDirectory")
            if (
                jit_tempdir_constructor_calls != 0
                or args != ()
                or type(kwargs) is not dict
                or kwargs
            ):
                fail("CPU static preflight Torch JIT TemporaryDirectory call differs")
            jit_tempdir_constructor_calls += 1
            return jit_tempdir_value

        def no_write_atexit_register(
            callback: Any, *args: Any, **kwargs: Any
        ) -> Any:
            nonlocal jit_tempdir_registration_calls
            require_instantiator_caller(23, label="atexit.register")
            if (
                jit_tempdir_registration_calls != 0
                or not inspect.ismethod(callback)
                or callback.__self__ is not jit_tempdir_value
                or callback.__func__ is not NoWriteTemporaryDirectory.cleanup
                or args != ()
                or type(kwargs) is not dict
                or kwargs
            ):
                fail("CPU static preflight Torch JIT atexit registration differs")
            jit_tempdir_registration_calls += 1
            return callback

        temporary_directory_previous = tempfile_module.TemporaryDirectory
        atexit_register_previous = atexit_module.register
        if (
            not inspect.isclass(temporary_directory_previous)
            or temporary_directory_previous.__module__ != "tempfile"
            or temporary_directory_previous.__qualname__ != "TemporaryDirectory"
            or not callable(atexit_register_previous)
        ):
            fail("CPU static preflight tempfile/atexit authority differs")

        target_instantiator_module = (
            "torch.distributed.nn.jit.instantiator"
        )
        target_remote_module = "torch.distributed.nn.api.remote_module"

        class NoWriteRemoteTemplate:
            __slots__ = ()

        remote_template_value = NoWriteRemoteTemplate()

        def no_write_remote_template_factory(
            *args: Any, **kwargs: Any
        ) -> Any:
            nonlocal remote_template_factory_calls
            caller = sys._getframe(1)
            module = sys.modules.get(target_remote_module)
            code = caller.f_code
            if (
                remote_template_factory_calls != 0
                or args != ()
                or type(kwargs) is not dict
                or kwargs
                or caller.f_lineno != 30
                or module is None
                or vars(module) is not caller.f_globals
                or caller.f_globals.get("__name__") != target_remote_module
                or code.co_name != "<module>"
                or getattr(code, "co_qualname", code.co_name) != "<module>"
                or code.co_firstlineno != 1
                or tuple(code.co_freevars) != ()
                or Path(code.co_filename).resolve(strict=True)
                != remote_module_path
            ):
                fail("CPU static preflight Torch remote-template caller differs")
            remote_template_factory_calls += 1
            return remote_template_value

        class ScopedInstantiatorLoader:
            __slots__ = (
                "original",
                "spec",
                "origin",
                "cached",
                "parent",
                "submodule_search_locations",
            )

            def __init__(self, original: Any, spec: Any) -> None:
                self.original = original
                self.spec = spec
                self.origin = spec.origin
                self.cached = spec.cached
                self.parent = spec.parent
                self.submodule_search_locations = spec.submodule_search_locations

            def create_module(self, spec: Any) -> Any:
                nonlocal jit_loader_create_module_calls
                if (
                    jit_loader_create_module_calls != 0
                    or spec is not self.spec
                    or spec.loader is not self
                ):
                    fail("CPU static preflight Torch JIT create_module differs")
                jit_loader_create_module_calls += 1
                created = self.original.create_module(spec)
                if created is not None:
                    fail("CPU static preflight Torch JIT loader constructed a module")
                return None

            def exec_module(self, module: Any) -> None:
                nonlocal jit_loader_exec_module_calls
                nonlocal temporary_directory_scope_installed
                nonlocal temporary_directory_scope_restored
                nonlocal temporary_directory_scope_tampered
                nonlocal atexit_scope_installed
                nonlocal atexit_scope_restored
                nonlocal atexit_scope_tampered
                nonlocal remote_template_factory_previous
                nonlocal remote_template_factory_scope_installed
                if (
                    jit_loader_exec_module_calls != 0
                    or module is not sys.modules.get(target_instantiator_module)
                    or module.__spec__ is not self.spec
                    or module.__loader__ is not self
                    or self.spec.loader is not self
                    or self.spec.name != target_instantiator_module
                    or Path(self.spec.origin).resolve(strict=True)
                    != jit_instantiator_path
                    or type(self.original)
                    is not importlib.machinery.SourceFileLoader
                    or self.original.name != target_instantiator_module
                    or Path(self.original.path).resolve(strict=True)
                    != jit_instantiator_path
                    or tempfile_module.TemporaryDirectory
                    is not temporary_directory_previous
                    or atexit_module.register is not atexit_register_previous
                ):
                    fail("CPU static preflight Torch JIT loader authority differs")
                jit_loader_exec_module_calls += 1
                tempfile_module.TemporaryDirectory = no_write_temporary_directory
                temporary_directory_scope_installed = True
                atexit_module.register = no_write_atexit_register
                atexit_scope_installed = True
                try:
                    self.original.exec_module(module)
                finally:
                    temporary_directory_scope_tampered = (
                        tempfile_module.TemporaryDirectory
                        is not no_write_temporary_directory
                    )
                    atexit_scope_tampered = (
                        atexit_module.register is not no_write_atexit_register
                    )
                    tempfile_module.TemporaryDirectory = (
                        temporary_directory_previous
                    )
                    temporary_directory_scope_restored = (
                        tempfile_module.TemporaryDirectory
                        is temporary_directory_previous
                    )
                    temporary_directory_scope_installed = False
                    atexit_module.register = atexit_register_previous
                    atexit_scope_restored = (
                        atexit_module.register is atexit_register_previous
                    )
                    atexit_scope_installed = False
                    if module.__spec__ is self.spec:
                        self.spec.loader = self.original
                        module.__loader__ = self.original
                if (
                    temporary_directory_scope_tampered
                    or atexit_scope_tampered
                    or not temporary_directory_scope_restored
                    or not atexit_scope_restored
                    or module.__spec__ is not self.spec
                    or module.__loader__ is not self.original
                    or self.spec.loader is not self.original
                    or self.spec.origin != self.origin
                    or self.spec.cached != self.cached
                    or self.spec.parent != self.parent
                    or self.spec.submodule_search_locations
                    is not self.submodule_search_locations
                ):
                    fail("CPU static preflight Torch JIT loader scope changed")
                remote_in_progress = sys.modules.get(target_remote_module)
                factory = getattr(
                    module,
                    "instantiate_non_scriptable_remote_module_template",
                    None,
                )
                factory_parameters = (
                    tuple(inspect.signature(factory).parameters.values())
                    if inspect.isfunction(factory)
                    else ()
                )
                if (
                    remote_finder_calls != 1
                    or remote_loader_create_module_calls != 1
                    or remote_loader_exec_module_calls != 1
                    or remote_in_progress is None
                    or jit_finder.remote_loader is None
                    or remote_in_progress.__loader__
                    is not jit_finder.remote_loader
                    or remote_in_progress.__spec__.loader
                    is not jit_finder.remote_loader
                    or not inspect.isfunction(factory)
                    or factory.__module__ != target_instantiator_module
                    or factory.__qualname__
                    != "instantiate_non_scriptable_remote_module_template"
                    or factory.__code__.co_firstlineno != 143
                    or tuple(factory.__code__.co_freevars) != ()
                    or Path(factory.__code__.co_filename).resolve(strict=True)
                    != jit_instantiator_path
                    or factory_parameters != ()
                    or inspect.signature(factory).return_annotation
                    is not inspect.Signature.empty
                ):
                    fail("CPU static preflight Torch remote factory authority differs")
                remote_template_factory_previous = factory
                setattr(
                    module,
                    "instantiate_non_scriptable_remote_module_template",
                    no_write_remote_template_factory,
                )
                remote_template_factory_scope_installed = True

        class ScopedRemoteModuleLoader:
            __slots__ = (
                "original",
                "spec",
                "origin",
                "cached",
                "parent",
                "submodule_search_locations",
            )

            def __init__(self, original: Any, spec: Any) -> None:
                self.original = original
                self.spec = spec
                self.origin = spec.origin
                self.cached = spec.cached
                self.parent = spec.parent
                self.submodule_search_locations = spec.submodule_search_locations

            def create_module(self, spec: Any) -> Any:
                nonlocal remote_loader_create_module_calls
                if (
                    remote_loader_create_module_calls != 0
                    or spec is not self.spec
                    or spec.loader is not self
                ):
                    fail("CPU static preflight Torch remote create_module differs")
                remote_loader_create_module_calls += 1
                created = self.original.create_module(spec)
                if created is not None:
                    fail("CPU static preflight Torch remote loader constructed a module")
                return None

            def exec_module(self, module: Any) -> None:
                nonlocal remote_loader_exec_module_calls
                nonlocal remote_template_factory_previous
                nonlocal remote_template_factory_scope_installed
                nonlocal remote_template_factory_scope_restored
                nonlocal remote_template_factory_scope_tampered
                if (
                    remote_loader_exec_module_calls != 0
                    or module is not sys.modules.get(target_remote_module)
                    or module.__spec__ is not self.spec
                    or module.__loader__ is not self
                    or self.spec.loader is not self
                    or self.spec.name != target_remote_module
                    or Path(self.spec.origin).resolve(strict=True)
                    != remote_module_path
                    or type(self.original)
                    is not importlib.machinery.SourceFileLoader
                    or self.original.name != target_remote_module
                    or Path(self.original.path).resolve(strict=True)
                    != remote_module_path
                ):
                    fail("CPU static preflight Torch remote loader authority differs")
                remote_loader_exec_module_calls += 1
                try:
                    self.original.exec_module(module)
                finally:
                    instantiator_module = sys.modules.get(
                        target_instantiator_module
                    )
                    remote_template_factory_scope_tampered = (
                        instantiator_module is None
                        or not remote_template_factory_scope_installed
                        or remote_template_factory_previous is _NO_BOUND_OWNER
                        or getattr(
                            instantiator_module,
                            "instantiate_non_scriptable_remote_module_template",
                            None,
                        )
                        is not no_write_remote_template_factory
                    )
                    if (
                        instantiator_module is not None
                        and remote_template_factory_previous
                        is not _NO_BOUND_OWNER
                    ):
                        setattr(
                            instantiator_module,
                            "instantiate_non_scriptable_remote_module_template",
                            remote_template_factory_previous,
                        )
                        remote_template_factory_scope_restored = (
                            getattr(
                                instantiator_module,
                                "instantiate_non_scriptable_remote_module_template",
                                None,
                            )
                            is remote_template_factory_previous
                        )
                    remote_template_factory_scope_installed = False
                    if module.__spec__ is self.spec:
                        self.spec.loader = self.original
                        module.__loader__ = self.original
                if (
                    remote_template_factory_scope_tampered
                    or not remote_template_factory_scope_restored
                    or remote_template_factory_calls != 1
                    or getattr(
                        module,
                        "_NON_SCRIPTABLE_REMOTE_MODULE_MODULE",
                        None,
                    )
                    is not remote_template_value
                    or module.__spec__ is not self.spec
                    or module.__loader__ is not self.original
                    or self.spec.loader is not self.original
                    or self.spec.origin != self.origin
                    or self.spec.cached != self.cached
                    or self.spec.parent != self.parent
                    or self.spec.submodule_search_locations
                    is not self.submodule_search_locations
                ):
                    fail("CPU static preflight Torch remote loader scope changed")

        class ExactInstantiatorFinder:
            __slots__ = ("loader", "remote_loader")

            def __init__(self) -> None:
                self.loader: Any = None
                self.remote_loader: Any = None

            def find_spec(
                self, fullname: str, path: Any = None, target: Any = None
            ) -> Any:
                nonlocal jit_finder_calls
                nonlocal remote_finder_calls
                if fullname == target_remote_module:
                    if (
                        remote_finder_calls != 0
                        or target is not None
                    ):
                        fail("CPU static preflight Torch remote finder call differs")
                    spec = importlib.machinery.PathFinder.find_spec(
                        fullname, path, target
                    )
                    original_loader = getattr(spec, "loader", None)
                    if (
                        spec is None
                        or spec.name != target_remote_module
                        or Path(getattr(spec, "origin", "")).resolve(strict=True)
                        != remote_module_path
                        or type(original_loader)
                        is not importlib.machinery.SourceFileLoader
                        or original_loader.name != target_remote_module
                        or Path(original_loader.path).resolve(strict=True)
                        != remote_module_path
                        or spec.submodule_search_locations is not None
                    ):
                        fail("CPU static preflight Torch remote finder authority differs")
                    self.remote_loader = ScopedRemoteModuleLoader(
                        original_loader, spec
                    )
                    spec.loader = self.remote_loader
                    remote_finder_calls += 1
                    return spec
                if fullname != target_instantiator_module:
                    return None
                if jit_finder_calls != 0 or target is not None:
                    fail("CPU static preflight Torch JIT finder call differs")
                spec = importlib.machinery.PathFinder.find_spec(
                    fullname, path, target
                )
                original_loader = getattr(spec, "loader", None)
                if (
                    spec is None
                    or spec.name != target_instantiator_module
                    or Path(getattr(spec, "origin", "")).resolve(strict=True)
                    != jit_instantiator_path
                    or type(original_loader)
                    is not importlib.machinery.SourceFileLoader
                    or original_loader.name != target_instantiator_module
                    or Path(original_loader.path).resolve(strict=True)
                    != jit_instantiator_path
                    or spec.submodule_search_locations is not None
                ):
                    fail("CPU static preflight Torch JIT finder authority differs")
                self.loader = ScopedInstantiatorLoader(original_loader, spec)
                spec.loader = self.loader
                jit_finder_calls += 1
                return spec

        jit_finder = ExactInstantiatorFinder()
        if (
            target_instantiator_module in sys.modules
            or target_remote_module in sys.modules
            or jit_finder in sys.meta_path
            or len(meta_path_snapshot) != 4
            or not meta_path_identity_is(meta_path_snapshot)
        ):
            fail("CPU static preflight Torch JIT import scope is not fresh")
        sys.meta_path.insert(0, jit_finder)
        jit_finder_scope_installed = True
        if (
            bernini_root_text != PINNED_BERNINI_ROOT
            or veomni_root_text != PINNED_VEOMNI_ROOT
            or preexisting_scoped_modules
            or any(
                Path(entry or os.getcwd()).resolve(strict=False)
                in (Path(bernini_root_text), Path(veomni_root_text))
                for entry in path_snapshot
            )
        ):
            fail("CPU static preflight Bernini/VeOmni import scope is not fresh/exact")
        sys.path[0:0] = [bernini_root_text, veomni_root_text]
        scoped_root_installed = True
        import torch

        if (
            jit_tempdir_constructor_calls != 0
            or jit_tempdir_registration_calls != 0
            or jit_tempdir_cleanup_calls != 0
            or jit_finder_calls != 0
            or jit_loader_create_module_calls != 0
            or jit_loader_exec_module_calls != 0
            or remote_finder_calls != 0
            or remote_loader_create_module_calls != 0
            or remote_loader_exec_module_calls != 0
            or remote_template_factory_calls != 0
            or target_instantiator_module in sys.modules
            or target_remote_module in sys.modules
            or tempfile_module.TemporaryDirectory
            is not temporary_directory_previous
            or atexit_module.register is not atexit_register_previous
        ):
            fail("CPU static preflight pre-diffusers import scope differs")
        cuda_initialized_before = bool(torch.cuda.is_initialized())
        if cuda_initialized_before:
            fail("CPU static preflight began after CUDA initialization")
        from diffusers import __version__ as diffusers_version
        from diffusers.models import AutoencoderKLWan

        jit_instantiator_module = sys.modules.get(
            "torch.distributed.nn.jit.instantiator"
        )
        remote_module = sys.modules.get(target_remote_module)
        jit_importer_parent_live = sys.path_importer_cache.get(
            jit_importer_parent, importer_cache_sentinel
        )
        remote_importer_parent_live = sys.path_importer_cache.get(
            remote_importer_parent, importer_cache_sentinel
        )
        importer_cache_jit_parent_added = (
            jit_importer_parent_live is not importer_cache_sentinel
            and type(jit_importer_parent_live)
            is importlib.machinery.FileFinder
        )
        importer_cache_remote_parent_added = (
            remote_importer_parent_prior is importer_cache_sentinel
            and remote_importer_parent_live is not importer_cache_sentinel
            and type(remote_importer_parent_live)
            is importlib.machinery.FileFinder
        )
        if (
            jit_tempdir_constructor_calls != 1
            or jit_tempdir_registration_calls != 1
            or jit_tempdir_cleanup_calls != 0
            or jit_finder_calls != 1
            or jit_loader_create_module_calls != 1
            or jit_loader_exec_module_calls != 1
            or jit_finder.loader is None
            or remote_finder_calls != 1
            or remote_loader_create_module_calls != 1
            or remote_loader_exec_module_calls != 1
            or remote_template_factory_calls != 1
            or jit_finder.remote_loader is None
            or not importer_cache_jit_parent_added
            or (
                remote_importer_parent_prior is importer_cache_sentinel
                and not importer_cache_remote_parent_added
            )
            or (
                remote_importer_parent_prior is not importer_cache_sentinel
                and remote_importer_parent_live is not remote_importer_parent_prior
            )
            or jit_instantiator_module is None
            or remote_module is None
            or Path(getattr(jit_instantiator_module, "__file__", "")).resolve(
                strict=True
            )
            != jit_instantiator_path
            or getattr(jit_instantiator_module, "_TEMP_DIR", None)
            is not jit_tempdir_value
            or getattr(
                jit_instantiator_module,
                "INSTANTIATED_TEMPLATE_DIR_PATH",
                None,
            )
            is not jit_tempdir_name
            or sys.path.count(jit_tempdir_name) != 1
            or not sys.path
            or sys.path[-1] is not jit_tempdir_name
            or jit_instantiator_module.__loader__
            is not jit_finder.loader.original
            or jit_instantiator_module.__spec__.loader
            is not jit_finder.loader.original
            or remote_module.__loader__
            is not jit_finder.remote_loader.original
            or remote_module.__spec__.loader
            is not jit_finder.remote_loader.original
            or getattr(
                remote_module,
                "_NON_SCRIPTABLE_REMOTE_MODULE_MODULE",
                None,
            )
            is not remote_template_value
            or not remote_template_factory_scope_restored
            or getattr(
                jit_instantiator_module,
                "instantiate_non_scriptable_remote_module_template",
                None,
            )
            is not remote_template_factory_previous
            or not temporary_directory_scope_restored
            or not atexit_scope_restored
            or tempfile_module.TemporaryDirectory
            is not temporary_directory_previous
            or atexit_module.register is not atexit_register_previous
        ):
            fail("CPU static preflight Torch JIT tempfile suppression differs")
        sys.path.pop()
        if jit_tempdir_name in sys.path:
            fail("CPU static preflight Torch JIT tempfile path was not removed")
        jit_sys_path_append_removed = True
        if (
            len(sys.meta_path) != 7
            or sys.meta_path[0] is not jit_finder
            or any(
                sys.meta_path[index + 1] is not original
                for index, original in enumerate(meta_path_snapshot)
            )
            or any(
                candidate is jit_finder
                or any(candidate is original for original in meta_path_snapshot)
                for candidate in sys.meta_path[5:]
            )
        ):
            fail("CPU static preflight Torch JIT meta-path scope changed")
        botocore_six_importer = sys.meta_path[5]
        six_importer = sys.meta_path[6]
        botocore_six_importer_receipt = audit_six_meta_path_importer(
            botocore_six_importer,
            module_name="botocore.vendored.six",
            source_path=botocore_six_path,
            source_sha256=botocore_six_sha,
            source_identity=botocore_six_identity,
        )
        six_importer_receipt = audit_six_meta_path_importer(
            six_importer,
            module_name="six",
            source_path=six_path,
            source_sha256=six_sha,
            source_identity=six_identity,
        )
        six_importer_scope_installed = True
        if sys.meta_path.pop(0) is not jit_finder:
            fail("CPU static preflight owned meta-path finder removal differs")
        jit_finder_scope_installed = False
        if not meta_path_identity_is(
            (*meta_path_snapshot, botocore_six_importer, six_importer)
        ):
            fail("CPU static preflight six meta-path importer order differs")

        from transformers import AutoTokenizer, __version__ as transformers_version
        from transformers.models.t5.tokenization_t5 import T5Tokenizer
        from bernini.models.renderer import BerniniRendererModel
        from bernini.models.wan_diffusion import GEN_Wanx22

        urllib3_connection_module = sys.modules.get("urllib3.util.connection")
        has_ipv6 = getattr(urllib3_connection_module, "_has_ipv6", None)
        has_ipv6_parameters = (
            tuple(inspect.signature(has_ipv6).parameters.values())
            if inspect.isfunction(has_ipv6)
            else ()
        )
        if (
            urllib3_connection_module is None
            or getattr(urllib3_connection_module, "HAS_IPV6", None) is not False
            or not inspect.isfunction(has_ipv6)
            or getattr(has_ipv6, "__module__", None)
            != "urllib3.util.connection"
            or getattr(has_ipv6, "__qualname__", None) != "_has_ipv6"
            or has_ipv6.__code__.co_firstlineno != 114
            or tuple(has_ipv6.__code__.co_freevars) != ()
            or len(has_ipv6_parameters) != 1
            or has_ipv6_parameters[0].name != "host"
            or has_ipv6_parameters[0].kind
            is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or has_ipv6_parameters[0].default is not inspect.Signature.empty
            or has_ipv6_parameters[0].annotation != "str"
            or inspect.signature(has_ipv6).return_annotation != "bool"
        ):
            fail("CPU static preflight blocked IPv6 probe state differs")
        if (
            str(torch.__version__) != PINNED_TORCH_VERSION
            or diffusers_version != PINNED_DIFFUSERS_VERSION
            or transformers_version != PINNED_TRANSFORMERS_VERSION
        ):
            fail("CPU static preflight vendor versions differ")
        wrapper_authority = authority(
            "site-packages:huggingface_hub/utils/_validators.py"
        )
        loader_receipts = {}
        for (
            label,
            function,
            method_name,
            module_name,
            owner_qualname,
            qualname,
            logical_name,
        ) in (
            (
                "Wan VAE load_config",
                AutoencoderKLWan.load_config,
                "load_config",
                "diffusers.configuration_utils",
                "ConfigMixin",
                "ConfigMixin.load_config",
                "site-packages:diffusers/configuration_utils.py",
            ),
            (
                "Wan VAE from_pretrained",
                AutoencoderKLWan.from_pretrained,
                "from_pretrained",
                "diffusers.models.modeling_utils",
                "ModelMixin",
                "ModelMixin.from_pretrained",
                "site-packages:diffusers/models/modeling_utils.py",
            ),
        ):
            definition_authority = authority(logical_name)
            loader_receipts[label] = (
                _audit_inherited_wrapped_classmethod_against_authenticated_sources(
                    function,
                    label=label,
                    expected_bound_owner=AutoencoderKLWan,
                    method_name=method_name,
                    expected_definition_module=module_name,
                    expected_definition_owner_qualname=owner_qualname,
                    expected_definition_qualname=qualname,
                    expected_definition_path=definition_authority["path"],
                    expected_definition_sha256=definition_authority["sha256"],
                    expected_wrapper_module="huggingface_hub.utils._validators",
                    expected_wrapper_factory_qualname="validate_hf_hub_args",
                    expected_wrapper_path=wrapper_authority["path"],
                    expected_wrapper_sha256=wrapper_authority["sha256"],
                )
            )

        # Bypass every constructor: these blank objects exist only to exercise
        # Python descriptor binding and are never called as models/tokenizers.
        blank_vae = object.__new__(AutoencoderKLWan)
        vae_receipt = _audit_loaded_vae_callables(
            blank_vae,
            expected_vae_class=AutoencoderKLWan,
            runtime_receipt=runtime_receipt,
        )
        tokenizer_factory_authority = authority(
            "site-packages:transformers/models/auto/tokenization_auto.py"
        )
        tokenizer_factory_receipt = _audit_callable_against_authenticated_source(
            AutoTokenizer.from_pretrained,
            label="AutoTokenizer from_pretrained",
            expected_module="transformers.models.auto.tokenization_auto",
            expected_qualname="AutoTokenizer.from_pretrained",
            expected_path=tokenizer_factory_authority["path"],
            expected_sha256=tokenizer_factory_authority["sha256"],
            expected_bound_owner=AutoTokenizer,
        )
        blank_tokenizer = object.__new__(T5Tokenizer)
        tokenizer_receipt = _audit_loaded_tokenizer_callable(
            blank_tokenizer,
            expected_tokenizer_class=T5Tokenizer,
            runtime_receipt=runtime_receipt,
        )
        blank_renderer = object.__new__(BerniniRendererModel)
        blank_diffusion = object.__new__(GEN_Wanx22)
        sample_pair = _audit_renderer_sample_pair_against_runtime(
            renderer=blank_renderer,
            diffusion=blank_diffusion,
            runtime_receipt=runtime_receipt,
            torch_module=torch,
        )

        shared_authority = authority("bernini:bernini/models/wan_diffusion.py")
        shared_descriptor = vars(GEN_Wanx22).get("shared_step")
        shared_definers = tuple(
            owner for owner in GEN_Wanx22.__mro__ if "shared_step" in vars(owner)
        )
        if (
            shared_definers != (GEN_Wanx22,)
            or not inspect.isfunction(shared_descriptor)
            or getattr(shared_descriptor, "__wrapped__", _NO_BOUND_OWNER)
            is not _NO_BOUND_OWNER
            or getattr(getattr(shared_descriptor, "__code__", None), "co_firstlineno", None)
            != 204
            or tuple(getattr(shared_descriptor.__code__, "co_freevars", ())) != ()
        ):
            fail("CPU static preflight shared_step descriptor differs")
        shared_receipt = _audit_callable_against_authenticated_source(
            shared_descriptor,
            label="native shared_step",
            expected_module="bernini.models.wan_diffusion",
            expected_qualname="GEN_Wanx22.shared_step",
            expected_path=shared_authority["path"],
            expected_sha256=shared_authority["sha256"],
        )
        shared_parameters = tuple(
            inspect.signature(shared_descriptor, follow_wrapped=False).parameters.values()
        )
        if (
            tuple(parameter.name for parameter in shared_parameters)
            != (
                "self",
                "model_id",
                "noisy_latents",
                "timesteps",
                "cond_embeds",
                "rotary_embs",
                "batch_vae_seqlen",
                "batch_text_seqlen",
                "kwargs",
            )
            or any(
                parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
                for parameter in shared_parameters[:-1]
            )
            or shared_parameters[-1].kind is not inspect.Parameter.VAR_KEYWORD
        ):
            fail("CPU static preflight shared_step signature differs")
        scoped_module_closure_receipt = audit_scoped_module_source_closure()
        if not meta_path_identity_is(
            (*meta_path_snapshot, botocore_six_importer, six_importer)
        ):
            fail("CPU static preflight six meta-path importer scope changed")
        if sys.meta_path.pop() is not six_importer:
            fail("CPU static preflight six importer removal order differs")
        if sys.meta_path.pop() is not botocore_six_importer:
            fail("CPU static preflight botocore six importer removal order differs")
        six_importer_scope_installed = False
        six_importer_scope_restored = meta_path_identity_is(meta_path_snapshot)
        jit_finder_scope_restored = six_importer_scope_restored
        if not six_importer_scope_restored:
            fail("CPU static preflight meta-path additions were not restored")
        cuda_initialized_after = bool(torch.cuda.is_initialized())
        if cuda_initialized_after:
            fail("CPU static preflight initialized CUDA")
    finally:
        if jit_finder_scope_installed:
            jit_finder_scope_tampered = (
                not sys.meta_path
                or sys.meta_path[0] is not jit_finder
                or len(sys.meta_path) != len(meta_path_snapshot) + 1
                or any(
                    sys.meta_path[index + 1] is not original
                    for index, original in enumerate(meta_path_snapshot)
                )
            )
            sys.meta_path[:] = list(meta_path_snapshot)
            jit_finder_scope_restored = (
                meta_path_identity_is(meta_path_snapshot)
            )
        if six_importer_scope_installed:
            six_importer_scope_tampered = not meta_path_identity_is(
                (*meta_path_snapshot, botocore_six_importer, six_importer)
            )
            sys.meta_path[:] = list(meta_path_snapshot)
            six_importer_scope_restored = (
                meta_path_identity_is(meta_path_snapshot)
            )
        if atexit_scope_installed:
            atexit_scope_tampered = (
                atexit_module.register is not no_write_atexit_register
            )
            atexit_module.register = atexit_register_previous
            atexit_scope_restored = (
                atexit_module.register is atexit_register_previous
            )
        if temporary_directory_scope_installed:
            temporary_directory_scope_tampered = (
                tempfile_module.TemporaryDirectory
                is not no_write_temporary_directory
            )
            tempfile_module.TemporaryDirectory = temporary_directory_previous
            temporary_directory_scope_restored = (
                tempfile_module.TemporaryDirectory
                is temporary_directory_previous
            )
        if tempfile_scope_installed:
            tempfile_scope_tampered = (
                tempfile_module.tempdir is not tempfile_override
            )
            tempfile_module.tempdir = tempfile_previous
            tempfile_scope_restored = (
                tempfile_previous is None and tempfile_module.tempdir is None
            )
        if scoped_root_installed:
            scope_tampered = (
                len(sys.path) < 2
                or sys.path[0] != bernini_root_text
                or sys.path[1] != veomni_root_text
                or tuple(sys.path[2:]) != path_snapshot
            )
            sys.path[:] = list(path_snapshot)
            for module_name in tuple(sys.modules):
                if (
                    module_name == "bernini"
                    or module_name.startswith("bernini.")
                    or module_name == "veomni"
                    or module_name.startswith("veomni.")
                ):
                    sys.modules.pop(module_name, None)
            if prior_importer_cache is importer_cache_sentinel:
                sys.path_importer_cache.pop(bernini_root_text, None)
            else:
                sys.path_importer_cache[bernini_root_text] = prior_importer_cache
            if prior_veomni_importer_cache is importer_cache_sentinel:
                sys.path_importer_cache.pop(veomni_root_text, None)
            else:
                sys.path_importer_cache[veomni_root_text] = (
                    prior_veomni_importer_cache
                )
        importer_cache_preexisting_tampered = any(
            key not in sys.path_importer_cache
            or sys.path_importer_cache[key] is not value
            for key, value in importer_cache_snapshot.items()
        )
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(importer_cache_snapshot)
        importer_cache_restored = (
            set(sys.path_importer_cache) == set(importer_cache_snapshot)
            and all(
                sys.path_importer_cache[key] is value
                for key, value in importer_cache_snapshot.items()
            )
        )
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if scope_tampered or (
            tempfile_scope_installed
            and (tempfile_scope_tampered or not tempfile_scope_restored)
        ) or (
            temporary_directory_scope_installed
            and (
                temporary_directory_scope_tampered
                or not temporary_directory_scope_restored
            )
        ) or (
            atexit_scope_installed
            and (atexit_scope_tampered or not atexit_scope_restored)
        ) or (
            jit_finder_scope_installed
            and (jit_finder_scope_tampered or not jit_finder_scope_restored)
        ) or (
            six_importer_scope_installed
            and (
                six_importer_scope_tampered
                or not six_importer_scope_restored
            )
        ) or importer_cache_preexisting_tampered or not importer_cache_restored:
            fail("CPU static preflight scoped import state changed during audit")

    if (
        not atexit_scope_restored
        or not temporary_directory_scope_restored
        or not remote_template_factory_scope_restored
        or not jit_finder_scope_restored
        or not six_importer_scope_restored
        or not importer_cache_restored
        or not meta_path_identity_is(meta_path_snapshot)
        or atexit_module.register is not atexit_register_previous
        or tempfile_module.TemporaryDirectory is not temporary_directory_previous
    ):
        fail("CPU static preflight Torch JIT import scopes were not restored")
    audit_guard_receipt = audit_guard.receipt(
        require_complete_static_import=True
    )

    validated_hashes = {
        key: str(value["sha256"])
        for key, value in sorted(rows.items())
        if isinstance(value, Mapping)
    }
    scoped_closure_rows = [
        dict(row) for row in scoped_module_closure_receipt
    ]
    scoped_closure_digest = object_sha256(scoped_closure_rows)
    six_meta_path_rows = [
        {"live_index_without_owned_finder": 4, **dict(botocore_six_importer_receipt)},
        {"live_index_without_owned_finder": 5, **dict(six_importer_receipt)},
    ]
    six_meta_path_digest = object_sha256(six_meta_path_rows)
    contract = {
        "runtime_release_manifest_sha256": runtime_receipt["release"][
            "manifest_sha256"
        ],
        "runtime_release_digest": runtime_receipt["release"]["release_digest"],
        "validated_vendor_source_sha256": validated_hashes,
        "validated_stdlib_socket_source_sha256": socket_source_sha,
        "pinned_bernini_root": bernini_root_text,
        "pinned_veomni_root": veomni_root_text,
        "scoped_module_source_closure_digest": scoped_closure_digest,
        "six_meta_path_importer_digest": six_meta_path_digest,
        "torch_version": PINNED_TORCH_VERSION,
        "diffusers_version": PINNED_DIFFUSERS_VERSION,
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "renderer_sample_parameters": list(RENDERER_SAMPLE_PARAMETERS),
        "diffusion_sample_parameters": list(INTERNAL_DIFFUSION_SAMPLE_PARAMETERS),
        "shared_step_parameters": [
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
            "kwargs",
        ],
    }
    unsigned = {
        "schema_version": CPU_STATIC_PREFLIGHT_SCHEMA,
        "authority": AUTHORITY,
        "complete": True,
        "contract_digest": object_sha256(contract),
        "exact_five_release_authenticated": True,
        "all_vendor_source_pins_rehashed": True,
        "scoped_module_source_closure": {
            "exact_module_count": len(scoped_closure_rows),
            "exact_bernini_module_count": sum(
                row["prefix"] == "bernini" for row in scoped_closure_rows
            ),
            "exact_veomni_module_count": sum(
                row["prefix"] == "veomni" for row in scoped_closure_rows
            ),
            "rows": scoped_closure_rows,
            "digest": scoped_closure_digest,
            "module_file_and_spec_origin_exact": True,
            "no_process_specific_repr_or_object_address_recorded": True,
        },
        "six_meta_path_importer_scope": {
            "snapshot_count": len(meta_path_snapshot),
            "owned_finder_live_index": 0,
            "addition_count": len(six_meta_path_rows),
            "deletion_count": 0,
            "original_snapshot_identity_order_preserved": True,
            "additions": six_meta_path_rows,
            "digest": six_meta_path_digest,
            "exact_append_order_verified": True,
            "exact_reverse_removal_verified": True,
            "restored_exactly": six_importer_scope_restored,
            "repr_or_object_address_in_receipt": False,
        },
        "vae_loader_wrappers_authenticated": len(loader_receipts) == 2,
        "vae_apply_forward_hook_and_annotations_authenticated": (
            vae_receipt.get("pep604_union_annotation_structure_verified") is True
        ),
        "tokenizer_factory_and_bound_call_authenticated": (
            bool(tokenizer_factory_receipt) and bool(tokenizer_receipt)
        ),
        "renderer_and_diffusion_no_grad_layers_authenticated": (
            sample_pair.get("exact_no_grad_sample_count") == 2
        ),
        "shared_step_signature_authenticated": bool(shared_receipt),
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "accelerator_visibility_environment": visibility_environment,
        "blas_import_environment": blas_import_environment,
        "blas_import_environment_preseeded_before_vendor_imports": True,
        "blas_import_environment_mutations_allowed": False,
        "veomni_logging_environment": veomni_logging_environment,
        "veomni_logging_environment_preseeded_before_vendor_imports": True,
        "weights_loaded": False,
        "model_constructors_called": False,
        "product_output_writes": False,
        "persistent_filesystem_writes": audit_guard_receipt[
            "persistent_filesystem_writes"
        ],
        "subprocesses_spawned": audit_guard_receipt["subprocesses_spawned"],
        "network_probe_blocked_before_socket_creation": (
            audit_guard_receipt["blocked_network_probe_count"] == 1
            and audit_guard_receipt["socket_objects_created"] is False
        ),
        "network_accessed": audit_guard_receipt["network_accessed"],
        "process_audit_guard": dict(audit_guard_receipt),
        "tempfile_import_probe_suppression": {
            "guard_installed_before_tempfile_import": True,
            "prior_tempfile_tempdir_was_none": True,
            "sentinel_path": tempfile_override,
            "sentinel_lstat_was_enoent": True,
            "scoped_tempdir_restored_by_identity": tempfile_scope_restored,
        },
        "numpy_blas_import_environment_seal": {
            "source_path": str(numpy_core_path),
            "source_sha256": numpy_core_sha,
            "source_size": numpy_core_identity["size"],
            "required_environment": blas_import_environment,
            "preseeded_before_vendor_imports": True,
            "putenv_and_unsetenv_audit_events_allowed": False,
            "process_environment_mutations_observed": 0,
        },
        "veomni_stdout_logging_suppression": {
            "source_path": str(veomni_logging_path),
            "source_sha256": veomni_logging_sha,
            "source_size": veomni_logging_identity["size"],
            "required_environment": veomni_logging_environment,
            "preseeded_before_vendor_imports": True,
            "putenv_and_unsetenv_audit_events_allowed": False,
            "timestamped_info_output_enabled": False,
        },
        "torch_jit_temporary_directory_suppression": {
            "source_path": str(jit_instantiator_path),
            "source_sha256": jit_instantiator_sha,
            "source_size": jit_instantiator_identity["size"],
            "exact_meta_path_finder_calls": jit_finder_calls,
            "exact_loader_create_module_calls": jit_loader_create_module_calls,
            "exact_loader_exec_module_calls": jit_loader_exec_module_calls,
            "exact_remote_finder_calls": remote_finder_calls,
            "exact_remote_loader_create_module_calls": (
                remote_loader_create_module_calls
            ),
            "exact_remote_loader_exec_module_calls": (
                remote_loader_exec_module_calls
            ),
            "meta_path_restored_exactly": jit_finder_scope_restored,
            "path_importer_cache_restored_exactly": importer_cache_restored,
            "jit_parent_importer_cache_was_missing_then_added": (
                jit_importer_parent_prior is importer_cache_sentinel
                and importer_cache_jit_parent_added
            ),
            "remote_parent_importer_cache_was_added": (
                importer_cache_remote_parent_added
            ),
            "module_loader_and_spec_loader_restored": (
                jit_instantiator_module.__loader__
                is jit_finder.loader.original
                and jit_instantiator_module.__spec__.loader
                is jit_finder.loader.original
            ),
            "constructor_call_line": 21,
            "constructor_calls_suppressed": jit_tempdir_constructor_calls,
            "atexit_registration_line": 23,
            "atexit_registrations_suppressed": jit_tempdir_registration_calls,
            "atexit_interception_only_during_exact_module_exec": True,
            "cleanup_calls_observed": jit_tempdir_cleanup_calls,
            "fixed_sentinel_name": jit_tempdir_name,
            "fixed_sentinel_lstat_was_enoent": True,
            "sys_path_append_removed": jit_sys_path_append_removed,
            "temporary_directory_class_restored_by_identity": (
                temporary_directory_scope_restored
            ),
            "atexit_register_restored_by_identity": atexit_scope_restored,
            "real_directory_created": False,
            "real_atexit_handler_registered": False,
        },
        "torch_remote_module_template_suppression": {
            "source_path": str(remote_module_path),
            "source_sha256": remote_module_sha,
            "source_size": remote_module_identity["size"],
            "module_call_line": 30,
            "instantiator_factory_firstlineno": 143,
            "factory_calls_suppressed": remote_template_factory_calls,
            "module_global_is_exact_opaque_sentinel": (
                getattr(
                    remote_module,
                    "_NON_SCRIPTABLE_REMOTE_MODULE_MODULE",
                    None,
                )
                is remote_template_value
            ),
            "factory_restored_by_identity": (
                remote_template_factory_scope_restored
                and getattr(
                    jit_instantiator_module,
                    "instantiate_non_scriptable_remote_module_template",
                    None,
                )
                is remote_template_factory_previous
            ),
            "template_source_written": False,
            "generated_template_imported": False,
        },
        "python_bytecode_writes_disabled_during_import": True,
        "pinned_bernini_and_veomni_roots_scoped_and_restored": True,
        "preexisting_bernini_or_veomni_modules_accepted": False,
    }
    return {**unsigned, "preflight_digest": object_sha256(unsigned)}


def _make_level_b_cpu_static_preflight(
    implementation: Any,
    runtime_validator: Any,
    expected_scoped_module_closure: tuple[
        tuple[str, str, str, str, int], ...
    ],
) -> Any:
    sealed_scoped_module_closure = tuple(
        tuple(row) for row in expected_scoped_module_closure
    )

    def run_level_b_cpu_static_runtime_preflight(
        *, verified_runtime: VerifiedLevelBRuntime
    ) -> Mapping[str, Any]:
        if type(verified_runtime) is not VerifiedLevelBRuntime:
            fail("Level-B static preflight requires one exact opaque verified runtime")
        runtime = runtime_validator(verified_runtime)
        return implementation(
            runtime,
            expected_scoped_module_closure=sealed_scoped_module_closure,
        )

    return run_level_b_cpu_static_runtime_preflight


run_level_b_cpu_static_runtime_preflight = _make_level_b_cpu_static_preflight(
    _run_level_b_cpu_static_runtime_preflight_authenticated,
    VerifiedLevelBRuntime.validate_at_use,
    PINNED_CPU_STATIC_SCOPED_MODULE_CLOSURE,
)
del _make_level_b_cpu_static_preflight
del _run_level_b_cpu_static_runtime_preflight_authenticated


run_level_b_pre_d0_offline_inference = _make_level_b_public_product_entrypoint(
    _run_level_b_pre_d0_offline_inference_authenticated,
    VerifiedLevelBRuntime.validate_at_use,
)
del _make_level_b_public_product_entrypoint
del _run_level_b_pre_d0_offline_inference_authenticated


__all__ = [
    "AUTHORITY",
    "BRIDGE_TRACE_SCHEMA",
    "CPU_STATIC_PREFLIGHT_SCHEMA",
    "INTERNAL_DIFFUSION_SAMPLE_PARAMETERS",
    "LEVEL_B_RELEASE_MEMBER_PATHS",
    "LEVEL_B_STATIC_SOURCE_MEMBER_PINS",
    "LevelBRendererError",
    "VerifiedLevelBRuntime",
    "METHOD",
    "NativeInitialNoiseObserver",
    "NUM_INFERENCE_STEPS",
    "OFFICIAL_RENDERER_SAMPLE_KEYWORDS",
    "OUTPUT_COMMIT_MARKER_SCHEMA",
    "PINNED_BASE_CHECKPOINT",
    "PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST",
    "PINNED_BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "PINNED_BASE_CHECKPOINT_TREE_SHA256",
    "PINNED_BERNINI_ROOT",
    "PINNED_VEOMNI_ROOT",
    "PINNED_FFMPEG_PATH",
    "PINNED_FFMPEG_SHA256",
    "PINNED_FFPROBE_PATH",
    "PINNED_FFPROBE_SHA256",
    "PINNED_UNIPC_SCHEDULE_SHA256",
    "PINNED_UNIPC_TIMESTEPS",
    "RECEIPT_SCHEMA",
    "RUNTIME_RELEASE_SCHEMA",
    "audit_official_renderer_sample_call",
    "authenticate_level_b_runtime_release",
    "observe_native_initial_noise",
    "run_level_b_pre_d0_offline_inference",
    "run_level_b_cpu_static_runtime_preflight",
    "validate_committed_level_b_product",
    "validate_public_product_signature",
]
