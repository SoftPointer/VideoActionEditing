#!/usr/bin/env python3
"""Pinned AUH E00 runtime adapter for the v15 source-role SP4 probe.

This module is deliberately an inference-only *site adapter*.  Importing it
does not import Bernini, initialize distributed state, construct a model, or
touch a GPU.  ``create_auh_bernini_source_role_adapter`` is the public
``module:factory`` entry point consumed by
``probe_source_owned_role_locator_v15_sp4.py`` after that harness has already
initialized a four-rank process group.

The adapter runs one frozen, adapter-free Bernini-R transformer forward on a
real E00 source latent at official UniPC 40-step schedule index 37
(t=291, sigma=0.2911904454231262).  It
splits the byte-pinned official ``WanTransformer3DModel.forward`` exactly at
``condition_embedder`` / ``prepare_inputs_for_sp`` so the observer context is
active before the real SP sharding call.  It neither calls a decoder nor
installs an action route.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as role_asset_v15b  # noqa: E402


ADAPTER_SCHEMA_VERSION = "bernini-source-owned-role-locator-sp4-adapter-v15"
BINDING_SCHEMA_VERSION = "bernini-auh-e00-source-role-runtime-binding-v15"
PREFLIGHT_SCHEMA_VERSION = "bernini-auh-e00-source-role-import-preflight-v15"
SP_SIZE = 4
SELECTED_BLOCKS = (4, 9, 14, 19, 24)

BERNINI_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_REVISION = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
CHECKPOINT_MANIFEST_SHA256 = "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
SOURCE_VIDEO_SHA256 = "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
SOURCE_MANIFEST_SHA256 = "ced0109cbd3a29b0c41826675793c188b19f0b1b342f03c9ab278091c20ed830"
CLEAN_LATENT_FILE_SHA256 = "d519727f5ba7cf8c39196bd1dcea3007eea09bca897621f54df73e268379dc3e"
CLEAN_LATENT_TENSOR_SHA256 = "f370b5a156275db335f20bb90c9cfb260de1590b92b5985de38f878225fd67ea"
NOISE_FILE_SHA256 = "8d6914d5bf0b4183bd9a11fad22064638344e44e0735cbf76f55b9d0b423ada7"
NOISE_TENSOR_SHA256 = "76034fae3be7c6170ea91686f5335f7c5fae7da1faf73137a0c71ef93196733d"
TIMESTEP_VALUE = 291
TIMESTEP_TENSOR_SHA256 = "06e98d7779c13d13358cae9e1277778c8da0676073c52ff0fe5699a7eb196078"
SCHEDULE_SIGMA = 0.2911904454231262
SCHEDULE_SIGMA_TENSOR_SHA256 = "7e72278017530771f7bfc16cdd1471a1cbd19f82f1c5af140c679646b2fa8749"
NOISY_SOURCE_TENSOR_SHA256 = "31d4bab37534c1612498395d40f627b60f096f332ebbd0b48005cdf3f631981b"
TOKEN_INPUT_IDS_SHA256 = "29c64e1005bc625c64a194d7056c6c1d9b15b78bb994c14793d46fa71d00983e"
TOKEN_ATTENTION_MASK_SHA256 = "86c7129afa1c6cc35f5104ce2bf534dff382c6a6e3c2c063b284ed328bcd14a3"
MODEL_TEXT_SHA256 = "c424fda6ec36b5c78a856f8abcba1db217c407f2920317832cc0e26095274451"
SOURCE_IID = "2f183dbf9e7a4d2e"
EVENT_ID = "pour-liquid-into-cup"
LATENT_SHAPE = (1, 16, 21, 74, 50)
SOURCE_GEOMETRY = (21, 37, 25)
RENDERER_TEXT_LENGTH = 512

_STAGE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1"
)
DFIX2_SOURCE_TREE = _STAGE / "source-online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2"
DFIX2_METHOD_ROOT = DFIX2_SOURCE_TREE / "methods" / "bernini_action_editing"
BERNINI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
)
VEOMNI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
)
CHECKPOINT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
CHECKPOINT_MANIFEST = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/"
    "runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
)
SOURCE_VIDEO = Path(
    "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/"
    "2f183dbf9e7a4d2e/source.mp4"
)
_SOURCE_RUN = (
    _STAGE
    / "interaction_complex8_large_lora_decode_v1"
    / "dpo_identity005_s4"
    / "event_00"
    / "step_0001"
)
SOURCE_MANIFEST = _SOURCE_RUN / "receipt.json"
CLEAN_LATENT = _SOURCE_RUN / "source.normalized-clean-latent.safetensors"
NOISE = _SOURCE_RUN / "rv2v.official-initial-gaussian.safetensors"

PINNED_RUNTIME_FILES = {
    BERNINI_ROOT / "bernini/models/renderer.py": (
        "fec319f3ede3482b28873dc55622208f1242ecba0caedea8e710093748dc7159"
    ),
    BERNINI_ROOT / "bernini/models/transformer_wan.py": (
        "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223"
    ),
    BERNINI_ROOT / "bernini/models/wan_diffusion.py": (
        "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
    ),
    BERNINI_ROOT / "bernini/models/scheduler.py": (
        "b6d729187fd784bf66831d5260a5c9482d89c452881d2f700c8887278f52ef97"
    ),
    BERNINI_ROOT / "bernini/parallel/__init__.py": (
        "ef16834c0af0e4e2201db37fbbd3a13be6622ac8e09d076a6e6bf68543c9bc29"
    ),
    BERNINI_ROOT / "bernini/parallel/state.py": (
        "32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa"
    ),
    BERNINI_ROOT / "bernini/parallel/ops.py": (
        "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30"
    ),
    DFIX2_METHOD_ROOT / "infer_anchor_sga_anc_event_v1.py": (
        "dd3558a4c38c5541ba6b7ad455ac599f43eb48b1b56f207a07776c9e1819145f"
    ),
    DFIX2_METHOD_ROOT / "infer_lora.py": (
        "0c79faa8417a40a5735571db3a5ba828d6aa977d7d0507a5bfcb63368c07728d"
    ),
    DFIX2_METHOD_ROOT / "train_lora.py": (
        "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85"
    ),
    DFIX2_METHOD_ROOT / "infer_source_aligned_controller_oracle.py": (
        "9ae3a41e52f520f66ebcddba331b26837a5c8291426d13379eaa4c8a01a80e02"
    ),
}

PINNED_VERSIONS = {
    "torch": "2.7.1+rocm6.3",
    "transformers": "5.5.4",
    "diffusers": "0.38.0",
    "safetensors": "0.8.0rc0",
}


class AUHSourceRoleAdapterError(RuntimeError):
    """Fail-closed AUH runtime binding or forward-seam violation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AUHSourceRoleAdapterError(f"cannot hash pinned file: {path}") from error
    return digest.hexdigest()


def _plain_file(path: Path, *, label: str, expected_sha256: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise AUHSourceRoleAdapterError(f"{label} must be an absolute non-symlink file")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise AUHSourceRoleAdapterError(f"{label} is missing") from error
    if not stat.S_ISREG(mode):
        raise AUHSourceRoleAdapterError(f"{label} is not a plain file")
    if _sha256_file(path) != expected_sha256:
        raise AUHSourceRoleAdapterError(f"{label} SHA-256 differs")
    return path.resolve(strict=True)


def _plain_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise AUHSourceRoleAdapterError(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AUHSourceRoleAdapterError(f"{label} is missing") from error
    if not resolved.is_dir():
        raise AUHSourceRoleAdapterError(f"{label} is not a directory")
    return resolved


def _exact_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AUHSourceRoleAdapterError(f"{label} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class _AdapterConfig:
    selected_block_indices: tuple[int, ...]


def _normalize_config(value: Mapping[str, Any]) -> _AdapterConfig:
    if not isinstance(value, Mapping):
        raise AUHSourceRoleAdapterError("adapter config must be one JSON object")
    if set(value) - {"selected_block_indices"}:
        raise AUHSourceRoleAdapterError("adapter config contains an unsupported override")
    raw = value.get("selected_block_indices", list(SELECTED_BLOCKS))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AUHSourceRoleAdapterError("selected block indices are malformed")
    blocks = tuple(raw)
    if blocks != SELECTED_BLOCKS:
        raise AUHSourceRoleAdapterError("this pinned adapter supports only blocks 4/9/14/19/24")
    return _AdapterConfig(selected_block_indices=blocks)


def pinned_e00_binding_registry() -> dict[str, Any]:
    """Return immutable public inputs without touching their AUH paths."""

    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "event_id": EVENT_ID,
        "source_iid": SOURCE_IID,
        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "source_video_sha256": SOURCE_VIDEO_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "clean_latent_file_sha256": CLEAN_LATENT_FILE_SHA256,
        "clean_latent_tensor_sha256": CLEAN_LATENT_TENSOR_SHA256,
        "noise_file_sha256": NOISE_FILE_SHA256,
        "noise_tensor_sha256": NOISE_TENSOR_SHA256,
        "official_schedule": {
            "scheduler": "diffusers.UniPCMultistepScheduler",
            "flow_shift": 5.0,
            "steps": 40,
            "index": 37,
            "timestep": TIMESTEP_VALUE,
            "sigma": SCHEDULE_SIGMA,
        },
        "timestep_tensor_sha256": TIMESTEP_TENSOR_SHA256,
        "schedule_sigma_tensor_sha256": SCHEDULE_SIGMA_TENSOR_SHA256,
        "noisy_source_tensor_sha256": NOISY_SOURCE_TENSOR_SHA256,
        "token_input_ids_sha256": TOKEN_INPUT_IDS_SHA256,
        "token_attention_mask_sha256": TOKEN_ATTENTION_MASK_SHA256,
        "model_text_sha256": MODEL_TEXT_SHA256,
        "tokenizer_tree_sha256": locator.PINNED_TOKENIZER_TREE_SHA256,
        "latent_shape": list(LATENT_SHAPE),
        "source_geometry": list(SOURCE_GEOMETRY),
        "selected_block_indices": list(SELECTED_BLOCKS),
        "role_asset_schema_version": role_asset_v15b.ASSET_SCHEMA_VERSION,
        "role_asset_sha256": role_asset_v15b.ASSET_SHA256,
        "role_event_sha256": role_asset_v15b.EVENT_SHA256,
        "role_names": list(role_asset_v15b.ROLE_NAMES),
        "vessel_competition_group": list(role_asset_v15b.VESSEL_COMPETITION_GROUP),
        "independent_roles": list(role_asset_v15b.INDEPENDENT_ROLES),
        "role_labels_are_source_instance_descriptors_not_action_ground_truth": True,
        "requested_action_graph": "stop(#1 -> #2), then execute(#2 -> #3), frame0 exact",
        "action_success_authorized": False,
        "frozen_base": True,
        "eval_mode": True,
        "adapters_disabled": True,
        "observer_only": True,
        "route_authorized": False,
        "training_authorized": False,
        "decode_authorized": False,
    }


def _validate_source_manifest(value: Mapping[str, Any]) -> None:
    """Authenticate the source clean latent and independent native Gaussian."""

    try:
        checkpoint = value["checkpoint"]
        source = value["input"]
        clean = value["source_condition_artifact"]
        noise = value["initial_noise_artifacts"]["rv2v"]
        runtime = value["runtime_versions"]
    except (KeyError, TypeError) as error:
        raise AUHSourceRoleAdapterError("source manifest lacks a pinned binding") from error
    required = (
        value.get("schema_version") == "bernini-native-identity-generation-canary-v1",
        value.get("bernini_commit") == BERNINI_REVISION,
        value.get("veomni_commit") == VEOMNI_REVISION,
        checkpoint.get("tree_sha256") == CHECKPOINT_TREE_SHA256,
        source.get("source_video_path") == str(SOURCE_VIDEO),
        source.get("source_video_sha256") == SOURCE_VIDEO_SHA256,
        source.get("target_video") is False,
        clean.get("path") == str(CLEAN_LATENT),
        clean.get("sha256") == CLEAN_LATENT_FILE_SHA256,
        clean.get("tensor_key") == "normalized_clean_latent",
        clean.get("shape") == list(LATENT_SHAPE),
        clean.get("source_video_vae_encode_before_any_decode") is True,
        clean.get("roundtrip_byte_exact_fp32") is True,
        noise.get("path") == str(NOISE),
        noise.get("sha256") == NOISE_FILE_SHA256,
        noise.get("tensor_key") == "official_initial_gaussian",
        noise.get("shape") == list(LATENT_SHAPE),
        noise.get("source_or_target_derived") is False,
        noise.get("captured_from_native_sampler") is True,
        noise.get("roundtrip_raw_value_exact") is True,
        runtime.get("torch") == PINNED_VERSIONS["torch"],
        runtime.get("transformers") == PINNED_VERSIONS["transformers"],
        runtime.get("diffusers") == PINNED_VERSIONS["diffusers"],
    )
    if not all(required):
        raise AUHSourceRoleAdapterError("source manifest binding differs")


def _all_rank_exact(value: Any, *, label: str) -> None:
    rows: list[Any] = [None for _ in range(SP_SIZE)]
    dist.all_gather_object(rows, value)
    if rows != [value] * SP_SIZE:
        raise AUHSourceRoleAdapterError(f"{label} differs across SP4 ranks")


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    try:
        a = left.untyped_storage()
        b = right.untyped_storage()
        return (
            a.data_ptr() == b.data_ptr()
            and a.nbytes() == b.nbytes()
            and getattr(a, "_cdata", None) == getattr(b, "_cdata", None)
        )
    except Exception:
        return False


def _activate_pinned_source_trees() -> tuple[Any, Any, Any]:
    """Import the exact dfix2 helpers and official Bernini source tree."""

    roots = (str(DFIX2_METHOD_ROOT), str(BERNINI_ROOT), str(VEOMNI_ROOT))
    for root in roots:
        while root in sys.path:
            sys.path.remove(root)
    sys.path[0:0] = list(roots)
    legacy = importlib.import_module("infer_lora")
    event_runtime = importlib.import_module("infer_anchor_sga_anc_event_v1")
    source_audit = importlib.import_module("infer_source_aligned_controller_oracle")
    expected_modules = {
        legacy: DFIX2_METHOD_ROOT / "infer_lora.py",
        event_runtime: DFIX2_METHOD_ROOT / "infer_anchor_sga_anc_event_v1.py",
        source_audit: DFIX2_METHOD_ROOT / "infer_source_aligned_controller_oracle.py",
    }
    for module, expected in expected_modules.items():
        raw_file = getattr(module, "__file__", None)
        if raw_file is None or Path(raw_file).resolve(strict=True) != expected:
            raise AUHSourceRoleAdapterError(
                f"Python reused a non-dfix2 module for {module.__name__}"
            )
    return legacy, event_runtime, source_audit


def _verify_runtime_files() -> None:
    for path, digest in PINNED_RUNTIME_FILES.items():
        _plain_file(path, label=f"pinned runtime file {path.name}", expected_sha256=digest)


def _version_receipt() -> dict[str, str]:
    result = {
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "diffusers": importlib.metadata.version("diffusers"),
        "safetensors": importlib.metadata.version("safetensors"),
    }
    if result != PINNED_VERSIONS:
        raise AUHSourceRoleAdapterError("AUH Python package versions differ")
    return result


def remote_import_preflight_plan() -> dict[str, Any]:
    """Describe the read-only check to run before any four-GPU model load."""

    value = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MODELING_BACKEND": "hf",
        },
        "module_factory": (
            "auh_source_owned_role_locator_v15_adapter:"
            "create_auh_bernini_source_role_adapter"
        ),
        "checks": [
            "adapter_import_does_not_initialize_torch_distributed_or_cuda",
            "dfix2_and_official_Bernini_file_sha256",
            "Bernini_and_VeOmni_revision",
            "checkpoint_manifest_and_tree_identity",
            "source_video_manifest_clean_latent_and_noise_sha256",
            "official_forward_and_prepare_inputs_for_sp_signatures",
            "transformers_5.5.4_tokenizer_tree_and_E00_token_ids",
            "no_model_construction_no_GPU_no_output_write",
        ],
        "then": (
            "launch probe_source_owned_role_locator_v15_sp4.py with exactly four "
            "torchrun ranks only after the import receipt passes"
        ),
    }
    return {**value, "plan_sha256": locator.object_sha256(value)}


@dataclass(frozen=True)
class _PreparedForward:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    timestep_proj: torch.Tensor
    temb: torch.Tensor
    rotary_emb: torch.Tensor
    batch_image_vae_seqlen: tuple[int, ...]
    text_features_length: tuple[int, ...]
    kwargs: Mapping[str, Any]


class AUHBerniniSourceRoleRuntimeAdapter:
    """Frozen E00 Bernini forward split at the real sequence-parallel seam."""

    def __init__(self, config: _AdapterConfig) -> None:
        if not isinstance(config, _AdapterConfig):
            raise AUHSourceRoleAdapterError("adapter received an unvalidated config")
        if not dist.is_available() or not dist.is_initialized():
            raise AUHSourceRoleAdapterError("factory requires an initialized torchrun group")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.world_size != SP_SIZE:
            raise AUHSourceRoleAdapterError("adapter requires WORLD4")
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
        except (KeyError, TypeError, ValueError) as error:
            raise AUHSourceRoleAdapterError("LOCAL_RANK is missing or invalid") from error
        if not torch.cuda.is_available():
            raise AUHSourceRoleAdapterError("AUH adapter requires ROCm CUDA compatibility")
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        if os.environ.get("MODELING_BACKEND") != "hf":
            raise AUHSourceRoleAdapterError("MODELING_BACKEND must be exactly hf")
        self.selected_block_indices = config.selected_block_indices

        _version_receipt()
        _verify_runtime_files()
        for path, label in (
            (DFIX2_SOURCE_TREE, "dfix2 source tree"),
            (BERNINI_ROOT, "Bernini root"),
            (VEOMNI_ROOT, "VeOmni root"),
            (CHECKPOINT, "checkpoint"),
        ):
            _plain_directory(path, label=label)
        _plain_file(
            CHECKPOINT_MANIFEST,
            label="checkpoint manifest",
            expected_sha256=CHECKPOINT_MANIFEST_SHA256,
        )
        _plain_file(SOURCE_VIDEO, label="real E00 source", expected_sha256=SOURCE_VIDEO_SHA256)
        _plain_file(
            SOURCE_MANIFEST,
            label="source authority manifest",
            expected_sha256=SOURCE_MANIFEST_SHA256,
        )
        _plain_file(CLEAN_LATENT, label="source clean latent", expected_sha256=CLEAN_LATENT_FILE_SHA256)
        _plain_file(NOISE, label="independent native Gaussian", expected_sha256=NOISE_FILE_SHA256)
        try:
            source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AUHSourceRoleAdapterError("cannot parse source authority manifest") from error
        if not isinstance(source_manifest, Mapping):
            raise AUHSourceRoleAdapterError("source authority manifest is not an object")
        _validate_source_manifest(source_manifest)

        legacy, event_runtime, source_audit = _activate_pinned_source_trees()
        self._legacy = legacy
        self._event_runtime = event_runtime
        self._source_audit = source_audit
        try:
            bernini_root, veomni_root, bernini_revision, veomni_revision = (
                legacy.trainer.validate_source_trees(
                    BERNINI_ROOT,
                    VEOMNI_ROOT,
                    expected_bernini_commit=BERNINI_REVISION,
                    expected_veomni_commit=VEOMNI_REVISION,
                )
            )
            checkpoint, transformer_config = legacy.trainer.validate_checkpoint(CHECKPOINT)
            source_audit.legacy = legacy
            source_audit.trainer = legacy.trainer
            checkpoint_identity = source_audit.validate_checkpoint_content(
                checkpoint, CHECKPOINT_MANIFEST
            )
        except Exception as error:
            raise AUHSourceRoleAdapterError("dfix2 checkpoint/source-tree audit failed") from error
        if (
            bernini_root != BERNINI_ROOT
            or veomni_root != VEOMNI_ROOT
            or bernini_revision != BERNINI_REVISION
            or veomni_revision != VEOMNI_REVISION
            or checkpoint != CHECKPOINT
            or transformer_config.get("num_layers") != 30
            or checkpoint_identity.get("manifest_sha256_computed") != CHECKPOINT_MANIFEST_SHA256
            or checkpoint_identity.get("verified_file_count") != 23
            or checkpoint_identity.get("every_file_sha256_verified") is not True
        ):
            raise AUHSourceRoleAdapterError("checkpoint/source-tree identity differs")

        from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
        from bernini.parallel import get_parallel_state, init_parallel_state
        from transformers import AutoTokenizer, UMT5EncoderModel

        renderer_module = importlib.import_module("bernini.models.renderer")
        transformer_module = importlib.import_module("bernini.models.transformer_wan")
        if (
            Path(renderer_module.__file__).resolve(strict=True)
            != BERNINI_ROOT / "bernini/models/renderer.py"
            or Path(transformer_module.__file__).resolve(strict=True)
            != BERNINI_ROOT / "bernini/models/transformer_wan.py"
        ):
            raise AUHSourceRoleAdapterError("Python imported a non-pinned Bernini tree")

        parallel = init_parallel_state(ulysses_size=SP_SIZE)
        if (
            parallel.world_size != SP_SIZE
            or parallel.dp_size != 1
            or parallel.ulysses_size != SP_SIZE
            or parallel.ulysses_rank != self.rank
            or parallel.ulysses_group is None
            or dist.get_world_size(parallel.ulysses_group) != SP_SIZE
            or dist.get_rank(parallel.ulysses_group) != self.rank
            or get_parallel_state() is not parallel
        ):
            raise AUHSourceRoleAdapterError("Bernini Ulysses group is not the WORLD4 rank set")

        renderer_config = BerniniRendererConfig.from_pretrained(
            str(BERNINI_ROOT / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        renderer_config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(renderer_config.to_dict(), checkpoint)
        if float(renderer_config.shift) != 5.0 or renderer_config.use_unipc is not True:
            raise AUHSourceRoleAdapterError("renderer schedule/config differs")
        with event_runtime._nonzero_rank_t5_load_bypass(
            distributed_rank=self.rank,
            t5_encoder_class=UMT5EncoderModel,
            expected_checkpoint=checkpoint,
            expected_dtype=torch.bfloat16,
            placeholder_factory=torch.nn.Identity,
        ) as t5_audit:
            model = BerniniRendererModel(renderer_config)
        if self.rank == 0:
            if not isinstance(model.t5_text_encoder, UMT5EncoderModel):
                raise AUHSourceRoleAdapterError("rank zero lacks the official T5 encoder")
        elif model.t5_text_encoder is not t5_audit["placeholder"]:
            raise AUHSourceRoleAdapterError("nonzero rank T5 bypass differs")
        model.requires_grad_(False)
        model.eval()

        scheduler = model.diff_dec.scheduler
        if (
            type(scheduler).__module__
            != "diffusers.schedulers.scheduling_unipc_multistep"
            or type(scheduler).__name__ != "UniPCMultistepScheduler"
            or float(scheduler.config.flow_shift) != 5.0
        ):
            raise AUHSourceRoleAdapterError("official UniPC scheduler identity differs")
        scheduler.set_timesteps(40)
        schedule_timestep_cpu = scheduler.timesteps[37:38].detach().cpu().contiguous()
        schedule_sigma_cpu = scheduler.sigmas[37:38].detach().float().cpu().contiguous()
        if (
            schedule_timestep_cpu.dtype != torch.int64
            or schedule_timestep_cpu.tolist() != [TIMESTEP_VALUE]
            or schedule_sigma_cpu.dtype != torch.float32
            or schedule_sigma_cpu.tolist() != [SCHEDULE_SIGMA]
            or locator.tensor_sha256(schedule_timestep_cpu) != TIMESTEP_TENSOR_SHA256
            or locator.tensor_sha256(schedule_sigma_cpu)
            != SCHEDULE_SIGMA_TENSOR_SHA256
        ):
            raise AUHSourceRoleAdapterError("official UniPC schedule index 37 differs")

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        model_text = (
            "You are a helpful assistant specialized in text-to-video generation."
            "An East Asian woman in a black floral blouse pours amber tea from a white "
            "bowl into a clear glass pitcher on a wooden tea table while a small white "
            "cup sits nearby."
        )
        if locator.text_sha256(model_text) != MODEL_TEXT_SHA256:
            raise AUHSourceRoleAdapterError("hard-bound E00 model text differs")
        token_probe = tokenizer(
            model_text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        ids = list(token_probe["input_ids"])
        mask = list(token_probe["attention_mask"])
        if (
            locator.object_sha256(ids) != TOKEN_INPUT_IDS_SHA256
            or locator.object_sha256(mask) != TOKEN_ATTENTION_MASK_SHA256
            or sum(mask) != 56
        ):
            raise AUHSourceRoleAdapterError("E00 tokenizer input identity differs")
        _all_rank_exact(
            {"ids": TOKEN_INPUT_IDS_SHA256, "mask": TOKEN_ATTENTION_MASK_SHA256},
            label="token input",
        )
        padded_ids, padded_mask = legacy._tokenize_training_prompt(tokenizer, model_text)

        if self.rank != 0:
            event_runtime._retire_t5_text_encoder(model, torch_module=torch)
        dist.barrier()
        raw_source_text = None
        status: list[Any] = [None]
        if self.rank == 0:
            try:
                model.t5_text_encoder.to(self.device)
                with torch.inference_mode():
                    raw_source_text = model.encode_prompt(
                        padded_ids.to(self.device), padded_mask.to(self.device)
                    ).contiguous()
                status[0] = {"ok": True}
            except Exception as error:
                status[0] = {"ok": False, "type": type(error).__name__, "message": str(error)}
            finally:
                event_runtime._retire_t5_text_encoder(model, torch_module=torch)
        dist.broadcast_object_list(status, src=0)
        if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
            raise AUHSourceRoleAdapterError(f"rank-zero source T5 encoding failed: {status[0]}")
        if self.rank != 0:
            raw_source_text = torch.empty(
                (1, RENDERER_TEXT_LENGTH, 4096),
                dtype=torch.bfloat16,
                device=self.device,
            )
        dist.broadcast(raw_source_text, src=0)
        if (
            not isinstance(raw_source_text, torch.Tensor)
            or tuple(raw_source_text.shape) != (1, RENDERER_TEXT_LENGTH, 4096)
            or raw_source_text.dtype != torch.bfloat16
            or raw_source_text.requires_grad
        ):
            raise AUHSourceRoleAdapterError("source T5 embedding geometry differs")
        self._raw_source_text = raw_source_text.detach().contiguous()
        self._raw_text_sha256 = locator.tensor_sha256(self._raw_source_text)
        _all_rank_exact(self._raw_text_sha256, label="raw source text embedding")
        self._tokenizer = tokenizer
        self._tokenizer_dir = checkpoint / "tokenizer"

        transformer = model.diff_dec.transformer
        if transformer is None or model.diff_dec.transformer_2 is not None:
            raise AUHSourceRoleAdapterError("renderer is not the frozen single-expert base")
        transformer.to(self.device)
        model.requires_grad_(False)
        model.eval()
        if transformer.gradient_checkpointing:
            raise AUHSourceRoleAdapterError("gradient checkpointing is enabled")
        if len(transformer.blocks) != 30:
            raise AUHSourceRoleAdapterError("transformer block count differs")
        for index in self.selected_block_indices:
            processor = transformer.blocks[index].attn2.processor
            if (
                type(processor).__module__ != locator.OFFICIAL_ATTN2_PROCESSOR_MODULE
                or type(processor).__name__ != locator.OFFICIAL_ATTN2_PROCESSOR_CLASS
            ):
                raise AUHSourceRoleAdapterError("selected attn2 processor is not official")
        adapter_markers = []
        for name, module in model.named_modules():
            identity = f"{type(module).__module__}.{type(module).__name__}".lower()
            lowered_name = name.lower()
            if (
                "peft" in identity
                or "lora" in identity
                or "adapter" in identity
                or "lora" in lowered_name
                or "adapter" in lowered_name
            ):
                adapter_markers.append((name, identity))
        parameter_markers = [
            name
            for name, _parameter in model.named_parameters()
            if "lora" in name.lower() or "adapter" in name.lower()
        ]
        if (
            adapter_markers
            or parameter_markers
            or any(parameter.requires_grad for parameter in model.parameters())
        ):
            raise AUHSourceRoleAdapterError("model contains an adapter or trainable parameter")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AUHSourceRoleAdapterError("model contains a pre-existing gradient")
        self.model = model
        self._transformer = transformer

        from safetensors.torch import load_file as load_safetensors_file

        clean_rows = load_safetensors_file(str(CLEAN_LATENT), device="cpu")
        noise_rows = load_safetensors_file(str(NOISE), device="cpu")
        if tuple(clean_rows) != ("normalized_clean_latent",):
            raise AUHSourceRoleAdapterError("source clean latent key differs")
        if tuple(noise_rows) != ("official_initial_gaussian",):
            raise AUHSourceRoleAdapterError("native Gaussian key differs")
        clean = clean_rows["normalized_clean_latent"].float().contiguous()
        noise = noise_rows["official_initial_gaussian"].float().contiguous()
        if (
            tuple(clean.shape) != LATENT_SHAPE
            or tuple(noise.shape) != LATENT_SHAPE
            or not bool(torch.isfinite(clean).all().item())
            or not bool(torch.isfinite(noise).all().item())
            or locator.tensor_sha256(clean) != CLEAN_LATENT_TENSOR_SHA256
            or locator.tensor_sha256(noise) != NOISE_TENSOR_SHA256
        ):
            raise AUHSourceRoleAdapterError("clean/noise tensor identity differs")
        timestep_cpu = schedule_timestep_cpu
        sigma = schedule_sigma_cpu[0]
        noisy_cpu = ((1.0 - sigma) * clean + sigma * noise).float().contiguous()
        if (
            locator.tensor_sha256(timestep_cpu) != TIMESTEP_TENSOR_SHA256
            or locator.tensor_sha256(noisy_cpu) != NOISY_SOURCE_TENSOR_SHA256
        ):
            raise AUHSourceRoleAdapterError("timestep/noised-source tensor identity differs")
        del clean_rows, noise_rows, clean, noise
        self._timestep = timestep_cpu.to(self.device)
        self._noisy_source = noisy_cpu.to(self.device).contiguous()
        del timestep_cpu, noisy_cpu

        # Wan keeps some modulation tables in fp32 even when the actual patch
        # convolution is bf16.  The official forward consumes VAE latents in
        # the patch-convolution dtype; using the first arbitrary parameter is
        # therefore neither stable nor correct.
        compute_dtype = transformer.patch_embedding.weight.dtype
        if compute_dtype != torch.bfloat16:
            raise AUHSourceRoleAdapterError("patch embedding compute dtype differs")
        with torch.inference_mode():
            visual_tokens, visual_rotary = transformer.patch_vae_latent(
                self._noisy_source.to(dtype=compute_dtype), source_id=0
            )
            temb, timestep_proj, conditioned, _ = transformer.condition_embedder(
                self._timestep, self._raw_source_text, None
            )
        if (
            tuple(visual_tokens.shape) != (1, 19425, 1536)
            or tuple(conditioned.shape) != (1, RENDERER_TEXT_LENGTH, 1536)
            or visual_tokens.dtype != torch.bfloat16
            or conditioned.dtype != torch.bfloat16
        ):
            raise AUHSourceRoleAdapterError("visual/text input embedding geometry differs")
        self._visual_tokens = visual_tokens.detach().contiguous()
        self._visual_rotary = visual_rotary.detach()
        self._visual_input_sha256 = locator.tensor_sha256(self._visual_tokens)
        self._conditioned_text_sha256 = locator.tensor_sha256(conditioned)
        self._temb_sha256 = locator.tensor_sha256(temb)
        self._timestep_proj_sha256 = locator.tensor_sha256(timestep_proj)
        _all_rank_exact(
            {
                "visual_input": self._visual_input_sha256,
                "conditioned_text": self._conditioned_text_sha256,
                "temb": self._temb_sha256,
                "timestep_proj": self._timestep_proj_sha256,
            },
            label="condition/input embedding",
        )
        self.source_geometry = locator.SourceVisualGeometry(height=37, width=25)
        self._pending_conditioned: torch.Tensor | None = None
        self._pending_temb: torch.Tensor | None = None
        self._pending_timestep_proj: torch.Tensor | None = None

        payload = {
            **pinned_e00_binding_registry(),
            "runtime_versions": _version_receipt(),
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "checkpoint_verified_entries_digest": checkpoint_identity[
                "verified_entries_digest"
            ],
            "pinned_runtime_file_sha256": {
                str(path): digest for path, digest in PINNED_RUNTIME_FILES.items()
            },
            "raw_source_text_embedding_sha256": self._raw_text_sha256,
            "conditioned_source_text_embedding_sha256": self._conditioned_text_sha256,
            "visual_input_embedding_sha256": self._visual_input_sha256,
            "time_embedding_sha256": self._temb_sha256,
            "timestep_projection_sha256": self._timestep_proj_sha256,
            "visual_source_id": 0,
            "visual_authority": "real_source_clean_plus_independent_native_gaussian",
            "input_embedding_binding": "computed_once_then_cross_rank_SHA_sealed",
            "forward_seam": (
                "official_condition_embedder_then_official_prepare_inputs_for_sp_then_"
                "30_official_blocks_then_official_output_gather"
            ),
        }
        self._binding_receipt = {
            **payload,
            "binding_receipt_sha256": locator.object_sha256(payload),
        }
        _all_rank_exact(self._binding_receipt, label="source binding receipt")

    def observer_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "checkpoint_sha256": CHECKPOINT_TREE_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_is_real_video": True,
            "frozen_base": True,
            "eval_mode": True,
            "adapters_disabled": True,
            "ulysses_group_is_world": True,
            "world_size": SP_SIZE,
            "selected_block_indices": list(self.selected_block_indices),
            "observer_only": True,
            "training_authorized": False,
            "route_authorized": False,
        }

    def binding_receipt(self) -> Mapping[str, Any]:
        return deepcopy(self._binding_receipt)

    def _check_event(self, event_spec: Any) -> None:
        if (
            not isinstance(event_spec, locator.SourceRoleEventSpec)
            or event_spec.event_id != EVENT_ID
            or event_spec.source_iid != SOURCE_IID
            or event_spec.model_text_sha256 != MODEL_TEXT_SHA256
            or event_spec.tokenizer_tree_sha256 != locator.PINNED_TOKENIZER_TREE_SHA256
            or event_spec.event_sha256 != role_asset_v15b.EVENT_SHA256
            or event_spec.role_names != role_asset_v15b.ROLE_NAMES
        ):
            raise AUHSourceRoleAdapterError(
                "probe event differs from the pinned E00 v15b instance roles"
            )

    def _derive_conditioned_source_text(self, raw: torch.Tensor) -> torch.Tensor:
        if (
            raw is not self._raw_source_text
            or not _same_storage(raw, self._raw_source_text)
            or locator.tensor_sha256(raw) != self._raw_text_sha256
        ):
            raise AUHSourceRoleAdapterError("condition embedder input is not pinned source T5")
        with torch.inference_mode():
            temb, timestep_proj, conditioned, image = self._transformer.condition_embedder(
                self._timestep, raw, None
            )
        if image is not None:
            raise AUHSourceRoleAdapterError("source-only condition unexpectedly produced image text")
        if (
            locator.tensor_sha256(temb) != self._temb_sha256
            or locator.tensor_sha256(timestep_proj) != self._timestep_proj_sha256
            or locator.tensor_sha256(conditioned) != self._conditioned_text_sha256
        ):
            raise AUHSourceRoleAdapterError("condition embedding changed after binding")
        self._pending_temb = temb.detach()
        self._pending_timestep_proj = timestep_proj.detach()
        self._pending_conditioned = conditioned.detach()
        return self._pending_conditioned

    def materialize_source(
        self, *, event_spec: Any, rank: int, world_size: int
    ) -> Mapping[str, Any]:
        self._check_event(event_spec)
        if rank != self.rank or world_size != self.world_size:
            raise AUHSourceRoleAdapterError("materialization rank/world differs")
        return {
            "tokenizer": self._tokenizer,
            "tokenizer_dir": self._tokenizer_dir,
            "raw_source_text_hidden_states": self._raw_source_text,
            "derive_conditioned_source_text": self._derive_conditioned_source_text,
            "renderer_text_length": RENDERER_TEXT_LENGTH,
            "geometry": self.source_geometry,
            "source_receipt_sha256": self._binding_receipt[
                "binding_receipt_sha256"
            ],
        }

    def prepare_inputs_for_sp(
        self,
        *,
        conditioned_source_text_hidden_states: torch.Tensor,
        source: Any,
        event_spec: Any,
        rank: int,
        world_size: int,
    ) -> _PreparedForward:
        self._check_event(event_spec)
        if rank != self.rank or world_size != self.world_size:
            raise AUHSourceRoleAdapterError("prepare rank/world differs")
        if (
            self._pending_conditioned is None
            or conditioned_source_text_hidden_states is not self._pending_conditioned
            or not _same_storage(
                conditioned_source_text_hidden_states, self._pending_conditioned
            )
            or locator.tensor_sha256(conditioned_source_text_hidden_states)
            != self._conditioned_text_sha256
            or getattr(source, "source_receipt_sha256", None)
            != self._binding_receipt["binding_receipt_sha256"]
        ):
            raise AUHSourceRoleAdapterError("prepare input is not the bound source text/storage")
        if self._pending_temb is None or self._pending_timestep_proj is None:
            raise AUHSourceRoleAdapterError("condition embedder outputs are absent")

        batch_image = (int(self._visual_tokens.shape[1]),)
        batch_text = (RENDERER_TEXT_LENGTH,)
        timestep_proj = self._pending_timestep_proj.unflatten(1, (6, -1))
        timestep_proj_indices = torch.repeat_interleave(
            torch.arange(len(batch_image), device=self.device),
            torch.tensor(batch_image, device=self.device),
        )
        temb = self._pending_temb[0:1].expand(batch_image[0], -1).unsqueeze(0)
        rotary = self._visual_rotary.transpose(1, 2)
        hidden, text, timestep_proj, temb, kwargs = (
            self._transformer.prepare_inputs_for_sp(
                self._visual_tokens,
                conditioned_source_text_hidden_states,
                timestep_proj,
                batch_image,
                batch_text,
                timestep_proj_indices,
                temb,
            )
        )
        expected_local = math.ceil(batch_image[0] / SP_SIZE)
        if (
            tuple(hidden.shape) != (1, expected_local, 1536)
            or tuple(text.shape) != (1, RENDERER_TEXT_LENGTH, 1536)
            or not _same_storage(text, conditioned_source_text_hidden_states)
            or tuple(timestep_proj.shape) != (1, expected_local, 6, 1536)
            or tuple(temb.shape) != (1, expected_local, 1536)
        ):
            raise AUHSourceRoleAdapterError("official prepare_inputs_for_sp geometry differs")
        return _PreparedForward(
            hidden_states=hidden,
            encoder_hidden_states=text,
            timestep_proj=timestep_proj,
            temb=temb,
            rotary_emb=rotary,
            batch_image_vae_seqlen=batch_image,
            text_features_length=batch_text,
            kwargs=kwargs,
        )

    def run_frozen_forward(self, *, prepared: _PreparedForward) -> torch.Tensor:
        if not isinstance(prepared, _PreparedForward):
            raise AUHSourceRoleAdapterError("prepared forward object differs")
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise AUHSourceRoleAdapterError("model left frozen eval state")
        hidden = prepared.hidden_states
        for block in self._transformer.blocks:
            hidden = block(
                hidden,
                prepared.encoder_hidden_states,
                prepared.timestep_proj,
                prepared.rotary_emb,
                prepared.batch_image_vae_seqlen,
                prepared.text_features_length,
                **prepared.kwargs,
            )
        shift_table, scale_table = self._transformer.scale_shift_table.float().chunk(
            2, dim=1
        )
        shift = shift_table + prepared.temb.float()
        scale = scale_table + prepared.temb.float()
        hidden = (
            self._transformer.norm_out(hidden.float()) * (1 + scale) + shift
        ).type_as(hidden)
        hidden = self._transformer.proj_out(hidden)
        from bernini.parallel import gather_outputs

        hidden = gather_outputs(
            hidden,
            gather_dim=1,
            padding_dim=1,
            unpad_dim_size=int(prepared.kwargs["origin_hidden_states_seq_len"]),
        )
        if (
            tuple(hidden.shape) != (1, 19425, 64)
            or hidden.device != self.device
            or hidden.requires_grad
            or hidden.grad_fn is not None
            or not bool(torch.isfinite(hidden).all().item())
        ):
            raise AUHSourceRoleAdapterError("full frozen forward output differs")
        self._pending_conditioned = None
        self._pending_temb = None
        self._pending_timestep_proj = None
        return hidden


def create_auh_bernini_source_role_adapter(
    config: Mapping[str, Any],
) -> AUHBerniniSourceRoleRuntimeAdapter:
    """Public ``module:factory`` for the observer-only SP4 harness."""

    return AUHBerniniSourceRoleRuntimeAdapter(_normalize_config(config))


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "AUHBerniniSourceRoleRuntimeAdapter",
    "AUHSourceRoleAdapterError",
    "BINDING_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "SELECTED_BLOCKS",
    "create_auh_bernini_source_role_adapter",
    "pinned_e00_binding_registry",
    "remote_import_preflight_plan",
]
