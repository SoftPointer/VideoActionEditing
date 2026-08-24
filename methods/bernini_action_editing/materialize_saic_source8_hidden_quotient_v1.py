#!/usr/bin/env python3
"""Materialize a preregistered source8 block-15 hidden quotient population.

This is an optimizer-free follow-up to the exploratory STARC core4 hidden
quotient audit.  It consumes the already sealed 60-candidate SAIC pure-T2V
bank, deterministically selects the minimum registered seed for each of the
eight registered sources, and materializes exactly the forward, no-op, and
reverse candidate states.  Every state is queried with the source-fixed
forward/no-op prompt pair at native schedule index 33.  Only the fixed spatial
sketch of ``block.15.output(forward_prompt) - block.15.output(noop_prompt)`` is
persisted.

The dog and human families run as independent SP4 groups.  Candidate RGB is
never opened, the generated latent is only a frozen hidden-state query, and no
editor forward, selection, parameter mutation, or optimizer exists here.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_contract  # noqa: E402
import generate_saic_pure_t2v_event_bank_v1 as generation_contract  # noqa: E402
import materialize_starc_core4_hidden_v1 as starc  # noqa: E402


SCHEMA_VERSION = "bernini-saic-source8-hidden-quotient-arm-v1"
GROUP_SCHEMA_VERSION = "bernini-saic-source8-hidden-quotient-group-v1"
MASTER_SCHEMA_VERSION = "bernini-saic-source8-hidden-quotient-master-v1"
ARM_RECEIPT_FILENAME = "saic-source8-hidden-arm-receipt-v1.json"
GROUP_RECEIPT_FILENAME = "saic-source8-hidden-group-{actor_family}-v1.json"
MASTER_RECEIPT_FILENAME = "saic-source8-hidden-master-v1.json"
TENSOR_FILENAME = "saic-source8-block15-hidden-residual.safetensors"

SOURCE_MANIFEST_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
    "t2v-events-topup-r6-umaskfix-72f3a40-r1/"
    "sealed-saic-source-manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
)
SOURCE_MANIFEST_CONTENT_SHA256 = (
    "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
)
SOURCE_VALIDATOR_SUMMARY_SHA256 = (
    "257d3aafaaee126ff2c1a061413d01bd0457676eb5d1ee027671221a5a794218"
)
ATTEMPTS_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
    "t2v-events-v1-533074b-r3/attempts"
)
ROOT_SPEC_SHA256 = (
    "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
)
ACTOR_FAMILIES = ("dog", "human")
BRANCH_ORDER = ("forward", "noop", "reverse")
SOURCE_COUNT = 8
CANDIDATE_COUNT = 60
SOURCES_PER_GROUP = 4
ARMS_PER_GROUP = 12
MODEL_FORWARDS_PER_ARM = 2
MODEL_FORWARDS_PER_GROUP = ARMS_PER_GROUP * MODEL_FORWARDS_PER_ARM
SOURCE8_LATENT_SHAPES = (
    (1, 16, 21, 60, 62),
    (1, 16, 21, 64, 58),
    (1, 16, 21, 82, 44),
    (1, 16, 21, 68, 54),
    (1, 16, 21, 74, 50),
)
SOURCE8_PATCH_GRIDS = ((30, 31), (32, 29), (41, 22), (34, 27), (37, 25))
SOURCE8_SKETCH_DIGESTS_BY_PATCH_POSITIONS = {
    930: (
        "5a75404b60cadddb29ac7473fc4596d7ebfcd306acfb3fa1a6bc6575a228a246",
        "be43863f6a000fb00083798610e3993200c24e5fd94dcb2ef7d4e3858618dde7",
        "4a8330c77079671f6515bda07acc21f0d060176c4c07d2609ad2553acf657561",
    ),
    928: (
        "260d47275c7d407512ff4fca9fa20d2223eaa29b6e4d151b7495e51721980df4",
        "9fdee154009d0d4283716a4e93abe4df2dde5241065040eaf05bd2c9a9f2fa64",
        "be52cac4d90f0a5a70368d25fef2fb1edb4d346fb10598329f5bb7e8e7285ede",
    ),
    902: (
        "b99fbc92380a0702ce7e1aca7d8a182c072d2b9f8ee5b746c51a14827ab3e5de",
        "99c97d50d1e28cb29adaf8fd8921800ca4946caae11f35e2cd8e1840b4d596aa",
        "0845cadd9958b1181a381fff44817d2acfa1c35636093f61bc20e929d80c33d2",
    ),
    918: (
        "f48f9577ec829cc67bd5f9da09721bebccec7e6c92b18f5322e25ab76f19192a",
        "d05582d93963ae8de876171526f00671b7fbe0ca27841b1ab4c32b196afbc911",
        "9cc6e96d5909542189ca43ea2ff54efda6a44b302483890629b82d2ecad7f7ba",
    ),
    925: (
        "da14e575bd6a7b7e05619bce4ffe3a5b59be9a8eb1b845e8030f7e2568778be4",
        "ed60b427e6b49d831a357af707cc7191b641427ea5b01b43076150b24ec41d72",
        "a4a6457463d2238cf0bd8ba1ca987f9f6202e6cd547b5e1d87151ada794492bd",
    ),
}


class Source8HiddenMaterializationError(RuntimeError):
    """An immutable input, frozen forward, artifact, or receipt failed closed."""


def source8_latent_geometry(value: Any) -> tuple[tuple[int, ...], int, int, int]:
    """Resolve only the five geometries sealed by the registered source8 inputs."""

    return starc.latent_geometry(
        value,
        allowed_latent_shapes=SOURCE8_LATENT_SHAPES,
        allowed_patch_grids=SOURCE8_PATCH_GRIDS,
        geometry_label="registered source8",
    )


def validate_group_spatial_bindings(
    value: Any, *, source_order: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Authenticate the canonical-JSON map without relying on map key order."""

    if (
        not isinstance(value, dict)
        or len(source_order) != SOURCES_PER_GROUP
        or len(set(source_order)) != SOURCES_PER_GROUP
        or set(value) != set(source_order)
    ):
        raise Source8HiddenMaterializationError(
            "group spatial binding topology differs"
        )
    for iid in source_order:
        binding = value.get(iid)
        if not isinstance(binding, dict):
            raise Source8HiddenMaterializationError(
                "group spatial binding value differs"
            )
        grid = binding.get("patch_grid_height_width")
        if not isinstance(grid, list) or len(grid) != 2:
            raise Source8HiddenMaterializationError(
                "group spatial binding grid differs"
            )
        patch_grid = (grid[0], grid[1])
        if patch_grid not in SOURCE8_PATCH_GRIDS:
            raise Source8HiddenMaterializationError(
                "group spatial binding grid is unregistered"
            )
        positions = patch_grid[0] * patch_grid[1]
        expected = SOURCE8_SKETCH_DIGESTS_BY_PATCH_POSITIONS.get(positions)
        observed = (
            binding.get("matrix_raw_bytes_sha256"),
            binding.get("matrix_value_sha256"),
            binding.get("critic_tensor_sha256"),
        )
        if (
            binding.get("matrix_shape") != [starc.SKETCH_COORDINATES, positions]
            or binding.get("patch_positions") != positions
            or observed != expected
            or binding.get("data_dependent") is not False
            or binding.get("full_support_no_mask_or_localizer") is not True
        ):
            raise Source8HiddenMaterializationError(
                "group spatial binding authentication differs"
            )
    return value


def _fail(message: str, error: Exception | None = None) -> Source8HiddenMaterializationError:
    result = Source8HiddenMaterializationError(message)
    if error is not None:
        result.__cause__ = error
    return result


def _plain_file(value: str | Path, *, label: str, expected: str | None = None) -> Path:
    path = Path(value)
    if expected is not None and str(path) != expected:
        raise Source8HiddenMaterializationError(f"{label} lexical path differs")
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Source8HiddenMaterializationError(f"{label} must be an absolute plain file")
    return path


def _plain_dir(value: str | Path, *, label: str, expected: str | None = None) -> Path:
    path = Path(value)
    if expected is not None and str(path) != expected:
        raise Source8HiddenMaterializationError(f"{label} lexical path differs")
    if (
        not path.is_absolute()
        or not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Source8HiddenMaterializationError(
            f"{label} must be an absolute plain directory"
        )
    return path


def _read_json(path: Path, *, expected_sha256: str | None, label: str) -> dict[str, Any]:
    before = path.stat()
    raw = path.read_bytes()
    observed = starc.file_sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise Source8HiddenMaterializationError(f"{label} SHA-256 differs")

    def reject_constant(token: str) -> Any:
        raise Source8HiddenMaterializationError(f"{label} contains {token}")

    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Source8HiddenMaterializationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _fail(f"{label} is invalid JSON", error)
    after = path.stat()
    if (
        type(value) is not dict
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or starc.file_sha256(path) != observed
    ):
        raise Source8HiddenMaterializationError(f"{label} changed while reading")
    return value


def _candidate_id(iid: str, branch: str, seed: int) -> str:
    return f"saic-{iid}-{branch}-s{seed}"


def _source_branch_instruction(source: Mapping[str, Any], branch: str) -> str:
    field = {
        "forward": "forward_instruction",
        "noop": "noop_instruction",
        "reverse": "inverse_instruction",
    }.get(branch)
    if field is None:
        raise Source8HiddenMaterializationError("candidate branch differs")
    value = source.get(field)
    if not isinstance(value, str):
        raise Source8HiddenMaterializationError("source instruction differs")
    return value


def load_registered_population(
    *, source_manifest: str | Path, attempts_root: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate all 60 receipts, then return the fixed 24-row population."""

    manifest_path = _plain_file(
        source_manifest, label="source manifest", expected=SOURCE_MANIFEST_PATH
    )
    manifest = _read_json(
        manifest_path,
        expected_sha256=SOURCE_MANIFEST_SHA256,
        label="source manifest",
    )
    try:
        summary = dict(source_contract.validate_manifest(manifest, verify_bound_files=True))
    except Exception as error:
        raise _fail("source manifest validation failed", error)
    if (
        summary.get("manifest_content_sha256") != SOURCE_MANIFEST_CONTENT_SHA256
        or summary.get("row_count") != SOURCE_COUNT
        or summary.get("bound_files_verified") is not True
        or starc.object_sha256(summary) != SOURCE_VALIDATOR_SUMMARY_SHA256
    ):
        raise Source8HiddenMaterializationError("source manifest closure differs")

    sources = manifest.get("rows")
    if not isinstance(sources, list) or len(sources) != SOURCE_COUNT:
        raise Source8HiddenMaterializationError("source manifest is not exact8")
    source_by_iid = {row.get("iid"): row for row in sources if isinstance(row, Mapping)}
    if len(source_by_iid) != SOURCE_COUNT or None in source_by_iid:
        raise Source8HiddenMaterializationError("source IID closure differs")
    real_paths = {str(row["source_video"]) for row in sources}
    real_hashes = {str(row["source_video_sha256"]) for row in sources}

    attempt_root = _plain_dir(
        attempts_root, label="attempts root", expected=ATTEMPTS_ROOT
    )
    receipt_paths = sorted(attempt_root.glob(f"*/{generation_contract.ATTEMPT_RECEIPT_BASENAME}"))
    if len(receipt_paths) != CANDIDATE_COUNT:
        raise Source8HiddenMaterializationError("sealed candidate receipt count differs")
    validated: dict[str, dict[str, Any]] = {}
    for path in receipt_paths:
        path = _plain_file(path, label="generation receipt")
        raw = generation_contract._load_json(path, label="generation receipt")
        candidate = raw.get("candidate")
        if not isinstance(candidate, Mapping):
            raise Source8HiddenMaterializationError("generation candidate differs")
        group = {
            "group_id": raw.get("group_id"),
            "actor_family": raw.get("actor_family"),
            "visible_gpus": raw.get("visible_gpus"),
        }
        try:
            checked = generation_contract._load_attempt_receipt(
                path,
                candidate=candidate,
                group=group,
                root_spec_sha256=ROOT_SPEC_SHA256,
                real_source_paths=real_paths,
                real_source_hashes=real_hashes,
            )
        except Exception as error:
            raise _fail(f"generation receipt replay failed: {path}", error)
        candidate_id = candidate.get("candidate_id")
        iid = candidate.get("iid")
        branch = candidate.get("branch")
        seed = candidate.get("seed")
        source = source_by_iid.get(iid)
        if (
            not isinstance(candidate_id, str)
            or source is None
            or branch not in BRANCH_ORDER
            or type(seed) is not int
            or candidate_id != _candidate_id(str(iid), str(branch), seed)
            or candidate.get("row_id") != source.get("row_id")
            or candidate.get("analysis_split") != source.get("analysis_split")
            or candidate.get("actor_family") != source.get("actor_family")
            or candidate.get("action_family_id") != source.get("action_family_id")
            or candidate.get("source_media_sha256_for_nonuse_audit")
            != source.get("source_video_sha256")
            or seed not in source.get("rollout_seeds", ())
            or candidate.get("branch_instruction")
            != _source_branch_instruction(source, str(branch))
            or checked.get("root_spec_raw_sha256") != ROOT_SPEC_SHA256
            or checked.get("event_verified") is not False
            or checked.get("training_target_authorized") is not False
            or checked.get("optimizer_or_parameter_update_authorized") is not False
        ):
            raise Source8HiddenMaterializationError(
                f"candidate/source join differs: {candidate_id}"
            )
        if candidate_id in validated:
            raise Source8HiddenMaterializationError("candidate ID repeats")
        validated[candidate_id] = {
            "source": dict(source),
            "candidate": dict(candidate),
            "generation_receipt": dict(checked),
            "generation_receipt_path": str(path),
            "generation_receipt_file_sha256": starc.file_sha256(path),
        }

    expected_all = {
        _candidate_id(str(row["iid"]), branch, seed)
        for row in sources
        for seed in row["rollout_seeds"]
        for branch in BRANCH_ORDER
    }
    if set(validated) != expected_all or len(expected_all) != CANDIDATE_COUNT:
        raise Source8HiddenMaterializationError("registered 60-candidate closure differs")

    selected: list[dict[str, Any]] = []
    for source in sources:
        seed = min(source["rollout_seeds"])
        for branch in BRANCH_ORDER:
            selected.append(validated[_candidate_id(source["iid"], branch, seed)])
    if (
        len(selected) != SOURCE_COUNT * len(BRANCH_ORDER)
        or len({row["candidate"]["candidate_id"] for row in selected}) != len(selected)
    ):
        raise Source8HiddenMaterializationError("deterministic source8 selection differs")
    return manifest, selected


def selected_group(
    rows: Sequence[Mapping[str, Any]], *, actor_family: str
) -> list[dict[str, Any]]:
    if actor_family not in ACTOR_FAMILIES:
        raise Source8HiddenMaterializationError("actor family differs")
    group = [dict(row) for row in rows if row["source"]["actor_family"] == actor_family]
    if (
        len(group) != ARMS_PER_GROUP
        or [row["candidate"]["branch"] for row in group]
        != list(BRANCH_ORDER) * SOURCES_PER_GROUP
        or sorted({row["source"]["analysis_split"] for row in group})
        != ["confirmation", "fit"]
        or len({row["source"]["iid"] for row in group}) != SOURCES_PER_GROUP
        or sum(row["source"]["analysis_split"] == "fit" for row in group)
        != 2 * len(BRANCH_ORDER)
    ):
        raise Source8HiddenMaterializationError("family group topology differs")
    return group


def _artifact_from_receipt(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    artifacts = row["generation_receipt"].get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Source8HiddenMaterializationError("generation artifacts differ")
    artifact = artifacts.get(key)
    if not isinstance(artifact, Mapping):
        raise Source8HiddenMaterializationError(f"generation artifact {key} differs")
    return artifact


def _runtime_binding(args: argparse.Namespace) -> dict[str, Any]:
    import diffusers
    import torch
    import transformers

    return {
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "materializer_source_sha256": args.expected_materializer_source_sha256,
        "starc_materializer_source_sha256": args.expected_starc_source_sha256,
        "source_contract_source_sha256": args.expected_source_contract_sha256,
        "generation_contract_source_sha256": args.expected_generation_contract_sha256,
        "torch": str(torch.__version__),
        "torch_hip": str(torch.version.hip),
        "transformers": str(transformers.__version__),
        "diffusers": str(diffusers.__version__),
        "ulysses_world": starc.EXPECTED_SP_WORLD,
        "materialization_only": True,
        "optimizer_constructed": False,
        "editor_forward_performed": False,
    }


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_materializer_source_sha256",
        "expected_starc_source_sha256",
        "expected_source_contract_sha256",
        "expected_generation_contract_sha256",
        "method_source_archive_sha256",
    ):
        starc._sha256(getattr(args, name), label=name)
    for name in ("method_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        starc._sha1(getattr(args, name), label=name)
    if (
        args.expected_bernini_commit != starc.temporal_contract.REQUIRED_BERNINI_REVISION
        or args.expected_veomni_commit != starc.temporal_contract.REQUIRED_VEOMNI_REVISION
    ):
        raise Source8HiddenMaterializationError("Bernini source revisions differ")
    expected_sources = {
        Path(__file__).resolve(): args.expected_materializer_source_sha256,
        METHOD_ROOT / "materialize_starc_core4_hidden_v1.py": args.expected_starc_source_sha256,
        METHOD_ROOT / "build_saic_reversible_source_set_v1.py": args.expected_source_contract_sha256,
        METHOD_ROOT / "generate_saic_pure_t2v_event_bank_v1.py": args.expected_generation_contract_sha256,
    }
    for path, expected in expected_sources.items():
        if starc.file_sha256(_plain_file(path, label="runtime source")) != expected:
            raise Source8HiddenMaterializationError(f"runtime source hash differs: {path.name}")
    if not all(
        getattr(args, name)
        for name in (
            "ack_hidden_diagnostic_only",
            "ack_no_generated_media_editor_use",
            "ack_no_optimizer_or_editor_update",
        )
    ):
        raise Source8HiddenMaterializationError("mandatory authority acknowledgement missing")
    output = _plain_dir(args.output_root, label="output root")
    if (output / args.actor_family).exists():
        raise Source8HiddenMaterializationError("group output already exists")


def materialize_group(args: argparse.Namespace) -> int:
    _validate_cli(args)
    frozen = starc.temporal_scorer._frozen_d541801_runtime()
    starc.temporal_scorer.validate_native_coordinate_runtime(frozen)
    coordinate = next(
        (row for row in starc.temporal_contract.NATIVE_SIGMA_COORDINATES if row[0] == starc.SCHEDULE_INDEX),
        None,
    )
    if (
        coordinate is None
        or float(coordinate[1]).hex() != float(starc.SIGMA).hex()
        or coordinate[2] != starc.NATIVE_TIMESTEP
    ):
        raise Source8HiddenMaterializationError("registered schedule coordinate differs")

    native_generation = frozen.native_generation
    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise _fail("Bernini runtime validation failed", error)
    if transformer_config.get("num_attention_heads") != 12:
        raise Source8HiddenMaterializationError("Bernini head count differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != starc.EXPECTED_SP_WORLD
        or not torch.cuda.is_available()
        or torch.version.hip is None
    ):
        raise Source8HiddenMaterializationError("materializer requires AUH ROCm SP4")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=starc.EXPECTED_SP_WORLD)
    device = torch.device("cuda", distributed.local_rank)
    observer: Optional[starc.Block15SpatialSketchObserver] = None
    try:
        source_manifest, selected = load_registered_population(
            source_manifest=args.source_manifest,
            attempts_root=args.attempts_root,
        )
        group_rows = selected_group(selected, actor_family=args.actor_family)

        checkpoint_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                identity = native_generation.source_audit.validate_checkpoint_content(
                    checkpoint, Path(args.checkpoint_content_manifest)
                )
                checkpoint_rows[0] = {"ok": True, "identity": identity}
            except Exception as error:
                checkpoint_rows[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
            raise Source8HiddenMaterializationError(
                f"checkpoint audit failed: {checkpoint_result}"
            )
        checkpoint_identity = dict(checkpoint_result["identity"])

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
        freeze_before = native_generation.source_audit.model_freeze_certificate(renderer)
        checkpoint_binding = frozen.checkpoint_content_binding(
            checkpoint_identity, freeze_before
        )
        diffusion = renderer.diff_dec
        transformer = diffusion.transformer
        if (
            transformer is None
            or diffusion.transformer_2 is not None
            or any(parameter.requires_grad for parameter in renderer.parameters())
        ):
            raise Source8HiddenMaterializationError("frozen transformer closure differs")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        builder_contract = frozen.prompt_builder_contract()
        runtime_binding = _runtime_binding(args)
        model_binding = {
            "checkpoint_content_binding": checkpoint_binding,
            "checkpoint_receipt_digest": frozen.object_sha256(checkpoint_identity),
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "native_schedule_digest": starc.temporal_contract.NATIVE_SCHEDULE_DIGEST,
            "native_schedule_index": starc.SCHEDULE_INDEX,
            "native_timestep": starc.NATIVE_TIMESTEP,
            "sigma": starc.SIGMA,
            "hook_coordinate": starc.HOOK_COORDINATE,
            "transformer_1_only": True,
            "adapter_loaded": False,
            "all_parameters_frozen": True,
        }
        source_manifest_binding = {
            "path": SOURCE_MANIFEST_PATH,
            "file_sha256": SOURCE_MANIFEST_SHA256,
            "content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
            "validator_summary_sha256": SOURCE_VALIDATOR_SUMMARY_SHA256,
            "source_count": SOURCE_COUNT,
            "all_bound_files_verified": True,
        }
        group_root = Path(args.output_root) / args.actor_family
        starc._rank0_action(
            dist=dist,
            rank=distributed.rank,
            label="create source8 group output",
            action=lambda: (group_root.mkdir(), str(group_root.resolve(strict=True)))[1],
        )

        arm_bindings: list[dict[str, Any]] = []
        source_order: list[str] = []
        split_by_iid: dict[str, str] = {}
        spatial_bindings_by_iid: dict[str, dict[str, Any]] = {}
        for source_offset in range(0, len(group_rows), len(BRANCH_ORDER)):
            source_rows = group_rows[source_offset : source_offset + len(BRANCH_ORDER)]
            source = source_rows[0]["source"]
            iid = source["iid"]
            if [row["candidate"]["branch"] for row in source_rows] != list(BRANCH_ORDER):
                raise Source8HiddenMaterializationError("source branch order differs")
            if any(row["source"] != source for row in source_rows):
                raise Source8HiddenMaterializationError("source metadata differs across branches")
            source_order.append(iid)
            split_by_iid[iid] = source["analysis_split"]

            cached: dict[str, dict[str, Any]] = {}
            first_gaussian: Any = None
            first_gaussian_sha: Optional[str] = None
            geometry: Optional[tuple[tuple[int, ...], int, int, int]] = None
            for row in source_rows:
                candidate_id = row["candidate"]["candidate_id"]
                clean_artifact = _artifact_from_receipt(row, "predecode_clean_latent")
                gaussian_artifact = _artifact_from_receipt(row, "official_initial_gaussian")
                clean = frozen._load_exact81_tensor(
                    clean_artifact,
                    key="normalized_clean_latent",
                    label=f"{candidate_id} clean latent",
                )
                gaussian = frozen._load_exact81_tensor(
                    gaussian_artifact,
                    key="official_initial_gaussian",
                    label=f"{candidate_id} official Gaussian",
                )
                clean_geometry = source8_latent_geometry(clean)
                gaussian_geometry = source8_latent_geometry(gaussian)
                if clean_geometry != gaussian_geometry:
                    raise Source8HiddenMaterializationError("clean/Gaussian geometry differs")
                if geometry is None:
                    geometry = clean_geometry
                elif geometry != clean_geometry:
                    raise Source8HiddenMaterializationError("source branch geometries differ")
                clean_auth = starc.verify_authenticated_native_clean_tensor_identity(
                    clean,
                    clean_artifact,
                    label=f"{candidate_id} clean latent",
                    frozen=frozen,
                    allowed_latent_shapes=SOURCE8_LATENT_SHAPES,
                    allowed_patch_grids=SOURCE8_PATCH_GRIDS,
                    geometry_label="registered source8",
                )
                frozen.verify_native_tensor_value_identity(
                    gaussian,
                    gaussian_artifact,
                    label=f"{candidate_id} official Gaussian",
                )
                gaussian_sha = starc.tensor_sha256(gaussian)
                if first_gaussian is None:
                    first_gaussian = gaussian
                    first_gaussian_sha = gaussian_sha
                elif gaussian_sha != first_gaussian_sha or not torch.equal(
                    gaussian, first_gaussian
                ):
                    raise Source8HiddenMaterializationError(
                        "same-source/seed branch Gaussians differ"
                    )
                cached[row["candidate"]["branch"]] = {
                    "row": row,
                    "clean": clean,
                    "clean_tensor_sha256": starc.tensor_sha256(clean),
                    "clean_authentication": clean_auth,
                    "gaussian": gaussian,
                    "gaussian_tensor_sha256": gaussian_sha,
                }
            assert geometry is not None and first_gaussian is not None and first_gaussian_sha
            source_shape, patch_height, patch_width, _patch_positions = geometry
            spatial = starc.fixed_spatial_sketch(
                patch_height=patch_height,
                patch_width=patch_width,
                device=device,
                allowed_patch_grids=SOURCE8_PATCH_GRIDS,
                geometry_label="registered source8",
            )
            current_spatial_binding = starc.sketch_binding(
                spatial,
                patch_height=patch_height,
                patch_width=patch_width,
                allowed_patch_grids=SOURCE8_PATCH_GRIDS,
                expected_digests_by_patch_positions=(
                    SOURCE8_SKETCH_DIGESTS_BY_PATCH_POSITIONS
                ),
                geometry_label="registered source8",
            )
            if iid in spatial_bindings_by_iid:
                raise Source8HiddenMaterializationError("source8 spatial binding repeats")
            spatial_bindings_by_iid[iid] = current_spatial_binding
            observer = starc.Block15SpatialSketchObserver(
                transformer,
                sp_rank=distributed.rank,
                patch_height=patch_height,
                patch_width=patch_width,
                spatial_sketch=spatial,
                allowed_latent_shapes=SOURCE8_LATENT_SHAPES,
                allowed_patch_grids=SOURCE8_PATCH_GRIDS,
                geometry_label="registered source8",
            ).install()

            forward_candidate = cached["forward"]["row"]["candidate"]
            noop_candidate = cached["noop"]["row"]["candidate"]
            conditions, condition_hashes, prompt_text = starc._encode_prompt_pair(
                renderer,
                tokenizer,
                action_caption=forward_candidate["full_t2v_caption"],
                noop_caption=noop_candidate["full_t2v_caption"],
                device=device,
                frozen=frozen,
            )
            prompt_binding = starc.temporal_scorer._prompt_binding(
                target_action_caption_sha256=forward_candidate[
                    "full_t2v_caption_utf8_sha256"
                ],
                target_noop_caption_sha256=noop_candidate[
                    "full_t2v_caption_utf8_sha256"
                ],
                action_prompt=prompt_text["action_prompt"],
                noop_prompt=prompt_text["noop_prompt"],
                condition_hashes=condition_hashes,
                prompt_builder_contract_digest=builder_contract["contract_digest"],
            )
            prompt_binding.update(
                {
                    "target_action_candidate_id": forward_candidate["candidate_id"],
                    "target_noop_candidate_id": noop_candidate["candidate_id"],
                    "all_three_states_use_source_fixed_prompt_pair": True,
                    "owner_branch_label_never_enters_model_condition": True,
                }
            )

            epsilon = first_gaussian.to(device=device).float().contiguous().detach()
            try:
                for branch in BRANCH_ORDER:
                    owner = cached[branch]
                    row = owner["row"]
                    candidate = row["candidate"]
                    clean_device = owner["clean"].to(device=device).contiguous().detach()
                    sigma = torch.tensor(starc.SIGMA, dtype=torch.float32, device=device)
                    x_sigma = (
                        clean_device
                        + sigma.reshape(1, 1, 1, 1, 1) * (epsilon - clean_device)
                    ).float().contiguous().detach()
                    residual, same_state, hidden = starc.forward_same_state_hidden_pair(
                        diffusion=diffusion,
                        transformer=transformer,
                        observer=observer,
                        x_sigma=x_sigma,
                        action_condition=conditions["target_action"],
                        noop_condition=conditions["noop"],
                        arm_key=f"{iid}:{branch}",
                        dist_module=dist,
                    )
                    arm_dir = group_root / iid / branch

                    def write_artifact() -> dict[str, Any]:
                        arm_dir.parent.mkdir(exist_ok=True)
                        arm_dir.mkdir()
                        return starc.save_residual_artifact(
                            arm_dir / TENSOR_FILENAME, residual
                        )

                    artifact = starc._rank0_action(
                        dist=dist,
                        rank=distributed.rank,
                        label=f"write {iid}/{branch} hidden artifact",
                        action=write_artifact,
                    )
                    if artifact["tensor_sha256"] != hidden["residual_tensor_sha256"]:
                        raise Source8HiddenMaterializationError("artifact tensor digest differs")
                    clean_artifact = _artifact_from_receipt(row, "predecode_clean_latent")
                    gaussian_artifact = _artifact_from_receipt(
                        row, "official_initial_gaussian"
                    )
                    generation = row["generation_receipt"]
                    unsigned = {
                        "schema_version": SCHEMA_VERSION,
                        "actor_family": args.actor_family,
                        "iid": iid,
                        "row_id": source["row_id"],
                        "analysis_split": source["analysis_split"],
                        "action_family_id": source["action_family_id"],
                        "actor_group_id": source["actor_group_id"],
                        "scene_group_id": source["scene_group_id"],
                        "branch": branch,
                        "seed": candidate["seed"],
                        "minimum_registered_seed": True,
                        "candidate_binding": {
                            "candidate_id": candidate["candidate_id"],
                            "candidate_envelope_path": generation[
                                "candidate_envelope_path"
                            ],
                            "candidate_envelope_sha256": generation[
                                "candidate_envelope_sha256"
                            ],
                            "generation_receipt_path": row[
                                "generation_receipt_path"
                            ],
                            "generation_receipt_file_sha256": row[
                                "generation_receipt_file_sha256"
                            ],
                            "generation_receipt_digest": generation[
                                "receipt_digest"
                            ],
                            "native_receipt_path": generation["native_receipt_path"],
                            "native_receipt_sha256": generation[
                                "native_receipt_sha256"
                            ],
                            "native_receipt_digest": generation[
                                "native_receipt_digest"
                            ],
                            "all_60_generation_receipts_authenticated_before_selection": True,
                        },
                        "source_manifest_binding": source_manifest_binding,
                        "clean_latent_binding": {
                            **starc._artifact_native_binding(
                                clean_artifact,
                                tensor_digest=owner["clean_tensor_sha256"],
                                authenticated_identity=owner["clean_authentication"],
                            ),
                            "clean_latent_authentication": dict(
                                owner["clean_authentication"]
                            ),
                            "generated_latent_used_only_as_frozen_hidden_query": True,
                        },
                        "official_gaussian_binding": {
                            **starc._artifact_native_binding(
                                gaussian_artifact,
                                tensor_digest=owner["gaussian_tensor_sha256"],
                            ),
                            "same_source_seed_shared_across_branches": True,
                            "same_source_seed_tensor_sha256": first_gaussian_sha,
                        },
                        "prompt_binding": prompt_binding,
                        "same_state_query_binding": same_state,
                        "hidden_binding": hidden,
                        "spatial_sketch_binding": current_spatial_binding,
                        "artifact": artifact,
                        "model_binding": model_binding,
                        "runtime_binding": runtime_binding,
                        "candidate_rgb_opened": False,
                        "generated_media_editor_use_authorized": False,
                        "representation_selection_authorized": False,
                        "training_performed": False,
                        "optimizer_authorized": False,
                        "editor_optimizer_authorized": False,
                        "scientific_claim_authorized": False,
                    }
                    receipt = starc._seal(unsigned)
                    digest_rows: list[Any] = [None] * starc.EXPECTED_SP_WORLD
                    dist.all_gather_object(digest_rows, receipt["receipt_digest"])
                    if len(set(digest_rows)) != 1:
                        raise Source8HiddenMaterializationError(
                            "SP4 arm receipt digests differ"
                        )

                    def write_receipt() -> dict[str, Any]:
                        receipt_path = arm_dir / ARM_RECEIPT_FILENAME
                        receipt_sha = starc._write_json_create_only(receipt_path, receipt)
                        return {
                            "iid": iid,
                            "analysis_split": source["analysis_split"],
                            "branch": branch,
                            "candidate_id": candidate["candidate_id"],
                            "receipt_path": str(receipt_path.resolve(strict=True)),
                            "receipt_file_sha256": receipt_sha,
                            "receipt_digest": receipt["receipt_digest"],
                            "artifact_path": artifact["path"],
                            "artifact_file_sha256": artifact["file_sha256"],
                            "artifact_tensor_sha256": artifact["tensor_sha256"],
                        }

                    arm_bindings.append(
                        starc._rank0_action(
                            dist=dist,
                            rank=distributed.rank,
                            label=f"write {iid}/{branch} arm receipt",
                            action=write_receipt,
                        )
                    )
                    del clean_device, sigma, x_sigma, residual
            finally:
                observer.remove()
                observer = None
            del conditions, epsilon, spatial
            cached.clear()

        freeze_after = native_generation.source_audit.model_freeze_certificate(renderer)
        if freeze_after != freeze_before or any(
            parameter.requires_grad for parameter in renderer.parameters()
        ):
            raise Source8HiddenMaterializationError("frozen renderer changed")
        if list(spatial_bindings_by_iid) != source_order:
            raise Source8HiddenMaterializationError("source8 spatial bindings differ")
        unsigned_group = {
            "schema_version": GROUP_SCHEMA_VERSION,
            "actor_family": args.actor_family,
            "source_manifest_binding": source_manifest_binding,
            "attempts_root": ATTEMPTS_ROOT,
            "root_spec_sha256": ROOT_SPEC_SHA256,
            "selection_rule": "minimum_registered_seed_per_source",
            "branch_order": list(BRANCH_ORDER),
            "source_order": source_order,
            "split_by_iid": split_by_iid,
            "source_count": SOURCES_PER_GROUP,
            "arm_count": ARMS_PER_GROUP,
            "model_forward_count": MODEL_FORWARDS_PER_GROUP,
            "arm_bindings": arm_bindings,
            "spatial_sketch_bindings_by_iid": spatial_bindings_by_iid,
            "model_binding": model_binding,
            "runtime_binding": runtime_binding,
            "all_60_generation_receipts_authenticated_before_selection": True,
            "candidate_rgb_opened": False,
            "confirmation_used_for_structure_or_parameter_selection": False,
            "training_performed": False,
            "optimizer_authorized": False,
            "editor_optimizer_authorized": False,
            "scientific_claim_authorized": False,
        }
        group_receipt = starc._seal(unsigned_group)
        group_digest_rows: list[Any] = [None] * starc.EXPECTED_SP_WORLD
        dist.all_gather_object(group_digest_rows, group_receipt["receipt_digest"])
        if len(set(group_digest_rows)) != 1:
            raise Source8HiddenMaterializationError("SP4 group receipt digests differ")

        def write_group() -> dict[str, Any]:
            path = group_root / GROUP_RECEIPT_FILENAME.format(
                actor_family=args.actor_family
            )
            digest = starc._write_json_create_only(path, group_receipt)
            for iid in source_order:
                for branch in BRANCH_ORDER:
                    os.chmod(group_root / iid / branch, 0o500)
                os.chmod(group_root / iid, 0o500)
            os.chmod(group_root, 0o500)
            return {"path": str(path.resolve(strict=True)), "file_sha256": digest}

        starc._rank0_action(
            dist=dist,
            rank=distributed.rank,
            label="write family group receipt",
            action=write_group,
        )
        dist.barrier()
        return 0
    finally:
        if observer is not None and observer.active:
            observer.abort()
        if observer is not None and observer.installed:
            observer.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


def _verify_group(path: Path, expected_sha256: str, actor_family: str) -> dict[str, Any]:
    group = _read_json(
        _plain_file(path, label=f"{actor_family} group receipt"),
        expected_sha256=expected_sha256,
        label=f"{actor_family} group receipt",
    )
    starc._verify_seal(group, schema=GROUP_SCHEMA_VERSION, label="source8 group")
    if (
        group.get("actor_family") != actor_family
        or group.get("source_count") != SOURCES_PER_GROUP
        or group.get("arm_count") != ARMS_PER_GROUP
        or group.get("model_forward_count") != MODEL_FORWARDS_PER_GROUP
        or group.get("branch_order") != list(BRANCH_ORDER)
        or group.get("selection_rule") != "minimum_registered_seed_per_source"
        or group.get("all_60_generation_receipts_authenticated_before_selection") is not True
        or group.get("candidate_rgb_opened") is not False
        or group.get("confirmation_used_for_structure_or_parameter_selection") is not False
        or group.get("training_performed") is not False
        or group.get("optimizer_authorized") is not False
        or group.get("editor_optimizer_authorized") is not False
        or group.get("scientific_claim_authorized") is not False
    ):
        raise Source8HiddenMaterializationError("group topology/authority differs")
    bindings = group.get("arm_bindings")
    if not isinstance(bindings, list) or len(bindings) != ARMS_PER_GROUP:
        raise Source8HiddenMaterializationError("group arm binding count differs")
    spatial_bindings = validate_group_spatial_bindings(
        group.get("spatial_sketch_bindings_by_iid"),
        source_order=group.get("source_order", ()),
    )
    expected_pairs = [
        (iid, branch) for iid in group["source_order"] for branch in BRANCH_ORDER
    ]
    if [(row.get("iid"), row.get("branch")) for row in bindings] != expected_pairs:
        raise Source8HiddenMaterializationError("group arm order differs")
    for binding in bindings:
        receipt_path = _plain_file(binding["receipt_path"], label="arm receipt")
        receipt = _read_json(
            receipt_path,
            expected_sha256=binding["receipt_file_sha256"],
            label="arm receipt",
        )
        starc._verify_seal(receipt, schema=SCHEMA_VERSION, label="source8 arm")
        if (
            receipt.get("receipt_digest") != binding.get("receipt_digest")
            or receipt.get("iid") != binding.get("iid")
            or receipt.get("analysis_split") != binding.get("analysis_split")
            or receipt.get("branch") != binding.get("branch")
            or receipt.get("candidate_binding", {}).get("candidate_id")
            != binding.get("candidate_id")
            or receipt.get("artifact", {}).get("path") != binding.get("artifact_path")
            or receipt.get("artifact", {}).get("file_sha256")
            != binding.get("artifact_file_sha256")
            or receipt.get("artifact", {}).get("tensor_sha256")
            != binding.get("artifact_tensor_sha256")
            or receipt.get("spatial_sketch_binding")
            != spatial_bindings.get(binding.get("iid"))
        ):
            raise Source8HiddenMaterializationError("group-to-arm binding differs")
        starc._validate_artifact(receipt["artifact"], verify_file=True)
    return group


def aggregate_master(args: argparse.Namespace) -> int:
    output_root = _plain_dir(args.output_root, label="output root")
    inputs = {
        "dog": (Path(args.dog_group_receipt), args.expected_dog_group_sha256),
        "human": (Path(args.human_group_receipt), args.expected_human_group_sha256),
    }
    groups = {
        family: _verify_group(path, expected, family)
        for family, (path, expected) in inputs.items()
    }
    if groups["dog"]["source_manifest_binding"] != groups["human"][
        "source_manifest_binding"
    ]:
        raise Source8HiddenMaterializationError("groups disagree on source population")
    source_order = [
        iid for family in ACTOR_FAMILIES for iid in groups[family]["source_order"]
    ]
    if len(source_order) != SOURCE_COUNT or len(set(source_order)) != SOURCE_COUNT:
        raise Source8HiddenMaterializationError("master source closure differs")
    group_bindings = []
    for family in ACTOR_FAMILIES:
        path, expected = inputs[family]
        group = groups[family]
        group_bindings.append(
            {
                "actor_family": family,
                "path": str(path.resolve(strict=True)),
                "file_sha256": expected,
                "receipt_digest": group["receipt_digest"],
                "source_order": group["source_order"],
                "split_by_iid": group["split_by_iid"],
                "arm_count": group["arm_count"],
                "model_forward_count": group["model_forward_count"],
            }
        )
    unsigned = {
        "schema_version": MASTER_SCHEMA_VERSION,
        "source_manifest_binding": groups["dog"]["source_manifest_binding"],
        "attempts_root": ATTEMPTS_ROOT,
        "root_spec_sha256": ROOT_SPEC_SHA256,
        "selection_rule": "minimum_registered_seed_per_source",
        "actor_family_order": list(ACTOR_FAMILIES),
        "branch_order": list(BRANCH_ORDER),
        "group_bindings": group_bindings,
        "source_order": source_order,
        "source_count": SOURCE_COUNT,
        "arm_count": SOURCE_COUNT * len(BRANCH_ORDER),
        "model_forward_count": SOURCE_COUNT
        * len(BRANCH_ORDER)
        * MODEL_FORWARDS_PER_ARM,
        "fit_source_count": 4,
        "confirmation_source_count": 4,
        "single_preregistered_coordinate": {
            "hook": starc.HOOK_COORDINATE,
            "schedule_index": starc.SCHEDULE_INDEX,
            "sigma": starc.SIGMA,
            "native_timestep": starc.NATIVE_TIMESTEP,
        },
        "all_60_generation_receipts_authenticated_before_selection": True,
        "candidate_rgb_opened": False,
        "confirmation_used_for_structure_or_parameter_selection": False,
        "training_performed": False,
        "optimizer_authorized": False,
        "editor_optimizer_authorized": False,
        "representation_selection_authorized": False,
        "scientific_claim_authorized": False,
    }
    master = starc._seal(unsigned)
    output = output_root / MASTER_RECEIPT_FILENAME
    starc._write_json_create_only(output, master)
    print(
        json.dumps(
            {
                "output": str(output),
                "receipt_digest": master["receipt_digest"],
                "arms": master["arm_count"],
                "optimizer": master["optimizer_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize-group")
    materialize.add_argument("--actor-family", choices=ACTOR_FAMILIES, required=True)
    materialize.add_argument("--source-manifest", required=True)
    materialize.add_argument("--attempts-root", required=True)
    materialize.add_argument("--bernini-root", required=True)
    materialize.add_argument("--veomni-root", required=True)
    materialize.add_argument("--checkpoint", required=True)
    materialize.add_argument("--checkpoint-content-manifest", required=True)
    materialize.add_argument("--output-root", required=True)
    materialize.add_argument("--expected-bernini-commit", required=True)
    materialize.add_argument("--expected-veomni-commit", required=True)
    materialize.add_argument("--method-source-revision", required=True)
    materialize.add_argument("--method-source-archive-sha256", required=True)
    materialize.add_argument("--expected-materializer-source-sha256", required=True)
    materialize.add_argument("--expected-starc-source-sha256", required=True)
    materialize.add_argument("--expected-source-contract-sha256", required=True)
    materialize.add_argument("--expected-generation-contract-sha256", required=True)
    materialize.add_argument("--ack-hidden-diagnostic-only", action="store_true")
    materialize.add_argument("--ack-no-generated-media-editor-use", action="store_true")
    materialize.add_argument("--ack-no-optimizer-or-editor-update", action="store_true")

    aggregate = commands.add_parser("aggregate-master")
    aggregate.add_argument("--output-root", required=True)
    aggregate.add_argument("--dog-group-receipt", required=True)
    aggregate.add_argument("--expected-dog-group-sha256", required=True)
    aggregate.add_argument("--human-group-receipt", required=True)
    aggregate.add_argument("--expected-human-group-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize-group":
        return materialize_group(args)
    if args.command == "aggregate-master":
        return aggregate_master(args)
    raise Source8HiddenMaterializationError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTOR_FAMILIES",
    "ATTEMPTS_ROOT",
    "BRANCH_ORDER",
    "GROUP_SCHEMA_VERSION",
    "MASTER_SCHEMA_VERSION",
    "ROOT_SPEC_SHA256",
    "SCHEMA_VERSION",
    "SOURCE_MANIFEST_PATH",
    "SOURCE_MANIFEST_SHA256",
    "Source8HiddenMaterializationError",
    "aggregate_master",
    "load_registered_population",
    "main",
    "materialize_group",
    "selected_group",
]
