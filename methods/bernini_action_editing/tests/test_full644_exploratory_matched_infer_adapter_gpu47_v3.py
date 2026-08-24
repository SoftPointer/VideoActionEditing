from __future__ import annotations

from contextlib import nullcontext
import json
import hashlib
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_infer_adapter_gpu47_v3 as adapter

GPU47_ENV = {"ROCR_VISIBLE_DEVICES": "4,5,6,7"}


def retained_ffmpeg_fixture(root: Path) -> tuple[dict, int, Path]:
    executable = root / "ffmpeg"
    executable.write_bytes(b"fixture-ffmpeg-executable\n")
    executable.chmod(0o555)
    descriptor = os.open(executable, os.O_RDONLY)
    os.set_inheritable(descriptor, False)
    row = {
        "role": "ffmpeg_executable",
        "fd": descriptor,
        "source_path": str(executable),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "identity": adapter._exec_identity(os.fstat(descriptor)),
    }
    binding = {
        "schema_version": adapter.FFMPEG_AUTHORITY_SCHEMA,
        "row": row,
    }
    binding["authority_digest"] = adapter.model_authority.object_sha256(binding)
    return binding, descriptor, executable


class MatchedInferAdapterV2Tests(unittest.TestCase):
    def test_empty_pread_requires_explicit_vendor_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            empty = Path(raw).resolve(strict=True) / "tracked-empty.py"
            empty.touch()
            descriptor = os.open(empty, os.O_RDONLY)
            try:
                with self.assertRaisesRegex(
                    adapter.MatchedInferAdapterError,
                    "retained ffmpeg pread is unavailable",
                ):
                    adapter._pread_exact(descriptor, 0)
                self.assertEqual(
                    adapter._pread_exact(descriptor, 0, allow_empty=True), b""
                )
                with self.assertRaisesRegex(
                    adapter.MatchedInferAdapterError,
                    "retained ffmpeg pread is unavailable",
                ):
                    adapter._pread_exact(descriptor, 0, allow_empty=1)
            finally:
                os.close(descriptor)

    def test_retained_ffmpeg_exact_exec_fd_and_restoration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            binding, ffmpeg_fd, executable = retained_ffmpeg_fixture(root)
            output = root / "anonymous-output"
            output_fd = os.open(output, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            os.set_inheritable(output_fd, False)
            owner_name = "_matched_fake_imageio_owner"
            owner = types.ModuleType(owner_name)
            launched: list[tuple[tuple, dict]] = []

            def original_popen(*args, **kwargs):
                launched.append((args, kwargs))
                return object()

            original_subprocess = types.SimpleNamespace(Popen=original_popen)
            owner.subprocess = original_subprocess

            def write_frames(*args, **kwargs):
                return None

            write_frames.__module__ = owner_name
            imageio = types.ModuleType("imageio_ffmpeg")
            imageio.write_frames = write_frames
            imageio.get_ffmpeg_exe = lambda: str(executable)
            literal = adapter.model_authority.canonical_json_bytes(binding).decode(
                "utf-8"
            )
            try:
                with mock.patch.dict(
                    sys.modules,
                    {owner_name: owner, "imageio_ffmpeg": imageio},
                ), mock.patch.dict(
                    os.environ,
                    {adapter.FFMPEG_AUTHORITY_ENV: literal},
                    clear=True,
                ):
                    with adapter.retained_ffmpeg_execution(
                        binding, expected_calls=1
                    ) as calls:
                        owner.subprocess.Popen(
                            [str(executable), "-f", "mp4", f"/proc/self/fd/{output_fd}"],
                            close_fds=True,
                            pass_fds=(output_fd,),
                        )
                    self.assertEqual(len(calls), 1)
                self.assertIs(owner.subprocess, original_subprocess)
                self.assertEqual(len(launched), 1)
                self.assertEqual(
                    launched[0][1]["executable"],
                    f"/proc/self/fd/{ffmpeg_fd}",
                )
                self.assertEqual(
                    launched[0][1]["pass_fds"],
                    tuple(sorted((output_fd, ffmpeg_fd))),
                )
                self.assertEqual(
                    launched[0][1]["env"],
                    {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                )
                self.assertFalse(adapter._FFMPEG_PATCH_ACTIVE)
            finally:
                os.close(output_fd)
                os.close(ffmpeg_fd)

    def test_retained_ffmpeg_named_swap_and_inheritable_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            binding, descriptor, executable = retained_ffmpeg_fixture(root)
            literal = adapter.model_authority.canonical_json_bytes(binding).decode(
                "utf-8"
            )
            try:
                with mock.patch.dict(
                    os.environ,
                    {adapter.FFMPEG_AUTHORITY_ENV: literal},
                    clear=True,
                ):
                    self.assertEqual(
                        adapter.load_retained_ffmpeg_authority(), binding
                    )
                    os.set_inheritable(descriptor, True)
                    with self.assertRaises(adapter.MatchedInferAdapterError):
                        adapter.load_retained_ffmpeg_authority()
                    os.set_inheritable(descriptor, False)
                    held = root / "ffmpeg-held"
                    executable.rename(held)
                    executable.write_bytes(b"hostile-replacement\n")
                    executable.chmod(0o555)
                    with self.assertRaises(adapter.MatchedInferAdapterError):
                        adapter.load_retained_ffmpeg_authority()
            finally:
                os.close(descriptor)

    def test_isolated_entry_fails_before_hostile_sitecustomize_without_fds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "sitecustomize-ran"
            (root / "sitecustomize.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root)
            environment.update(GPU47_ENV)
            for key in adapter._SECONDARY_GPU_MASKS:
                environment.pop(key, None)
            environment.pop(
                adapter.model_authority.INHERITED_FD_BINDING_ENV, None
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(Path(adapter.__file__).resolve(strict=True)),
                ],
                check=False,
                capture_output=True,
                env=environment,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"FD environment is absent", completed.stderr)
            self.assertFalse(marker.exists())

    def test_rank_cache_is_fresh_isolated_and_exact_world4(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            with mock.patch.dict(
                os.environ,
                {
                    "FULL644_MATCHED_RANK_CACHE_ROOT": str(base),
                    "LOCAL_RANK": "2",
                },
                clear=True,
            ):
                root = adapter.configure_rank_cache()
                self.assertEqual(root, base / "rank-2")
                self.assertEqual(
                    os.environ["MIOPEN_USER_DB_PATH"], str(root / "miopen-user")
                )
                self.assertEqual(
                    set(path.name for path in root.iterdir()),
                    {
                        "miopen-user",
                        "miopen-custom",
                        "xdg",
                        "tmp",
                        "triton",
                        "inductor",
                        "extensions",
                        "pycache",
                        "home",
                        "hf",
                        "torch",
                    },
                )
                with self.assertRaises(adapter.MatchedInferAdapterError):
                    adapter.configure_rank_cache()

    def test_rank_cache_rejects_missing_or_non_world4_rank(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            for environment in (
                {},
                {
                    "FULL644_MATCHED_RANK_CACHE_ROOT": raw,
                    "LOCAL_RANK": "4",
                },
                {
                    "FULL644_MATCHED_RANK_CACHE_ROOT": raw,
                    "LOCAL_RANK": "not-an-int",
                },
            ):
                with self.subTest(environment=environment), mock.patch.dict(
                    os.environ, environment, clear=True
                ):
                    with self.assertRaises(adapter.MatchedInferAdapterError):
                        adapter.configure_rank_cache()

    def test_rank0_and_nonzero_publication_call_contract(self) -> None:
        for rank, expected in ((0, 1), (1, 0), (2, 0), (3, 0)):
            with self.subTest(rank=rank), mock.patch.dict(
                os.environ,
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": "4",
                    "LOCAL_WORLD_SIZE": "4",
                },
                clear=True,
            ):
                self.assertEqual(
                    adapter.distributed_publication_contract(), (rank, expected)
                )
        for environment in (
            {},
            {"RANK": "00", "LOCAL_RANK": "0", "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"},
            {"RANK": "1", "LOCAL_RANK": "0", "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"},
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4"},
        ):
            with self.subTest(environment=environment), mock.patch.dict(
                os.environ, environment, clear=True
            ):
                with self.assertRaises(adapter.MatchedInferAdapterError):
                    adapter.distributed_publication_contract()

    def test_gpu47_mapping_is_rocr_only_and_fail_closed(self) -> None:
        value = adapter.validate_gpu47_mapping_environment(GPU47_ENV)
        self.assertEqual(value["physical_gpu_indices"], [4, 5, 6, 7])
        self.assertEqual(value["logical_gpu_indices"], [0, 1, 2, 3])
        self.assertEqual(
            value["slurm_step_reserved_gpu_indices"], list(range(8))
        )
        for hostile in (
            {},
            {"ROCR_VISIBLE_DEVICES": "0,1,2,3"},
            {"ROCR_VISIBLE_DEVICES": "4-7"},
            {"ROCR_VISIBLE_DEVICES": "7,6,5,4"},
            {**GPU47_ENV, "HIP_VISIBLE_DEVICES": "0,1,2,3"},
            {**GPU47_ENV, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            {**GPU47_ENV, "GPU_DEVICE_ORDINAL": "0,1,2,3"},
        ):
            with self.subTest(hostile=hostile), self.assertRaises(
                adapter.MatchedInferAdapterError
            ):
                adapter.validate_gpu47_mapping_environment(hostile)
        source = Path(adapter.__file__).read_text(encoding="utf-8")
        entry_source = source[source.index("if __name__ == \"__main__\":") :]
        self.assertLess(
            entry_source.index(
                "_EARLY_GPU_MAPPING = validate_gpu47_mapping_environment()"
            ),
            entry_source.index("infer_lora = load_frozen_inference_sources"),
        )
        self.assertNotIn(
            "import torch",
            entry_source[
                : entry_source.index(
                    "_EARLY_GPU_MAPPING = validate_gpu47_mapping_environment()"
                )
            ],
        )

    def _binding_mocks(self, root: Path, descriptor: int):
        binding = {"binding": "fixture"}
        validated = {"validated": "fixture"}
        task = {
            "fd": descriptor,
            "source_path": str(root),
            "scope": "task",
            "role": "publication_root",
        }
        return (
            binding,
            mock.patch.object(
                adapter.model_authority,
                "validate_inherited_fd_binding",
                return_value=validated,
            ),
            mock.patch.object(
                adapter.model_authority,
                "inherited_fd_row",
                return_value=task,
            ),
        )

    def test_frozen_origins_are_exact(self) -> None:
        closed_path = [
            value
            for value in sys.path
            if Path(value or os.curdir).resolve(strict=False)
            != adapter._METHOD_ROOT
        ]
        with mock.patch.object(sys, "path", closed_path):
            self.assertEqual(
                adapter.validate_frozen_origins(),
                {
                    "infer_lora": adapter.INFER_LORA_SHA256,
                    "model_authority": adapter.MODEL_AUTHORITY_SHA256,
                    "train_lora": adapter.TRAIN_LORA_SHA256,
                    "self_generated_action_preservation_v2": (
                        adapter.SELF_GENERATED_PRESERVATION_SHA256
                    ),
                    "tools.materialize_vae": adapter.MATERIALIZE_VAE_SHA256,
                    "tools.build_renderer_dataset": (
                        adapter.BUILD_RENDERER_DATASET_SHA256
                    ),
                },
            )

    def test_source_bootstrap_scrubs_method_root_and_seals_tools_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            tools = root / "tools"
            site = root / "pinned-site"
            bernini_tree = root / "bernini-tree"
            veomni_tree = root / "veomni-tree"
            tools.mkdir()
            site.mkdir()
            (bernini_tree / "bernini/models").mkdir(parents=True)
            (bernini_tree / "configs/bernini_renderer_wan21_1p3b").mkdir(
                parents=True
            )
            (veomni_tree / "veomni").mkdir(parents=True)
            marker = root / "hostile-peft-loaded"
            bernini_marker = root / "hostile-site-bernini-loaded"
            veomni_marker = root / "hostile-site-veomni-loaded"
            tools_marker = root / "hostile-tools-loaded"
            (bernini_tree / "bernini/__init__.py").write_text(
                "ORIGIN = 'captured-bernini-tree'\n", encoding="utf-8"
            )
            (bernini_tree / "bernini/cli.py").write_text(
                "VALUE = 'captured-cli'\n", encoding="utf-8"
            )
            (bernini_tree / "bernini/models/__init__.py").write_text(
                "VALUE = 'models'\n", encoding="utf-8"
            )
            (bernini_tree / "bernini/models/renderer.py").write_text(
                textwrap.dedent(
                    """\
                    import json
                    from transformers.configuration_utils import PretrainedConfig
                    VALUE = 'captured-renderer'
                    class BerniniRendererConfig(PretrainedConfig):
                        pass
                    """
                ),
                encoding="utf-8",
            )
            transformers = site / "transformers"
            transformers.mkdir()
            (transformers / "__init__.py").write_text(
                "from .configuration_utils import PretrainedConfig, PreTrainedConfig\n"
                "ORIGIN = 'pinned-site-transformers'\n",
                encoding="utf-8",
            )
            (transformers / "configuration_utils.py").write_text(
                textwrap.dedent(
                    """\
                    import json
                    class PreTrainedConfig:
                        @classmethod
                        def from_pretrained(cls, path, **kwargs):
                            value = cls()
                            with open(path, 'r', encoding='utf-8') as handle:
                                value.payload = json.load(handle)
                            value.kwargs = dict(kwargs)
                            return value
                    PretrainedConfig = PreTrainedConfig
                    """
                ),
                encoding="utf-8",
            )
            (bernini_tree / "bernini/asset.txt").write_text(
                "captured-resource\n", encoding="utf-8"
            )
            (
                bernini_tree
                / "configs/bernini_renderer_wan21_1p3b/config.json"
            ).write_text(
                '{"marker":"captured-config"}\n', encoding="utf-8"
            )
            (veomni_tree / "veomni/__init__.py").write_text(
                "ORIGIN = 'captured-veomni-tree'\n", encoding="utf-8"
            )

            def commit_tree(path: Path) -> str:
                subprocess.run(["git", "init", "-q", str(path)], check=True)
                subprocess.run(
                    ["git", "-C", str(path), "add", "--all"], check=True
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(path),
                        "-c",
                        "user.name=fixture",
                        "-c",
                        "user.email=fixture@example.invalid",
                        "commit",
                        "-q",
                        "-m",
                        "fixture",
                    ],
                    check=True,
                )
                return subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                ).strip()

            bernini_commit = commit_tree(bernini_tree)
            veomni_commit = commit_tree(veomni_tree)
            sources = {
                "train_lora.py": textwrap.dedent(
                    f"""\
                    import sys
                    BERNINI_OFFICIAL_COMMIT = {bernini_commit!r}
                    VEOMNI_TESTED_COMMIT = {veomni_commit!r}
                    def validate_source_trees(*args, **kwargs):
                        raise RuntimeError('unscoped validator executed')
                    def activate_source_trees(bernini_root, veomni_root):
                        roots = [str(bernini_root), str(veomni_root)]
                        for root in roots:
                            while root in sys.path:
                                sys.path.remove(root)
                        sys.path[0:0] = roots
                    """
                ),
                "tools/build_renderer_dataset.py": "VALUE = 'builder'\n",
                "tools/materialize_vae.py": textwrap.dedent(
                    """\
                    import sys
                    from pathlib import Path
                    METHOD_ROOT = Path(__file__).resolve().parents[1]
                    if str(METHOD_ROOT) not in sys.path:
                        sys.path.insert(0, str(METHOD_ROOT))
                    from tools import build_renderer_dataset as raw_builder
                    """
                ),
                "self_generated_action_preservation_v2.py": "VALUE = 'preservation'\n",
                "infer_lora.py": textwrap.dedent(
                    """\
                    import pkgutil,sys
                    from pathlib import Path
                    METHOD_ROOT = Path(__file__).resolve().parent
                    if str(METHOD_ROOT) not in sys.path:
                        sys.path.insert(0, str(METHOD_ROOT))
                    import train_lora as trainer
                    def lazy_import(bernini_root, veomni_root):
                        bernini_root,veomni_root,_,_=trainer.validate_source_trees(
                            bernini_root,veomni_root,
                            expected_bernini_commit=trainer.BERNINI_OFFICIAL_COMMIT,
                            expected_veomni_commit=trainer.VEOMNI_TESTED_COMMIT,
                        )
                        trainer.activate_source_trees(bernini_root,veomni_root)
                        sys.path_importer_cache[str(bernini_root)]=object()
                        sys.path_importer_cache[sys.path[-1]]=object()
                        import torch, diffusers, peft, transformers
                        import bernini, veomni
                        from bernini.cli import VALUE as cli_value
                        from bernini.models.renderer import BerniniRendererConfig, VALUE as renderer_value
                        resource=pkgutil.get_data('bernini','asset.txt').decode('utf-8')
                        config_path=bernini_root/'configs/bernini_renderer_wan21_1p3b/config.json'
                        config_path.chmod(0o600)
                        config_path.write_text('{"marker":"hostile-named-config"}'+chr(10),encoding='utf-8')
                        config_path.chmod(0o400)
                        config=BerniniRendererConfig.from_pretrained(
                            str(bernini_root/'configs/bernini_renderer_wan21_1p3b'),
                            diff_dec_config_path='/abs/checkpoint',
                            ema_decay=None,
                            local_files_only=True,
                            max_sequence_length=512,
                            scratch=False,
                            shift=5.0,
                            skip_transformer_1=False,
                            skip_transformer_2=True,
                            switch_dit_boundary=0.0,
                            use_src_id_rotary_emb=True,
                            use_unipc=True,
                            wan22_base='/abs/checkpoint',
                        )
                        config_path.chmod(0o600)
                        config_path.write_text('{"marker":"captured-config"}'+chr(10),encoding='utf-8')
                        config_path.chmod(0o400)
                        return (torch.ORIGIN, diffusers.ORIGIN, peft.ORIGIN, transformers.ORIGIN, bernini.ORIGIN, veomni.ORIGIN, cli_value, renderer_value, resource, config.payload['marker'], config.kwargs['shift'])
                    """
                ),
            }
            for relative, source in sources.items():
                path = root / relative
                path.write_text(source, encoding="utf-8")
                path.chmod(0o444)
            hostile_source = root / "hostile_peft_source.py"
            hostile_source.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('loaded', encoding='utf-8')\n"
                "ORIGIN = 'hostile-method-root'\n",
                encoding="utf-8",
            )
            py_compile.compile(
                str(hostile_source),
                cfile=str(root / "peft.pyc"),
                doraise=True,
            )
            hostile_source.unlink()
            (tools / "foo.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(tools_marker)!r}).write_text('loaded', encoding='utf-8')\n",
                encoding="utf-8",
            )
            for name in ("torch", "diffusers", "peft"):
                (site / f"{name}.py").write_text(
                    f"ORIGIN = 'pinned-site-{name}'\n", encoding="utf-8"
                )
            (site / "bernini.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(bernini_marker)!r}).write_text('loaded', encoding='utf-8')\n"
                "ORIGIN = 'hostile-installed-bernini'\n",
                encoding="utf-8",
            )
            (site / "veomni.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(veomni_marker)!r}).write_text('loaded', encoding='utf-8')\n"
                "ORIGIN = 'hostile-installed-veomni'\n",
                encoding="utf-8",
            )
            hashes = {
                relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for relative in sources
            }
            script = textwrap.dedent(
                f"""\
                import importlib,json,os,pkgutil,sys
                from pathlib import Path
                sys.path.insert(0,{str(MODULE_ROOT)!r})
                import full644_exploratory_matched_infer_adapter_gpu47_v3 as adapter
                root=Path({str(root)!r})
                site=Path({str(site)!r})
                bernini_tree=Path({str(bernini_tree)!r})
                veomni_tree=Path({str(veomni_tree)!r})
                marker=Path({str(marker)!r})
                bernini_marker=Path({str(bernini_marker)!r})
                veomni_marker=Path({str(veomni_marker)!r})
                tools_marker=Path({str(tools_marker)!r})
                for name in ('train_lora','tools','tools.build_renderer_dataset','tools.materialize_vae','self_generated_action_preservation_v2','infer_lora','torch','diffusers','peft','transformers','bernini','veomni','tools.foo'):
                    sys.modules.pop(name,None)
                adapter._METHOD_ROOT=root
                adapter.TRAIN_LORA_SHA256={hashes['train_lora.py']!r}
                adapter.BUILD_RENDERER_DATASET_SHA256={hashes['tools/build_renderer_dataset.py']!r}
                adapter.MATERIALIZE_VAE_SHA256={hashes['tools/materialize_vae.py']!r}
                adapter.SELF_GENERATED_PRESERVATION_SHA256={hashes['self_generated_action_preservation_v2.py']!r}
                adapter.INFER_LORA_SHA256={hashes['infer_lora.py']!r}
                sys.path[:]=[str(site)]
                module=adapter.load_frozen_inference_sources(require_absent=True)
                adapter.infer_lora=module
                rank_cache=root/'rank-cache'
                rank_cache.mkdir()
                adapter._EARLY_RANK_CACHE=rank_cache
                importer_cache_expected={{}}
                def fixture_preload(site_root):
                    import torch,diffusers,peft,transformers
                    from transformers.configuration_utils import PretrainedConfig,PreTrainedConfig
                    if PretrainedConfig is not PreTrainedConfig: raise RuntimeError('config alias differs')
                    importer_cache_expected.update(sys.path_importer_cache)
                    return {{'schema_version':'fixture-third-party-preload','authority_digest':'a'*64}}
                adapter._preload_pinned_dependencies=fixture_preload
                if sys.path != [str(site)]: raise RuntimeError('method root survived')
                with adapter.pinned_dependency_import_paths(site) as dependency:
                    observed=module.lazy_import(bernini_tree,veomni_tree)
                    if observed != ('pinned-site-torch','pinned-site-diffusers','pinned-site-peft','pinned-site-transformers','captured-bernini-tree','captured-veomni-tree','captured-cli','captured-renderer','captured-resource'+chr(10),'captured-config',5.0): raise RuntimeError('pinned dependency lost')
                if dependency.get('activation_call_count') != 1 or dependency.get('package_specific_finder_first') is not True or dependency.get('live_roots_never_used_for_import_or_config') is not True or dependency.get('renderer_config_memfd_authority',{{}}).get('native_from_pretrained_call_count') != 1: raise RuntimeError('dependency authority lost')
                if sys.path != [str(site)]: raise RuntimeError('source roots survived')
                if set(sys.path_importer_cache)!=set(importer_cache_expected) or any(sys.path_importer_cache[key] is not value for key,value in importer_cache_expected.items()): raise RuntimeError('importer cache survived')
                try: importlib.import_module('tools.foo')
                except ModuleNotFoundError: pass
                else: raise RuntimeError('tools namespace escaped')
                if marker.exists() or bernini_marker.exists() or veomni_marker.exists() or tools_marker.exists(): raise RuntimeError('hostile local loaded')
                adapter.validate_frozen_origins()
                print('CLOSED_LOCAL_IMPORT_PATH_OK',flush=True)
                """
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout.decode("utf-8").strip(),
                "CLOSED_LOCAL_IMPORT_PATH_OK",
            )
            self.assertFalse(marker.exists())
            self.assertFalse(bernini_marker.exists())
            self.assertFalse(veomni_marker.exists())
            self.assertFalse(tools_marker.exists())

    def test_vendor_capture_rejects_internal_package_shadow_and_ignored_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            vendor = root / "vendor"
            (vendor / "bernini/models").mkdir(parents=True)
            (vendor / "configs/bernini_renderer_wan21_1p3b").mkdir(parents=True)
            (vendor / "bernini/__init__.py").write_text("VALUE=1\n")
            (vendor / "bernini/cli.py").write_text("VALUE=2\n")
            (vendor / "bernini/models/__init__.py").write_text("VALUE=3\n")
            (vendor / "bernini/models/renderer.py").write_text("VALUE=4\n")
            (vendor / "configs/bernini_renderer_wan21_1p3b/config.json").write_text(
                "{}\n"
            )
            (vendor / ".gitignore").write_text("*.pyc\n")
            subprocess.run(["git", "init", "-q", str(vendor)], check=True)
            subprocess.run(["git", "-C", str(vendor), "add", "--all"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(vendor),
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "fixture",
                ],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
            ).strip()
            marker = root / "hostile-marker"
            hostile_cases = (
                vendor / "bernini/cli/__init__.py",
                vendor / "bernini/models/renderer/__init__.py",
                vendor / "bernini/ignored.pyc",
            )
            for hostile in hostile_cases:
                with self.subTest(hostile=hostile.relative_to(vendor)):
                    hostile.parent.mkdir(parents=True, exist_ok=True)
                    hostile.write_text(
                        "from pathlib import Path\n"
                        f"Path({str(marker)!r}).write_text('executed')\n"
                    )
                    with self.assertRaisesRegex(
                        adapter.MatchedInferAdapterError,
                        "exact physical tree closure differs",
                    ):
                        adapter._capture_git_vendor_tree(
                            vendor,
                            expected_commit=commit,
                            scopes=adapter._BERNINI_TREE_SCOPES,
                            label="Bernini-fixture",
                        )
                    self.assertFalse(marker.exists())
                    hostile.unlink()
                    parent = hostile.parent
                    while parent != vendor and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent

    def test_vendor_git_audit_does_not_execute_fsmonitor_or_lazy_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            vendor = root / "vendor"
            (vendor / "veomni").mkdir(parents=True)
            (vendor / "veomni/__init__.py").write_text("VALUE=1\n")
            subprocess.run(["git", "init", "-q", str(vendor)], check=True)
            subprocess.run(["git", "-C", str(vendor), "add", "--all"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(vendor),
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "fixture",
                ],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
            ).strip()
            marker = root / "git-hook-marker"
            hook = root / "hostile-fsmonitor.sh"
            hook.write_text(f"#!/bin/sh\ntouch {str(marker)!r}\n")
            hook.chmod(0o755)
            subprocess.run(
                ["git", "-C", str(vendor), "config", "core.fsmonitor", str(hook)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(vendor), "config", "remote.origin.promisor", "true"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(vendor),
                    "config",
                    "remote.origin.partialclonefilter",
                    "blob:none",
                ],
                check=True,
            )
            capture = adapter._capture_git_vendor_tree(
                vendor,
                expected_commit=commit,
                scopes=adapter._VEOMNI_TREE_SCOPES,
                label="VeOmni-fixture",
            )
            self.assertEqual(capture.expected_commit, commit)
            self.assertFalse(marker.exists())

    def test_vendor_capture_accepts_tracked_zero_byte_and_rejects_extra_empty(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            vendor = root / "vendor"
            trainer = vendor / "veomni/trainer"
            trainer.mkdir(parents=True)
            (vendor / "veomni/__init__.py").write_text("VALUE=1\n")
            tracked_empty = trainer / "__init__.py"
            tracked_empty.touch()
            subprocess.run(["git", "init", "-q", str(vendor)], check=True)
            subprocess.run(
                ["git", "-C", str(vendor), "add", "--all"], check=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(vendor),
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "zero-byte-fixture",
                ],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
            ).strip()
            capture = adapter._capture_git_vendor_tree(
                vendor,
                expected_commit=commit,
                scopes=adapter._VEOMNI_TREE_SCOPES,
                label="VeOmni-zero-byte-fixture",
            )
            relative = "veomni/trainer/__init__.py"
            self.assertEqual(capture.file_bytes[relative], b"")
            self.assertEqual(
                capture.file_sha256[relative], hashlib.sha256(b"").hexdigest()
            )
            self.assertEqual(
                capture.file_git_blobs[relative],
                "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
            )
            snapshot = adapter._materialize_captured_vendor_tree(
                capture, root / "captured-snapshot"
            )
            snapshot_authority = adapter._verify_captured_snapshot(
                capture, snapshot
            )
            materialized_empty = snapshot / relative
            self.assertEqual(materialized_empty.read_bytes(), b"")
            self.assertEqual(materialized_empty.stat().st_size, 0)
            self.assertEqual(materialized_empty.stat().st_mode & 0o777, 0o400)
            self.assertTrue(snapshot_authority["snapshot_exact_physical_closure"])
            empty_row = adapter._CapturedVendorModule(
                fullname="veomni.trainer",
                origin=materialized_empty,
                raw=capture.file_bytes[relative],
                sha256=capture.file_sha256[relative],
                is_package=True,
                is_namespace=False,
            )
            empty_loader = adapter._CapturedVendorLoader(
                empty_row,
                {os.path.normpath(str(materialized_empty)): b""},
                mock.Mock(),
            )
            empty_module = types.ModuleType(empty_row.fullname)
            empty_loader.exec_module(empty_module)
            self.assertIn("__builtins__", empty_module.__dict__)
            self.assertEqual(empty_loader.get_source(empty_row.fullname), "")
            self.assertEqual(empty_loader.get_data(str(materialized_empty)), b"")

            namespace_row = adapter._CapturedVendorModule(
                fullname="veomni.synthetic_namespace",
                origin=materialized_empty.parent,
                raw=b"",
                sha256=hashlib.sha256(b"").hexdigest(),
                is_package=True,
                is_namespace=True,
            )
            namespace_loader = adapter._CapturedVendorLoader(
                namespace_row, {}, mock.Mock()
            )
            namespace_module = types.ModuleType(namespace_row.fullname)
            namespace_loader.exec_module(namespace_module)
            self.assertNotIn("__builtins__", namespace_module.__dict__)
            for directory, child_directories, _ in os.walk(snapshot):
                os.chmod(directory, 0o700)
                for child in child_directories:
                    os.chmod(Path(directory) / child, 0o700)

            hostile_empty = vendor / "veomni/hostile.py"
            hostile_empty.touch()
            with self.assertRaisesRegex(
                adapter.MatchedInferAdapterError,
                "exact physical tree closure differs",
            ):
                adapter._capture_git_vendor_tree(
                    vendor,
                    expected_commit=commit,
                    scopes=adapter._VEOMNI_TREE_SCOPES,
                    label="VeOmni-zero-byte-hostile",
                )

    def test_vendor_git_exec_uses_retained_proc_fd_and_exact_environment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            executable = root / "git"
            executable.write_bytes(b"retained-git-fixture\n")
            git_fd = os.open(executable, os.O_RDONLY)
            root_fd = os.open(root, os.O_RDONLY)
            authority = {
                "path": "/usr/bin/git",
                "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "identity": adapter._exec_identity(os.fstat(git_fd)),
                "authority_digest": "a" * 64,
            }
            completed = types.SimpleNamespace(
                returncode=0,
                stdout=b"commit\n",
                stderr=b"",
            )
            try:
                with mock.patch.object(
                    adapter,
                    "_open_git_executable_authority",
                    return_value=(git_fd, authority),
                ), mock.patch.object(
                    adapter,
                    "_git_authority_from_open_fd",
                    return_value=authority,
                ) as replay, mock.patch.object(
                    adapter.Path,
                    "exists",
                    return_value=True,
                ), mock.patch.object(
                    adapter.subprocess,
                    "run",
                    return_value=completed,
                ) as execute:
                    observed = adapter._run_git(
                        root_fd,
                        ["cat-file", "-t", "1" * 40],
                        label="fixture",
                    )
                self.assertEqual(observed, b"commit\n")
                replay.assert_called_once_with(git_fd)
                kwargs = execute.call_args.kwargs
                self.assertEqual(kwargs["executable"], f"/proc/self/fd/{git_fd}")
                self.assertEqual(kwargs["pass_fds"], (root_fd, git_fd))
                self.assertEqual(kwargs["env"], adapter._GIT_ENV)
                self.assertEqual(kwargs["cwd"], "/")
                self.assertTrue(kwargs["close_fds"])
                self.assertEqual(
                    execute.call_args.args[0],
                    [
                        "/usr/bin/git",
                        "--no-replace-objects",
                        "-C",
                        f"/proc/self/fd/{root_fd}",
                        "cat-file",
                        "-t",
                        "1" * 40,
                    ],
                )
                with self.assertRaises(OSError):
                    os.fstat(git_fd)
            finally:
                os.close(root_fd)

    def test_renderer_config_redirect_is_exact_once_and_restores_inherited_method(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            logical = root / "snapshot/configs/bernini_renderer_wan21_1p3b"
            logical.mkdir(parents=True)

            class PreTrainedConfig:
                calls: list[tuple[str, dict]] = []

                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    cls.calls.append((path, dict(kwargs)))
                    return {"path": path, "kwargs": dict(kwargs)}

            PreTrainedConfig.__module__ = "transformers.configuration_utils"
            PreTrainedConfig.__qualname__ = "PreTrainedConfig"
            PreTrainedConfig.__dict__["from_pretrained"].__func__.__module__ = (
                "transformers.configuration_utils"
            )
            PreTrainedConfig.__dict__["from_pretrained"].__func__.__qualname__ = (
                "PreTrainedConfig.from_pretrained"
            )
            configuration_utils = types.ModuleType(
                "transformers.configuration_utils"
            )
            configuration_utils.PreTrainedConfig = PreTrainedConfig
            configuration_utils.PretrainedConfig = PreTrainedConfig

            class BerniniRendererConfig(PreTrainedConfig):
                pass

            module = types.ModuleType("bernini.models.renderer")
            BerniniRendererConfig.__module__ = module.__name__
            BerniniRendererConfig.__qualname__ = "BerniniRendererConfig"
            module.BerniniRendererConfig = BerniniRendererConfig
            redirect = None
            with mock.patch.object(adapter, "_EARLY_RANK_CACHE", root), mock.patch.object(
                adapter, "_EARLY_INBOUND_BINDING", None
            ):
                redirect = adapter._create_sealed_renderer_config_redirect(
                    b'{"marker":"captured"}\n', logical
                )
            descriptor = redirect.descriptor
            with mock.patch.dict(
                sys.modules,
                {"transformers.configuration_utils": configuration_utils},
            ):
                try:
                    redirect.install(module)
                    valid = {
                        "diff_dec_config_path": "/abs/checkpoint",
                        "ema_decay": None,
                        "local_files_only": True,
                        "max_sequence_length": 512,
                        "scratch": False,
                        "shift": 5.0,
                        "skip_transformer_1": False,
                        "skip_transformer_2": True,
                        "switch_dit_boundary": 0.0,
                        "use_src_id_rotary_emb": True,
                        "use_unipc": True,
                        "wan22_base": "/abs/checkpoint",
                    }
                    with self.assertRaisesRegex(
                        adapter.MatchedInferAdapterError,
                        "keyword set differs",
                    ):
                        BerniniRendererConfig.from_pretrained(
                            str(logical), local_files_only=True
                        )
                    self.assertEqual(PreTrainedConfig.calls, [])
                    observed = BerniniRendererConfig.from_pretrained(
                        str(logical), **valid
                    )
                    self.assertEqual(observed["path"], redirect.fd_path)
                    self.assertEqual(observed["kwargs"], valid)
                    with self.assertRaisesRegex(
                        adapter.MatchedInferAdapterError, "call differs"
                    ):
                        BerniniRendererConfig.from_pretrained(
                            str(logical), **valid
                        )
                    authority = redirect.finalize_authority()
                    self.assertEqual(
                        authority["native_from_pretrained_call_count"], 1
                    )
                    self.assertTrue(
                        authority["native_from_pretrained_call"][
                            "native_function_identity_verified"
                        ]
                    )
                finally:
                    redirect.restore_and_close()
            self.assertNotIn(
                "from_pretrained", BerniniRendererConfig.__dict__
            )
            self.assertIs(
                BerniniRendererConfig.from_pretrained.__func__,
                PreTrainedConfig.from_pretrained.__func__,
            )
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_renderer_config_redirect_rejects_unrelated_inherited_classmethod(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            logical = root / "snapshot/configs/bernini_renderer_wan21_1p3b"
            logical.mkdir(parents=True)

            class PreTrainedConfig:
                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    return (cls, path, kwargs)

            PreTrainedConfig.__module__ = "transformers.configuration_utils"
            PreTrainedConfig.__qualname__ = "PreTrainedConfig"
            PreTrainedConfig.__dict__["from_pretrained"].__func__.__module__ = (
                "transformers.configuration_utils"
            )
            PreTrainedConfig.__dict__["from_pretrained"].__func__.__qualname__ = (
                "PreTrainedConfig.from_pretrained"
            )
            configuration_utils = types.ModuleType(
                "transformers.configuration_utils"
            )
            configuration_utils.PreTrainedConfig = PreTrainedConfig
            configuration_utils.PretrainedConfig = PreTrainedConfig

            class UnrelatedConfig:
                @classmethod
                def from_pretrained(cls, path, **kwargs):
                    return (cls, path, kwargs)

            class BerniniRendererConfig(UnrelatedConfig):
                pass

            module = types.ModuleType("bernini.models.renderer")
            BerniniRendererConfig.__module__ = module.__name__
            BerniniRendererConfig.__qualname__ = "BerniniRendererConfig"
            module.BerniniRendererConfig = BerniniRendererConfig
            with mock.patch.object(
                adapter, "_EARLY_RANK_CACHE", root
            ), mock.patch.object(
                adapter, "_EARLY_INBOUND_BINDING", None
            ):
                redirect = adapter._create_sealed_renderer_config_redirect(
                    b'{"marker":"captured"}\n', logical
                )
            try:
                with mock.patch.dict(
                    sys.modules,
                    {"transformers.configuration_utils": configuration_utils},
                ), self.assertRaisesRegex(
                    adapter.MatchedInferAdapterError, "class origin differs"
                ):
                    redirect.install(module)
            finally:
                redirect.restore_and_close()

    def test_resolve_binds_named_parent_to_held_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            descriptor = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)
            binding, validate_patch, row_patch = self._binding_mocks(
                root, descriptor
            )
            with validate_patch, row_patch:
                paths = adapter.resolve_publication_paths(
                    root / "case00-base.mp4", binding
                )
            self.assertEqual(paths.logical_output, root / "case00-base.mp4")
            self.assertEqual(
                paths.runtime_output,
                Path(f"/proc/self/fd/{descriptor}/case00-base.mp4"),
            )
            self.assertEqual(
                paths.runtime_receipt,
                Path(f"/proc/self/fd/{descriptor}/case00-base.mp4.receipt.json"),
            )

    def test_resolve_rejects_other_parent_existing_output_and_replaced_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            root = parent / "media"
            root.mkdir()
            other = parent / "other"
            other.mkdir()
            descriptor = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)
            binding, validate_patch, row_patch = self._binding_mocks(
                root, descriptor
            )
            with validate_patch, row_patch:
                with self.assertRaises(adapter.MatchedInferAdapterError):
                    adapter.resolve_publication_paths(
                        other / "case00-base.mp4", binding
                    )
                existing = root / "case00-base.mp4"
                existing.write_bytes(b"old")
                with self.assertRaises(adapter.MatchedInferAdapterError):
                    adapter.resolve_publication_paths(existing, binding)
                existing.unlink()
                held_name = parent / "held-media"
                root.rename(held_name)
                root.mkdir()
                with self.assertRaises(adapter.MatchedInferAdapterError):
                    adapter.resolve_publication_paths(
                        root / "case00-base.mp4", binding
                    )

    def test_output_argument_must_be_unique_normalized_and_mp4(self) -> None:
        self.assertEqual(
            adapter._extract_exact_output(["--output", "/tmp/case00-base.mp4"]),
            Path("/tmp/case00-base.mp4"),
        )
        for argv in (
            [],
            ["--output", "relative.mp4"],
            ["--output", "/tmp/not-video.json"],
            ["--output", "/tmp/a.mp4", "--output=/tmp/b.mp4"],
            ["--output", "/tmp/../tmp/a.mp4"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(adapter.MatchedInferAdapterError):
                    adapter._extract_exact_output(argv)

    def test_translation_is_exactly_once_and_always_restored(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            logical_output = root / "case00-base.mp4"
            logical_receipt = root / "case00-base.mp4.receipt.json"
            task_fd = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, task_fd)
            paths = adapter.PublicationPaths(
                logical_output=logical_output,
                logical_receipt=logical_receipt,
                runtime_output=logical_output,
                runtime_receipt=logical_receipt,
                task_fd=task_fd,
                task_root=root,
            )

            def encoded(path: Path, *args, **kwargs):
                Path(path).write_bytes(b"video")
                Path(path).chmod(0o444)
                return {"path": str(path)}

            def receipt(path: Path, value):
                Path(path).write_text(
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                Path(path).chmod(0o400)

            fake = types.SimpleNamespace(
                _create_retained_encoded_output=encoded,
                _atomic_write_json=receipt,
            )
            original_encoded = fake._create_retained_encoded_output
            original_receipt = fake._atomic_write_json
            with adapter.translated_publication(paths, inference_module=fake) as calls:
                fake._create_retained_encoded_output(logical_output)
                output_raw = logical_output.read_bytes()
                receipt_value = {
                    "output": {
                        "path": str(logical_output),
                        "sha256": hashlib.sha256(output_raw).hexdigest(),
                        "size": len(output_raw),
                        "publication_identity": adapter._exec_identity(
                            logical_output.lstat()
                        ),
                    }
                }
                receipt_value["receipt_digest"] = (
                    adapter.model_authority.object_sha256(receipt_value)
                )
                fake._atomic_write_json(
                    logical_receipt,
                    receipt_value,
                )
                with self.assertRaises(adapter.MatchedInferAdapterError):
                    fake._create_retained_encoded_output(logical_output)
            self.assertEqual((calls.encoded_output, calls.receipt), (1, 1))
            self.assertIs(fake._create_retained_encoded_output, original_encoded)
            self.assertIs(fake._atomic_write_json, original_receipt)
            replay = adapter.replay_publication(paths, calls)
            self.assertEqual(replay["logical_output"], str(logical_output))
            for descriptor in (calls.output_fd, calls.receipt_fd):
                if descriptor is not None:
                    os.close(descriptor)

            try:
                with adapter.translated_publication(paths, inference_module=fake):
                    raise RuntimeError("injected failure")
            except RuntimeError:
                pass
            self.assertIs(fake._create_retained_encoded_output, original_encoded)
            self.assertIs(fake._atomic_write_json, original_receipt)
            self.assertFalse(adapter._PATCH_ACTIVE)

            with self.assertRaisesRegex(
                adapter.MatchedInferAdapterError, "were not restored"
            ):
                with adapter.translated_publication(
                    paths, inference_module=fake
                ):
                    fake._atomic_write_json = lambda path, value: None
            self.assertIs(fake._create_retained_encoded_output, original_encoded)
            self.assertIs(fake._atomic_write_json, original_receipt)
            self.assertFalse(adapter._PATCH_ACTIVE)

    def test_rank_entry_rejects_lost_or_still_inheritable_fds(self) -> None:
        error = adapter.model_authority.ModelConsumptionAuthorityError(
            "rank FD lost/inheritable differs"
        )
        with mock.patch.object(
            adapter.model_authority,
            "load_inherited_fd_environment",
            side_effect=error,
        ):
            with self.assertRaisesRegex(
                adapter.MatchedInferAdapterError, "lost/inheritable"
            ):
                adapter.load_bootstrap_sealed_authority_fds()

    def test_run_keeps_logical_receipt_path_and_restores_origins(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            logical_output = root / "case07-full644.mp4"
            logical_receipt = root / "case07-full644.mp4.receipt.json"
            descriptor = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)
            paths = adapter.PublicationPaths(
                logical_output=logical_output,
                logical_receipt=logical_receipt,
                runtime_output=logical_output,
                runtime_receipt=logical_receipt,
                task_fd=descriptor,
                task_root=root,
            )
            original_encoded = adapter.infer_lora._create_retained_encoded_output
            original_receipt = adapter.infer_lora._atomic_write_json

            def encoded(path: Path, *args, **kwargs):
                Path(path).write_bytes(b"video")
                Path(path).chmod(0o444)
                return {"path": str(path)}

            def receipt(path: Path, value):
                Path(path).write_bytes(
                    adapter.model_authority.canonical_json_bytes(value) + b"\n"
                )
                Path(path).chmod(0o400)

            def fake_main(argv):
                adapter.infer_lora._create_retained_encoded_output(logical_output)
                output_raw = logical_output.read_bytes()
                receipt_value = {
                    "output": {
                        "path": str(logical_output),
                        "sha256": hashlib.sha256(output_raw).hexdigest(),
                        "size": len(output_raw),
                        "publication_identity": adapter._exec_identity(
                            logical_output.lstat()
                        ),
                    }
                }
                receipt_value["receipt_digest"] = (
                    adapter.model_authority.object_sha256(receipt_value)
                )
                adapter.infer_lora._atomic_write_json(
                    logical_receipt,
                    receipt_value,
                )
                return 0

            try:
                adapter.infer_lora._create_retained_encoded_output = encoded
                adapter.infer_lora._atomic_write_json = receipt
                with mock.patch.object(
                    adapter, "resolve_publication_paths", return_value=paths
                ), mock.patch.object(
                    adapter,
                    "retained_ffmpeg_execution",
                    side_effect=lambda authority, expected_calls: nullcontext(
                        [{"fixture": True}] if expected_calls == 1 else []
                    ),
                ), mock.patch.dict(
                    os.environ,
                    {
                        **GPU47_ENV,
                        "RANK": "0",
                        "LOCAL_RANK": "0",
                        "WORLD_SIZE": "4",
                        "LOCAL_WORLD_SIZE": "4",
                    },
                    clear=True,
                ), mock.patch.object(
                    adapter,
                    "publish_publication_handoff",
                    return_value={"payload_digest": "e" * 64},
                ):
                    result = adapter.run(
                        ["--output", str(logical_output)],
                        ffmpeg_authority={"authority_digest": "f" * 64},
                        publication_handoff={
                            "task_id": "shared8-07-full644",
                            "authority_digest": "d" * 64,
                        },
                        gpu_visibility_contract=(
                            adapter.validate_gpu47_mapping_environment()
                        ),
                        inference_main=fake_main,
                        binding_loader=lambda **kwargs: {
                            "binding": "fixture",
                            "task_id": "shared8-07-full644",
                        },
                        verify_origins=False,
                    )
                self.assertEqual(result["return_code"], 0)
                self.assertEqual(result["rank"], 0)
                self.assertTrue(result["publication_functions_restored"])
                self.assertEqual(
                    json.loads(logical_receipt.read_text())["output"]["path"],
                    str(logical_output),
                )
                self.assertIs(
                    adapter.infer_lora._create_retained_encoded_output, encoded
                )
                self.assertIs(adapter.infer_lora._atomic_write_json, receipt)
            finally:
                adapter.infer_lora._create_retained_encoded_output = original_encoded
                adapter.infer_lora._atomic_write_json = original_receipt

    def test_nonzero_rank_requires_zero_local_calls_but_replays_rank0_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            logical_output = root / "case01-base.mp4"
            logical_receipt = root / "case01-base.mp4.receipt.json"
            paths = adapter.PublicationPaths(
                logical_output=logical_output,
                logical_receipt=logical_receipt,
                runtime_output=logical_output,
                runtime_receipt=logical_receipt,
                task_fd=7,
                task_root=root,
            )
            original_encoded = adapter.infer_lora._create_retained_encoded_output
            original_receipt = adapter.infer_lora._atomic_write_json

            def encoded(path: Path, *args, **kwargs):
                Path(path).write_bytes(b"rank0-video")
                Path(path).chmod(0o444)
                return {"path": str(path)}

            def receipt(path: Path, value):
                Path(path).write_text(json.dumps(value) + "\n", encoding="utf-8")
                Path(path).chmod(0o400)

            def fake_nonzero_main(argv):
                # Simulate rank 0's concurrent publication without invoking the
                # functions patched in this nonzero-rank interpreter.
                encoded(logical_output)
                receipt(
                    logical_receipt,
                    {"output": {"path": str(logical_output)}},
                )
                return 0

            try:
                adapter.infer_lora._create_retained_encoded_output = encoded
                adapter.infer_lora._atomic_write_json = receipt
                with mock.patch.object(
                    adapter, "resolve_publication_paths", return_value=paths
                ), mock.patch.object(
                    adapter,
                    "retained_ffmpeg_execution",
                    side_effect=lambda authority, expected_calls: nullcontext(
                        [{"fixture": True}] if expected_calls == 1 else []
                    ),
                ), mock.patch.dict(
                    os.environ,
                    {
                        **GPU47_ENV,
                        "RANK": "3",
                        "LOCAL_RANK": "3",
                        "WORLD_SIZE": "4",
                        "LOCAL_WORLD_SIZE": "4",
                    },
                    clear=True,
                ):
                    result = adapter.run(
                        ["--output", str(logical_output)],
                        ffmpeg_authority={"authority_digest": "f" * 64},
                        publication_handoff={
                            "task_id": "shared8-01-base",
                            "authority_digest": "d" * 64,
                        },
                        gpu_visibility_contract=(
                            adapter.validate_gpu47_mapping_environment()
                        ),
                        inference_main=fake_nonzero_main,
                        binding_loader=lambda **kwargs: {
                            "binding": "fixture",
                            "task_id": "shared8-01-base",
                        },
                        verify_origins=False,
                    )
                self.assertEqual(result["rank"], 3)
                self.assertEqual(result["local_publication_call_count"], 0)
                self.assertTrue(result["publication_functions_restored"])
            finally:
                adapter.infer_lora._create_retained_encoded_output = original_encoded
                adapter.infer_lora._atomic_write_json = original_receipt


if __name__ == "__main__":
    unittest.main()
