from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SEALED_METHOD_FIXTURE = Path(
    "/tmp/case01_object_trajectory_v1_sealed_methods_fixture"
)
V2_SOURCE = METHOD_ROOT / "full644_exploratory_matched_infer_adapter_v2.py"
V3_SOURCE = METHOD_ROOT / "full644_exploratory_matched_infer_adapter_v3.py"

OLD_ACTIVATION_ORDER = (
    """
        original_activate(bernini, veomni)
        active_path = [str(bernini), str(veomni), *path_snapshot]
        if sys.path != active_path:
            raise MatchedInferAdapterError(
                "original source-tree activation path delta differs"
            )
        sys.meta_path.insert(0, finder)
        if sys.meta_path != [finder, *meta_path_snapshot]:
            raise MatchedInferAdapterError(
                "captured vendor finder installation differs"
            )
    """.lstrip("\n")
)
NEW_ACTIVATION_ORDER = (
    """
        sys.meta_path.insert(0, finder)
        if (
            sys.meta_path is not meta_path_owner
            or sys.meta_path != [finder, *meta_path_snapshot]
        ):
            raise MatchedInferAdapterError(
                "captured vendor finder installation differs"
            )
        original_activate(bernini, veomni)
        if (
            sys.meta_path is not meta_path_owner
            or sys.meta_path != [finder, *meta_path_snapshot]
        ):
            raise MatchedInferAdapterError(
                "captured vendor finder changed during source-tree activation"
            )
        active_path = [str(bernini), str(veomni), *path_snapshot]
        if sys.path != active_path:
            raise MatchedInferAdapterError(
                "original source-tree activation path delta differs"
            )
    """.lstrip("\n")
)
OLD_META_PATH_CAPTURE = "    meta_path_snapshot = list(sys.meta_path)\n"
NEW_META_PATH_CAPTURE = (
    "    meta_path_owner = sys.meta_path\n"
    "    if type(meta_path_owner) is not list:\n"
    "        raise MatchedInferAdapterError(\"import finder container differs\")\n"
    "    meta_path_snapshot = list(meta_path_owner)\n"
)
OLD_META_PATH_CLEANUP = (
    """
            if finder is not None:
                while finder in sys.meta_path:
                    sys.meta_path.remove(finder)
            trainer.validate_source_trees = original_validate
    """.lstrip("\n")
)
NEW_META_PATH_CLEANUP = (
    """
            # The context treats any callback mutation of ``sys.meta_path`` as
            # an authority failure.  Restore both the original list object and
            # its exact captured contents even when the callback rebound the
            # public attribute to a non-list object.
            sys.meta_path = meta_path_owner
            meta_path_owner[:] = meta_path_snapshot
            trainer.validate_source_trees = original_validate
    """.lstrip("\n")
)
OLD_META_PATH_SNAPSHOT_CHECK = "            or sys.meta_path != meta_path_snapshot\n"
NEW_META_PATH_SNAPSHOT_CHECK = (
    "            or sys.meta_path is not meta_path_owner\n"
    "            or sys.meta_path != meta_path_snapshot\n"
)
OLD_ACTIVE_META_PATH_CHECK = (
    "                or sys.meta_path != [finder, *meta_path_snapshot]\n"
)
NEW_ACTIVE_META_PATH_CHECK = (
    "                or sys.meta_path is not meta_path_owner\n"
    "                or sys.meta_path != [finder, *meta_path_snapshot]\n"
)


def _commit(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "--all"], check=True)
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


def _build_vendor_fixture(root: Path, *, import_error: bool) -> tuple[Path, Path, str, str]:
    bernini = root / "Bernini"
    veomni = root / "VeOmni"
    (bernini / "bernini").mkdir(parents=True)
    (bernini / "configs/bernini_renderer_wan21_1p3b").mkdir(parents=True)
    (veomni / "veomni").mkdir(parents=True)
    (bernini / "bernini/__init__.py").write_text(
        "ORIGIN = 'captured-bernini'\n", encoding="utf-8"
    )
    pipeline = (
        "raise RuntimeError('activation-import-boom')\n"
        if import_error
        else "ORIGIN = 'captured-pipeline'\n"
    )
    (bernini / "bernini/pipeline.py").write_text(pipeline, encoding="utf-8")
    (bernini / "configs/bernini_renderer_wan21_1p3b/config.json").write_text(
        '{"fixture":true}\n', encoding="utf-8"
    )
    (veomni / "veomni/__init__.py").write_text(
        "ORIGIN = 'captured-veomni'\n", encoding="utf-8"
    )
    return bernini, veomni, _commit(bernini), _commit(veomni)


def _scenario_script(
    *,
    adapter_name: str,
    module_root: Path,
    root: Path,
    bernini: Path,
    veomni: Path,
    bernini_commit: str,
    veomni_commit: str,
    callback_error: bool,
    meta_path_tamper: bool,
    meta_path_rebind: bool,
    meta_path_equal_rebind: bool,
) -> str:
    return textwrap.dedent(
        f"""\
        import importlib,json,os,sys,types
        from pathlib import Path

        sys.path.insert(0,{str(module_root)!r})
        adapter=importlib.import_module({adapter_name!r})
        root=Path({str(root)!r})
        site=root/'site-packages'
        site.mkdir()
        bernini_root=Path({str(bernini)!r})
        veomni_root=Path({str(veomni)!r})
        for name in tuple(sys.modules):
            if name in ('bernini','veomni') or name.startswith(('bernini.','veomni.')):
                sys.modules.pop(name,None)
        sys.path[:]=[str(site)]

        third_party={{}}
        for name in ('torch','diffusers','peft','transformers'):
            path=site/(name+'.py')
            path.write_text('ORIGIN='+repr(name)+'\\n',encoding='utf-8')
            module=types.ModuleType(name)
            module.__file__=str(path)
            module.__package__=''
            module.__cached__=None
            third_party[name]=module

        def fixture_preload(site_root):
            if Path(site_root)!=site or sys.path!=[str(site)]:
                raise RuntimeError('fixture preload path differs')
            sys.modules.update(third_party)
            return {{'schema_version':'fixture-preload','authority_digest':'a'*64}}

        class Redirect:
            def __init__(self):
                self.closed=False
            def install(self,module):
                raise RuntimeError('renderer redirect must not be installed in focused fixture')
            def finalize_authority(self):
                return {{'schema_version':'fixture-redirect','authority_digest':'b'*64}}
            def restore_and_close(self):
                if self.closed:
                    raise RuntimeError('fixture redirect closed twice')
                self.closed=True

        redirects=[]
        def make_redirect(raw,logical_directory):
            if not raw or logical_directory.name!='bernini_renderer_wan21_1p3b':
                raise RuntimeError('fixture config redirect input differs')
            value=Redirect();redirects.append(value);return value

        observations=[]
        activation_calls=[]
        def original_validate(*args,**kwargs):
            raise RuntimeError('unscoped validator executed')
        def original_activate(bernini_value,veomni_value):
            activation_calls.append((str(bernini_value),str(veomni_value)))
            roots=[str(bernini_value),str(veomni_value)]
            for value in roots:
                while value in sys.path: sys.path.remove(value)
            sys.path[0:0]=roots
            module=importlib.import_module('bernini.pipeline')
            spec=getattr(module,'__spec__',None)
            observations.append({{
                'loader_type':type(getattr(module,'__loader__',None)).__name__,
                'spec_loader_type':type(getattr(spec,'loader',None)).__name__,
                'captured_loader':isinstance(getattr(module,'__loader__',None),adapter._CapturedVendorLoader),
                'cached_is_none':getattr(module,'__cached__',None) is None,
                'finder_count':sum(isinstance(value,adapter._CapturedVendorFinder) for value in sys.meta_path),
            }})
            if {callback_error!r}:
                raise RuntimeError('activation-callback-boom')
            if {meta_path_tamper!r}:
                sys.meta_path.append(object())
            if {meta_path_rebind!r}:
                sys.meta_path=tuple(sys.meta_path)+(object(),)
            if {meta_path_equal_rebind!r}:
                sys.meta_path=list(sys.meta_path)

        trainer=types.SimpleNamespace(
            BERNINI_OFFICIAL_COMMIT={bernini_commit!r},
            VEOMNI_TESTED_COMMIT={veomni_commit!r},
            validate_source_trees=original_validate,
            activate_source_trees=original_activate,
        )
        adapter.infer_lora=types.SimpleNamespace(trainer=trainer)
        adapter._preload_pinned_dependencies=fixture_preload
        adapter._create_sealed_renderer_config_redirect=make_redirect
        rank_cache=root/'rank-cache'
        rank_cache.mkdir()
        adapter._EARLY_RANK_CACHE=rank_cache

        path_before=list(sys.path)
        meta_owner_before=sys.meta_path
        meta_before=list(sys.meta_path)
        importer_before=dict(sys.path_importer_cache)
        error=None
        chain=[]
        authority=None
        try:
            with adapter.pinned_dependency_import_paths(site) as authority:
                values=trainer.validate_source_trees(
                    bernini_root,veomni_root,
                    expected_bernini_commit=trainer.BERNINI_OFFICIAL_COMMIT,
                    expected_veomni_commit=trainer.VEOMNI_TESTED_COMMIT,
                )
                trainer.activate_source_trees(values[0],values[1])
                importlib.import_module('veomni')
        except BaseException as caught:
            error=type(caught).__name__+':'+str(caught)
            cursor=caught
            while cursor is not None and len(chain)<8:
                chain.append(type(cursor).__name__+':'+str(cursor))
                cursor=getattr(cursor,'__context__',None)

        restored=(
            sys.path==path_before
            and sys.meta_path is meta_owner_before
            and sys.meta_path==meta_before
            and set(sys.path_importer_cache)==set(importer_before)
            and all(sys.path_importer_cache[key] is value for key,value in importer_before.items())
            and trainer.validate_source_trees is original_validate
            and trainer.activate_source_trees is original_activate
            and not any(name in ('bernini','veomni') or name.startswith(('bernini.','veomni.')) for name in sys.modules)
            and not any(isinstance(value,adapter._CapturedVendorFinder) for value in sys.meta_path)
            and len(redirects)==1 and redirects[0].closed
        )
        result={{
            'error':error,
            'chain':chain,
            'restored':restored,
            'observations':observations,
            'activation_call_count':len(activation_calls),
            'authority_activation_count':None if authority is None else authority.get('activation_call_count'),
            'loaded_rows':[] if authority is None else authority.get('loaded_vendor_modules',[]),
        }}
        print(json.dumps(result,sort_keys=True,separators=(',',':')))
        """
    )


class MatchedInferAdapterV3ActivationImportTests(unittest.TestCase):
    maxDiff = None

    def _run(
        self,
        adapter_name: str,
        *,
        callback_error: bool = False,
        import_error: bool = False,
        meta_path_tamper: bool = False,
        meta_path_rebind: bool = False,
        meta_path_equal_rebind: bool = False,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            self.assertTrue(SEALED_METHOD_FIXTURE.is_dir())
            release = root / "release"
            shutil.copytree(SEALED_METHOD_FIXTURE, release)
            if adapter_name.endswith("_v3"):
                shutil.copy2(V3_SOURCE, release / f"{adapter_name}.py")
            bernini, veomni, bernini_commit, veomni_commit = _build_vendor_fixture(
                root, import_error=import_error
            )
            script = _scenario_script(
                adapter_name=adapter_name,
                module_root=release,
                root=root,
                bernini=bernini,
                veomni=veomni,
                bernini_commit=bernini_commit,
                veomni_commit=veomni_commit,
                callback_error=callback_error,
                meta_path_tamper=meta_path_tamper,
                meta_path_rebind=meta_path_rebind,
                meta_path_equal_rebind=meta_path_equal_rebind,
            )
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                check=False,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            return json.loads(completed.stdout.decode("utf-8", "strict"))

    def test_v3_is_exact_v2_derivative_with_only_finder_order_changed(self) -> None:
        v2 = V2_SOURCE.read_text(encoding="utf-8")
        v3 = V3_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(v2.count(OLD_ACTIVATION_ORDER), 1)
        self.assertEqual(v3.count(NEW_ACTIVATION_ORDER), 1)
        expected = v2.replace(OLD_ACTIVATION_ORDER, NEW_ACTIVATION_ORDER)
        self.assertEqual(expected.count(OLD_META_PATH_CAPTURE), 1)
        expected = expected.replace(OLD_META_PATH_CAPTURE, NEW_META_PATH_CAPTURE)
        self.assertEqual(expected.count(OLD_META_PATH_CLEANUP), 1)
        expected = expected.replace(OLD_META_PATH_CLEANUP, NEW_META_PATH_CLEANUP)
        self.assertEqual(expected.count(OLD_META_PATH_SNAPSHOT_CHECK), 3)
        expected = expected.replace(
            OLD_META_PATH_SNAPSHOT_CHECK, NEW_META_PATH_SNAPSHOT_CHECK
        )
        self.assertEqual(expected.count(OLD_ACTIVE_META_PATH_CHECK), 1)
        expected = expected.replace(
            OLD_ACTIVE_META_PATH_CHECK, NEW_ACTIVE_META_PATH_CHECK
        )
        self.assertEqual(expected, v3)
        for optimize in (0, 1, 2):
            compile(v3, str(V3_SOURCE), "exec", optimize=optimize)

    def test_activation_time_import_is_rejected_by_v2_and_captured_by_v3(self) -> None:
        old = self._run("full644_exploratory_matched_infer_adapter_v2")
        self.assertTrue(old["restored"])
        self.assertEqual(old["activation_call_count"], 1)
        self.assertIn("captured vendor module origin differs: bernini", old["error"])
        self.assertEqual(old["observations"][0]["loader_type"], "SourceFileLoader")
        self.assertFalse(old["observations"][0]["captured_loader"])
        self.assertEqual(old["observations"][0]["finder_count"], 0)

        new = self._run("full644_exploratory_matched_infer_adapter_v3")
        self.assertIsNone(new["error"])
        self.assertTrue(new["restored"])
        self.assertEqual(new["activation_call_count"], 1)
        self.assertEqual(new["authority_activation_count"], 1)
        self.assertTrue(new["observations"][0]["captured_loader"])
        self.assertEqual(new["observations"][0]["loader_type"], "_CapturedVendorLoader")
        self.assertEqual(new["observations"][0]["spec_loader_type"], "_CapturedVendorLoader")
        self.assertTrue(new["observations"][0]["cached_is_none"])
        self.assertEqual(new["observations"][0]["finder_count"], 1)
        self.assertEqual(
            [row["module"] for row in new["loaded_rows"]],
            ["bernini", "bernini.pipeline", "veomni"],
        )

    def test_v3_restores_state_after_callback_and_import_failures(self) -> None:
        callback = self._run(
            "full644_exploratory_matched_infer_adapter_v3", callback_error=True
        )
        self.assertTrue(callback["restored"])
        self.assertEqual(callback["activation_call_count"], 1)
        self.assertTrue(callback["observations"][0]["captured_loader"])
        self.assertTrue(
            any("activation-callback-boom" in value for value in callback["chain"])
        )

        imported = self._run(
            "full644_exploratory_matched_infer_adapter_v3", import_error=True
        )
        self.assertTrue(imported["restored"])
        self.assertEqual(imported["activation_call_count"], 1)
        self.assertEqual(imported["observations"], [])
        self.assertTrue(
            any("activation-import-boom" in value for value in imported["chain"])
        )

        tampered = self._run(
            "full644_exploratory_matched_infer_adapter_v3", meta_path_tamper=True
        )
        self.assertTrue(tampered["restored"])
        self.assertEqual(tampered["activation_call_count"], 1)
        self.assertTrue(tampered["observations"][0]["captured_loader"])
        self.assertTrue(
            any(
                "captured vendor finder changed during source-tree activation" in value
                for value in tampered["chain"]
            )
        )

        rebound = self._run(
            "full644_exploratory_matched_infer_adapter_v3", meta_path_rebind=True
        )
        self.assertTrue(rebound["restored"])
        self.assertEqual(rebound["activation_call_count"], 1)
        self.assertTrue(rebound["observations"][0]["captured_loader"])
        self.assertTrue(
            any(
                "captured vendor finder changed during source-tree activation" in value
                for value in rebound["chain"]
            )
        )

        equal_rebound = self._run(
            "full644_exploratory_matched_infer_adapter_v3",
            meta_path_equal_rebind=True,
        )
        self.assertTrue(equal_rebound["restored"])
        self.assertEqual(equal_rebound["activation_call_count"], 1)
        self.assertTrue(equal_rebound["observations"][0]["captured_loader"])
        self.assertTrue(
            any(
                "captured vendor finder changed during source-tree activation" in value
                for value in equal_rebound["chain"]
            )
        )


if __name__ == "__main__":
    unittest.main()
