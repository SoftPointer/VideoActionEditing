from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_generic_action_phi_v1_authority_release_v1 as release  # noqa: E402
import materialize_phi_v1_authority_controller_v1 as controller  # noqa: E402


AUTHORING = METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
POPULATION = METHOD_ROOT / "assets/mosaic_event_population_compact6_topup20_v1.json"


def _exact81() -> dict:
    return {"decoder": "tools.materialize_vae._decode_exact_video", "decoder_source_sha256": "d" * 64, "all_integer_frames_0_through_80_decoded": True, "frame_count": 81, "fps": 25, "height": 32, "width": 48, "channels": 3, "dtype": "uint8"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_bytes(controller.canonical_json_bytes(value) + b"\n")


class PhiAuthorityControllerTests(unittest.TestCase):
    def _release(self, root: Path) -> tuple[Path, str]:
        archive = root / "authority-overlay.tar"
        manifest = root / "authority-overlay.json"
        built = release.build(METHOD_ROOT.resolve(), archive, manifest)
        controller._activate_installed_closure(manifest, built["manifest_sha256"])
        return manifest, built["manifest_sha256"]

    def _fit_fixture(self, root: Path) -> tuple[list[dict], dict, dict, Path, Path, str]:
        release_manifest, release_sha = self._release(root)
        authoring = json.loads(AUTHORING.read_text())
        population = json.loads(POPULATION.read_text())
        cells = controller._expected_cells(authoring, population, "fit")
        registered_rows, _ = controller.review_authority._population_context(authoring, population)
        registered = {row["candidate_id"]: row for row in registered_rows}
        generation = {}
        authority_rows = []
        for cell in cells:
            for branch in controller.legacy_phi.ALL_BRANCHES:
                candidate_id = cell["candidate_ids"][branch]
                receipt_path = root / f"{hashlib.sha256(candidate_id.encode()).hexdigest()[:20]}.json"
                receipt_path.write_text("fixture\n")
                receipt_sha = _sha(receipt_path)
                media_sha = hashlib.sha256((candidate_id + "-media").encode()).hexdigest()
                generation[candidate_id] = (receipt_path, {
                    "_file_sha256": receipt_sha,
                    "root_spec_raw_sha256": registered[candidate_id]["root_spec_raw_sha256"],
                    "candidate": {"candidate_id": candidate_id, "analysis_split": "fit", "seed": cell["seed"], "semantic_branch": branch, "calibration_group_id": f"cell-{cell['source_iid']}-s{cell['seed']}"},
                    "artifacts": {"mp4": {"sha256": media_sha}},
                })
                authority_rows.append({
                    "candidate_id": candidate_id, "branch": branch,
                    "source_iid": cell["source_iid"], "seed": cell["seed"],
                    "analysis_split": "fit", "profile_id": "core4-v2" if "core4" in candidate_id else "reserve4-v1",
                    "generation_receipt_file_sha256": receipt_sha,
                    "media_sha256": media_sha,
                    "review_receipt_path": str(root / f"review-{hashlib.sha256(candidate_id.encode()).hexdigest()[:20]}.json"),
                    "review_receipt_file_sha256": hashlib.sha256((candidate_id + "-review").encode()).hexdigest(),
                })
        # Full authority includes confirmation too; these rows are not read by fit plan construction.
        for ordinal in range(80):
            authority_rows.append({"candidate_id": f"confirmation-fixture-{ordinal:03d}", "analysis_split": "confirmation"})
        authority_path = root / "authority.json"; authority_path.write_text("external-authority-fixture\n")
        authority = {"row_count": 160, "authority_digest": "a" * 64, "rows": authority_rows}
        return cells, generation, authority, authority_path, release_manifest, release_sha

    def test_plan_requires_reserve_fit40_not_core4_fit40(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cells, generation, authority, authority_path, release_manifest, release_sha = self._fit_fixture(root)
            core4 = {key: value for key, value in generation.items() if "core4" in key}
            self.assertEqual(len(core4), 40)
            with mock.patch.object(controller.review_authority, "load_authority", return_value=authority), mock.patch.object(controller.review_authority, "_scan_generation", return_value=core4), self.assertRaisesRegex(controller.PhiAuthorityControllerError, "closure is incomplete"):
                controller.build_authorized_plan(authoring_path=AUTHORING, population_path=POPULATION, analysis_split="fit", generation_roots=[root], external_review_authority=authority_path, expected_external_review_authority_sha256="b" * 64, authority_release_manifest=release_manifest, expected_authority_release_manifest_sha256=release_sha, output=root / "plan.json", gap_output=root / "gap.json")
            gap = json.loads((root / "gap.json").read_text())
            self.assertEqual(gap["observed_generation_candidate_count"], 40)
            self.assertEqual(len(gap["missing_generation_candidate_ids"]), 40)
            self.assertFalse(gap["phi_v1_materialization_authorized"])
            self.assertFalse((root / "plan.json").exists())
            self.assertEqual(len(cells), 8)

    def test_plan_rejects_unregistered_generation_receipt_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, generation, authority, authority_path, release_manifest, release_sha = self._fit_fixture(root)
            foreign = root / "foreign.json"; foreign.write_text("foreign\n")
            contaminated = dict(generation)
            contaminated["unregistered-candidate"] = (foreign, {"_file_sha256": _sha(foreign), "candidate": {"candidate_id": "unregistered-candidate"}, "artifacts": {"mp4": {"sha256": "0" * 64}}})
            with mock.patch.object(controller.review_authority, "load_authority", return_value=authority), mock.patch.object(controller.review_authority, "_scan_generation", return_value=contaminated), self.assertRaisesRegex(controller.PhiAuthorityControllerError, "contaminated"):
                controller.build_authorized_plan(authoring_path=AUTHORING, population_path=POPULATION, analysis_split="fit", generation_roots=[root], external_review_authority=authority_path, expected_external_review_authority_sha256="b" * 64, authority_release_manifest=release_manifest, expected_authority_release_manifest_sha256=release_sha, output=root / "plan.json", gap_output=root / "gap.json")
            gap = json.loads((root / "gap.json").read_text())
            self.assertEqual(gap["unexpected_generation_candidate_ids"], ["unregistered-candidate"])
            self.assertFalse((root / "plan.json").exists())

    def _authorized_plan(self, root: Path) -> tuple[dict, dict, dict, Path, str]:
        _, generation, authority, authority_path, release_manifest, release_sha = self._fit_fixture(root)
        with mock.patch.object(controller.review_authority, "load_authority", return_value=authority), mock.patch.object(controller.review_authority, "_scan_generation", return_value=generation):
            plan = controller.build_authorized_plan(authoring_path=AUTHORING, population_path=POPULATION, analysis_split="fit", generation_roots=[root], external_review_authority=authority_path, expected_external_review_authority_sha256="b" * 64, authority_release_manifest=release_manifest, expected_authority_release_manifest_sha256=release_sha, output=root / "plan.json", gap_output=root / "gap.json")
        self.assertFalse((root / "gap.json").exists())
        return dict(plan), authority, generation, release_manifest, release_sha

    def _validate_plan_fixture(self, root: Path, authority: dict, generation: dict, release_manifest: Path, release_sha: str, *, plan_path: Path | None = None):
        generation_by_path = {str(path): value for path, value in generation.values()}
        review_by_path = {
            row["review_receipt_path"]: {
                "candidate_id": row["candidate_id"], "branch": row["branch"],
                "media_sha256": row["media_sha256"],
            }
            for row in authority["rows"] if row.get("analysis_split") == "fit"
        }
        selected = plan_path or (root / "plan.json")
        with mock.patch.object(controller.review_authority, "load_authority", return_value=authority), mock.patch.object(controller.legacy_phi, "_candidate_receipt", side_effect=lambda path: generation_by_path[str(path)]), mock.patch.object(controller.manifests, "validate_review_receipt", side_effect=lambda path, *_args, **_kwargs: review_by_path[str(path)]):
            return controller.validate_authorized_plan(selected, _sha(selected), authority_release_manifest=release_manifest, expected_authority_release_manifest_sha256=release_sha)

    def _forged_plan_with_legacy(self, root: Path, mutate) -> Path:
        plan = json.loads((root / "plan.json").read_text())
        legacy_path = Path(plan["legacy_materializer_plan"]["path"])
        legacy = json.loads(legacy_path.read_text())
        mutate(plan, legacy)
        legacy.pop("plan_digest", None)
        legacy["plan_digest"] = controller.object_sha256(legacy)
        forged_legacy = root / f"forged-legacy-{len(list(root.glob('forged-legacy-*.json')))}.json"
        _json(forged_legacy, legacy)
        plan["legacy_materializer_plan"] = {
            "path": str(forged_legacy), "file_sha256": _sha(forged_legacy),
            "plan_digest": legacy["plan_digest"],
        }
        plan.pop("plan_digest", None)
        plan["plan_digest"] = controller.object_sha256(plan)
        forged_plan = root / f"forged-plan-{len(list(root.glob('forged-plan-*.json')))}.json"
        _json(forged_plan, plan)
        return forged_plan

    def test_validate_plan_rebuilds_exact_split80_and_binds_authority_receipt_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, authority, generation, release_manifest, release_sha = self._authorized_plan(root)
            validated, _, _, _ = self._validate_plan_fixture(root, authority, generation, release_manifest, release_sha)
            self.assertEqual(len(validated["rows"]), 8)
            self.assertEqual(sum(len(row["candidate_ids"]) for row in validated["rows"]), 80)
            hostile = json.loads(json.dumps(authority))
            hostile["rows"][0]["generation_receipt_file_sha256"] = "0" * 64
            with self.assertRaisesRegex(controller.PhiAuthorityControllerError, "generation receipt/pinned authority differs"):
                self._validate_plan_fixture(root, hostile, generation, release_manifest, release_sha)

    def test_validate_plan_rejects_registry_coordinate_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, authority, generation, release_manifest, release_sha = self._authorized_plan(root)
            def mutate(plan: dict, legacy: dict) -> None:
                plan["rows"][0]["seed"] += 1
                legacy["rows"][0]["seed"] += 1
            forged = self._forged_plan_with_legacy(root, mutate)
            with self.assertRaisesRegex(controller.PhiAuthorityControllerError, "pinned cell seed differs"):
                self._validate_plan_fixture(root, authority, generation, release_manifest, release_sha, plan_path=forged)

    def test_validate_plan_rejects_legacy_forward_contract_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, authority, generation, release_manifest, release_sha = self._authorized_plan(root)
            forged = self._forged_plan_with_legacy(root, lambda _plan, legacy: legacy.__setitem__("model_forwards", 81))
            with self.assertRaisesRegex(controller.PhiAuthorityControllerError, "legacy materializer plan binding differs"):
                self._validate_plan_fixture(root, authority, generation, release_manifest, release_sha, plan_path=forged)

    def test_fit_operator_manifest_is_exact16_and_binds_separate_raw_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan, authority, _, release_manifest, release_sha = self._authorized_plan(root)
            output = root / "phi-output"; output.mkdir()
            sidecar_refs = []
            sidecar_values = {}
            captured = []
            q0 = {}
            for cell_index, cell in enumerate(plan["rows"]):
                camera = [0.0] * 672; appearance = [0.0] * 672
                camera[32] = float(cell_index + 1); camera[64] = -float(cell_index + 1)
                appearance[33] = float(cell_index + 11); appearance[65] = -float(cell_index + 11)
                captured.append((camera, appearance))
                camera_sha = hashlib.sha256(struct.pack("<672f", *camera)).hexdigest()
                appearance_sha = hashlib.sha256(struct.pack("<672f", *appearance)).hexdigest()
                source = root / f"q0-{cell_index}.mp4"; source.write_bytes(f"q0-{cell_index}".encode())
                q0[cell["source_iid"]] = {"q0_source_video_path": str(source), "q0_source_video_sha256": _sha(source), "_exact81": _exact81()}
                for branch in controller.legacy_phi.SELECTED_BRANCHES:
                    row_id = f"gaav1:fit:{cell['source_iid']}:s{cell['seed']}:{branch}"
                    ref = {"row_id": row_id, "path": str(root / f"sidecar-{len(sidecar_refs)}.json"), "file_sha256": hashlib.sha256(row_id.encode()).hexdigest(), "receipt_digest": "c" * 64, "review_status": "PASS_SEALED_BEFORE_EXTRACTION"}
                    sidecar_refs.append(ref)
                    quotient = root / f"quotient-{len(sidecar_refs)}.f32le"; quotient.write_bytes(b"\x00" * 2688)
                    sidecar_values[ref["path"]] = {
                        "row_id": row_id, "candidate_id": cell["candidate_ids"][branch],
                        "tensor": {"path": str(quotient), "raw_sha256": _sha(quotient), "dtype": "float32", "byte_order": "little", "shape": [21, 32], "normalization": "global_l2_unit"},
                        "nuisance_projection": {"camera_raw_sha256": camera_sha, "appearance_raw_sha256": appearance_sha},
                        "phi_v1": {"p32_raw_sha256": "d" * 64}, "receipt_digest": "c" * 64,
                    }
            run_unsigned = {
                "schema_version": controller.legacy_phi.RUN_SCHEMA,
                "plan_path": str(root / "legacy-plan.json"), "plan_file_sha256": "e" * 64,
                "plan_digest": "f" * 64, "mode": "OFFICIAL_REVIEWED", "world_size": 4,
                "model_forwards": 82, "sidecar_count": 32, "sidecars": sidecar_refs,
                "p32_raw_path": str(root / "p32.f32le"), "p32_raw_sha256": "d" * 64,
                "bernini_revision": "2d2b4591", "veomni_revision": "f90b3dc6",
                "training_performed": False, "optimizer_created": False,
                "generated_media_is_optimizer_input_or_target": False,
                "optimizer_authorized": False,
            }
            run = {**run_unsigned, "receipt_digest": controller.object_sha256(run_unsigned)}
            _json(output / "phi-v1-sidecar-run-receipt.json", run)

            def save(path: Path, values: list[float]) -> str:
                raw = struct.pack("<672f", *values); path.write_bytes(raw); return hashlib.sha256(raw).hexdigest()

            with mock.patch.object(controller.manifests, "validate_sidecar_receipt", side_effect=lambda path, *_args, **_kwargs: sidecar_values[str(path)]), mock.patch.object(controller, "preflight_q0_sources", return_value=q0), mock.patch.object(controller.legacy_phi, "_save_f32le", side_effect=save):
                manifest = controller.materialize_operator_coordinate_manifest(plan=plan, authority=authority, output_dir=output, captured_cells=captured, q0_authority_path=root / "q0.json")
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual(manifest["row_count"], 16)
            self.assertEqual(manifest["core4_only_operator_rows"], 8)
            self.assertEqual(len(manifest["cell_coordinates"]), 8)
            self.assertEqual(len({row["row_id"] for row in manifest["rows"]}), 16)
            self.assertEqual({row["branch"] for row in manifest["rows"]}, {"action", "incomplete"})
            self.assertTrue(all(row["camera_coordinate"]["path"] != row["appearance_coordinate"]["path"] for row in manifest["rows"]))
            self.assertTrue(all(row["projection_contract"]["weighted_metric_mixing"] is False for row in manifest["rows"]))
            self.assertFalse(manifest["generated_rgb_or_latent_is_editor_input_or_target"])
            manifest_path = output / "phi-v1-operator-coordinate-manifest.json"
            with mock.patch.object(controller.review_authority, "load_authority", return_value=authority), mock.patch.object(controller.review_authority, "_probe_full81", return_value=_exact81()), mock.patch.object(controller, "_q0_rows", return_value=q0), mock.patch.object(controller.manifests, "validate_sidecar_receipt", side_effect=lambda path, *_args, **_kwargs: sidecar_values[str(path)]):
                replayed = controller.validate_operator_coordinate_manifest(manifest_path, _sha(manifest_path), authority_release_manifest=release_manifest, expected_authority_release_manifest_sha256=release_sha)
            self.assertEqual(replayed["manifest_digest"], manifest["manifest_digest"])

            core4_cells = [item for item in zip(plan["rows"], captured) if "core4" in item[0]["candidate_ids"]["action"]]
            core4_plan = {**plan, "rows": [item[0] for item in core4_cells]}
            core4_output = root / "core4-output"; core4_output.mkdir()
            with self.assertRaisesRegex(controller.PhiAuthorityControllerError, "capture closure"):
                controller.materialize_operator_coordinate_manifest(plan=core4_plan, authority=authority, output_dir=core4_output, captured_cells=[item[1] for item in core4_cells], q0_authority_path=root / "q0.json")

    def test_authority_overlay_release_is_deterministic_and_exact_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive1 = root / "one.tar"; manifest1 = root / "one.json"
            archive2 = root / "two.tar"; manifest2 = root / "two.json"
            first = release.build(METHOD_ROOT.resolve(), archive1, manifest1)
            second = release.build(METHOD_ROOT.resolve(), archive2, manifest2)
            self.assertEqual(archive1.read_bytes(), archive2.read_bytes())
            self.assertEqual(manifest1.read_bytes(), manifest2.read_bytes())
            audited = release.audit(archive1, manifest1, first["archive_sha256"], first["manifest_sha256"])
            installed = release.validate_installed_closure(METHOD_ROOT.resolve(), manifest1, first["manifest_sha256"])
            self.assertEqual(audited["file_count"], 6)
            self.assertEqual(audited["authority"]["full_first8_external_review_required"], 160)
            self.assertEqual(audited["authority"]["fit_operator_coordinate_row_count"], 16)
            self.assertEqual(audited["authority"]["core4_only_operator_coordinate_row_count"], 8)
            self.assertTrue(audited["authority"]["precommitted_reviewer_tool_source_required"])
            self.assertTrue(audited["authority"]["signed_execution_credential_required"])
            self.assertTrue(audited["authority"]["authority_replay_full81_decode_required"])
            self.assertTrue(audited["authority"]["pinned_split80_coordinate_reconstruction_required"])
            self.assertFalse(audited["authority"]["private_reviewer_key_embedded"])
            self.assertFalse(audited["authority"]["optimizer_authorized"])
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertEqual(installed["manifest_digest"], audited["manifest_digest"])

    def test_installed_release_rejects_a_forged_required_base_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / "source.tar"; manifest_path = root / "source.json"
            release.build(METHOD_ROOT.resolve(), archive, manifest_path)
            manifest = json.loads(manifest_path.read_text())
            first_base = sorted(manifest["required_base_file_sha256"])[0]
            manifest["required_base_file_sha256"][first_base] = "0" * 64
            manifest.pop("manifest_digest")
            manifest["manifest_digest"] = release.object_sha256(manifest)
            forged = root / "forged.json"
            forged.write_bytes(release.canonical_json_bytes(manifest) + b"\n")
            with self.assertRaisesRegex(release.PhiAuthorityReleaseError, "required base member differs"):
                release.validate_installed_closure(METHOD_ROOT.resolve(), forged, _sha(forged))


if __name__ == "__main__":
    unittest.main()
