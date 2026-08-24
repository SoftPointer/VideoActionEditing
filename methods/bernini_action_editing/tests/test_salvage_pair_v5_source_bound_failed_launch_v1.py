from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for root in (METHOD_ROOT, TEST_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import pair_v5_source_bound_preservation_evaluator_v1 as evaluator  # noqa: E402
import salvage_pair_v5_source_bound_failed_launch_v1 as salvage  # noqa: E402
import test_pair_v5_source_bound_preservation_evaluator_v1 as fixtures  # noqa: E402


ROLLOUT_SPEC = METHOD_ROOT / "assets/pair_v5_native_rv2v4_core4_action_population_v1.json"
PATCHED_VALIDATOR = METHOD_ROOT / "pair_v5_source_bound_preservation_evaluator_v1.py"
SALVAGE_IMPLEMENTATION = METHOD_ROOT / "salvage_pair_v5_source_bound_failed_launch_v1.py"
LAUNCHER = METHOD_ROOT / "scripts/auh_salvage_pair_v5_source_bound_failed_launch_v1_cpu.sbatch"


def _write_canonical(path: Path, value: dict) -> str:
    raw = evaluator.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _resign(row: dict, field: str) -> None:
    unsigned = dict(row)
    unsigned.pop(field, None)
    row[field] = evaluator.object_sha256(unsigned)


class LegacyFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run = root / "legacy-run"
        self.run.mkdir()
        self.archive = root / "legacy-method.tar"
        self.salvage_archive = root / "salvage-source.tar"
        self.evaluator_spec = root / "legacy-evaluator-spec.json"
        self.output_dir = root / "salvage-output"
        self.output_dir.mkdir()
        self.output = self.output_dir / "salvage.json"
        self.method_revision = "f" * 40
        self.salvage_revision = "e" * 40

        implementation_raw = b"# sealed legacy evaluator implementation\n"
        contract_raw = PATCHED_VALIDATOR.read_bytes()
        with tarfile.open(
            self.archive,
            "w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": self.method_revision},
        ) as handle:
            for name, raw in (
                (salvage.LEGACY_IMPLEMENTATION_MEMBER, implementation_raw),
                (salvage.LEGACY_CONTRACT_MEMBER, contract_raw),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                member.mode = 0o444
                handle.addfile(member, io.BytesIO(raw))
        archive_sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()

        with tarfile.open(
            self.salvage_archive,
            "w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": self.salvage_revision},
        ) as handle:
            for name, raw in (
                (salvage.SALVAGE_TOOL_MEMBER, SALVAGE_IMPLEMENTATION.read_bytes()),
                (salvage.SALVAGE_VALIDATOR_MEMBER, PATCHED_VALIDATOR.read_bytes()),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                member.mode = 0o444
                handle.addfile(member, io.BytesIO(raw))
        salvage_archive_sha = hashlib.sha256(self.salvage_archive.read_bytes()).hexdigest()

        spec = fixtures._spec()
        spec["implementation_sha256"] = hashlib.sha256(implementation_raw).hexdigest()
        spec["contract_sha256"] = hashlib.sha256(contract_raw).hexdigest()
        spec["method_source_revision"] = self.method_revision
        spec["method_source_archive_sha256"] = archive_sha
        _resign(spec, "spec_digest")
        spec_raw_sha = _write_canonical(self.evaluator_spec, spec)
        self.spec = evaluator.validate_evaluator_spec(spec)

        rows = []
        candidate_file_hashes: dict[str, str] = {}
        for ordinal in range(8):
            row = fixtures._receipt(self.spec, ordinal=ordinal)
            row["evaluator_spec_raw_sha256"] = spec_raw_sha
            _resign(row, "receipt_digest")
            rows.append(evaluator.validate_candidate_receipt(
                row,
                evaluator_spec=self.spec,
                evaluator_spec_raw_sha256=spec_raw_sha,
            ))

        groups = []
        group_file_hashes: dict[str, str] = {}
        for group_id in evaluator.EXPECTED_GROUPS:
            group_dir = self.run / group_id
            group_dir.mkdir()
            selected = [row for row in rows if row["group_id"] == group_id]
            for row in selected:
                path = group_dir / f"{row['candidate_id']}.json"
                candidate_file_hashes[row["candidate_id"]] = _write_canonical(path, row)
            group = evaluator.make_group_receipt(
                evaluator_spec=self.spec,
                evaluator_spec_raw_sha256=spec_raw_sha,
                group_id=group_id,
                candidate_receipts=selected,
                candidate_receipt_file_sha256_by_id={
                    row["candidate_id"]: candidate_file_hashes[row["candidate_id"]]
                    for row in selected
                },
            )
            group_path = group_dir / salvage.LEGACY_GROUP_NAME.format(group_id=group_id)
            group_file_hashes[group_id] = _write_canonical(group_path, group)
            groups.append(group)

        topology = {
            "group_world_size": 4,
            "group_ulysses_size": 4,
            "groups": {
                group_id: evaluator.EXPECTED_GROUP_GPUS[group_id]
                for group_id in evaluator.EXPECTED_GROUPS
            },
            "total_physical_gpus": 8,
            "concurrent_disjoint_groups": True,
        }
        root_receipt = evaluator.make_root_receipt(
            evaluator_spec=self.spec,
            evaluator_spec_raw_sha256=spec_raw_sha,
            group_receipts=groups,
            group_receipt_file_sha256_by_id=group_file_hashes,
            candidate_receipts=rows,
            candidate_receipt_file_sha256_by_id=candidate_file_hashes,
            topology=topology,
        )
        root_sha = _write_canonical(self.run / salvage.LEGACY_ROOT_NAME, root_receipt)
        self.seal = salvage.LegacySeal(
            slurm_job_id=131222,
            evaluator_spec_raw_sha256=spec_raw_sha,
            evaluator_spec_digest=self.spec["spec_digest"],
            method_source_revision=self.method_revision,
            method_source_archive_sha256=archive_sha,
            legacy_implementation_sha256=self.spec["implementation_sha256"],
            legacy_contract_sha256=self.spec["contract_sha256"],
            legacy_root_file_sha256=root_sha,
            legacy_root_digest=root_receipt["root_digest"],
            candidate_order=tuple(root_receipt["candidate_order"]),
            group_receipt_digest_by_id=tuple(
                root_receipt["group_receipt_digest_by_id"].items()
            ),
            group_receipt_file_sha256_by_id=tuple(
                root_receipt["group_receipt_file_sha256_by_id"].items()
            ),
            candidate_receipt_digest_by_id=tuple(
                root_receipt["candidate_receipt_digest_by_id"].items()
            ),
            candidate_receipt_file_sha256_by_id=tuple(
                root_receipt["candidate_receipt_file_sha256_by_id"].items()
            ),
            patched_validator_source_revision=salvage.PATCHED_VALIDATOR_SOURCE_REVISION,
            patched_validator_source_sha256=hashlib.sha256(contract_raw).hexdigest(),
        )
        self.salvage_source_seal = salvage.verify_salvage_source_archive(
            self.salvage_archive,
            expected_source_revision=self.salvage_revision,
            expected_source_archive_sha256=salvage_archive_sha,
            executed_tool_path=SALVAGE_IMPLEMENTATION,
            imported_validator_path=PATCHED_VALIDATOR,
        )

    def audit(self) -> dict:
        return salvage.audit_legacy_run(
            rollout_spec_path=ROLLOUT_SPEC,
            evaluator_spec_path=self.evaluator_spec,
            legacy_method_archive_path=self.archive,
            legacy_run_dir=self.run,
            salvage_source_archive_path=self.salvage_archive,
            executed_tool_path=SALVAGE_IMPLEMENTATION,
            patched_validator_source_path=PATCHED_VALIDATOR,
            salvage_source_seal=self.salvage_source_seal,
            seal=self.seal,
        )


class SourceBoundSalvageTests(unittest.TestCase):
    def test_registered_auh_131222_constants_match_synced_real_receipts_when_available(self) -> None:
        synced_run = Path("/tmp/source_bound_131222")
        synced_inputs = Path("/tmp/source_bound_131222_inputs")
        required = (
            synced_run / salvage.LEGACY_ROOT_NAME,
            synced_inputs / "pair_v5_native_rv2v4_core4_action_population_v1.json",
            synced_inputs / "pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json",
            synced_inputs / "pair_v5_source_bound_preservation_7c4c837_minimal.tar",
        )
        if not all(path.is_file() for path in required):
            self.skipTest("synced immutable AUH 131222 fixture is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            source_archive = Path(temporary) / "salvage-source.tar"
            revision = "c" * 40
            with tarfile.open(
                source_archive,
                "w",
                format=tarfile.PAX_FORMAT,
                pax_headers={"comment": revision},
            ) as handle:
                for name, raw in (
                    (salvage.SALVAGE_TOOL_MEMBER, SALVAGE_IMPLEMENTATION.read_bytes()),
                    (salvage.SALVAGE_VALIDATOR_MEMBER, PATCHED_VALIDATOR.read_bytes()),
                ):
                    member = tarfile.TarInfo(name)
                    member.size = len(raw)
                    handle.addfile(member, io.BytesIO(raw))
            source_seal = salvage.verify_salvage_source_archive(
                source_archive,
                expected_source_revision=revision,
                expected_source_archive_sha256=hashlib.sha256(
                    source_archive.read_bytes()
                ).hexdigest(),
                executed_tool_path=SALVAGE_IMPLEMENTATION,
                imported_validator_path=PATCHED_VALIDATOR,
            )
            receipt = salvage.audit_legacy_run(
                rollout_spec_path=required[1],
                evaluator_spec_path=required[2],
                legacy_method_archive_path=required[3],
                legacy_run_dir=synced_run,
                salvage_source_archive_path=source_archive,
                executed_tool_path=SALVAGE_IMPLEMENTATION,
                patched_validator_source_path=PATCHED_VALIDATOR,
                salvage_source_seal=source_seal,
            )
            self.assertEqual(
                receipt["candidate_order"],
                list(salvage.REGISTERED_AUH_131222.candidate_order),
            )
            self.assertEqual(
                receipt["candidate_receipt_file_sha256_by_id"],
                dict(
                    salvage.REGISTERED_AUH_131222.candidate_receipt_file_sha256_by_id
                ),
            )

    def test_reopens_full_closure_and_emits_no_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LegacyFixture(Path(temporary))
            before = {
                path.relative_to(fixture.run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in fixture.run.rglob("*.json")
            }
            receipt = fixture.audit()
            output_sha = salvage.write_fresh_receipt(
                fixture.output,
                receipt,
                legacy_run_dir=fixture.run,
                salvage_source_seal=fixture.salvage_source_seal,
                seal=fixture.seal,
            )
            after = {
                path.relative_to(fixture.run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in fixture.run.rglob("*.json")
            }
            self.assertEqual(before, after)
            self.assertEqual(receipt["artifact_reopen_counts"], {
                "candidate_receipts": 8, "group_receipts": 2, "root_receipts": 1,
            })
            self.assertTrue(receipt["authority_closure"]["execution_and_raw_evidence_closure_authenticated"])
            self.assertFalse(receipt["authority_closure"]["dino_recomputed"])
            self.assertFalse(receipt["authority_closure"]["absolute_source_preservation_pass"])
            self.assertFalse(receipt["authority_closure"]["optimizer_go"])
            self.assertFalse(receipt["authority_closure"]["training_authorized"])
            self.assertEqual(output_sha, hashlib.sha256(fixture.output.read_bytes()).hexdigest())
            durable = json.loads(fixture.output.read_bytes())
            salvage.validate_salvage_receipt(
                durable,
                salvage_source_seal=fixture.salvage_source_seal,
                seal=fixture.seal,
            )
            repeated = fixture.audit()
            self.assertEqual(
                output_sha,
                salvage.verify_existing_receipt(
                    fixture.output,
                    expected_receipt=repeated,
                    salvage_source_seal=fixture.salvage_source_seal,
                    seal=fixture.seal,
                ),
            )

    def test_output_is_o_excl_and_cannot_enter_legacy_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LegacyFixture(Path(temporary))
            receipt = fixture.audit()
            salvage.write_fresh_receipt(
                fixture.output,
                receipt,
                legacy_run_dir=fixture.run,
                salvage_source_seal=fixture.salvage_source_seal,
                seal=fixture.seal,
            )
            with self.assertRaisesRegex(salvage.SourceBoundSalvageError, "O_EXCL"):
                salvage.write_fresh_receipt(
                    fixture.output,
                    receipt,
                    legacy_run_dir=fixture.run,
                    salvage_source_seal=fixture.salvage_source_seal,
                    seal=fixture.seal,
                )
            forbidden = fixture.run / "salvage.json"
            with self.assertRaisesRegex(salvage.SourceBoundSalvageError, "legacy run tree"):
                salvage.write_fresh_receipt(
                    forbidden,
                    receipt,
                    legacy_run_dir=fixture.run,
                    salvage_source_seal=fixture.salvage_source_seal,
                    seal=fixture.seal,
                )
            self.assertFalse(forbidden.exists())

    def test_noncanonical_or_nested_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LegacyFixture(Path(temporary))
            candidate = next((fixture.run / "sp4-a").glob("pair5-*.json"))
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            with self.assertRaisesRegex(salvage.SourceBoundSalvageError, "canonical JSON"):
                fixture.audit()

    def test_authority_escalation_is_rejected_even_when_resigned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LegacyFixture(Path(temporary))
            forged = copy.deepcopy(fixture.audit())
            forged["authority_closure"]["optimizer_go"] = True
            _resign(forged, "salvage_digest")
            with self.assertRaisesRegex(salvage.SourceBoundSalvageError, "authority closure"):
                salvage.validate_salvage_receipt(
                    forged,
                    salvage_source_seal=fixture.salvage_source_seal,
                    seal=fixture.seal,
                )

    def test_all_provenance_and_exact_map_mutations_fail_after_resigning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LegacyFixture(Path(temporary))
            original = fixture.audit()

            def old_implementation(row: dict) -> None:
                row["legacy_evaluator"]["implementation_sha256"] = "0" * 64

            def old_contract(row: dict) -> None:
                row["legacy_evaluator"]["contract_sha256"] = "1" * 64

            def salvage_tool(row: dict) -> None:
                row["salvage_source"]["tool_member_sha256"] = "2" * 64

            def candidate_id(row: dict) -> None:
                row["candidate_order"][0] = "pair5-forged-candidate"

            mutations = [old_implementation, old_contract, salvage_tool, candidate_id]
            for field in (
                "group_receipt_digest_by_id",
                "group_receipt_file_sha256_by_id",
                "candidate_receipt_digest_by_id",
                "candidate_receipt_file_sha256_by_id",
            ):
                def mutate_map(row: dict, field: str = field) -> None:
                    first = next(iter(row[field]))
                    row[field][first] = "3" * 64

                mutations.append(mutate_map)

            for mutate in mutations:
                with self.subTest(mutation=mutate.__name__):
                    forged = copy.deepcopy(original)
                    mutate(forged)
                    _resign(forged, "salvage_digest")
                    with self.assertRaises(salvage.SourceBoundSalvageError):
                        salvage.validate_salvage_receipt(
                            forged,
                            salvage_source_seal=fixture.salvage_source_seal,
                            seal=fixture.seal,
                        )

    def test_salvage_archive_rejects_unsafe_members_and_member_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case in ("unsafe", "tool-drift"):
                archive = root / f"{case}.tar"
                with tarfile.open(
                    archive,
                    "w",
                    format=tarfile.PAX_FORMAT,
                    pax_headers={"comment": "d" * 40},
                ) as handle:
                    members = [
                        (
                            salvage.SALVAGE_TOOL_MEMBER,
                            b"# drift\n" if case == "tool-drift" else SALVAGE_IMPLEMENTATION.read_bytes(),
                        ),
                        (salvage.SALVAGE_VALIDATOR_MEMBER, PATCHED_VALIDATOR.read_bytes()),
                    ]
                    if case == "unsafe":
                        members.append(("../escape", b"bad"))
                    for name, raw in members:
                        member = tarfile.TarInfo(name)
                        member.size = len(raw)
                        handle.addfile(member, io.BytesIO(raw))
                with self.subTest(case=case), self.assertRaises(
                    salvage.SourceBoundSalvageError
                ):
                    salvage.verify_salvage_source_archive(
                        archive,
                        expected_source_revision="d" * 40,
                        expected_source_archive_sha256=hashlib.sha256(
                            archive.read_bytes()
                        ).hexdigest(),
                        executed_tool_path=SALVAGE_IMPLEMENTATION,
                        imported_validator_path=PATCHED_VALIDATOR,
                    )

    def test_runtime_source_is_standard_library_json_only(self) -> None:
        source = SALVAGE_IMPLEMENTATION.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import av", "transformers", "AutoModel", "DinoV2Model"):
            self.assertNotIn(forbidden, source)
        self.assertIn("os.O_EXCL", source)
        self.assertIn('"optimizer_go": False', source)
        self.assertIn('"dino_recomputed": False', source)

    def test_auh_launcher_is_receipt_only_and_hides_minimum_gpu_allocation(self) -> None:
        import subprocess

        syntax = subprocess.run(
            ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:1", source)
        self.assertIn(
            "unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL",
            source,
        )
        self.assertIn("compute=cpu-json-only", source)
        self.assertIn("legacy_job=131222", source)
        self.assertIn("absolute_preservation_pass=false optimizer_go=false", source)
        for token in (
            "PurePosixPath",
            "path.is_absolute()",
            '".." in path.parts',
            "member.issym()",
            "member.islnk()",
            "member.isdev()",
            "member.isfifo()",
            "tool_member_sha256",
            "validator_member_sha256",
            "executed tool/archive member differs",
            "imported validator/archive member differs",
            "--salvage-source-archive",
            "--salvage-source-revision",
            "--salvage-source-archive-sha256",
            "--verify-existing-output",
        ):
            self.assertIn(token, source)
        self.assertEqual(source.count('--rollout-spec "${rollout_spec}"'), 2)
        for forbidden in ("torchrun", "srun", "score_pair_v5_source_bound_preservation_v1.py"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
