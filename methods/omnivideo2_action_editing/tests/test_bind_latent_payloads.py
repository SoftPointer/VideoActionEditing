from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pact.dataset import (  # noqa: E402
    PAYLOAD_FORMAT,
    PAYLOAD_PROVENANCE_BINDINGS,
    AtomicLatentDataset,
    encoder_contract_sha256,
)
from pact.manifest import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
)
from tests.test_manifest import authorized_atom_fixture, track  # noqa: E402
from tests.test_dataset import encoder_contract  # noqa: E402
from tools.bind_latent_payloads import (  # noqa: E402
    PayloadBindingError,
    bind_latent_payloads,
)


_DEFAULT_ATOM_ID = object()


def payload(atom: dict, *, atom_id: object = _DEFAULT_ATOM_ID) -> dict:
    shape = (16, 3, 4, 5)
    value = {
        "format": PAYLOAD_FORMAT,
        "encoder_contract": encoder_contract(),
        "source_latent": torch.randn(shape),
        "global_target_latent": torch.randn(shape),
        "source_component_mask": torch.zeros(1, *shape[1:]),
        "target_component_mask": torch.ones(1, *shape[1:]),
        "text_context": torch.randn(2, 4096),
        "vlm_context": torch.randn(3, 2048),
    }
    for payload_field, atomic_field in PAYLOAD_PROVENANCE_BINDINGS:
        value[payload_field] = atom[atomic_field]
    if atom_id is _DEFAULT_ATOM_ID:
        value["atom_id"] = atom["atom_id"]
    elif atom_id is not None:
        value["atom_id"] = atom_id
    return value


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")


def atomize(media_root: Path, *tracks: dict) -> list[dict]:
    media_root.mkdir(parents=True)
    return authorized_atom_fixture(media_root, list(tracks))


class BindLatentPayloadsTest(unittest.TestCase):
    def test_binds_complete_set_with_canonical_relative_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atoms = atomize(
                root / "media", track("subject_02"), track("subject_01")
            )
            manifest = root / "atomic.jsonl"
            write_manifest(manifest, list(reversed(atoms)))
            payload_root = root / "payloads"
            payload_root.mkdir()
            for atom in atoms:
                torch.save(payload(atom), payload_root / f"{atom['atom_id']}.pt")

            output = root / "bound"
            summary = bind_latent_payloads(manifest, payload_root, output)

            self.assertEqual(summary["payload_files"], 2)
            self.assertEqual(
                summary["encoder_contract_sha256"],
                encoder_contract_sha256(encoder_contract()),
            )
            self.assertEqual(summary["atomic_input_files_verified"], 6)
            self.assertEqual(summary["payload_root"], str(payload_root.resolve()))
            self.assertEqual(
                summary["payload_provenance_bindings"],
                dict(PAYLOAD_PROVENANCE_BINDINGS),
            )
            self.assertTrue(summary["strict_one_to_one_payload_set"])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"training_manifest.jsonl", "summary.json", "done.json"},
            )
            manifest_lines = (output / "training_manifest.jsonl").read_bytes().splitlines()
            bound_rows = [json.loads(line) for line in manifest_lines]
            self.assertEqual(
                [row["atom_id"] for row in bound_rows],
                sorted(atom["atom_id"] for atom in atoms),
            )
            for line, row in zip(manifest_lines, bound_rows):
                self.assertEqual(line, canonical_json_bytes(row))
                self.assertFalse(Path(row["latent_payload_path"]).is_absolute())
                self.assertNotIn("..", Path(row["latent_payload_path"]).parts)
                self.assertEqual(
                    row["latent_payload_path"], f"{row['atom_id']}.pt"
                )
                payload_path = (payload_root / row["latent_payload_path"]).resolve()
                self.assertEqual(row["latent_payload_sha256"], file_sha256(payload_path))
                self.assertEqual(row["latent_payload_format"], PAYLOAD_FORMAT)

            for filename in ("summary.json", "done.json"):
                raw = (output / filename).read_bytes()
                self.assertEqual(
                    raw,
                    canonical_json_bytes(json.loads(raw.decode("utf-8"))) + b"\n",
                )
            done = json.loads((output / "done.json").read_text(encoding="utf-8"))
            self.assertEqual(
                done["training_manifest_sha256"],
                file_sha256(output / "training_manifest.jsonl"),
            )

            dataset = AtomicLatentDataset(
                output / "training_manifest.jsonl", payload_root=payload_root
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                {dataset[index]["atom_id"] for index in range(len(dataset))},
                {atom["atom_id"] for atom in atoms},
            )

    def test_rejects_mixed_encoder_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atoms = atomize(
                root / "media", track("subject_01"), track("subject_02")
            )
            manifest = root / "atomic.jsonl"
            write_manifest(manifest, atoms)
            payload_root = root / "payloads"
            payload_root.mkdir()
            first = payload(atoms[0])
            second = payload(atoms[1])
            second["encoder_contract"] = encoder_contract(vlm_digest="4" * 64)
            torch.save(first, payload_root / f"{atoms[0]['atom_id']}.pt")
            torch.save(second, payload_root / f"{atoms[1]['atom_id']}.pt")
            output = root / "bound"
            with self.assertRaisesRegex(PayloadBindingError, "mixes incompatible"):
                bind_latent_payloads(manifest, payload_root, output)
            self.assertFalse(output.exists())

    def test_rejects_missing_and_unexpected_payloads_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = atomize(root / "media", track())[0]
            manifest = root / "atomic.jsonl"
            write_manifest(manifest, [atom])
            payload_root = root / "payloads"
            payload_root.mkdir()

            missing_output = root / "missing-output"
            with self.assertRaisesRegex(PayloadBindingError, "missing=.*component_01"):
                bind_latent_payloads(manifest, payload_root, missing_output)
            self.assertFalse(missing_output.exists())

            expected_path = payload_root / f"{atom['atom_id']}.pt"
            torch.save(payload(atom), expected_path)
            torch.save(payload(atom, atom_id="unexpected"), payload_root / "unexpected.pt")
            unexpected_output = root / "unexpected-output"
            with self.assertRaisesRegex(PayloadBindingError, "unexpected=.*unexpected.pt"):
                bind_latent_payloads(manifest, payload_root, unexpected_output)
            self.assertFalse(unexpected_output.exists())

    def test_requires_exact_payload_atom_id_and_valid_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = atomize(root / "media", track())[0]
            manifest = root / "atomic.jsonl"
            write_manifest(manifest, [atom])
            payload_root = root / "payloads"
            payload_root.mkdir()
            payload_path = payload_root / f"{atom['atom_id']}.pt"

            torch.save(payload(atom, atom_id=None), payload_path)
            with self.assertRaisesRegex(PayloadBindingError, "atom_id"):
                bind_latent_payloads(manifest, payload_root, root / "missing-id")

            bad = payload(atom)
            bad["text_context"] = torch.randn(2, 123)
            torch.save(bad, payload_path)
            with self.assertRaisesRegex(PayloadBindingError, "4096"):
                bind_latent_payloads(manifest, payload_root, root / "bad-contract")

            wrong_provenance = payload(atom)
            wrong_provenance["edit_instruction_sha256"] = "0" * 64
            torch.save(wrong_provenance, payload_path)
            with self.assertRaisesRegex(
                PayloadBindingError, "edit_instruction_sha256 does not match"
            ):
                bind_latent_payloads(manifest, payload_root, root / "bad-provenance")

            legacy = payload(atom)
            for key in (
                "source_video_sha256",
                "global_counterfactual_target_video_sha256",
                "source_component_mask_sha256",
                "target_component_mask_sha256",
                "track_record_sha256",
            ):
                del legacy[key]
            torch.save(legacy, payload_path)
            with self.assertRaisesRegex(PayloadBindingError, "payload fields missing"):
                bind_latent_payloads(manifest, payload_root, root / "legacy-bypass")

    def test_rejects_swapped_video_mask_and_track_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = atomize(root / "media", track())[0]
            manifest = root / "atomic.jsonl"
            write_manifest(manifest, [atom])
            payload_root = root / "payloads"
            payload_root.mkdir()
            payload_path = payload_root / f"{atom['atom_id']}.pt"

            swapped_video = payload(atom)
            swapped_video["source_video_sha256"], swapped_video[
                "global_counterfactual_target_video_sha256"
            ] = (
                swapped_video["global_counterfactual_target_video_sha256"],
                swapped_video["source_video_sha256"],
            )
            swapped_masks = payload(atom)
            swapped_masks["source_component_mask_sha256"], swapped_masks[
                "target_component_mask_sha256"
            ] = (
                swapped_masks["target_component_mask_sha256"],
                swapped_masks["source_component_mask_sha256"],
            )
            copied_track = payload(atom)
            copied_track["track_record_sha256"] = copied_track[
                "parent_row_sha256"
            ]
            cases = (
                ("video", "source_video_sha256", swapped_video),
                ("mask", "source_component_mask_sha256", swapped_masks),
                ("track", "track_record_sha256", copied_track),
            )
            for label, expected_field, wrong_payload in cases:
                with self.subTest(label=label):
                    torch.save(wrong_payload, payload_path)
                    with self.assertRaisesRegex(
                        PayloadBindingError, expected_field
                    ):
                        bind_latent_payloads(
                            manifest, payload_root, root / f"bad-{label}"
                        )

    def test_rejects_unauthorized_duplicate_rebound_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = atomize(root / "media", track())[0]
            manifest = root / "atomic.jsonl"
            payload_root = root / "payloads"
            payload_root.mkdir()
            torch.save(
                payload(atom), payload_root / f"{atom['atom_id']}.pt"
            )

            unauthorized = copy.deepcopy(atom)
            unauthorized["training_authorized"] = False
            unauthorized["training_use_forbidden"] = True
            unauthorized["parent_preview_only"] = True
            unauthorized.pop("post_generation_release")
            write_manifest(manifest, [unauthorized])
            with self.assertRaisesRegex(PayloadBindingError, "not authorized"):
                bind_latent_payloads(manifest, payload_root, root / "unauthorized")

            write_manifest(manifest, [atom, atom])
            with self.assertRaisesRegex(PayloadBindingError, "duplicate atom_id"):
                bind_latent_payloads(manifest, payload_root, root / "duplicate")

            rebound = dict(atom)
            rebound["latent_payload_path"] = "old.pt"
            rebound["latent_payload_sha256"] = "0" * 64
            write_manifest(manifest, [rebound])
            with self.assertRaisesRegex(PayloadBindingError, "already payload-bound"):
                bind_latent_payloads(manifest, payload_root, root / "rebound")

            write_manifest(manifest, [atom])
            output = root / "already-exists"
            output.mkdir()
            with self.assertRaisesRegex(PayloadBindingError, "already exists"):
                bind_latent_payloads(manifest, payload_root, output)

    def test_rejects_payload_symlink_outside_declared_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom = atomize(root / "media", track())[0]
            manifest = root / "atomic.jsonl"
            write_manifest(manifest, [atom])
            external = root / "external.pt"
            torch.save(payload(atom), external)
            payload_root = root / "payloads"
            payload_root.mkdir()
            (payload_root / f"{atom['atom_id']}.pt").symlink_to(external)

            output = root / "bound"
            with self.assertRaisesRegex(PayloadBindingError, "symlinks are forbidden"):
                bind_latent_payloads(manifest, payload_root, output)
            self.assertFalse(output.exists())

    def test_rehashes_all_atomic_inputs_and_rejects_input_symlink(self) -> None:
        digest_fields = {
            "source_video_path": "source_video_sha256",
            "global_counterfactual_target_video_path": (
                "global_counterfactual_target_video_sha256"
            ),
            "source_component_mask_path": "source_component_mask_sha256",
            "target_component_mask_path": "target_component_mask_sha256",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (path_field, digest_field) in enumerate(digest_fields.items()):
                with self.subTest(path_field=path_field):
                    case_root = root / f"case-{index}"
                    case_root.mkdir()
                    atom = atomize(case_root / "media", track())[0]
                    manifest = case_root / "atomic.jsonl"
                    write_manifest(manifest, [atom])
                    payload_root = case_root / "payloads"
                    payload_root.mkdir()
                    torch.save(
                        payload(atom), payload_root / f"{atom['atom_id']}.pt"
                    )
                    Path(atom[path_field]).write_bytes(b"replaced-after-atomization")
                    output = case_root / "bound"
                    with self.assertRaisesRegex(
                        PayloadBindingError, f"atomic input {digest_field} differs"
                    ):
                        bind_latent_payloads(manifest, payload_root, output)
                    self.assertFalse(output.exists())

            symlink_root = root / "symlink-case"
            symlink_root.mkdir()
            atom = atomize(symlink_root / "media", track())[0]
            manifest = symlink_root / "atomic.jsonl"
            write_manifest(manifest, [atom])
            payload_root = symlink_root / "payloads"
            payload_root.mkdir()
            torch.save(payload(atom), payload_root / f"{atom['atom_id']}.pt")
            source_mask = Path(atom["source_component_mask_path"])
            external = symlink_root / "same-mask-bytes.pt"
            external.write_bytes(source_mask.read_bytes())
            source_mask.unlink()
            source_mask.symlink_to(external)
            with self.assertRaisesRegex(
                PayloadBindingError, "regular non-symlink"
            ):
                bind_latent_payloads(
                    manifest, payload_root, symlink_root / "bound"
                )


if __name__ == "__main__":
    unittest.main()
