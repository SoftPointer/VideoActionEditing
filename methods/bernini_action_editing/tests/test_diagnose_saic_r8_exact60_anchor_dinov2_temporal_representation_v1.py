from __future__ import annotations

from collections import Counter
import copy
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import diagnose_saic_r8_exact60_anchor_dinov2_temporal_representation_v1 as diagnostic


def _rows() -> list[dict]:
    source_cells = [
        ("fit-dog-0", "dog", "fit", (11, 12)),
        ("fit-dog-1", "dog", "fit", (13, 14)),
        ("fit-human-0", "human", "fit", (15, 16)),
        ("fit-human-1", "human", "fit", (17, 18)),
        ("confirmation-dog-0", "dog", "confirmation", (21, 22, 23)),
        ("confirmation-dog-1", "dog", "confirmation", (24, 25, 26)),
        ("confirmation-human-0", "human", "confirmation", (27, 28, 29)),
        ("confirmation-human-1", "human", "confirmation", (30, 31, 32)),
    ]
    branch_vectors = {
        "incomplete": np.asarray([1.0, 0.2, 0.0, 0.0], dtype=np.float32),
        "camera_only": np.asarray([0.0, 1.0, 0.2, 0.0], dtype=np.float32),
        "appearance_only": np.asarray([0.0, 0.0, 1.0, 0.2], dtype=np.float32),
    }
    family_offset = {
        "dog": np.asarray([0.0, 0.0, 0.0, 0.01], dtype=np.float32),
        "human": np.asarray([0.01, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    rows = []
    for iid, family, split, seeds in source_cells:
        for seed in seeds:
            for branch in diagnostic.BRANCHES:
                value = diagnostic.anchor.unit(
                    branch_vectors[branch]
                    + family_offset[family]
                    + np.asarray([0.0, 0.0, 0.0, seed * 1.0e-5], dtype=np.float32)
                )
                rows.append({
                    "candidate_id": f"candidate-{iid}-{seed}-{branch}",
                    "iid": iid,
                    "seed": seed,
                    "branch": branch,
                    "actor_family": family,
                    "analysis_split": split,
                    "representations": {"probe": value},
                })
    return rows


def _manifest_and_master() -> tuple[dict, dict]:
    rows = _rows()
    attempts, master_attempts = [], []
    proofs = []
    for row in rows:
        binding = {
            "candidate_id": row["candidate_id"],
            "iid": row["iid"],
            "seed": row["seed"],
            "branch": row["branch"],
            "actor_family": row["actor_family"],
            "analysis_split": row["analysis_split"],
            "receipt_sha256": hashlib.sha256(
                (row["candidate_id"] + "-receipt").encode()
            ).hexdigest(),
            "mp4_sha256": hashlib.sha256(
                (row["candidate_id"] + "-mp4").encode()
            ).hexdigest(),
            "correct_source": {"iid": row["iid"]},
        }
        attempts.append(binding)
        master_attempts.append({key: binding[key] for key in (
            "candidate_id", "iid", "seed", "branch", "receipt_sha256", "mp4_sha256"
        )})
    for iid, seed in sorted({(row["iid"], row["seed"]) for row in rows}):
        proofs.append({
            "iid": iid,
            "seed": seed,
            "branch_order": list(diagnostic.BRANCHES),
            "official_gaussian_tensor_values_byte_equal": True,
            "official_gaussian_identity_digest": hashlib.sha256(
                f"{iid}:{seed}".encode()
            ).hexdigest(),
        })
    manifest = {"attempts": attempts}
    master = {
        "schema_version": diagnostic.cyclic.MASTER_SCHEMA,
        "receipt_digest": "a" * 64,
        "attempt_count": 60,
        "seed_cell_count": 20,
        "branch_order": list(diagnostic.BRANCHES),
        "same_seed_official_gaussian_proofs": proofs,
        "attempts": master_attempts,
    }
    return manifest, master


class R8TemporalRepresentationContractTest(unittest.TestCase):
    def test_launcher_hard_pins_source_terminal_and_single_gpu_isolation(self) -> None:
        launcher = (
            METHOD_ROOT / "scripts"
            / "auh_diagnose_saic_r8_exact60_anchor_dinov2_temporal_representation_v1.sh"
        ).read_text("utf-8")
        source_sha = hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest()
        self.assertIn(f"readonly expected_diagnostic_sha={source_sha}", launcher)
        self.assertIn(
            "readonly expected_terminal_evidence_sha="
            "07a6ec7ccbe165d89aa8757985537ef18d62eea5d08e245e452b607dee5bd29a",
            launcher,
        )
        self.assertIn('if [[ "$#" -ne 17 ]]', launcher)
        self.assertIn('ROCR_VISIBLE_DEVICES="$logical_gpu"', launcher)
        self.assertIn("MIOPEN_USER_DB_PATH=", launcher)
        self.assertIn("find \"$runtime_scratch\" -xdev -depth -mindepth 1 -delete", launcher)

    def test_dependency_and_upstream_pins_are_exact(self) -> None:
        self.assertEqual(
            hashlib.sha256((METHOD_ROOT / diagnostic.ANCHOR_SOURCE_NAME).read_bytes()).hexdigest(),
            "3a1705dfb44522357bd9f136a40e65eefa468b70c771d1207edf4ff9a47ed2d1",
        )
        self.assertEqual(
            hashlib.sha256((METHOD_ROOT / diagnostic.CYCLIC_SOURCE_NAME).read_bytes()).hexdigest(),
            "2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e",
        )
        self.assertEqual(
            diagnostic.EXPECTED_INPUT_MANIFEST_SHA256,
            "28ff1e40f4dd314548616050013afdfb5e2a2a768aba9f0cbd4f00c9f6718c62",
        )
        self.assertEqual(
            diagnostic.EXPECTED_TERMINAL_EVIDENCE_SHA256,
            "07a6ec7ccbe165d89aa8757985537ef18d62eea5d08e245e452b607dee5bd29a",
        )
        self.assertEqual(
            diagnostic.EXPECTED_TERMINAL_EVIDENCE_RECEIPT_DIGEST,
            "a8fe672840d597445a2164660a38bdfeb4fa51ccfbc3b822c3af8adb6d6519e5",
        )
        self.assertEqual(
            diagnostic.EXPECTED_MASTER_RECEIPT_SHA256,
            "c5528a08fa976c0dbfb16984a35df3169c2d013a73fabd982ad45f45d5defc61",
        )
        self.assertEqual(
            diagnostic.EXPECTED_MASTER_RECEIPT_DIGEST,
            "8d28c170f5c8fdc5e76bdfb55bb89a5a819f02beb483c005f87d6898c5d8ae33",
        )

    def test_registered_semantics_never_relabel_action_noop(self) -> None:
        self.assertEqual(
            diagnostic.BRANCHES,
            ("incomplete", "camera_only", "appearance_only"),
        )
        self.assertEqual(
            diagnostic.PAIRWISE_CONTRASTS,
            (
                ("incomplete", "camera_only"),
                ("incomplete", "appearance_only"),
                ("camera_only", "appearance_only"),
            ),
        )
        source = Path(diagnostic.__file__).read_text("utf-8")
        self.assertNotIn("action_noop_contrast_evaluation(", source)
        self.assertNotIn("typed_subspace_evaluation(", source)
        self.assertIn('"forward_reverse_noop_mapping_performed": False', source)
        self.assertIn('"action_minus_noop_computation_performed": False', source)
        self.assertIn('"typed_action_subspace_computation_performed": False', source)

    def test_exact60_branch_split_and_cell_closure(self) -> None:
        manifest, _ = _manifest_and_master()
        closure = diagnostic._validate_manifest_semantics(manifest)
        self.assertEqual(
            closure["branch_counts"],
            {branch: 20 for branch in diagnostic.BRANCHES},
        )
        self.assertEqual(closure["split_counts"], {"fit": 24, "confirmation": 36})
        self.assertEqual(closure["same_iid_seed_cell_count"], 20)
        hostile = copy.deepcopy(manifest)
        hostile["attempts"][0]["branch"] = "noop"
        with self.assertRaises(diagnostic.R8TemporalRepresentationError):
            diagnostic._validate_manifest_semantics(hostile)

    def test_master_requires_exact20_gaussian_proofs_and_manifest_list(self) -> None:
        manifest, master = _manifest_and_master()
        checked = diagnostic._validate_master(master, "b" * 64, manifest)
        self.assertTrue(checked["all20_same_cell_official_gaussians_byte_equal"])
        hostile = copy.deepcopy(master)
        hostile["same_seed_official_gaussian_proofs"][0][
            "official_gaussian_tensor_values_byte_equal"
        ] = False
        with self.assertRaises(diagnostic.R8TemporalRepresentationError):
            diagnostic._validate_master(hostile, "b" * 64, manifest)
        with self.assertRaises(diagnostic.R8TemporalRepresentationError):
            diagnostic._validate_master(master, "b" * 64, {"attempts": None})

    def test_fit_only_centroid_and_pairwise_diagnostics_cover_confirmation(self) -> None:
        rows = _rows()
        self.assertEqual(Counter(row["branch"] for row in rows), Counter({
            "incomplete": 20, "camera_only": 20, "appearance_only": 20,
        }))
        centroid = diagnostic.branch_label_centroid_diagnostic(rows, "probe")
        self.assertEqual(centroid["fit_row_count"], 24)
        self.assertEqual(centroid["confirmation_row_count"], 36)
        self.assertTrue(centroid["confirmation_rows_never_extend_centroids"])
        self.assertEqual(len(centroid["predictions"]), 36)
        pairwise = diagnostic.pairwise_branch_label_difference_diagnostic(rows, "probe")
        self.assertTrue(pairwise["no_action_or_noop_semantics"])
        self.assertEqual(len(pairwise["contrasts"]), 3)
        for value in pairwise["contrasts"].values():
            self.assertEqual(value["fit_vector_count"], 8)
            self.assertEqual(value["confirmation_vector_count"], 12)
            self.assertTrue(value["same_iid_seed_gaussian_cell_required"])
        nearest = diagnostic.nearest_neighbor_diagnostic(rows, "probe")
        self.assertEqual(nearest["row_count"], 60)

    def test_authority_closure_has_no_ranking_selection_or_training(self) -> None:
        self.assertTrue(diagnostic.AUTHORITY["diagnostic_only"])
        self.assertTrue(diagnostic.AUTHORITY["raw_proxy_evidence_only"])
        for field in (
            "identity_authority", "identity_preservation_verified",
            "event_authority", "event_verified", "scientific_claim_authorized",
            "representation_selection_authorized", "selection_authorized",
            "ranking_authorized", "training_target_authorized",
            "optimizer_or_parameter_update_authorized",
        ):
            self.assertIs(diagnostic.AUTHORITY[field], False)


if __name__ == "__main__":
    unittest.main()
