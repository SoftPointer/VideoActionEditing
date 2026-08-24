from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from motive.r10_path_contract import (
    R10PathContractError,
    ensure_experiment_directory,
    require_attempt_receipt_path,
    require_experiment_path,
)


class R10PathContractTests(unittest.TestCase):
    def test_exact_rooted_path_and_missing_leaf_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_dir = ensure_experiment_directory(
                root
                / "provenance"
                / "job_attempts"
                / "seed_260108837",
                root,
                "provenance/job_attempts/seed_260108837",
            )
            receipt = receipt_dir / "attempt_1.json"
            self.assertEqual(
                require_attempt_receipt_path(
                    receipt,
                    root,
                    260108837,
                    1,
                    allow_missing=True,
                ),
                receipt,
            )
            receipt.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                require_attempt_receipt_path(
                    receipt,
                    root,
                    260108837,
                    1,
                ),
                receipt,
            )
            with self.assertRaises(R10PathContractError):
                require_attempt_receipt_path(
                    receipt,
                    root,
                    260108837,
                    2,
                )

    def test_rejects_leaf_and_ancestor_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "experiment"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "artifact").mkdir()

            (root / "representation").symlink_to(
                outside,
                target_is_directory=True,
            )
            with self.assertRaises(R10PathContractError):
                require_experiment_path(
                    root / "representation" / "artifact",
                    root,
                    "representation/artifact",
                    kind="dir",
                )

            root_alias = base / "experiment-alias"
            root_alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(R10PathContractError):
                require_experiment_path(
                    root_alias / "representation",
                    root_alias,
                    "representation",
                    kind="dir",
                )
            (root / "representation").unlink()
            (root / "representation").mkdir()
            (root / "representation" / "artifact").symlink_to(
                outside / "artifact",
                target_is_directory=True,
            )
            with self.assertRaises(R10PathContractError):
                require_experiment_path(
                    root / "representation" / "artifact",
                    root,
                    "representation/artifact",
                    kind="dir",
                )

    def test_rejects_noncanonical_and_out_of_root_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "provenance").mkdir()
            with self.assertRaises(R10PathContractError):
                require_experiment_path(
                    root / "provenance" / ".." / "escape",
                    root,
                    "escape",
                    allow_missing=True,
                )
            with self.assertRaises(R10PathContractError):
                require_experiment_path(
                    Path(os.path.dirname(root)) / "escape",
                    root,
                    "escape",
                    allow_missing=True,
                )
            with self.assertRaises(R10PathContractError):
                require_experiment_path(
                    f"{root}//escape",
                    root,
                    "escape",
                    allow_missing=True,
                )
            with self.assertRaises(R10PathContractError):
                require_experiment_path(
                    root,
                    root,
                    ".",
                    kind="dir",
                )


if __name__ == "__main__":
    unittest.main()
