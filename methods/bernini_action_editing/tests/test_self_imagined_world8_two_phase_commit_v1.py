from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import mosaic_starc_stateless_jacobian_qp as qp
    import self_imagined_world8_two_phase_commit_v1 as world8

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    qp = None  # type: ignore[assignment]
    world8 = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _FakeCollectiveWorld:
    def __init__(
        self,
        *,
        fail_ordinal: int | None = None,
        fail_exception=RuntimeError,
    ) -> None:
        self.size = 8
        self.fail_ordinal = fail_ordinal
        self.fail_exception = fail_exception
        self.condition = threading.Condition()
        self.slots = {}
        self.readers = {}

    def endpoint(self, rank: int):
        return _FakeCollective(self, rank)

    def exchange(self, key, rank, value, src):
        if self.fail_ordinal == key[0]:
            raise self.fail_exception(f"injected collective failure {key}")
        with self.condition:
            slot = self.slots.setdefault(key, {})
            slot[rank] = value
            self.condition.notify_all()
            if not self.condition.wait_for(
                lambda: len(self.slots[key]) == self.size, timeout=30.0
            ):
                raise TimeoutError(f"fake collective timed out: {key}")
            ordered = tuple(self.slots[key][item] for item in range(self.size))
            result = ordered if src is None else ordered[src]
            self.readers[key] = self.readers.get(key, 0) + 1
            if self.readers[key] == self.size:
                del self.slots[key]
                del self.readers[key]
            return result


class _FakeCollective:
    def __init__(self, world: _FakeCollectiveWorld, rank: int) -> None:
        self.world = world
        self._rank = rank
        self.ordinal = 0

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return 8

    def all_gather_small(self, value):
        key = (self.ordinal, "all_gather")
        self.ordinal += 1
        return self.world.exchange(key, self.rank, value, None)

    def broadcast_small(self, value, *, src):
        key = (self.ordinal, "broadcast")
        self.ordinal += 1
        return self.world.exchange(key, self.rank, value, src)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class World8ShadowTransactionTests(unittest.TestCase):
    checkpoint_digest = _sha("checkpoint-content")
    topology_digest = _sha("topology")
    verifier_policy = None

    def setUp(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.verifier_private_key = Ed25519PrivateKey.from_private_bytes(
            bytes(range(32))
        )
        public_key_hex = self.verifier_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        self.verifier_policy = world8.ExternalVerifierPolicy(
            verifier_id="frozen-external-verifier-v1",
            verifier_executable_sha256=_sha("verifier-executable"),
            verifier_ed25519_public_key_hex=public_key_hex,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def immutable_file(self, name: str, raw: bytes):
        path = self.root / name
        path.write_bytes(raw)
        path.chmod(0o444)
        return world8.CanonicalFileRef(path, hashlib.sha256(raw).hexdigest())

    def external_receipt(self, label: str, audit_type: str, bindings):
        artifact = self.immutable_file(
            f"{label}.artifact", f"artifact:{label}".encode("ascii")
        )
        unsigned = {
            "schema_version": world8.EXTERNAL_AUDIT_SCHEMA,
            "verifier_id": self.verifier_policy.verifier_id,
            "verifier_executable_sha256": (
                self.verifier_policy.verifier_executable_sha256
            ),
            "audit_type": audit_type,
            "verdict": "PASS",
            "artifact_sha256": artifact.sha256,
            "artifact_size": len(f"artifact:{label}".encode("ascii")),
            "bindings": bindings,
        }
        receipt = {
            **unsigned,
            "signature_ed25519": self.verifier_private_key.sign(
                world8.canonical_json_bytes(unsigned)
            ).hex(),
        }
        raw = world8.canonical_json_bytes(receipt)
        receipt_ref = self.immutable_file(f"{label}.receipt.json", raw)
        return world8.ExternalReceiptRef(receipt_ref, artifact)

    def parameters(self):
        all_b = []
        all_a = []
        all_trust = []
        for _ in range(8):
            b_rows = [
                (
                    name,
                    torch.zeros(
                        qp.CANONICAL_B_SHAPE,
                        dtype=torch.float32,
                        requires_grad=True,
                    ),
                )
                for name in qp.CANONICAL_PARAMETER_NAMES
            ]
            a_rows = []
            trust = []
            for name in qp.CANONICAL_PARAMETER_NAMES:
                a_name = name.replace(
                    "action_lora_b.weight", "action_lora_a.weight"
                )
                value = torch.zeros(qp.CANONICAL_A_SHAPE, dtype=torch.float32)
                for ordinal in range(qp.LORA_RANK):
                    value[ordinal, ordinal] = 1.0
                a_rows.append((a_name, value))
                trust.append(
                    qp.LayerTrustRadius(
                        parameter_name=name,
                        fixed_lora_a_parameter_name=a_name,
                        fixed_lora_a=value.clone(),
                        maximum_relative_delta=1.0,
                        reference_effective_weight_norm=1.0,
                        fixed_gauge_receipt_digest=_sha("fixed-gauge"),
                        reference_weight_receipt_digest=_sha(f"reference-{name}"),
                    )
                )
            all_b.append(b_rows)
            all_a.append(a_rows)
            all_trust.append(tuple(trust))
        return all_b, all_a, all_trust

    def checkpoint_manifest(self, b_rows, a_rows):
        payload = world8.checkpoint_manifest_payload(
            checkpoint_content_receipt_digest=self.checkpoint_digest,
            ordered_parameters=b_rows,
            ordered_fixed_lora_a=a_rows,
        )
        return self.immutable_file(
            "checkpoint-ab-manifest.json", world8.canonical_json_bytes(payload)
        )

    def evidence(self, b_rows):
        layout = qp.FixedParameterLayout.from_ordered_parameters(b_rows)
        per_tensor = qp.HIDDEN_SIZE * qp.LORA_RANK

        def vector(index):
            value = torch.zeros(layout.total_numel, dtype=torch.float32)
            value[index] = 1.0
            return value

        actor_rows = {}
        for actor in world8.ARM_ORDER:
            action = qp.ActionConstraintRow(
                row_id=f"{actor}-action",
                actor_family=actor,
                values=vector(0 if actor == "dog" else per_tensor),
                minimum_dot=0.1,
                layout_digest=layout.layout_digest,
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                parameter_state_sha256=layout.parameter_state_sha256,
                gradient_computation_receipt_digest=_sha(f"{actor}-action-gradient"),
            )
            base = 2 if actor == "dog" else 8
            preservation = tuple(
                qp.PreservationConstraintRow(
                    row_id=f"{actor}-{family}",
                    family=family,
                    values=vector((base + ordinal) * per_tensor),
                    maximum_absolute_dot=0.02,
                    layout_digest=layout.layout_digest,
                    checkpoint_content_receipt_digest=self.checkpoint_digest,
                    parameter_state_sha256=layout.parameter_state_sha256,
                    gradient_computation_receipt_digest=_sha(
                        f"{actor}-{family}-gradient"
                    ),
                )
                for ordinal, family in enumerate(world8.PRESERVATION_FAMILIES)
            )
            actor_rows[actor] = ((action,), preservation)
        result = []
        for rank in range(8):
            actor = "dog" if rank < 4 else "human"
            action, preservation = actor_rows[actor]
            result.append(
                qp.SPRankEvidence(
                    global_rank=rank,
                    action_rows=action,
                    preservation_rows=preservation,
                    rank_evidence_receipt_digest=_sha(f"rank-{rank}-evidence"),
                )
            )
        return result

    def gate_files(
        self,
        evidence,
        checkpoint_manifest,
        fresh_inputs,
        b_rows,
        a_rows,
        trust,
        *,
        config=None,
    ):
        config = config or qp.JacobianQPConfig()
        union = qp.DP2SP4Evidence(
            dp_arms=(
                qp.DPArmEvidence("dog", tuple(evidence[:4])),
                qp.DPArmEvidence("human", tuple(evidence[4:])),
            ),
            topology_receipt_digest=self.topology_digest,
        )
        qp_bindings = world8.build_signed_qp_contract_bindings(
            checkpoint_manifest=checkpoint_manifest,
            ordered_parameters=b_rows,
            ordered_fixed_lora_a=a_rows,
            evidence=union,
            topology_receipt_digest=self.topology_digest,
            global_trust_radius=1.0,
            layer_trust_radii=trust,
            config=config,
        )
        qp_contract = self.external_receipt(
            "qp-contract", "qp_contract", qp_bindings
        )
        arms = []
        for actor, first_rank in (("dog", evidence[0]), ("human", evidence[4])):
            owner_bindings = {
                "actor_family": actor,
                "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
                "checkpoint_content_receipt_digest": self.checkpoint_digest,
            }
            owner = self.external_receipt(
                f"{actor}-owner", "owner_audit", owner_bindings
            )
            seed_gates = []
            query_hashes = []
            for seed in (11, 29):
                common = {
                    "actor_family": actor,
                    "query_seed": seed,
                    "owner_receipt_sha256": owner.receipt.sha256,
                    "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
                }
                specificity = self.external_receipt(
                    f"{actor}-{seed}-specificity",
                    "two_seed_specificity",
                    common,
                )
                direction = self.external_receipt(
                    f"{actor}-{seed}-direction",
                    "plus_q_minus_q_direction",
                    {
                        **common,
                        "specificity_receipt_sha256": specificity.receipt.sha256,
                    },
                )
                seed_gates.append(
                    world8.QuerySeedGateFiles(seed, specificity, direction)
                )
                query_hashes.append(
                    [specificity.receipt.sha256, direction.receipt.sha256]
                )
            action_refs = tuple(
                self.external_receipt(
                    f"{actor}-action-{ordinal}",
                    "action_gradient",
                    {
                        "actor_family": actor,
                        "row_id": row.row_id,
                        "row_sha256": world8.tensor_sha256(row.values),
                        "gradient_computation_receipt_digest": (
                            row.gradient_computation_receipt_digest
                        ),
                        "minimum_dot": float(row.minimum_dot),
                        "layout_digest": row.layout_digest,
                        "checkpoint_content_receipt_digest": (
                            row.checkpoint_content_receipt_digest
                        ),
                        "parameter_state_sha256": row.parameter_state_sha256,
                        "owner_receipt_sha256": owner.receipt.sha256,
                        "query_gate_receipt_sha256s": query_hashes,
                        "qp_contract_receipt_sha256": qp_contract.receipt.sha256,
                        "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
                    },
                )
                for ordinal, row in enumerate(first_rank.action_rows)
            )
            preservation_refs = tuple(
                self.external_receipt(
                    f"{actor}-preservation-{ordinal}",
                    "preservation_gradient",
                    {
                        "actor_family": actor,
                        "family": row.family,
                        "row_id": row.row_id,
                        "row_sha256": world8.tensor_sha256(row.values),
                        "gradient_computation_receipt_digest": (
                            row.gradient_computation_receipt_digest
                        ),
                        "maximum_absolute_dot": float(row.maximum_absolute_dot),
                        "layout_digest": row.layout_digest,
                        "checkpoint_content_receipt_digest": (
                            row.checkpoint_content_receipt_digest
                        ),
                        "parameter_state_sha256": row.parameter_state_sha256,
                        "qp_contract_receipt_sha256": qp_contract.receipt.sha256,
                        "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
                    },
                )
                for ordinal, row in enumerate(first_rank.preservation_rows)
            )
            arms.append(
                world8.ArmScientificGateFiles(
                    actor,
                    owner,
                    tuple(seed_gates),
                    action_refs,
                    preservation_refs,
                )
            )
        plan = self.external_receipt(
            "fresh-plan",
            "fresh_exact81_plan",
            {
                "checkpoint_manifest_sha256": checkpoint_manifest.sha256,
                "checkpoint_content_receipt_digest": self.checkpoint_digest,
                "qp_contract_receipt_sha256": qp_contract.receipt.sha256,
                "source_manifest_sha256": fresh_inputs.source_manifest.sha256,
                "official_gaussian_sha256": fresh_inputs.official_gaussian.sha256,
                "exact_frame_count": 81,
                "actor_families": list(world8.ARM_ORDER),
                "preservation_families": list(world8.PRESERVATION_FAMILIES),
                "fresh_rollout_required": True,
                "source_disjoint_confirmation_required": True,
                "rollback_on_any_failure": True,
                "post_hoc_selection_allowed": False,
            },
        )
        return world8.ScientificGateFiles(
            self.verifier_policy, tuple(arms), qp_contract, plan
        )

    def fixture(self):
        b_rows, a_rows, trust = self.parameters()
        checkpoint = self.checkpoint_manifest(b_rows[0], a_rows[0])
        evidence = self.evidence(b_rows[0])
        source = self.immutable_file("source-manifest.json", b"source-manifest")
        gaussian = self.immutable_file("official-gaussian.bin", b"gaussian")
        fresh_inputs = world8.FreshInputFiles(source, gaussian)
        gates = self.gate_files(
            evidence,
            checkpoint,
            fresh_inputs,
            b_rows[0],
            a_rows[0],
            trust[0],
        )
        transaction = self.root / "transaction"
        world8.create_transaction_directory(
            transaction,
            transaction_id="tx-test",
            verifier_policy=self.verifier_policy,
        )
        return {
            "b": b_rows,
            "a": a_rows,
            "trust": trust,
            "checkpoint": checkpoint,
            "evidence": evidence,
            "fresh_inputs": fresh_inputs,
            "gates": gates,
            "transaction": transaction,
        }

    def run_prepare(self, fixture, *, world=None, checkpoint_by_rank=None):
        fake = world or _FakeCollectiveWorld()
        checkpoint_by_rank = checkpoint_by_rank or [fixture["checkpoint"]] * 8

        def worker(rank):
            return world8.prepare_world8_shadow(
                collective=fake.endpoint(rank),
                transaction_directory=fixture["transaction"],
                ordered_parameters=fixture["b"][rank],
                ordered_fixed_lora_a=fixture["a"][rank],
                checkpoint_manifest=checkpoint_by_rank[rank],
                local_rank_evidence=fixture["evidence"][rank],
                scientific_gate_files=fixture["gates"],
                fresh_inputs=fixture["fresh_inputs"],
                topology_receipt_digest=self.topology_digest,
                global_trust_radius=1.0,
                layer_trust_radii=fixture["trust"][rank],
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, rank) for rank in range(8)]
            return [future.result(timeout=60.0) for future in futures]

    def assert_authoritative_zero(self, fixture):
        for rank in range(8):
            self.assertTrue(
                all(torch.count_nonzero(tensor).item() == 0 for _, tensor in fixture["b"][rank])
            )

    def fresh_receipts(self, prepared):
        request = json.loads(
            prepared.fresh_render_request.path.read_text("ascii")
        )
        refs = {}
        for actor in world8.ARM_ORDER:
            refs[actor] = self.external_receipt(
                f"fresh-{actor}",
                "fresh_exact81_endpoint",
                {
                    "actor_family": actor,
                    "fresh_request_sha256": prepared.fresh_render_request.sha256,
                    "candidate_delta_sha256": request["candidate_delta_sha256"],
                    "shadow_ab_state_digest": request["shadow_ab_state_digest"],
                    "shadow_manifest_sha256": request["shadow_manifest_sha256"],
                    "fresh_plan_receipt_sha256": request[
                        "fresh_plan_receipt_sha256"
                    ],
                    "source_manifest_sha256": request["source_manifest_sha256"],
                    "official_gaussian_sha256": request[
                        "official_gaussian_sha256"
                    ],
                    "exact_frame_count": 81,
                    "full_action_conjunction_passed": True,
                    "preservation_families_noninferior": list(
                        world8.PRESERVATION_FAMILIES
                    ),
                    "source_disjoint_confirmation_passed": True,
                },
            )
        return [refs["dog"]] * 4 + [refs["human"]] * 4

    def test_checkpoint_manifest_rejects_alias_and_omission(self):
        b_rows, a_rows, _ = self.parameters()
        alias = list(a_rows[0])
        alias[1] = (alias[1][0], alias[0][1])
        with self.assertRaisesRegex(world8.World8ShadowError, "aliased"):
            world8.checkpoint_manifest_payload(
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                ordered_parameters=b_rows[0],
                ordered_fixed_lora_a=alias,
            )
        with self.assertRaisesRegex(world8.World8ShadowError, "exactly 32"):
            world8.checkpoint_manifest_payload(
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                ordered_parameters=b_rows[0][:-1],
                ordered_fixed_lora_a=a_rows[0],
            )

    def test_nonzero_b_rounding_is_audited_as_realized_not_requested_delta(self):
        base = torch.tensor([2**24], dtype=torch.float32)
        delta = torch.tensor([1.0], dtype=torch.float32)
        after, realized = world8.compute_anticipated_realized(base, delta)
        self.assertTrue(torch.equal(after, base))
        self.assertTrue(torch.equal(realized, torch.zeros_like(realized)))
        self.assertFalse(torch.equal(realized, delta))

        b_rows, a_rows, trust = self.parameters()
        evidence_rows = self.evidence(b_rows[0])
        evidence = qp.DP2SP4Evidence(
            dp_arms=(
                qp.DPArmEvidence("dog", tuple(evidence_rows[:4])),
                qp.DPArmEvidence("human", tuple(evidence_rows[4:])),
            ),
            topology_receipt_digest=self.topology_digest,
        )
        layout = qp.FixedParameterLayout.from_ordered_parameters(b_rows[0])
        solution = qp.solve_stateless_jacobian_qp(
            layout=layout,
            evidence=evidence,
            global_trust_radius=1.0,
            layer_trust_radii=trust[0],
        )
        nonzero_b = [tensor.detach().clone() for _, tensor in b_rows[0]]
        nonzero_b[0].reshape(-1)[0] = float(2**24)
        snapshot = world8._ABSnapshot(
            a_tensors=tuple(tensor.detach().clone() for _, tensor in a_rows[0]),
            b_tensors=tuple(nonzero_b),
            state_digest=_sha("nonzero-state"),
            manifest_sha256=_sha("nonzero-manifest"),
            checkpoint_content_receipt_digest=self.checkpoint_digest,
        )
        _, realized_rows, _, _ = world8._anticipated_state(
            snapshot=snapshot,
            delta_by_parameter=solution.delta_by_parameter,
        )
        with self.assertRaisesRegex(world8.World8ShadowError, "violates"):
            world8._audit_anticipated_constraints(
                solution=solution, realized=realized_rows
            )

    def test_external_verifier_rehashes_artifact_and_rejects_self_pass_bool(self):
        bindings = {"actor_family": "dog"}
        ref = self.external_receipt("verifier-good", "owner_audit", bindings)
        verifier = world8.CanonicalExternalVerifier(self.verifier_policy)
        self.assertEqual(
            verifier.verify(ref, audit_type="owner_audit", exact_bindings=bindings)[
                "receipt_sha256"
            ],
            ref.receipt.sha256,
        )
        value = json.loads(ref.receipt.path.read_text("ascii"))
        value["passed"] = True
        forged_raw = world8.canonical_json_bytes(value)
        forged_receipt = self.immutable_file("forged-self-pass.json", forged_raw)
        with self.assertRaisesRegex(world8.World8ShadowError, "key closure"):
            verifier.verify(
                world8.ExternalReceiptRef(forged_receipt, ref.artifact),
                audit_type="owner_audit",
                exact_bindings=bindings,
            )
        signed = json.loads(ref.receipt.path.read_text("ascii"))
        signed["bindings"] = {"actor_family": "human"}
        forged_signature_raw = world8.canonical_json_bytes(signed)
        forged_signature = self.immutable_file(
            "forged-signature.json", forged_signature_raw
        )
        with self.assertRaisesRegex(world8.World8ShadowError, "authentication"):
            verifier.verify(
                world8.ExternalReceiptRef(forged_signature, ref.artifact),
                audit_type="owner_audit",
                exact_bindings={"actor_family": "human"},
            )

    def test_prepare_is_shadow_only_and_writes_durable_wal(self):
        fixture = self.fixture()
        results = self.run_prepare(fixture)
        self.assertTrue(all(result.prepared for result in results))
        self.assert_authoritative_zero(fixture)
        self.assertEqual(
            world8.recover_world8_transaction(fixture["transaction"])["state"],
            "PREPARED_QUARANTINED",
        )
        self.assertTrue((fixture["transaction"] / "PREPARED.json").exists())
        self.assertFalse((fixture["transaction"] / "COMMIT.json").exists())
        self.assertTrue(
            all((fixture["transaction"] / f"rank-{rank}.PREPARED.json").exists() for rank in range(8))
        )
        with self.assertRaisesRegex(world8.World8ShadowError, "publication is disabled"):
            world8.publish_committed_shadow(
                transaction_directory=fixture["transaction"],
                published_directory=self.root / "must-not-be-published",
            )
        resumed = self.run_prepare(fixture)
        self.assertTrue(all(result.prepared for result in resumed))
        self.assertTrue(
            all(
                result.receipt["resumed_from_durable_state"] is True
                and result.receipt["durable_commit_recorded"] is False
                for result in resumed
            )
        )
        self.assert_authoritative_zero(fixture)

    def test_finalize_and_publish_are_unconditionally_disabled(self):
        fixture = self.fixture()
        prepared = self.run_prepare(fixture)
        fresh_refs = self.fresh_receipts(prepared[0])
        with self.assertRaisesRegex(world8.World8ShadowError, "finalize/COMMIT is disabled"):
            world8.finalize_world8_shadow(
                collective=_FakeCollectiveWorld().endpoint(0),
                transaction_directory=fixture["transaction"],
                ordered_parameters=fixture["b"][0],
                ordered_fixed_lora_a=fixture["a"][0],
                checkpoint_manifest=fixture["checkpoint"],
                verifier_policy=self.verifier_policy,
                local_fresh_receipt=fresh_refs[0],
            )
        with self.assertRaisesRegex(world8.World8ShadowError, "publication is disabled"):
            world8.publish_committed_shadow(
                transaction_directory=fixture["transaction"],
                published_directory=self.root / "published-adapter",
            )
        self.assertEqual(
            world8.recover_world8_transaction(fixture["transaction"])["state"],
            "PREPARED_QUARANTINED",
        )
        self.assert_authoritative_zero(fixture)

    def test_rank_preflight_failure_is_globally_aborted_without_mutation(self):
        fixture = self.fixture()
        wrong = list([fixture["checkpoint"]] * 8)
        wrong[3] = world8.CanonicalFileRef(
            fixture["checkpoint"].path, _sha("wrong-checkpoint-manifest")
        )
        results = self.run_prepare(fixture, checkpoint_by_rank=wrong)
        self.assertTrue(all(result.aborted for result in results))
        self.assert_authoritative_zero(fixture)
        self.assertTrue((fixture["transaction"] / "DECISION.json").exists())
        aborted_sha = hashlib.sha256(
            (fixture["transaction"] / "DECISION.json").read_bytes()
        ).hexdigest()
        resumed = self.run_prepare(fixture)
        self.assertTrue(all(result.aborted for result in resumed))
        self.assertEqual(
            hashlib.sha256(
                (fixture["transaction"] / "DECISION.json").read_bytes()
            ).hexdigest(),
            aborted_sha,
        )
        self.assertFalse((fixture["transaction"] / "PREPARED.json").exists())

    def test_forged_action_gradient_receipt_prevents_prepare(self):
        fixture = self.fixture()
        dog = fixture["gates"].arms[0]
        forged_ref = self.external_receipt(
            "forged-action-gradient",
            "action_gradient",
            {"actor_family": "dog", "row_id": "dog-action"},
        )
        forged_dog = world8.ArmScientificGateFiles(
            dog.actor_family,
            dog.owner,
            dog.query_seed_gates,
            (forged_ref,),
            dog.preservation_gradient_receipts,
        )
        fixture["gates"] = world8.ScientificGateFiles(
            fixture["gates"].verifier_policy,
            (forged_dog, fixture["gates"].arms[1]),
            fixture["gates"].qp_contract,
            fixture["gates"].fresh_plan,
        )
        results = self.run_prepare(fixture)
        self.assertTrue(all(result.aborted for result in results))
        self.assert_authoritative_zero(fixture)

    def test_action_row_mutation_after_signed_gate_prevents_prepare(self):
        fixture = self.fixture()
        fixture["evidence"][0].action_rows[0].values[0] = 2.0
        results = self.run_prepare(fixture)
        self.assertTrue(all(result.aborted for result in results))
        self.assert_authoritative_zero(fixture)

    def test_no_renderer_callback_surface_can_write_during_prepare(self):
        marker = self.root / "callback-wrote.txt"

        def malicious_callback():
            marker.write_text("bad")

        self.assertNotIn(
            "fresh_endpoint_audit",
            inspect.signature(world8.prepare_world8_shadow).parameters,
        )
        with self.assertRaises(TypeError):
            world8.prepare_world8_shadow(fresh_endpoint_audit=malicious_callback)
        self.assertFalse(marker.exists())

    def test_v1_transport_has_no_subgroup_or_backend_injection_surface(self):
        self.assertEqual(
            list(inspect.signature(world8.TorchWorld8SmallCollective).parameters),
            [],
        )
        prepare_parameters = inspect.signature(
            world8.prepare_world8_shadow
        ).parameters
        self.assertNotIn("group", prepare_parameters)
        self.assertNotIn("backend", prepare_parameters)
        self.assertNotIn("_backend", prepare_parameters)

    def test_post_shadow_collective_failure_fail_stops_unchanged(self):
        fixture = self.fixture()
        fake = _FakeCollectiveWorld(fail_ordinal=3, fail_exception=RuntimeError)
        with self.assertRaises(world8.World8FailStopError):
            self.run_prepare(fixture, world=fake)
        self.assert_authoritative_zero(fixture)
        self.assertFalse((fixture["transaction"] / "COMMIT.json").exists())

    def test_systemexit_at_preflight_is_caught_at_baseexception_boundary(self):
        fixture = self.fixture()
        fake = _FakeCollectiveWorld(fail_ordinal=0, fail_exception=SystemExit)
        with self.assertRaises(world8.World8FailStopError):
            self.run_prepare(fixture, world=fake)
        self.assert_authoritative_zero(fixture)
        self.assertFalse((fixture["transaction"] / "COMMIT.json").exists())

    def test_signed_bounds_cannot_be_reused_after_constraint_weakening(self):
        fixture = self.fixture()
        fixture["evidence"] = [
            dataclasses.replace(
                evidence,
                action_rows=tuple(
                    dataclasses.replace(row, minimum_dot=1.0e-12)
                    for row in evidence.action_rows
                ),
                preservation_rows=tuple(
                    dataclasses.replace(row, maximum_absolute_dot=1.0e12)
                    for row in evidence.preservation_rows
                ),
            )
            for evidence in fixture["evidence"]
        ]
        results = self.run_prepare(fixture)
        self.assertTrue(all(result.aborted for result in results))
        self.assert_authoritative_zero(fixture)

    def test_trust_radius_a_must_equal_checkpoint_a_byte_for_byte(self):
        fixture = self.fixture()
        fixture["trust"] = [
            tuple(
                dataclasses.replace(
                    bound,
                    fixed_lora_a=bound.fixed_lora_a * 1000.0,
                )
                for bound in rank_trust
            )
            for rank_trust in fixture["trust"]
        ]
        results = self.run_prepare(fixture)
        self.assertTrue(all(result.aborted for result in results))
        self.assert_authoritative_zero(fixture)

    def test_evidence_checkpoint_digest_must_equal_manifest(self):
        fixture = self.fixture()
        other = _sha("other-checkpoint")
        fixture["evidence"] = [
            dataclasses.replace(
                evidence,
                action_rows=tuple(
                    dataclasses.replace(
                        row, checkpoint_content_receipt_digest=other
                    )
                    for row in evidence.action_rows
                ),
                preservation_rows=tuple(
                    dataclasses.replace(
                        row, checkpoint_content_receipt_digest=other
                    )
                    for row in evidence.preservation_rows
                ),
            )
            for evidence in fixture["evidence"]
        ]
        results = self.run_prepare(fixture)
        self.assertTrue(all(result.aborted for result in results))
        self.assert_authoritative_zero(fixture)

    def test_checkpoint_manifest_is_explicitly_first_step_zero_b_only(self):
        b_rows, a_rows, _ = self.parameters()
        b_rows[0][0][1].detach().reshape(-1)[0] = 1.0
        with self.assertRaisesRegex(world8.World8ShadowError, "first-step-only"):
            world8.checkpoint_manifest_payload(
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                ordered_parameters=b_rows[0],
                ordered_fixed_lora_a=a_rows[0],
            )

    def test_one_rank_bootstrap_path_failure_is_enveloped_without_peer_hang(self):
        fixture = self.fixture()
        fake = _FakeCollectiveWorld()
        paths = [fixture["transaction"]] * 8
        paths[3] = self.root / "missing-transaction"

        def worker(rank):
            try:
                world8.prepare_world8_shadow(
                    collective=fake.endpoint(rank),
                    transaction_directory=paths[rank],
                    ordered_parameters=fixture["b"][rank],
                    ordered_fixed_lora_a=fixture["a"][rank],
                    checkpoint_manifest=fixture["checkpoint"],
                    local_rank_evidence=fixture["evidence"][rank],
                    scientific_gate_files=fixture["gates"],
                    fresh_inputs=fixture["fresh_inputs"],
                    topology_receipt_digest=self.topology_digest,
                    global_trust_radius=1.0,
                    layer_trust_radii=fixture["trust"][rank],
                )
            except BaseException as error:
                return type(error).__name__
            return "UNEXPECTED_SUCCESS"

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = [
                future.result(timeout=15.0)
                for future in [executor.submit(worker, rank) for rank in range(8)]
            ]
        self.assertEqual(outcomes, ["World8FailStopError"] * 8)

    def test_external_artifact_rejects_hardlink_and_size_overflow(self):
        bindings = {"actor_family": "dog"}
        ref = self.external_receipt("file-policy", "owner_audit", bindings)
        hardlink = self.root / "artifact-hardlink"
        os.link(ref.artifact.path, hardlink)
        verifier = world8.CanonicalExternalVerifier(self.verifier_policy)
        with self.assertRaisesRegex(world8.World8ShadowError, "hard link"):
            verifier.verify(ref, audit_type="owner_audit", exact_bindings=bindings)
        hardlink.unlink()
        with mock.patch.object(world8, "MAX_EXTERNAL_ARTIFACT_BYTES", 1):
            with self.assertRaisesRegex(world8.World8ShadowError, "artifact size"):
                verifier.verify(
                    ref, audit_type="owner_audit", exact_bindings=bindings
                )

    def test_transaction_creation_is_exclusive_and_decision_schema_is_strict(self):
        transaction = self.root / "exclusive-transaction"
        world8.create_transaction_directory(
            transaction,
            transaction_id="exclusive",
            verifier_policy=self.verifier_policy,
        )
        with self.assertRaisesRegex(world8.World8ShadowError, "exclusively"):
            world8.create_transaction_directory(
                transaction,
                transaction_id="exclusive",
                verifier_policy=self.verifier_policy,
            )
        world8._atomic_publish_record(
            transaction / "DECISION.json",
            {"schema_version": world8.DECISION_SCHEMA, "decision": "ABORTED"},
        )
        with self.assertRaisesRegex(world8.World8ShadowError, "DECISION"):
            world8.recover_world8_transaction(transaction)

    def test_prepare_has_no_authoritative_restore_or_copy_surface(self):
        source = inspect.getsource(world8.prepare_world8_shadow)
        self.assertNotIn("copy_(", source)
        self.assertFalse(hasattr(world8, "_restore_authoritative_snapshot"))


if __name__ == "__main__":
    unittest.main()
