from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import fewshot_episode_parallel as parallel  # noqa: E402


class FakeCollective:
    def __init__(self, *, reduce_result=None, gathered=()):
        self.reduce_result = reduce_result
        self.gathered = tuple(gathered)
        self.calls = []

    def all_reduce_sum(self, value, *, group):
        self.calls.append(("all_reduce_sum", value, group))
        return self.reduce_result

    def all_gather_object(self, value, *, group):
        self.calls.append(("all_gather_object", value, group))
        return self.gathered


@dataclass(frozen=True)
class FakeCode:
    value: float
    requires_grad: bool = False
    grad_fn: object = None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _losses(offset: float = 0.0):
    return {
        "zero": 4.0 + offset,
        "correct": 2.0 + offset,
        "reverse": 3.0 + offset,
        "shuffle": 3.5 + offset,
    }


def _payload(
    support_index: int,
    iid: str,
    producer_rank: int,
    value: float,
    *,
    loss_offset: float = 0.0,
):
    return parallel.freeze_support_payload(
        support_index=support_index,
        iid=iid,
        producer_rank=producer_rank,
        code=FakeCode(value),
        code_sha256=_digest(f"code-{support_index}-{value}"),
        held_losses=_losses(loss_offset),
    )


class TopologyTests(unittest.TestCase):
    def test_world8_dp2_ulysses4_rank_to_support_mapping_is_closed(self):
        topology = parallel.EpisodeParallelTopology()
        self.assertEqual(
            [group.ranks for group in topology.ulysses_groups],
            [(0, 1, 2, 3), (4, 5, 6, 7)],
        )
        self.assertEqual(
            [group.ranks for group in topology.data_parallel_groups],
            [(0, 4), (1, 5), (2, 6), (3, 7)],
        )
        self.assertEqual(
            [topology.assignment(rank).support_index for rank in range(8)],
            [1, 1, 1, 1, 2, 2, 2, 2],
        )
        self.assertEqual(
            [topology.assignment(rank).ulysses_rank for rank in range(8)],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        receipt = topology.receipt()
        self.assertFalse(receipt["cross_dp_gradient_sync"])
        self.assertEqual(receipt["gradient_divisor"], 4)

    def test_any_other_parallel_shape_fails_closed(self):
        invalid = (
            {"world_size": 4},
            {"data_parallel_size": 1},
            {"ulysses_size": 8},
        )
        for update in invalid:
            with self.subTest(update=update):
                with self.assertRaisesRegex(
                    parallel.FewShotEpisodeParallelError, "world=8"
                ):
                    parallel.EpisodeParallelTopology(**update)

    def test_invalid_and_boolean_ranks_fail_closed(self):
        for rank in (-1, 8, True):
            with self.subTest(rank=rank):
                with self.assertRaises(parallel.FewShotEpisodeParallelError):
                    parallel.DEFAULT_TOPOLOGY.assignment(rank)


class GradientGroupTests(unittest.TestCase):
    def test_support_gradient_uses_only_own_ulysses_group_and_divides_by_four(self):
        topology = parallel.DEFAULT_TOPOLOGY
        expected_group = topology.ulysses_groups[1]
        collective = FakeCollective(reduce_result=12.0)
        result = parallel.mean_support_gradient(
            3.0,
            rank=6,
            support_index=2,
            group=expected_group,
            divisor=4,
            collective=collective,
        )
        self.assertEqual(result, 3.0)
        self.assertEqual(
            collective.calls,
            [("all_reduce_sum", 3.0, expected_group)],
        )

    def test_world_or_dp_gradient_group_is_rejected_before_collective(self):
        topology = parallel.DEFAULT_TOPOLOGY
        bad_groups = (topology.world_group, topology.data_parallel_groups[0])
        for group in bad_groups:
            with self.subTest(group=group.name):
                collective = FakeCollective(reduce_result=8.0)
                with self.assertRaisesRegex(
                    parallel.FewShotEpisodeParallelError, "never WORLD/DP"
                ):
                    parallel.mean_support_gradient(
                        1.0,
                        rank=0,
                        support_index=1,
                        group=group,
                        divisor=4,
                        collective=collective,
                    )
                self.assertEqual(collective.calls, [])

    def test_division_by_eight_or_two_is_rejected(self):
        group = parallel.DEFAULT_TOPOLOGY.ulysses_groups[0]
        for divisor in (8, 2):
            with self.subTest(divisor=divisor):
                collective = FakeCollective(reduce_result=8.0)
                with self.assertRaisesRegex(
                    parallel.FewShotEpisodeParallelError, "exactly 4"
                ):
                    parallel.mean_support_gradient(
                        2.0,
                        rank=1,
                        support_index=1,
                        group=group,
                        divisor=divisor,
                        collective=collective,
                    )
                self.assertEqual(collective.calls, [])

    def test_rank_cannot_optimize_the_other_support(self):
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "rank-to-support"
        ):
            parallel.mean_support_gradient(
                1.0,
                rank=4,
                support_index=1,
                group=parallel.DEFAULT_TOPOLOGY.ulysses_groups[1],
                divisor=4,
                collective=FakeCollective(reduce_result=4.0),
            )


class DetachedPayloadTests(unittest.TestCase):
    def test_autograd_bearing_code_is_rejected_and_detach_is_injected(self):
        attached = FakeCode(0.2, requires_grad=True, grad_fn=object())
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "requires gradients"
        ):
            parallel.freeze_support_payload(
                support_index=1,
                iid="dog-z",
                producer_rank=0,
                code=attached,
                code_sha256=_digest("attached"),
                held_losses=_losses(),
            )

        frozen = parallel.freeze_support_payload(
            support_index=1,
            iid="dog-z",
            producer_rank=0,
            code=attached,
            detach_code=lambda code: FakeCode(code.value),
            code_sha256=_digest("detached"),
            held_losses=_losses(),
        )
        self.assertFalse(frozen.code.requires_grad)
        self.assertIsNone(frozen.code.grad_fn)

    def test_payload_rank_must_own_support(self):
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "other support"
        ):
            _payload(2, "dog-b", 0, 0.2)

    def test_ulysses_replica_certificate_is_group_scoped_and_exact(self):
        payload = _payload(1, "dog-z", 0, 0.1)
        group = parallel.DEFAULT_TOPOLOGY.ulysses_groups[0]
        collective = FakeCollective(gathered=[payload.semantic_sha256] * 4)
        certificate = parallel.certify_support_replicas(
            payload,
            rank=0,
            group=group,
            collective=collective,
        )
        self.assertEqual(certificate.semantic_sha256, payload.semantic_sha256)
        self.assertEqual(certificate.group, group)

        wrong = FakeCollective(gathered=[payload.semantic_sha256] * 4)
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "exact Ulysses"
        ):
            parallel.certify_support_replicas(
                payload,
                rank=0,
                group=parallel.DEFAULT_TOPOLOGY.world_group,
                collective=wrong,
            )
        self.assertEqual(wrong.calls, [])

    def test_nonidentical_ulysses_replica_digest_is_rejected(self):
        payload = _payload(1, "dog-z", 0, 0.1)
        observed = [payload.semantic_sha256] * 3 + [_digest("different")]
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "byte-identical"
        ):
            parallel.certify_support_replicas(
                payload,
                rank=0,
                group=parallel.DEFAULT_TOPOLOGY.ulysses_groups[0],
                collective=FakeCollective(gathered=observed),
            )


class CanonicalExchangeTests(unittest.TestCase):
    def _certificate(self, payload):
        return parallel.SupportReplicaCertificate(
            support_index=payload.support_index,
            iid=payload.iid,
            semantic_sha256=payload.semantic_sha256,
            group=parallel.DEFAULT_TOPOLOGY.ulysses_groups[
                payload.support_index - 1
            ],
        )

    def test_dp_exchange_returns_support_order_not_arrival_order(self):
        support_one = _payload(1, "dog-z", 0, 0.1)
        support_two = _payload(2, "dog-a", 4, 0.5, loss_offset=2.0)
        group = parallel.DEFAULT_TOPOLOGY.data_parallel_groups[0]
        collective = FakeCollective(gathered=(support_two, support_one))
        ordered = parallel.canonical_exchange_support_payloads(
            support_one,
            self._certificate(support_one),
            rank=0,
            group=group,
            collective=collective,
        )
        self.assertEqual([item.support_index for item in ordered], [1, 2])
        self.assertEqual([item.iid for item in ordered], ["dog-z", "dog-a"])
        self.assertEqual(collective.calls[0][2], group)

    def test_exchange_rejects_world_or_ulysses_group(self):
        support_one = _payload(1, "dog-z", 0, 0.1)
        support_two = _payload(2, "dog-a", 4, 0.5)
        for group in (
            parallel.DEFAULT_TOPOLOGY.world_group,
            parallel.DEFAULT_TOPOLOGY.ulysses_groups[0],
        ):
            with self.subTest(group=group.name):
                collective = FakeCollective(gathered=(support_one, support_two))
                with self.assertRaisesRegex(
                    parallel.FewShotEpisodeParallelError, "DP column"
                ):
                    parallel.canonical_exchange_support_payloads(
                        support_one,
                        self._certificate(support_one),
                        rank=0,
                        group=group,
                        collective=collective,
                    )
                self.assertEqual(collective.calls, [])

    def test_duplicate_iid_or_support_is_rejected(self):
        support_one = _payload(1, "same-dog", 0, 0.1)
        same_iid = _payload(2, "same-dog", 4, 0.5)
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "distinct IIDs"
        ):
            parallel.canonical_two_support_payloads((support_one, same_iid))

        duplicate_support = _payload(1, "other-dog", 1, 0.2)
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "support 1 and support 2"
        ):
            parallel.canonical_two_support_payloads(
                (support_one, duplicate_support)
            )

    def test_prototype_midpoint_receives_support1_then_support2(self):
        support_one = _payload(1, "dog-z", 0, 0.1)
        support_two = _payload(2, "dog-a", 4, 0.5)
        observed = []

        def midpoint(left, right):
            observed.append((left.value, right.value))
            return FakeCode((left.value + right.value) / 2.0)

        prototype = parallel.build_canonical_prototype(
            (support_two, support_one), midpoint=midpoint
        )
        self.assertEqual(observed, [(0.1, 0.5)])
        self.assertEqual(prototype.code.value, 0.3)
        self.assertEqual(prototype.support_iids, ("dog-z", "dog-a"))
        self.assertEqual(
            prototype.aggregation_rule,
            "exact_arithmetic_midpoint_in_decoded_fp32_gate_space",
        )


class GateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.support_one = _payload(1, "dog-z", 0, 0.1, loss_offset=0.0)
        self.support_two = _payload(2, "dog-a", 4, 0.5, loss_offset=2.0)
        self.payloads = (self.support_two, self.support_one)

    def test_probe_evidence_is_support1_only_and_canonical(self):
        probes = (
            parallel.ProbeEvidence(1, "dog-z", "block_only", True),
            parallel.ProbeEvidence(1, "dog-z", "phase_only", True),
        )
        ordered = parallel.canonical_reference_probes(probes, self.payloads)
        self.assertEqual(
            [item.family for item in ordered], ["phase_only", "block_only"]
        )

    def test_support2_probe_cannot_change_reference_gate(self):
        probes = (
            parallel.ProbeEvidence(1, "dog-z", "phase_only", True),
            parallel.ProbeEvidence(2, "dog-a", "block_only", True),
        )
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "only support 1"
        ):
            parallel.canonical_reference_probes(probes, self.payloads)

    def test_duplicate_probe_family_is_rejected(self):
        probes = (
            parallel.ProbeEvidence(1, "dog-z", "phase_only", True),
            parallel.ProbeEvidence(1, "dog-z", "phase_only", True),
        )
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "exactly once"
        ):
            parallel.canonical_reference_probes(probes, self.payloads)

    def test_held_controls_sort_by_iid_then_average_losses(self):
        aggregate = parallel.aggregate_held_control_losses(self.payloads)
        self.assertEqual(aggregate.ordered_iids, ("dog-a", "dog-z"))
        self.assertEqual(
            aggregate.mean_loss_map,
            {
                "zero": 5.0,
                "correct": 3.0,
                "reverse": 4.0,
                "shuffle": 4.5,
            },
        )
        self.assertEqual(
            aggregate.aggregation,
            "sort_by_iid_then_arithmetic_mean_losses_before_ratios",
        )

    def test_missing_or_extra_held_control_is_rejected(self):
        invalid = _losses()
        invalid.pop("shuffle")
        with self.assertRaisesRegex(
            parallel.FewShotEpisodeParallelError, "exactly"
        ):
            parallel.freeze_support_payload(
                support_index=1,
                iid="dog-z",
                producer_rank=0,
                code=FakeCode(0.1),
                code_sha256=_digest("bad-held"),
                held_losses=invalid,
            )


if __name__ == "__main__":
    unittest.main()
