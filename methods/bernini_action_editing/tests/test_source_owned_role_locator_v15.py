from __future__ import annotations

import inspect
import json
from dataclasses import replace
import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

try:
    import torch
except ImportError as error:  # pragma: no cover - lightweight default Python
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_owned_role_locator_v15 as locator  # noqa: E402


ASSET = METHOD_ROOT / "assets" / "interaction_complex4_source_role_token_spans_v15.json"
TOKENIZER_SHA = "1" * 64


class TinyTokenizer:
    def __init__(self, *, overflow: bool = False, fix_mistral_regex: bool = True) -> None:
        self.overflow = overflow
        self.padding_side = "right"
        self.init_kwargs = {"fix_mistral_regex": fix_mistral_regex}

    def __call__(self, text, **kwargs):
        if kwargs != {
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_offsets_mapping": True,
        }:
            raise AssertionError(kwargs)
        if self.overflow:
            return {
                "input_ids": list(range(513)),
                "attention_mask": [1] * 513,
                "offset_mapping": [(0, 0)] * 513,
            }
        offsets = []
        ids = []
        cursor = 0
        for index, word in enumerate(text.split(" ")):
            start = cursor
            end = start + len(word)
            offsets.append((start, end))
            ids.append(100 + index)
            cursor = end + 1
        offsets.append((0, 0))
        ids.append(1)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


def make_lock(
    *,
    role: str,
    substring: str,
    char_start: int,
    token_start: int,
    token_ids,
    protected: bool = True,
):
    return locator.LockedRoleSpan.create(
        role=role,
        kind="actor" if role.startswith("actor") else "manipulated_object",
        protected=protected,
        substring=substring,
        char_start=char_start,
        token_start=token_start,
        token_ids=token_ids,
    )


def make_spec(*, swapped: bool = False):
    text = "red dog blue cup"
    if swapped:
        roles = (
            make_lock(
                role="actor_first",
                substring="blue cup",
                char_start=8,
                token_start=2,
                token_ids=(102, 103),
            ),
            make_lock(
                role="object_second",
                substring="red dog",
                char_start=0,
                token_start=0,
                token_ids=(100, 101),
            ),
        )
    else:
        roles = (
            make_lock(
                role="actor_first",
                substring="red dog",
                char_start=0,
                token_start=0,
                token_ids=(100, 101),
            ),
            make_lock(
                role="object_second",
                substring="blue cup",
                char_start=8,
                token_start=2,
                token_ids=(102, 103),
            ),
        )
    payload = {
        "event_id": "synthetic-source-event",
        "source_iid": "source-001",
        "source_caption": text,
        "source_caption_sha256": locator.text_sha256(text),
        "model_text": text,
        "model_text_sha256": locator.text_sha256(text),
        "source_caption_char_start": 0,
        "tokenizer_tree_sha256": TOKENIZER_SHA,
        "roles": [item.as_dict() for item in roles],
    }
    return locator.SourceRoleEventSpec(
        **{key: value for key, value in payload.items() if key != "roles"},
        roles=roles,
        event_sha256=locator.object_sha256(payload),
    )


def with_tokenizer_sha(spec, digest):
    payload = {
        "event_id": spec.event_id,
        "source_iid": spec.source_iid,
        "source_caption": spec.source_caption,
        "source_caption_sha256": spec.source_caption_sha256,
        "model_text": spec.model_text,
        "model_text_sha256": spec.model_text_sha256,
        "source_caption_char_start": spec.source_caption_char_start,
        "tokenizer_tree_sha256": digest,
        "roles": [item.as_dict() for item in spec.roles],
    }
    return locator.SourceRoleEventSpec(
        **{key: value for key, value in payload.items() if key != "roles"},
        roles=spec.roles,
        event_sha256=locator.object_sha256(payload),
    )


def make_tokenizer_tree(root):
    root = Path(root)
    for index, name in enumerate(locator.PINNED_TOKENIZER_FILES):
        (root / name).write_bytes(f"test-tokenizer-file-{index}".encode())
    return locator.tokenizer_tree_sha256(root)


class FakeOfficialAttn2:
    def __init__(self) -> None:
        self.base_calls = 0
        self.project_calls = 0
        self.last_output = None
        self.last_projection = None

    def __call__(self, _attn, hidden_states, **kwargs):
        self.base_calls += 1
        self.last_output = hidden_states.square() + 0.125
        self.last_projection = kwargs
        return self.last_output

    def _project_qkv(
        self,
        _attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        origin_hidden_states_seq_len,
        is_cross_attn,
    ):
        self.project_calls += 1
        if rotary_emb is not None or is_cross_attn is not True:
            raise AssertionError("not official cross-attention projection")
        if origin_hidden_states_seq_len <= 0:
            raise AssertionError("origin length is invalid")
        query = hidden_states.unsqueeze(2)
        key = encoder_hidden_states.unsqueeze(2)
        value = key.clone()
        return query, key, value


# Preserve the exact production type gate while keeping this unit test free of
# the remote Bernini package.
FakeOfficialAttn2.__name__ = locator.OFFICIAL_ATTN2_PROCESSOR_CLASS
FakeOfficialAttn2.__module__ = locator.OFFICIAL_ATTN2_PROCESSOR_MODULE


def role_pattern_hidden():
    rows = []
    for _phase in range(locator.LATENT_PHASES):
        rows.extend(([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]))
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


def source_text_hidden():
    prefix = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    result = torch.zeros((1, locator.MAX_TEXT_TOKENS, 4), dtype=torch.float32)
    result[:, : prefix.shape[1]].copy_(prefix)
    return result


def invoke(processor, *, spec, bank, text=None):
    hidden = role_pattern_hidden()
    raw_source_text = source_text_hidden() if text is None else text
    geometry = locator.SourceVisualGeometry(height=1, width=2)
    with tempfile.TemporaryDirectory() as tmp:
        digest = make_tokenizer_tree(tmp)
        runtime_spec = with_tokenizer_sha(spec, digest)
        with mock.patch.object(
            locator.importlib.metadata, "version", return_value="5.5.4"
        ), mock.patch.object(locator, "PINNED_TOKENIZER_TREE_SHA256", digest):
            provenance, conditioned = locator.bind_source_text_provenance(
                tokenizer=TinyTokenizer(),
                tokenizer_dir=tmp,
                transformers_version=locator.PINNED_TRANSFORMERS_VERSION,
                event_spec=runtime_spec,
                raw_source_text_hidden_states=raw_source_text,
                derive_conditioned_source_text=lambda value: value.clone(),
                renderer_text_length=5,
            )
        invocation = locator.SourceRoleObserverInvocation(
            capture_bank=bank,
            event_spec=runtime_spec,
            geometry=geometry,
            source_text_provenance=provenance,
            step_index=3,
            ulysses=locator.UlyssesVisualShard(geometry=geometry, rank=0, size=1),
        )
        source_text_view = conditioned[:, :5]
        with locator.observe_source_roles(invocation):
            output = processor(
                object(),
                hidden,
                encoder_hidden_states=source_text_view,
                batch_image_vae_seqlen=[42],
                text_features_length=[5],
                origin_hidden_states_seq_len=42,
            )
    return hidden, source_text_view, output


class TokenSpanTests(unittest.TestCase):
    def test_exact_substring_locks_char_token_ids_and_hash(self):
        lock = locator.resolve_exact_substring_token_span(
            TinyTokenizer(),
            model_text="red dog blue cup",
            role="actor_first",
            kind="actor",
            protected=True,
            substring="red dog",
            expected_char_start=0,
        )
        self.assertEqual((lock.char_start, lock.char_end), (0, 7))
        self.assertEqual((lock.token_start, lock.token_end), (0, 2))
        self.assertEqual(lock.token_ids, (100, 101))
        self.assertEqual(len(lock.span_sha256), 64)
        self.assertEqual(
            locator.resolve_exact_substring_token_span(
                TinyTokenizer(),
                model_text="red dog blue cup",
                role="actor_first",
                kind="actor",
                protected=True,
                substring="red dog",
                expected_lock=lock,
            ),
            lock,
        )

    def test_missing_duplicate_wrong_lock_and_truncation_fail(self):
        with self.assertRaises(locator.SourceOwnedRoleLocatorError):
            locator.resolve_exact_substring_token_span(
                TinyTokenizer(), model_text="red dog", role="actor", kind="actor",
                protected=True, substring="cat"
            )
        with self.assertRaises(locator.SourceOwnedRoleLocatorError):
            locator.resolve_exact_substring_token_span(
                TinyTokenizer(), model_text="red dog red", role="actor", kind="actor",
                protected=True, substring="red"
            )
        lock = locator.resolve_exact_substring_token_span(
            TinyTokenizer(), model_text="red dog", role="actor", kind="actor",
            protected=True, substring="red"
        )
        with self.assertRaises(locator.SourceOwnedRoleLocatorError):
            locator.resolve_exact_substring_token_span(
                TinyTokenizer(), model_text="blue cat", role="actor", kind="actor",
                protected=True, substring="blue", expected_lock=lock
            )
        with self.assertRaises(locator.SourceOwnedRoleLocatorError):
            locator.resolve_exact_substring_token_span(
                TinyTokenizer(overflow=True), model_text="red", role="actor", kind="actor",
                protected=True, substring="red"
            )

    def test_overlapping_character_or_token_spans_fail(self):
        text = "red dog blue cup"
        first = make_lock(
            role="actor_first", substring="red dog", char_start=0,
            token_start=0, token_ids=(100, 101)
        )
        overlap = make_lock(
            role="actor_second", substring="dog blue", char_start=4,
            token_start=1, token_ids=(101, 102)
        )
        payload = {
            "event_id": "bad-overlap",
            "source_iid": "source",
            "source_caption": text,
            "source_caption_sha256": locator.text_sha256(text),
            "model_text": text,
            "model_text_sha256": locator.text_sha256(text),
            "source_caption_char_start": 0,
            "tokenizer_tree_sha256": TOKENIZER_SHA,
            "roles": [first.as_dict(), overlap.as_dict()],
        }
        with self.assertRaises(locator.SourceOwnedRoleLocatorError):
            locator.SourceRoleEventSpec(
                **{key: value for key, value in payload.items() if key != "roles"},
                roles=(first, overlap), event_sha256=locator.object_sha256(payload)
            )

    def test_real_asset_is_exact_complex4_and_hash_closed(self):
        specs = locator.load_role_span_asset(ASSET)
        raw_asset = json.loads(ASSET.read_text())
        self.assertEqual(raw_asset["status"], "observer_only_diagnostic_not_route")
        self.assertIs(raw_asset["route_authorized"], False)
        self.assertIs(raw_asset["training_authorized"], False)
        self.assertEqual(raw_asset["transformers_version"], "5.5.4")
        self.assertIs(raw_asset["fix_mistral_regex"], True)
        self.assertEqual(
            tuple(item.event_id for item in specs),
            (
                "pour-liquid-into-cup",
                "twist-pull-mushroom",
                "close-door-then-drawer",
                "players-contact-then-separate",
            ),
        )
        self.assertEqual(specs[0].roles[0].token_ids, (8330, 9751, 29606, 27502))
        self.assertEqual(
            {item.tokenizer_tree_sha256 for item in specs},
            {"0e7e4b06b2c321420e2fb97c07d2329837539b09a39bdf4bcbaa6ec1977da616"},
        )

    def test_runtime_version_regex_and_tokenizer_tree_are_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            digest = make_tokenizer_tree(tmp)
            spec = with_tokenizer_sha(make_spec(), digest)
            with mock.patch.object(
                locator.importlib.metadata, "version", return_value="5.5.4"
            ), mock.patch.object(locator, "PINNED_TOKENIZER_TREE_SHA256", digest):
                receipt = locator.validate_pinned_tokenizer_runtime(
                    TinyTokenizer(), spec, tokenizer_dir=tmp,
                    transformers_version="5.5.4",
                )
            self.assertTrue(receipt.fix_mistral_regex)
            with self.assertRaises(locator.SourceOwnedRoleLocatorError):
                locator.validate_pinned_tokenizer_runtime(
                    TinyTokenizer(), spec, tokenizer_dir=tmp,
                    transformers_version="5.5.3",
                )
            with mock.patch.object(
                locator.importlib.metadata, "version", return_value="5.5.3"
            ), mock.patch.object(locator, "PINNED_TOKENIZER_TREE_SHA256", digest):
                with self.assertRaises(locator.SourceOwnedRoleLocatorError):
                    locator.validate_pinned_tokenizer_runtime(
                        TinyTokenizer(), spec, tokenizer_dir=tmp,
                        transformers_version="5.5.4",
                    )
            with mock.patch.object(
                locator.importlib.metadata, "version", return_value="5.5.4"
            ), mock.patch.object(locator, "PINNED_TOKENIZER_TREE_SHA256", digest):
                with self.assertRaises(locator.SourceOwnedRoleLocatorError):
                    locator.validate_pinned_tokenizer_runtime(
                        TinyTokenizer(fix_mistral_regex=False), spec,
                        tokenizer_dir=tmp, transformers_version="5.5.4",
                    )
            (Path(tmp) / "tokenizer.json").write_bytes(b"mutated")
            with mock.patch.object(
                locator.importlib.metadata, "version", return_value="5.5.4"
            ), mock.patch.object(locator, "PINNED_TOKENIZER_TREE_SHA256", digest):
                with self.assertRaises(locator.SourceOwnedRoleLocatorError):
                    locator.validate_pinned_tokenizer_runtime(
                        TinyTokenizer(), spec, tokenizer_dir=tmp,
                        transformers_version="5.5.4",
                    )


class ObserverAndMaskTests(unittest.TestCase):
    def test_public_observer_authority_has_no_action_carrier_parameter(self):
        signatures = (
            inspect.signature(locator.SourceRoleObserverInvocation),
            inspect.signature(locator.SourceOwnedRoleAttn2Observer),
            inspect.signature(locator.bind_source_text_provenance),
            inspect.signature(locator.observe_source_roles),
            inspect.signature(locator.install_source_owned_role_observer),
        )
        forbidden_fragments = ("donor", "anchor")
        for signature in signatures:
            names = tuple(signature.parameters)
            self.assertFalse(any(fragment in name for fragment in forbidden_fragments for name in names))
        with self.assertRaises(TypeError):
            locator.SourceRoleObserverInvocation(action_anchor=object())

    def test_observer_returns_official_output_object_bit_exact(self):
        bank = locator.SourceRoleCaptureBank((4,))
        base = FakeOfficialAttn2()
        processor = locator.SourceOwnedRoleAttn2Observer(
            base, block_index=4, capture_bank=bank
        )
        hidden, _text, output = invoke(processor, spec=make_spec(), bank=bank)
        self.assertIs(output, base.last_output)
        self.assertTrue(torch.equal(output, hidden.square() + 0.125))
        self.assertEqual(output.data_ptr(), base.last_output.data_ptr())
        self.assertEqual((base.base_calls, base.project_calls), (1, 1))
        self.assertEqual(processor.statistics()["output_modified"], False)
        shard = bank.shards_for(
            event_id="synthetic-source-event", step_index=3, block_index=4
        )[0]
        self.assertEqual(tuple(shard.affinity.shape), (2, 42))
        self.assertFalse(shard.affinity.requires_grad)

    def test_inactive_observer_is_plain_delegation_without_projection(self):
        bank = locator.SourceRoleCaptureBank((2,))
        base = FakeOfficialAttn2()
        processor = locator.SourceOwnedRoleAttn2Observer(
            base, block_index=2, capture_bank=bank
        )
        hidden = role_pattern_hidden()
        result = processor(object(), hidden, encoder_hidden_states=source_text_hidden())
        self.assertIs(result, base.last_output)
        self.assertEqual((base.base_calls, base.project_calls), (1, 0))
        self.assertEqual(bank.capture_count, 0)

    def test_wrong_text_storage_and_non_source_geometry_fail_after_base(self):
        bank = locator.SourceRoleCaptureBank((4,))
        base = FakeOfficialAttn2()
        processor = locator.SourceOwnedRoleAttn2Observer(base, block_index=4, capture_bank=bank)
        spec = make_spec()
        geometry = locator.SourceVisualGeometry(height=1, width=2)
        with tempfile.TemporaryDirectory() as tmp:
            runtime_spec = with_tokenizer_sha(spec, make_tokenizer_tree(tmp))
            with mock.patch.object(
                locator.importlib.metadata, "version", return_value="5.5.4"
            ), mock.patch.object(
                locator, "PINNED_TOKENIZER_TREE_SHA256", runtime_spec.tokenizer_tree_sha256
            ):
                provenance, conditioned = locator.bind_source_text_provenance(
                    tokenizer=TinyTokenizer(), tokenizer_dir=tmp,
                    transformers_version=locator.PINNED_TRANSFORMERS_VERSION,
                    event_spec=runtime_spec,
                    raw_source_text_hidden_states=source_text_hidden(),
                    derive_conditioned_source_text=lambda value: value.clone(),
                    renderer_text_length=5,
                )
            invocation = locator.SourceRoleObserverInvocation(
                capture_bank=bank, event_spec=runtime_spec, geometry=geometry,
                source_text_provenance=provenance,
                step_index=3, ulysses=locator.UlyssesVisualShard(geometry, 0, 1)
            )
            with locator.observe_source_roles(invocation):
                with self.assertRaises(locator.SourceOwnedRoleLocatorError):
                    processor(
                        object(), role_pattern_hidden(),
                        encoder_hidden_states=conditioned[:, :5].clone(),
                        batch_image_vae_seqlen=[42], text_features_length=[5],
                        origin_hidden_states_seq_len=42,
                    )
        self.assertEqual(base.base_calls, 1)
        self.assertEqual(base.project_calls, 0)

    def test_sp4_replicated_root_text_is_accepted_but_equal_clone_is_not(self):
        bank = locator.SourceRoleCaptureBank((4,))
        base = FakeOfficialAttn2()
        processor = locator.SourceOwnedRoleAttn2Observer(base, block_index=4, capture_bank=bank)
        geometry = locator.SourceVisualGeometry(height=1, width=2)
        with tempfile.TemporaryDirectory() as tmp:
            spec = with_tokenizer_sha(make_spec(), make_tokenizer_tree(tmp))
            with mock.patch.object(
                locator.importlib.metadata, "version", return_value="5.5.4"
            ), mock.patch.object(
                locator, "PINNED_TOKENIZER_TREE_SHA256", spec.tokenizer_tree_sha256
            ):
                provenance, conditioned = locator.bind_source_text_provenance(
                    tokenizer=TinyTokenizer(), tokenizer_dir=tmp,
                    transformers_version=locator.PINNED_TRANSFORMERS_VERSION,
                    event_spec=spec,
                    raw_source_text_hidden_states=source_text_hidden(),
                    derive_conditioned_source_text=lambda value: value.clone(),
                    renderer_text_length=512,
                )
            invocation = locator.SourceRoleObserverInvocation(
                capture_bank=bank, event_spec=spec, geometry=geometry,
                source_text_provenance=provenance, step_index=0,
                ulysses=locator.UlyssesVisualShard(geometry, 0, 4),
            )
            local_hidden = role_pattern_hidden()[:, :11].contiguous()
            rank_local_view = conditioned
            with locator.observe_source_roles(invocation):
                output = processor(
                    object(), local_hidden,
                    encoder_hidden_states=rank_local_view,
                    batch_image_vae_seqlen=[42], text_features_length=[512],
                    origin_hidden_states_seq_len=42,
                )
            self.assertIs(output, base.last_output)
            shard = bank.shards_for(
                event_id=spec.event_id, step_index=0, block_index=4
            )[0]
            self.assertEqual(tuple(shard.affinity.shape), (2, 11))

            bank2 = locator.SourceRoleCaptureBank((4,))
            processor2 = locator.SourceOwnedRoleAttn2Observer(
                FakeOfficialAttn2(), block_index=4, capture_bank=bank2
            )
            invocation2 = locator.SourceRoleObserverInvocation(
                capture_bank=bank2, event_spec=spec, geometry=geometry,
                source_text_provenance=provenance, step_index=1,
                ulysses=locator.UlyssesVisualShard(geometry, 0, 4),
            )
            with locator.observe_source_roles(invocation2):
                with self.assertRaises(locator.SourceOwnedRoleLocatorError):
                    processor2(
                        object(), local_hidden,
                        encoder_hidden_states=rank_local_view.clone(),
                        batch_image_vae_seqlen=[42], text_features_length=[512],
                        origin_hidden_states_seq_len=42,
                    )

    def test_ulysses_explicit_shard_roundtrip_and_global_21hw(self):
        geometry = locator.SourceVisualGeometry(height=1, width=2)
        rows = []
        full = torch.arange(84, dtype=torch.float32).reshape(2, 42)
        for rank in range(4):
            layout = locator.UlyssesVisualShard(geometry=geometry, rank=rank, size=4)
            shard = locator.RoleAffinityShard(
                event_id="synthetic-source-event",
                source_text_provenance_sha256="2" * 64,
                step_index=1, block_index=7,
                role_names=("actor_first", "object_second"), layout=layout,
                affinity=full[:, layout.global_start:layout.global_stop].contiguous(),
                null_affinity=torch.zeros(
                    layout.valid_local_tokens, dtype=torch.float32
                ).contiguous(),
                shuffled_affinity=full.flip(0)[
                    :, layout.global_start:layout.global_stop
                ].contiguous(),
            )
            rebuilt = locator.RoleAffinityShard.from_collective(
                shard.padded_collective_tensor(), shard.collective_metadata()
            )
            self.assertTrue(torch.equal(rebuilt.affinity, shard.affinity))
            rows.append(rebuilt)
        global_value = locator.assemble_global_role_affinity(rows)
        self.assertEqual(tuple(global_value.affinity.shape), (2, 21, 1, 2))
        self.assertTrue(torch.equal(global_value.affinity.reshape(2, 42), full))
        self.assertEqual(tuple(global_value.null_affinity.shape), (21, 1, 2))
        self.assertEqual(tuple(global_value.shuffled_affinity.shape), (2, 21, 1, 2))
        with self.assertRaises(locator.SourceOwnedRoleLocatorError):
            locator.assemble_global_role_affinity(rows[:-1])
        wrong_authority = list(rows)
        wrong_authority[-1] = replace(
            wrong_authority[-1], source_text_provenance_sha256="3" * 64
        )
        with self.assertRaises(locator.SourceOwnedRoleLocatorError):
            locator.assemble_global_role_affinity(wrong_authority)

    def test_role_permutation_changes_exclusive_phase0_masks(self):
        def observed_masks(spec):
            bank = locator.SourceRoleCaptureBank((4,))
            processor = locator.SourceOwnedRoleAttn2Observer(
                FakeOfficialAttn2(), block_index=4, capture_bank=bank
            )
            invoke(processor, spec=spec, bank=bank)
            global_value = locator.assemble_global_role_affinity(
                bank.shards_for(
                    event_id="synthetic-source-event", step_index=3, block_index=4
                )
            )
            return locator.build_source_owned_role_masks(
                global_value, spec,
                policy=locator.RoleMaskPolicy(keep_fraction=0.5)
            )

        normal = observed_masks(make_spec(swapped=False))
        swapped = observed_masks(make_spec(swapped=True))
        self.assertFalse(torch.equal(normal.masks, swapped.masks))
        self.assertTrue(torch.equal(normal.masks[0], swapped.masks[1]))
        self.assertTrue(torch.equal(normal.masks[1], swapped.masks[0]))
        self.assertEqual(tuple(normal.phase0_masks.shape), (2, 1, 2))
        self.assertFalse(bool((normal.masks.sum(dim=0) > 1).any().item()))
        self.assertTrue(bool(normal.protected_union.all().item()))
        self.assertEqual(len(normal.receipt_sha256), 64)

    def test_flat_affinity_yields_unqualified_empty_masks(self):
        geometry = locator.SourceVisualGeometry(height=1, width=2)
        affinity = locator.GlobalRoleAffinity(
            event_id="synthetic-source-event",
            source_text_provenance_sha256="2" * 64,
            step_index=0, block_index=0,
            role_names=make_spec().role_names, geometry=geometry,
            affinity=torch.ones((2, 21, 1, 2), dtype=torch.float32).contiguous(),
            null_affinity=torch.ones((21, 1, 2), dtype=torch.float32).contiguous(),
            shuffled_affinity=torch.ones((2, 21, 1, 2), dtype=torch.float32).contiguous(),
        )
        masks = locator.build_source_owned_role_masks(affinity, make_spec())
        self.assertFalse(masks.qualified)
        self.assertEqual(int(masks.masks.sum().item()), 0)
        self.assertEqual(len(masks.affinity_sha256), 64)
        self.assertEqual(len(masks.mask_sha256), 64)


class InstallationTests(unittest.TestCase):
    class Attn2:
        def __init__(self):
            self.processor = FakeOfficialAttn2()

        def set_processor(self, processor):
            self.processor = processor

    class Model:
        def __init__(self):
            self.blocks = [SimpleNamespace(attn2=InstallationTests.Attn2()) for _ in range(30)]

    def test_install_is_exact_attn2_only_and_reversible(self):
        model = self.Model()
        originals = (model.blocks[1].attn2.processor, model.blocks[7].attn2.processor)
        bank = locator.SourceRoleCaptureBank((1, 7))
        handle = locator.install_source_owned_role_observer(model, capture_bank=bank)
        self.assertIsInstance(model.blocks[1].attn2.processor, locator.SourceOwnedRoleAttn2Observer)
        self.assertIsInstance(model.blocks[7].attn2.processor, locator.SourceOwnedRoleAttn2Observer)
        self.assertEqual(handle.receipt()["parameters_added"], 0)
        handle.restore()
        self.assertIs(model.blocks[1].attn2.processor, originals[0])
        self.assertIs(model.blocks[7].attn2.processor, originals[1])


if __name__ == "__main__":
    unittest.main()
