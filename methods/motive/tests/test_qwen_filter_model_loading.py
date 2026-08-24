from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from motive.qwen_filter import LocalQwenBackend


def _fake_loader(
    name: str,
    calls: dict[str, list[tuple[tuple, dict]]],
) -> type:
    class Loader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls[name].append((args, kwargs))
            model = SimpleNamespace(
                config=SimpleNamespace(
                    _commit_hash=f"{name}-revision",
                    _name_or_path=args[0],
                ),
                device="fake-device",
                eval_calls=0,
            )

            def eval_model():
                model.eval_calls += 1
                return model

            model.eval = eval_model
            return model

    Loader.__name__ = name
    return Loader


def _fake_runtime(
    *,
    model_type: str,
    qwen2: bool = True,
    qwen3: bool = True,
    auto_image_text: bool = True,
) -> tuple[ModuleType, ModuleType, dict[str, list[tuple[tuple, dict]]]]:
    names = (
        "config",
        "processor",
        "tokenizer",
        "qwen2",
        "qwen3",
        "auto_image_text",
        "causal",
    )
    calls: dict[str, list[tuple[tuple, dict]]] = {
        name: [] for name in names
    }

    torch_module = ModuleType("torch")
    torch_module.bfloat16 = object()

    transformers_module = ModuleType("transformers")
    transformers_module.__version__ = "fake-transformers-4.57"

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["config"].append((args, kwargs))
            return SimpleNamespace(model_type=model_type)

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["processor"].append((args, kwargs))
            return SimpleNamespace(kind="processor")

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["tokenizer"].append((args, kwargs))
            return SimpleNamespace(kind="tokenizer")

    transformers_module.AutoConfig = AutoConfig
    transformers_module.AutoProcessor = AutoProcessor
    transformers_module.AutoTokenizer = AutoTokenizer
    transformers_module.AutoModelForCausalLM = _fake_loader(
        "causal",
        calls,
    )
    if qwen2:
        transformers_module.Qwen2_5_VLForConditionalGeneration = (
            _fake_loader("qwen2", calls)
        )
    if qwen3:
        transformers_module.Qwen3VLForConditionalGeneration = (
            _fake_loader("qwen3", calls)
        )
    if auto_image_text:
        transformers_module.AutoModelForImageTextToText = _fake_loader(
            "auto_image_text",
            calls,
        )
    return torch_module, transformers_module, calls


def _load_backend(
    torch_module: ModuleType,
    transformers_module: ModuleType,
    *,
    allow_download: bool = False,
    attn_implementation: str = "auto",
) -> LocalQwenBackend:
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch_module,
            "transformers": transformers_module,
        },
    ):
        return LocalQwenBackend(
            model_path="/models/frozen",
            mode="visual",
            attn_implementation=attn_implementation,
            allow_download=allow_download,
            max_new_tokens=321,
        )


class LocalQwenBackendModelLoadingTests(unittest.TestCase):
    def test_qwen25_preserves_specific_loader_and_runtime_kwargs(self) -> None:
        torch_module, transformers_module, calls = _fake_runtime(
            model_type="qwen2_5_vl",
        )
        backend = _load_backend(
            torch_module,
            transformers_module,
            attn_implementation="sdpa",
        )

        self.assertEqual(len(calls["qwen2"]), 1)
        self.assertEqual(calls["qwen3"], [])
        self.assertEqual(calls["auto_image_text"], [])
        self.assertEqual(calls["causal"], [])
        self.assertEqual(
            calls["qwen2"][0],
            (
                ("/models/frozen",),
                {
                    "local_files_only": True,
                    "device_map": "auto",
                    "torch_dtype": torch_module.bfloat16,
                    "attn_implementation": "sdpa",
                },
            ),
        )
        self.assertEqual(
            calls["config"],
            [(("/models/frozen",), {"local_files_only": True})],
        )
        self.assertEqual(
            calls["processor"],
            [(("/models/frozen",), {"local_files_only": True})],
        )
        self.assertIsNone(backend.tokenizer)
        self.assertEqual(backend.model.eval_calls, 1)
        self.assertEqual(backend.model_revision, "qwen2-revision")

    def test_qwen3_prefers_architecture_specific_loader(self) -> None:
        torch_module, transformers_module, calls = _fake_runtime(
            model_type="QWEN3_VL",
        )
        backend = _load_backend(
            torch_module,
            transformers_module,
            allow_download=True,
        )

        self.assertEqual(len(calls["qwen3"]), 1)
        self.assertEqual(calls["auto_image_text"], [])
        self.assertEqual(calls["qwen2"], [])
        self.assertEqual(calls["causal"], [])
        self.assertEqual(
            calls["qwen3"][0],
            (
                ("/models/frozen",),
                {
                    "local_files_only": False,
                    "device_map": "auto",
                    "torch_dtype": torch_module.bfloat16,
                },
            ),
        )
        self.assertEqual(
            calls["processor"],
            [(("/models/frozen",), {"local_files_only": False})],
        )
        self.assertEqual(backend.model_revision, "qwen3-revision")

    def test_qwen3_uses_only_explicit_image_text_auto_fallback(self) -> None:
        torch_module, transformers_module, calls = _fake_runtime(
            model_type="qwen3_vl",
            qwen3=False,
        )
        backend = _load_backend(torch_module, transformers_module)

        self.assertEqual(calls["qwen3"], [])
        self.assertEqual(len(calls["auto_image_text"]), 1)
        self.assertEqual(calls["causal"], [])
        self.assertEqual(
            backend.model_revision,
            "auto_image_text-revision",
        )

    def test_unsupported_vl_model_rejects_before_model_or_processor(self) -> None:
        torch_module, transformers_module, calls = _fake_runtime(
            model_type="other_vl",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"unsupported VL checkpoint model_type=other_vl",
        ):
            _load_backend(torch_module, transformers_module)

        self.assertEqual(calls["qwen2"], [])
        self.assertEqual(calls["qwen3"], [])
        self.assertEqual(calls["auto_image_text"], [])
        self.assertEqual(calls["causal"], [])
        self.assertEqual(calls["processor"], [])

    def test_qwen3_missing_supported_loaders_has_no_causal_fallback(self) -> None:
        torch_module, transformers_module, calls = _fake_runtime(
            model_type="qwen3_vl",
            qwen3=False,
            auto_image_text=False,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"neither Qwen3VLForConditionalGeneration nor "
            r"AutoModelForImageTextToText is available",
        ):
            _load_backend(torch_module, transformers_module)

        self.assertEqual(calls["causal"], [])
        self.assertEqual(calls["processor"], [])

    def test_qwen25_missing_specific_loader_has_no_auto_fallback(self) -> None:
        torch_module, transformers_module, calls = _fake_runtime(
            model_type="qwen2_5_vl",
            qwen2=False,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"does not provide Qwen2_5_VLForConditionalGeneration",
        ):
            _load_backend(torch_module, transformers_module)

        self.assertEqual(calls["qwen3"], [])
        self.assertEqual(calls["auto_image_text"], [])
        self.assertEqual(calls["causal"], [])
        self.assertEqual(calls["processor"], [])


if __name__ == "__main__":
    unittest.main()
