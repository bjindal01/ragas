from __future__ import annotations

import asyncio
import typing as t

import pytest

from ragas.embeddings import utils as emb_utils
from ragas.embeddings.utils import (
    batch_texts,
    get_optimal_batch_size,
    run_async_in_current_loop,
    run_sync_in_async,
    safe_import,
    validate_texts,
)


class TestBatchTexts:
    def test_splits_into_even_batches(self):
        assert batch_texts(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]

    def test_last_batch_holds_remainder(self):
        assert batch_texts(["a", "b", "c", "d", "e"], 2) == [
            ["a", "b"],
            ["c", "d"],
            ["e"],
        ]

    def test_batch_larger_than_input_returns_single_batch(self):
        assert batch_texts(["a", "b"], 10) == [["a", "b"]]

    def test_empty_input_returns_no_batches(self):
        assert batch_texts([], 3) == []

    @pytest.mark.parametrize("size", [0, -1])
    def test_non_positive_batch_size_raises(self, size):
        with pytest.raises(ValueError, match="Batch size must be positive"):
            batch_texts(["a"], size)


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("openai", 100),
        ("AzureOpenAI", 100),
        ("cohere", 96),
        ("google", 5),
        ("Vertex", 5),
        ("huggingface", 32),
        ("litellm", 10),
        ("something-else", 10),
    ],
)
def test_get_optimal_batch_size(provider, expected):
    assert get_optimal_batch_size(provider, "any-model") == expected


class TestValidateTexts:
    def test_single_string_is_wrapped_in_list(self):
        assert validate_texts("hello") == ["hello"]

    def test_valid_list_is_returned_unchanged(self):
        texts = ["a", "b"]
        assert validate_texts(texts) is texts

    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a string or list"):
            validate_texts(("a", "b"))  # type: ignore[arg-type]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_texts([])

    def test_non_string_item_raises_with_index(self):
        with pytest.raises(ValueError, match="Text at index 1 must be a string"):
            validate_texts(["ok", 3])  # type: ignore[list-item]

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_blank_item_raises_with_index(self, blank):
        with pytest.raises(ValueError, match="Text at index 2 cannot be empty"):
            validate_texts(["a", "b", blank])

    def test_blank_single_string_raises(self):
        with pytest.raises(ValueError, match="index 0 cannot be empty"):
            validate_texts("  ")


class TestSafeImport:
    def test_imports_existing_module(self):
        import json

        assert safe_import("json") is json

    def test_imports_dotted_submodule(self):
        import os.path

        assert safe_import("os.path") is os.path

    def test_missing_module_uses_package_name_in_message(self):
        with pytest.raises(ImportError, match="pip install some-pkg") as exc_info:
            safe_import("ragas_definitely_missing_module_xyz", "some-pkg")
        assert "ragas_definitely_missing_module_xyz" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ImportError)

    def test_missing_module_defaults_package_name_to_module(self):
        with pytest.raises(
            ImportError, match="pip install ragas_definitely_missing_module_xyz"
        ):
            safe_import("ragas_definitely_missing_module_xyz")


class TestRunSyncInAsync:
    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs_and_returns_result(self):
        def add(a, b, scale=1):
            return (a + b) * scale

        assert await run_sync_in_async(add, 2, 3, scale=10) == 50

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self):
        def boom():
            raise KeyError("bad")

        with pytest.raises(KeyError, match="bad"):
            await run_sync_in_async(boom)


async def _double(x: int) -> int:
    await asyncio.sleep(0)
    return x * 2


async def _fail() -> t.NoReturn:
    await asyncio.sleep(0)
    raise ValueError("coroutine failed")


class TestRunAsyncInCurrentLoop:
    def test_uses_existing_non_running_loop(self, monkeypatch):
        loop = asyncio.new_event_loop()
        try:
            monkeypatch.setattr(emb_utils.asyncio, "get_event_loop", lambda: loop)
            assert run_async_in_current_loop(_double(4)) == 8
            assert not loop.is_closed()
        finally:
            loop.close()

    def test_creates_new_loop_when_none_available(self, monkeypatch):
        def no_loop():
            raise RuntimeError("no current event loop")

        set_calls: t.List[t.Any] = []
        real_set = asyncio.set_event_loop

        def record_set(loop):
            set_calls.append(loop)
            real_set(loop)

        monkeypatch.setattr(emb_utils.asyncio, "get_event_loop", no_loop)
        monkeypatch.setattr(emb_utils.asyncio, "set_event_loop", record_set)

        assert run_async_in_current_loop(_double(5)) == 10
        # A fresh loop is installed, then cleared after it is closed.
        assert len(set_calls) == 2
        assert set_calls[0].is_closed()
        assert set_calls[1] is None

    @pytest.mark.asyncio
    async def test_running_loop_executes_in_separate_thread(self):
        assert run_async_in_current_loop(_double(21)) == 42

    @pytest.mark.asyncio
    async def test_running_loop_reraises_coroutine_exception(self):
        with pytest.raises(ValueError, match="coroutine failed"):
            run_async_in_current_loop(_fail())
