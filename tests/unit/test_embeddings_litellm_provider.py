from __future__ import annotations

import sys
import types
import typing as t
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.embeddings.litellm_provider import LiteLLMEmbeddings


def _response(vectors: t.List[t.List[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [{"embedding": v} for v in vectors]
    return response


def _batch_echo(input, **kwargs):
    return _response([[float(len(text))] for text in input])


@pytest.fixture
def fake_litellm(monkeypatch):
    module = types.ModuleType("litellm")
    module.embedding = MagicMock(side_effect=_batch_echo)  # type: ignore[attr-defined]

    async def _aembedding(input, **kwargs):
        return _batch_echo(input, **kwargs)

    module.aembedding = AsyncMock(side_effect=_aembedding)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


def test_missing_litellm_raises_helpful_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)
    with pytest.raises(ImportError, match="pip install litellm"):
        LiteLLMEmbeddings(model="text-embedding-3-small")


def test_defaults(fake_litellm):
    emb = LiteLLMEmbeddings(model="text-embedding-3-small")
    assert emb.litellm is fake_litellm
    assert emb.batch_size == 10
    assert emb.timeout == 600
    assert emb.max_retries == 3
    assert emb.litellm_params == {}


def test_prepare_kwargs_minimal_omits_unset_credentials(fake_litellm):
    emb = LiteLLMEmbeddings(model="m")
    assert emb._prepare_kwargs() == {"model": "m", "timeout": 600, "num_retries": 3}


def test_prepare_kwargs_includes_credentials_params_and_overrides(fake_litellm):
    emb = LiteLLMEmbeddings(
        model="azure/emb",
        api_key="sk-test",
        api_base="https://example.invalid",
        api_version="2024-01-01",
        timeout=30,
        max_retries=1,
        dimensions=256,
    )
    kwargs = emb._prepare_kwargs(dimensions=64, user="u1")
    assert kwargs == {
        "model": "azure/emb",
        "timeout": 30,
        "num_retries": 1,
        "dimensions": 64,
        "user": "u1",
        "api_key": "sk-test",
        "api_base": "https://example.invalid",
        "api_version": "2024-01-01",
    }


def test_embed_text_calls_litellm_with_single_input(fake_litellm):
    fake_litellm.embedding.side_effect = None
    fake_litellm.embedding.return_value = _response([[0.1, 0.2, 0.3]])
    emb = LiteLLMEmbeddings(model="m", api_key="k")

    assert emb.embed_text("hello", dimensions=3) == [0.1, 0.2, 0.3]
    fake_litellm.embedding.assert_called_once_with(
        input=["hello"],
        model="m",
        timeout=600,
        num_retries=3,
        dimensions=3,
        api_key="k",
    )


@pytest.mark.asyncio
async def test_aembed_text_uses_async_client(fake_litellm):
    fake_litellm.aembedding.side_effect = None
    fake_litellm.aembedding.return_value = _response([[1.0, 2.0]])
    emb = LiteLLMEmbeddings(model="m")

    assert await emb.aembed_text("hi") == [1.0, 2.0]
    fake_litellm.aembedding.assert_awaited_once_with(
        input=["hi"], model="m", timeout=600, num_retries=3
    )
    fake_litellm.embedding.assert_not_called()


def test_embed_texts_batches_and_preserves_order(fake_litellm):
    emb = LiteLLMEmbeddings(model="m", batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    assert emb.embed_texts(texts) == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    batches = [c.kwargs["input"] for c in fake_litellm.embedding.call_args_list]
    assert batches == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]


def test_embed_texts_accepts_single_string(fake_litellm):
    emb = LiteLLMEmbeddings(model="m")
    assert emb.embed_texts("abc") == [[3.0]]  # type: ignore[arg-type]


def test_embed_texts_validates_input(fake_litellm):
    emb = LiteLLMEmbeddings(model="m")
    with pytest.raises(ValueError, match="index 1 cannot be empty"):
        emb.embed_texts(["ok", "  "])
    with pytest.raises(ValueError, match="cannot be empty"):
        emb.embed_texts([])
    fake_litellm.embedding.assert_not_called()


@pytest.mark.asyncio
async def test_aembed_texts_batches(fake_litellm):
    emb = LiteLLMEmbeddings(model="m", batch_size=3, api_base="http://b")
    result = await emb.aembed_texts(["a", "bb", "ccc", "dddd"], encoding_format="x")

    assert result == [[1.0], [2.0], [3.0], [4.0]]
    calls = fake_litellm.aembedding.await_args_list
    assert [c.kwargs["input"] for c in calls] == [["a", "bb", "ccc"], ["dddd"]]
    for c in calls:
        assert c.kwargs["encoding_format"] == "x"
        assert c.kwargs["api_base"] == "http://b"


@pytest.mark.asyncio
async def test_aembed_texts_validates_input(fake_litellm):
    emb = LiteLLMEmbeddings(model="m")
    with pytest.raises(ValueError, match="must be a string"):
        await emb.aembed_texts([1])  # type: ignore[list-item]
    fake_litellm.aembedding.assert_not_called()


def test_repr_with_defaults(fake_litellm):
    emb = LiteLLMEmbeddings(model="m")
    assert repr(emb) == "LiteLLMEmbeddings(provider='litellm', model='m')"
    assert str(emb) == repr(emb)


def test_repr_shows_non_default_config(fake_litellm):
    emb = LiteLLMEmbeddings(
        model="m",
        api_base="http://b",
        batch_size=4,
        timeout=5,
        max_retries=0,
        foo=1,
        bar=2,
    )
    assert repr(emb) == (
        "LiteLLMEmbeddings(provider='litellm', model='m', api_base='http://b', "
        "batch_size=4, timeout=5, max_retries=0, +2 litellm_params)"
    )


def test_cache_avoids_repeat_calls(fake_litellm, tmp_path):
    from ragas.cache import DiskCacheBackend

    emb = LiteLLMEmbeddings(model="m", cache=DiskCacheBackend(str(tmp_path)))
    first = emb.embed_text("hello")
    second = emb.embed_text("hello")

    assert first == second == [5.0]
    assert fake_litellm.embedding.call_count == 1
