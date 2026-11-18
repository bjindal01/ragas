from __future__ import annotations

import sys
import types
import typing as t
from unittest.mock import MagicMock

import numpy as np
import pytest

from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings


@pytest.fixture
def fake_st(monkeypatch):
    """Replace sentence_transformers so no model is downloaded."""
    module = types.ModuleType("sentence_transformers")
    model = MagicMock(name="SentenceTransformerInstance")

    def encode(inputs, **kwargs):
        if isinstance(inputs, str):
            return np.array([float(len(inputs)), 1.0])
        return np.array([[float(len(s)), 1.0] for s in inputs])

    model.encode.side_effect = encode
    module.SentenceTransformer = MagicMock(return_value=model)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace huggingface_hub so no network client is created."""
    module = types.ModuleType("huggingface_hub")
    client = MagicMock(name="InferenceClientInstance")
    module.InferenceClient = MagicMock(return_value=client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


class TestLocalModel:
    def test_init_loads_sentence_transformer_with_device_and_kwargs(self, fake_st):
        emb = HuggingFaceEmbeddings(
            model="all-MiniLM", device="cpu", trust_remote_code=True
        )
        fake_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM", device="cpu", trust_remote_code=True
        )
        assert emb.model_instance is fake_st.SentenceTransformer.return_value
        assert not hasattr(emb, "client")

    def test_missing_sentence_transformers_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match="pip install sentence-transformers"):
            HuggingFaceEmbeddings(model="m")

    def test_embed_text_returns_list_and_passes_normalize(self, fake_st):
        emb = HuggingFaceEmbeddings(model="m", normalize_embeddings=False)
        result = emb.embed_text("abcd", convert_to_numpy=True)

        assert result == [4.0, 1.0]
        assert isinstance(result, list)
        emb.model_instance.encode.assert_called_once_with(
            "abcd", normalize_embeddings=False, convert_to_numpy=True
        )

    def test_embed_texts_uses_batch_size(self, fake_st):
        emb = HuggingFaceEmbeddings(model="m", batch_size=7)
        assert emb.embed_texts(["a", "bcd"]) == [[1.0, 1.0], [3.0, 1.0]]
        emb.model_instance.encode.assert_called_once_with(
            ["a", "bcd"], normalize_embeddings=True, batch_size=7
        )

    def test_embed_texts_validates(self, fake_st):
        emb = HuggingFaceEmbeddings(model="m")
        with pytest.raises(ValueError, match="cannot be empty"):
            emb.embed_texts([])
        with pytest.raises(ValueError, match="index 0 cannot be empty"):
            emb.embed_texts([" "])
        emb.model_instance.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_paths(self, fake_st):
        emb = HuggingFaceEmbeddings(model="m")
        assert await emb.aembed_text("xyz") == [3.0, 1.0]
        assert await emb.aembed_texts(["ab", "c"]) == [[2.0, 1.0], [1.0, 1.0]]

    @pytest.mark.asyncio
    async def test_aembed_texts_validates(self, fake_st):
        emb = HuggingFaceEmbeddings(model="m")
        with pytest.raises(ValueError, match="must be a string"):
            await emb.aembed_texts([None])  # type: ignore[list-item]

    def test_repr_default(self, fake_st):
        emb = HuggingFaceEmbeddings(model="m")
        assert repr(emb) == (
            "HuggingFaceEmbeddings(provider='huggingface', model='m', "
            "client=<SentenceTransformer>, use_api=False)"
        )
        assert str(emb) == repr(emb)

    def test_repr_non_default(self, fake_st):
        emb = HuggingFaceEmbeddings(
            model="m",
            device="cuda",
            normalize_embeddings=False,
            batch_size=8,
            foo=1,
        )
        assert repr(emb) == (
            "HuggingFaceEmbeddings(provider='huggingface', model='m', "
            "client=<SentenceTransformer>, use_api=False, device='cuda', "
            "normalize_embeddings=False, batch_size=8, +1 model_kwargs)"
        )


class TestApiClient:
    def test_init_creates_inference_client(self, fake_hub, fake_st):
        emb = HuggingFaceEmbeddings(model="org/model", use_api=True, api_key="hf_x")
        fake_hub.InferenceClient.assert_called_once_with(
            model="org/model", token="hf_x"
        )
        assert emb.client is fake_hub.InferenceClient.return_value
        fake_st.SentenceTransformer.assert_not_called()

    def test_missing_hub_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ImportError, match="pip install huggingface-hub"):
            HuggingFaceEmbeddings(model="m", use_api=True)

    def test_embed_text_unwraps_nested_response(self, fake_hub):
        emb = HuggingFaceEmbeddings(model="m", use_api=True)
        emb.client.feature_extraction.return_value = [[0.1, 0.2]]

        assert emb.embed_text("hi", truncate=True) == [0.1, 0.2]
        emb.client.feature_extraction.assert_called_once_with("hi", truncate=True)

    def test_embed_text_flat_response(self, fake_hub):
        emb = HuggingFaceEmbeddings(model="m", use_api=True)
        emb.client.feature_extraction.return_value = (0.5, 0.6)

        assert emb.embed_text("hi") == [0.5, 0.6]

    def test_embed_texts_calls_per_text_across_batches(self, fake_hub):
        emb = HuggingFaceEmbeddings(model="m", use_api=True, batch_size=2)
        responses: t.Dict[str, t.Any] = {
            "a": [[1.0]],
            "b": [2.0],
            "c": [[3.0]],
        }
        emb.client.feature_extraction.side_effect = lambda text, **kw: responses[text]

        assert emb.embed_texts(["a", "b", "c"], normalize=True) == [
            [1.0],
            [2.0],
            [3.0],
        ]
        calls = emb.client.feature_extraction.call_args_list
        assert [c.args[0] for c in calls] == ["a", "b", "c"]
        assert all(c.kwargs == {"normalize": True} for c in calls)

    @pytest.mark.asyncio
    async def test_async_api_paths(self, fake_hub):
        emb = HuggingFaceEmbeddings(model="m", use_api=True)
        emb.client.feature_extraction.side_effect = lambda text, **kw: [
            [float(len(text))]
        ]

        assert await emb.aembed_text("abc") == [3.0]
        assert await emb.aembed_texts(["a", "bb"]) == [[1.0], [2.0]]

    def test_repr_api(self, fake_hub):
        emb = HuggingFaceEmbeddings(model="m", use_api=True, device="cpu")
        # Device and normalization are local-only settings and are not shown.
        assert repr(emb) == (
            "HuggingFaceEmbeddings(provider='huggingface', model='m', "
            "client=<HuggingFaceAPI>, use_api=True)"
        )
