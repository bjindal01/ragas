from __future__ import annotations

import sys
import types
import typing as t
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragas.embeddings import base as base_module
from ragas.embeddings.base import (
    BaseRagasEmbedding,
    LangchainEmbeddingsWrapper,
    LlamaIndexEmbeddingsWrapper,
    _create_modern_embedding,
    _get_provider_registry,
    _infer_embedding_provider_from_llm,
    _is_legacy_embedding_call,
    _looks_like_model_name,
    embedding_factory,
    modern_embedding_factory,
)
from ragas.run_config import RunConfig


@pytest.fixture(autouse=True)
def google_namespace(monkeypatch):
    """Provide an empty ``google`` package when none is installed.

    The provider does ``from google import genai``, which needs the parent
    package to exist before the fake ``google.genai`` in sys.modules is found.
    """
    try:
        import google  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("google")
        pkg.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google", pkg)


@pytest.fixture
def tracked(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(base_module, "track", mock)
    return mock


class LengthEmbedding(BaseRagasEmbedding):
    def __init__(self, cache=None):
        super().__init__(cache=cache)
        self.calls: t.List[t.Tuple[str, t.Dict[str, t.Any]]] = []

    def embed_text(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [float(len(text))]

    async def aembed_text(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [float(len(text)), 1.0]


class TestBaseRagasEmbeddingDefaults:
    def test_embed_texts_calls_embed_text_per_item(self):
        emb = LengthEmbedding()
        assert emb.embed_texts(["a", "bbb"], dim=2) == [[1.0], [3.0]]
        assert emb.calls == [("a", {"dim": 2}), ("bbb", {"dim": 2})]

    def test_embed_texts_validates(self):
        with pytest.raises(ValueError, match="index 0 cannot be empty"):
            LengthEmbedding().embed_texts([""])

    @pytest.mark.asyncio
    async def test_aembed_texts_gathers_in_order(self):
        emb = LengthEmbedding()
        assert await emb.aembed_texts(["abc", "d"]) == [[3.0, 1.0], [1.0, 1.0]]

    @pytest.mark.asyncio
    async def test_aembed_texts_validates(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            await LengthEmbedding().aembed_texts([])

    def test_check_client_async(self):
        emb = LengthEmbedding()

        async def acreate(**kw):
            return None

        async_client = types.SimpleNamespace(
            embeddings=types.SimpleNamespace(create=acreate)
        )
        sync_client = types.SimpleNamespace(
            embeddings=types.SimpleNamespace(create=lambda **kw: None)
        )
        assert emb._check_client_async(async_client) is True
        assert emb._check_client_async(sync_client) is False
        assert emb._check_client_async(object()) is False
        assert (
            emb._check_client_async(
                types.SimpleNamespace(run=acreate), method_path="run"
            )
            is True
        )

    def test_run_async_in_current_loop_delegates(self):
        async def coro():
            return 7

        assert LengthEmbedding()._run_async_in_current_loop(coro()) == 7


class _Provider(BaseRagasEmbedding):
    PROVIDER_NAME = "fake"
    DEFAULT_MODEL = "default-model"

    def __init__(self, model=None, client=None, cache=None, **kwargs):
        super().__init__(cache=cache)
        self.model = model
        self.client = client
        self.kwargs = kwargs

    def embed_text(self, text, **kwargs):
        return [0.0]

    async def aembed_text(self, text, **kwargs):
        return [0.0]


class TestFromFactory:
    def test_requires_client(self):
        cls = type("NeedsClient", (_Provider,), {"REQUIRES_CLIENT": True})
        with pytest.raises(ValueError, match="fake provider requires a client"):
            cls._from_factory(model="m")

    def test_requires_model(self):
        cls = type("NeedsModel", (_Provider,), {"REQUIRES_MODEL": True})
        with pytest.raises(ValueError, match="fake provider requires a model"):
            cls._from_factory()

    def test_uses_default_model_and_passes_kwargs(self):
        inst = _Provider._from_factory(extra=1)
        assert isinstance(inst, _Provider)
        assert inst.model == "default-model"
        assert inst.client is None
        assert inst.kwargs == {"extra": 1}

    def test_client_only_forwarded_when_required(self):
        client = object()
        assert _Provider._from_factory(model="m", client=client).client is None

        cls = type("NeedsClient", (_Provider,), {"REQUIRES_CLIENT": True})
        inst = cls._from_factory(model="m", client=client)
        assert inst.client is client
        assert inst.model == "m"


class TestLegacyDetection:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("text-embedding-3-small", True),
            ("ADA-002", True),
            ("gpt-4", True),
            ("openai", False),
            ("google", False),
            ("huggingface", False),
        ],
    )
    def test_looks_like_model_name(self, name, expected):
        assert _looks_like_model_name(name) is expected

    def test_explicit_interface_wins(self):
        assert _is_legacy_embedding_call("google", None, None, "legacy") is True
        assert _is_legacy_embedding_call("openai", None, None, "modern") is False

    def test_auto_detection(self):
        assert _is_legacy_embedding_call("openai", None, None, "auto") is True
        assert (
            _is_legacy_embedding_call("text-embedding-ada-002", None, None, "auto")
            is True
        )
        assert _is_legacy_embedding_call("openai", None, object(), "auto") is False
        assert _is_legacy_embedding_call("google", None, None, "auto") is False


class TestInferProvider:
    def test_none_defaults_to_openai(self):
        assert _infer_embedding_provider_from_llm(None) == "openai"

    def test_provider_attribute_is_lowercased(self):
        llm = types.SimpleNamespace(provider="Anthropic")
        assert _infer_embedding_provider_from_llm(llm) == "anthropic"

    def test_empty_provider_falls_back_to_class_name(self):
        llm = type("ChatGemini", (), {"provider": ""})()
        assert _infer_embedding_provider_from_llm(llm) == "google"

    @pytest.mark.parametrize(
        "class_name, expected",
        [
            ("ChatClaude", "anthropic"),
            ("VertexLLM", "google"),
            ("ChatGroq", "groq"),
            ("MistralLLM", "mistral"),
            ("CohereChat", "cohere"),
            ("AzureChat", "azure"),
            ("ChatOpenAI", "openai"),
            ("SomethingElse", "openai"),
        ],
    )
    def test_class_name_mapping(self, class_name, expected):
        llm = type(class_name, (), {})()
        assert _infer_embedding_provider_from_llm(llm) == expected


@pytest.fixture
def fake_litellm(monkeypatch):
    module = types.ModuleType("litellm")
    module.embedding = MagicMock()  # type: ignore[attr-defined]
    module.aembedding = AsyncMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


class TestModernFactory:
    def test_registry_contains_all_providers(self):
        registry = _get_provider_registry()
        assert set(registry) == {"openai", "google", "litellm", "huggingface"}

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider: nope"):
            _create_modern_embedding("nope", None, None)

    def test_provider_model_string_is_split(self, fake_litellm):
        from ragas.embeddings.litellm_provider import LiteLLMEmbeddings

        emb = _create_modern_embedding("litellm/cohere/embed-v3", None, None)
        assert isinstance(emb, LiteLLMEmbeddings)
        assert emb.model == "cohere/embed-v3"

    def test_requires_model_for_litellm(self, fake_litellm):
        with pytest.raises(ValueError, match="litellm provider requires a model"):
            _create_modern_embedding("litellm", None, None)

    def test_embedding_factory_modern_passes_cache_and_tracks(
        self, fake_litellm, tracked, tmp_path
    ):
        from ragas.cache import DiskCacheBackend

        cache = DiskCacheBackend(str(tmp_path))
        emb = embedding_factory("LiteLLM", model="m", cache=cache, timeout=5)

        assert emb.cache is cache
        assert emb.timeout == 5  # type: ignore[union-attr]
        event = tracked.call_args.args[0]
        assert event.embedding_type == "factory_modern"
        assert event.provider == "LiteLLM"
        assert event.model == "m"

    def test_embedding_factory_google_uses_default_model(self, tracked, monkeypatch):
        from ragas.embeddings.google_provider import GoogleEmbeddings

        class Client:
            models = types.SimpleNamespace(embed_content=lambda **kw: None)

        genai = types.ModuleType("google.genai")
        genai.Client = Client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google.genai", genai)

        emb = embedding_factory("google", interface="modern")
        assert isinstance(emb, GoogleEmbeddings)
        assert emb.model == GoogleEmbeddings.DEFAULT_MODEL
        assert isinstance(emb.client, Client)

    def test_modern_embedding_factory(self, fake_litellm, tracked):
        from ragas.embeddings.litellm_provider import LiteLLMEmbeddings

        emb = modern_embedding_factory("litellm", model="m")
        assert isinstance(emb, LiteLLMEmbeddings)

    def test_legacy_factory_returns_langchain_wrapper(self, tracked, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        with pytest.warns(DeprecationWarning):
            emb = embedding_factory(
                "text-embedding-3-small", run_config=RunConfig(timeout=12)
            )

        assert isinstance(emb, LangchainEmbeddingsWrapper)
        assert emb.embeddings.model == "text-embedding-3-small"  # type: ignore[attr-defined]
        assert emb.embeddings.request_timeout == 12  # type: ignore[attr-defined]
        from openai import RateLimitError

        assert emb.run_config.exception_types is RateLimitError
        event = tracked.call_args.args[0]
        assert event.embedding_type == "factory_legacy"
        assert event.model == "text-embedding-3-small"

    def test_legacy_factory_default_model(self, tracked, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        with pytest.warns(DeprecationWarning):
            emb = embedding_factory()
        assert emb.embeddings.model == "text-embedding-ada-002"  # type: ignore[union-attr]


class _FakeLangchainEmbeddings:
    model = "fake-lc"

    def embed_query(self, text):
        return [float(len(text))]

    def embed_documents(self, texts):
        return [[float(len(x))] for x in texts]

    async def aembed_query(self, text):
        return [float(len(text)), 1.0]

    async def aembed_documents(self, texts):
        return [[float(len(x)), 1.0] for x in texts]


class TestLangchainWrapper:
    def _wrapper(self, **kwargs):
        with pytest.warns(DeprecationWarning, match="LangchainEmbeddingsWrapper"):
            return LangchainEmbeddingsWrapper(_FakeLangchainEmbeddings(), **kwargs)  # type: ignore[arg-type]

    def test_default_run_config(self):
        wrapper = self._wrapper()
        assert isinstance(wrapper.run_config, RunConfig)
        assert repr(wrapper) == (
            "LangchainEmbeddingsWrapper(embeddings=_FakeLangchainEmbeddings(...))"
        )

    def test_sync_methods_delegate_and_track(self, tracked):
        wrapper = self._wrapper()
        assert wrapper.embed_query("abc") == [3.0]
        assert wrapper.embed_documents(["a", "bb"]) == [[1.0], [2.0]]

        events = [c.args[0] for c in tracked.call_args_list]
        assert [(e.num_requests, e.is_async) for e in events] == [
            (1, False),
            (2, False),
        ]
        assert all(e.provider == "langchain" and e.model == "fake-lc" for e in events)

    @pytest.mark.asyncio
    async def test_async_methods_delegate_and_track(self, tracked):
        wrapper = self._wrapper()
        assert await wrapper.aembed_query("ab") == [2.0, 1.0]
        assert await wrapper.aembed_documents(["a", "b", "c"]) == [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
        events = [c.args[0] for c in tracked.call_args_list]
        assert [(e.num_requests, e.is_async) for e in events] == [(1, True), (3, True)]

    @pytest.mark.asyncio
    async def test_embed_text_and_texts_async_and_sync_paths(self, tracked):
        wrapper = self._wrapper(run_config=RunConfig(max_retries=1))
        assert await wrapper.embed_text("abcd") == [4.0, 1.0]
        assert await wrapper.embed_texts(["ab"], is_async=False) == [[2.0]]

    def test_cache_wraps_methods(self, tmp_path, tracked):
        from ragas.cache import DiskCacheBackend

        inner = _FakeLangchainEmbeddings()
        inner.embed_query = MagicMock(return_value=[9.0])  # type: ignore[method-assign]
        with pytest.warns(DeprecationWarning):
            wrapper = LangchainEmbeddingsWrapper(
                inner,  # type: ignore[arg-type]
                cache=DiskCacheBackend(str(tmp_path)),
            )
        assert wrapper.embed_query("q") == [9.0]
        assert wrapper.embed_query("q") == [9.0]
        assert inner.embed_query.call_count == 1


class TestLlamaIndexWrapper:
    def test_delegates_to_llama_index_methods(self):
        inner = MagicMock()
        inner.get_query_embedding.return_value = [1.0]
        inner.get_text_embedding_batch.return_value = [[1.0], [2.0]]
        with pytest.warns(DeprecationWarning, match="LlamaIndexEmbeddingsWrapper"):
            wrapper = LlamaIndexEmbeddingsWrapper(inner)

        assert wrapper.embed_query("q") == [1.0]
        inner.get_query_embedding.assert_called_once_with("q")
        assert wrapper.embed_documents(["a", "b"]) == [[1.0], [2.0]]
        inner.get_text_embedding_batch.assert_called_once_with(["a", "b"])
        assert isinstance(wrapper.run_config, RunConfig)
        assert repr(wrapper) == "LlamaIndexEmbeddingsWrapper(embeddings=MagicMock(...))"

    @pytest.mark.asyncio
    async def test_async_delegation(self):
        inner = MagicMock()
        inner.aget_query_embedding = AsyncMock(return_value=[3.0])
        inner.aget_text_embedding_batch = AsyncMock(return_value=[[4.0]])
        run_config = RunConfig(timeout=3)
        with pytest.warns(DeprecationWarning):
            wrapper = LlamaIndexEmbeddingsWrapper(inner, run_config=run_config)

        assert wrapper.run_config is run_config
        assert await wrapper.aembed_query("q") == [3.0]
        assert await wrapper.aembed_documents(["x"]) == [[4.0]]
        inner.aget_text_embedding_batch.assert_awaited_once_with(["x"])
