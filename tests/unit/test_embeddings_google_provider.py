from __future__ import annotations

import sys
import types
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.embeddings.google_provider import GoogleEmbeddings


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


class _Values:
    def __init__(self, values: t.Sequence[float]):
        self.values = tuple(values)


class _Models:
    """Mimics ``genai.Client().models`` from the google-genai SDK."""

    def __init__(self):
        self.calls: t.List[t.Dict[str, t.Any]] = []

    def embed_content(self, model, contents, **kwargs):
        self.calls.append({"model": model, "contents": contents, **kwargs})
        items = [contents] if isinstance(contents, str) else contents
        result = MagicMock()
        result.embeddings = [_Values([float(len(s)), 0.5]) for s in items]
        return result


class Client:
    """Named ``Client`` so it is detected as a google-genai client."""

    def __init__(self, *args, **kwargs):
        self.models = _Models()


class _OldGenai:
    """Mimics the module-level API of google-generativeai."""

    def __init__(self):
        self.calls: t.List[t.Dict[str, t.Any]] = []

    def embed_content(self, model, content, **kwargs):
        self.calls.append({"model": model, "content": content, **kwargs})
        return {"embedding": [float(len(content))]}


@pytest.fixture
def no_google_sdks(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google.generativeai", None)


@pytest.fixture
def fake_vertex(monkeypatch):
    model = MagicMock(name="TextEmbeddingModelInstance")
    model.get_embeddings.side_effect = lambda texts, **kw: [
        types.SimpleNamespace(values=[float(len(s))]) for s in texts
    ]
    lm = types.ModuleType("vertexai.language_models")
    lm.TextEmbeddingModel = MagicMock()  # type: ignore[attr-defined]
    lm.TextEmbeddingModel.from_pretrained.return_value = model  # type: ignore[attr-defined]
    pkg = types.ModuleType("vertexai")
    pkg.language_models = lm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vertexai", pkg)
    monkeypatch.setitem(sys.modules, "vertexai.language_models", lm)
    return lm


class TestClientResolution:
    def test_vertex_requires_client(self):
        with pytest.raises(ValueError, match="Vertex AI embeddings require a client"):
            GoogleEmbeddings(use_vertex=True)

    def test_vertex_client_used_as_is(self):
        client = object()
        emb = GoogleEmbeddings(client=client, use_vertex=True, project_id="p")
        assert emb.client is client
        assert emb._use_new_sdk is False

    def test_new_sdk_client_detected(self):
        client = Client()
        emb = GoogleEmbeddings(client=client)
        assert emb.client is client
        assert emb._use_new_sdk is True

    def test_new_sdk_client_detected_by_module(self):
        client = MagicMock()
        client.__module__ = "google.genai.client"
        emb = GoogleEmbeddings(client=client)
        assert emb.client is client
        assert emb._use_new_sdk is True

    def test_old_sdk_module_with_embed_content(self):
        genai = _OldGenai()
        emb = GoogleEmbeddings(client=genai)
        assert emb.client is genai
        assert emb._use_new_sdk is False

    def test_generative_model_resolves_to_module_in_sys_modules(self, monkeypatch):
        genai_module = types.ModuleType("google.generativeai")
        genai_module.embed_content = _OldGenai().embed_content  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google.generativeai", genai_module)

        model_cls = type("GenerativeModel", (), {})
        model_cls.__module__ = "google.generativeai.generative_models"

        emb = GoogleEmbeddings(client=model_cls())
        assert emb.client is genai_module
        assert emb._use_new_sdk is False

    def test_new_sdk_object_without_client_interface_resolves_module(self, monkeypatch):
        genai_module = types.ModuleType("google.genai")
        genai_module.embed_content = _OldGenai().embed_content  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google.genai", genai_module)

        model_cls = type("SomeModel", (), {})
        model_cls.__module__ = "google.genai.models"

        emb = GoogleEmbeddings(client=model_cls())
        assert emb.client is genai_module
        assert emb._use_new_sdk is False

    def test_generative_model_resolves_via_importlib(self, monkeypatch):
        genai_module = types.ModuleType("google.generativeai")
        genai_module.embed_content = _OldGenai().embed_content  # type: ignore[attr-defined]
        monkeypatch.delitem(sys.modules, "google.generativeai", raising=False)

        import importlib

        real_import_module = importlib.import_module
        requested: t.List[str] = []

        def fake_import_module(name, *args, **kwargs):
            requested.append(name)
            if name == "google.generativeai":
                return genai_module
            return real_import_module(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        model_cls = type("GenerativeModel", (), {})
        model_cls.__module__ = "google.generativeai.generative_models"

        emb = GoogleEmbeddings(client=model_cls())
        assert emb.client is genai_module
        assert requested == ["google.generativeai"]

    def test_unresolvable_client_falls_back_and_raises_without_sdks(
        self, no_google_sdks
    ):
        model_cls = type("GenerativeModel", (), {})
        model_cls.__module__ = "google.generativeai.generative_models"
        with pytest.raises(ImportError, match="google-genai"):
            GoogleEmbeddings(client=model_cls())

    def test_auto_import_prefers_new_sdk(self, monkeypatch):
        genai = types.ModuleType("google.genai")
        genai.Client = Client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google.genai", genai)

        emb = GoogleEmbeddings()
        assert isinstance(emb.client, Client)
        assert emb._use_new_sdk is True

    def test_auto_import_falls_back_to_old_sdk_when_client_fails(self, monkeypatch):
        genai = types.ModuleType("google.genai")

        def failing_client():
            raise RuntimeError("missing API key")

        genai.Client = failing_client  # type: ignore[attr-defined]
        old = types.ModuleType("google.generativeai")
        monkeypatch.setitem(sys.modules, "google.genai", genai)
        monkeypatch.setitem(sys.modules, "google.generativeai", old)

        emb = GoogleEmbeddings()
        assert emb.client is old
        assert emb._use_new_sdk is False

    def test_auto_import_old_sdk_when_new_missing(self, monkeypatch):
        old = types.ModuleType("google.generativeai")
        monkeypatch.setitem(sys.modules, "google.genai", None)
        monkeypatch.setitem(sys.modules, "google.generativeai", old)

        emb = GoogleEmbeddings()
        assert emb.client is old
        assert emb._use_new_sdk is False

    def test_auto_import_without_any_sdk_raises(self, no_google_sdks):
        with pytest.raises(ImportError, match="google-generativeai"):
            GoogleEmbeddings()


class TestNewSdk:
    def test_embed_text_passes_model_and_merged_kwargs(self):
        client = Client()
        emb = GoogleEmbeddings(client=client, model="gemini-embedding-001", a=1, b=2)

        result = emb.embed_text("abc", b=3)
        assert result == [3.0, 0.5]
        assert isinstance(result, list)
        assert client.models.calls == [
            {"model": "gemini-embedding-001", "contents": "abc", "a": 1, "b": 3}
        ]

    def test_embed_texts_uses_single_batch_call(self):
        client = Client()
        emb = GoogleEmbeddings(client=client, model="m")

        assert emb.embed_texts(["a", "bb"]) == [[1.0, 0.5], [2.0, 0.5]]
        assert client.models.calls == [{"model": "m", "contents": ["a", "bb"]}]

    def test_embed_texts_validates(self):
        client = Client()
        emb = GoogleEmbeddings(client=client)
        with pytest.raises(ValueError, match="cannot be empty"):
            emb.embed_texts([])
        with pytest.raises(ValueError, match="index 1 cannot be empty"):
            emb.embed_texts(["a", ""])
        assert client.models.calls == []

    @pytest.mark.asyncio
    async def test_async_paths(self):
        client = Client()
        emb = GoogleEmbeddings(client=client, model="m")

        assert await emb.aembed_text("abcd") == [4.0, 0.5]
        assert await emb.aembed_texts(["a", "bcd"]) == [[1.0, 0.5], [3.0, 0.5]]
        assert [c["contents"] for c in client.models.calls] == ["abcd", ["a", "bcd"]]

    @pytest.mark.asyncio
    async def test_aembed_texts_validates(self):
        emb = GoogleEmbeddings(client=Client())
        with pytest.raises(ValueError, match="must be a string"):
            await emb.aembed_texts([42])  # type: ignore[list-item]


class TestOldSdk:
    def test_embed_text_prefixes_model_name(self):
        genai = _OldGenai()
        emb = GoogleEmbeddings(client=genai, model="text-embedding-004", task="x")

        assert emb.embed_text("hello", title="t") == [5.0]
        assert genai.calls == [
            {
                "model": "models/text-embedding-004",
                "content": "hello",
                "task": "x",
                "title": "t",
            }
        ]

    def test_embed_texts_calls_once_per_text(self):
        genai = _OldGenai()
        emb = GoogleEmbeddings(client=genai, model="m")

        assert emb.embed_texts(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]
        assert [c["content"] for c in genai.calls] == ["a", "bb", "ccc"]
        assert all(c["model"] == "models/m" for c in genai.calls)


class TestVertex:
    def test_embed_text(self, fake_vertex):
        emb = GoogleEmbeddings(client=object(), use_vertex=True, model="v-model", k=1)

        assert emb.embed_text("abc", j=2) == [3.0]
        fake_vertex.TextEmbeddingModel.from_pretrained.assert_called_once_with(
            "v-model"
        )
        model = fake_vertex.TextEmbeddingModel.from_pretrained.return_value
        model.get_embeddings.assert_called_once_with(["abc"], k=1, j=2)

    def test_embed_texts_batch(self, fake_vertex):
        emb = GoogleEmbeddings(client=object(), use_vertex=True)

        assert emb.embed_texts(["a", "bb"]) == [[1.0], [2.0]]
        model = fake_vertex.TextEmbeddingModel.from_pretrained.return_value
        model.get_embeddings.assert_called_once_with(["a", "bb"])

    @pytest.mark.asyncio
    async def test_aembed_text(self, fake_vertex):
        emb = GoogleEmbeddings(client=object(), use_vertex=True)
        assert await emb.aembed_text("abcd") == [4.0]

    @pytest.mark.parametrize("method", ["embed_text", "embed_texts"])
    def test_missing_vertex_sdk_raises(self, monkeypatch, method):
        monkeypatch.setitem(sys.modules, "vertexai", None)
        monkeypatch.setitem(sys.modules, "vertexai.language_models", None)
        emb = GoogleEmbeddings(client=object(), use_vertex=True)
        arg: t.Any = "x" if method == "embed_text" else ["x"]
        with pytest.raises(ImportError, match="google-cloud-aiplatform"):
            getattr(emb, method)(arg)


class TestRepr:
    def test_gemini_repr(self):
        emb = GoogleEmbeddings(client=Client(), model="m")
        assert repr(emb) == (
            "GoogleEmbeddings(provider='google', model='m', client=<Client>, "
            "use_vertex=False)"
        )
        assert str(emb) == repr(emb)

    def test_vertex_repr_defaults(self):
        emb = GoogleEmbeddings(client=object(), use_vertex=True, model="m")
        assert repr(emb) == (
            "GoogleEmbeddings(provider='google', model='m', client=<VertexAI>, "
            "use_vertex=True)"
        )

    def test_vertex_repr_with_project_and_location(self):
        emb = GoogleEmbeddings(
            client=object(),
            use_vertex=True,
            model="m",
            project_id="proj",
            location="europe-west4",
        )
        assert repr(emb) == (
            "GoogleEmbeddings(provider='google', model='m', client=<VertexAI>, "
            "use_vertex=True, project_id='proj', location='europe-west4')"
        )
