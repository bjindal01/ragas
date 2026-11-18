import typing as t
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain.chains.base")
pytest.importorskip("langsmith.evaluation")

from langchain.schema import RUN_KEY  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import FakeEmbeddings  # noqa: E402
from langchain_core.language_models import FakeListChatModel  # noqa: E402

from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings.base import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.integrations.langchain import EvaluatorChain  # noqa: E402
from ragas.llms.base import LangchainLLMWrapper  # noqa: E402
from ragas.metrics.base import (  # noqa: E402
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)


@dataclass
class ContextCount(SingleTurnMetric):
    """Scores a sample with the number of retrieved contexts."""

    name: str = "context_count"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )
    seen: t.List[SingleTurnSample] = field(default_factory=list)
    run_config: t.Any = None

    def init(self, run_config):
        self.run_config = run_config

    async def _single_turn_ascore(self, sample, callbacks):
        self.seen.append(sample)
        return float(len(sample.retrieved_contexts or []))


@dataclass
class ReferenceMetric(ContextCount):
    name: str = "reference_metric"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
            }
        }
    )


@dataclass
class LLMAndEmbeddingsMetric(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    name: str = "llm_and_embeddings"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )

    async def _single_turn_ascore(self, sample, callbacks):
        return 1.0


@pytest.fixture
def chain():
    return EvaluatorChain(ContextCount())


def test_keys(chain):
    assert sorted(chain.input_keys) == ["answer", "contexts", "question"]
    assert chain.output_keys == ["context_count"]


def test_init_passes_run_config_to_metric():
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=7)
    chain = EvaluatorChain(ContextCount(), run_config=run_config)
    assert chain.metric.run_config is run_config


def test_init_wraps_llm_and_embeddings():
    llm = FakeListChatModel(responses=["ok"])
    embeddings = FakeEmbeddings(size=4)
    chain = EvaluatorChain(LLMAndEmbeddingsMetric(), llm=llm, embeddings=embeddings)

    assert isinstance(chain.metric.llm, LangchainLLMWrapper)
    assert chain.metric.llm.langchain_llm is llm
    assert isinstance(chain.metric.embeddings, LangchainEmbeddingsWrapper)
    assert chain.metric.embeddings.embeddings is embeddings


def test_invoke_converts_v1_row_and_documents(chain):
    result = chain.invoke(
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "contexts": [Document(page_content="Paris is the capital."), "plain"],
        }
    )
    assert result["context_count"] == 2.0
    sample = chain.metric.seen[-1]
    assert sample.user_input == "What is the capital of France?"
    assert sample.response == "Paris"
    assert sample.retrieved_contexts == ["Paris is the capital.", "plain"]


async def _ainvoke(chain, inputs):
    return await chain.ainvoke(inputs)


@pytest.mark.asyncio
async def test_ainvoke(chain):
    result = await _ainvoke(
        chain,
        {
            "question": "q",
            "answer": "a",
            "contexts": [Document(page_content="c1"), "c2", "c3"],
        },
    )
    assert result["context_count"] == 3.0
    assert chain.metric.seen[-1].retrieved_contexts == ["c1", "c2", "c3"]


def test_validate_rejects_missing_columns(chain):
    with pytest.raises(ValueError, match='"retrieved_contexts" is required'):
        chain._validate(SingleTurnSample(user_input="q", response="a"))


def test_rejects_multi_turn_only_metric():
    from ragas.metrics.base import MultiTurnMetric

    @dataclass
    class MultiOnly(MultiTurnMetric):
        name: str = "multi_only"

        def init(self, run_config):
            pass

        async def _multi_turn_ascore(self, sample, callbacks):
            return 0.0

    with pytest.raises(AssertionError, match="SingleTurnMetric"):
        EvaluatorChain(MultiOnly())


def _run(outputs):
    return SimpleNamespace(outputs=outputs)


def _example(inputs=None, outputs=None):
    return SimpleNamespace(inputs=inputs, outputs=outputs)


class TestEvaluateRun:
    def test_scores_run(self, chain):
        run = _run({"answer": "Paris", "contexts": ["c1", "c2"]})
        example = _example({"question": "q"}, {"ground_truth": "Paris"})

        result = chain.evaluate_run(run, example)

        assert result.key == "context_count"
        assert result.score == 2.0
        assert RUN_KEY in result.evaluator_info

    def test_adds_ground_truth_when_metric_needs_reference(self):
        chain = EvaluatorChain(ReferenceMetric())
        run = _run({"answer": "Paris", "contexts": ["c1"]})
        example = _example({"question": "q"}, {"ground_truth": "Paris!"})

        result = chain.evaluate_run(run, example)

        assert result.score == 1.0
        assert chain.metric.seen[-1].reference == "Paris!"

    @pytest.mark.parametrize(
        "example, message",
        [
            (None, "expected example to be provided"),
            (_example(None, {"ground_truth": "x"}), "expected example.inputs"),
            (_example({"question": "q"}, None), "expected example.inputs"),
            (_example({"q": "q"}, {"ground_truth": "x"}), "Expected 'question'"),
            (_example({"question": "q"}, {"answer": "x"}), "Expected 'question'"),
        ],
    )
    def test_invalid_examples(self, chain, example, message):
        run = _run({"answer": "a", "contexts": []})
        with pytest.raises(ValueError, match=message):
            chain.evaluate_run(run, example)

    def test_run_without_outputs(self, chain):
        with pytest.raises(AssertionError, match="no outputs"):
            chain.evaluate_run(
                _run(None), _example({"question": "q"}, {"ground_truth": "x"})
            )

    def test_run_missing_output_keys(self, chain):
        with pytest.raises(ValueError, match="Expected 'answer' and 'contexts'"):
            chain.evaluate_run(
                _run({"answer": "a"}),
                _example({"question": "q"}, {"ground_truth": "x"}),
            )
