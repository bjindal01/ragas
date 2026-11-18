import typing as t
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langchain.smith")
pytest.importorskip("langchain.chains.base")

from langsmith.utils import LangSmithNotFoundError  # noqa: E402

from ragas.integrations import langsmith as ls_integration  # noqa: E402
from ragas.integrations.langchain import EvaluatorChain  # noqa: E402
from ragas.metrics.base import MetricType, SingleTurnMetric  # noqa: E402


@dataclass
class ContextCount(SingleTurnMetric):
    name: str = "context_count"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        return float(len(sample.retrieved_contexts or []))


@pytest.fixture
def client():
    with patch.object(ls_integration, "Client") as client_cls:
        yield client_cls.return_value


class TestUploadDataset:
    def test_uploads_when_dataset_is_new(self, client, capsys):
        client.read_dataset.side_effect = LangSmithNotFoundError("missing")
        uploaded = MagicMock()
        uploaded.name = "my-ds"
        uploaded.url = "https://smith.example/my-ds"
        client.upload_dataframe.return_value = uploaded
        testset = MagicMock()

        result = ls_integration.upload_dataset(testset, "my-ds", "desc")

        assert result is uploaded
        client.read_dataset.assert_called_once_with(dataset_name="my-ds")
        client.upload_dataframe.assert_called_once_with(
            df=testset.to_pandas.return_value,
            name="my-ds",
            input_keys=["question"],
            output_keys=["ground_truth"],
            description="desc",
        )
        assert "https://smith.example/my-ds" in capsys.readouterr().out

    def test_raises_when_dataset_exists(self, client):
        client.read_dataset.return_value = "existing"
        with pytest.raises(ValueError, match="already exists"):
            ls_integration.upload_dataset(MagicMock(), "my-ds")
        client.upload_dataframe.assert_not_called()


class TestEvaluate:
    def test_missing_dataset(self, client):
        client.read_dataset.side_effect = LangSmithNotFoundError("missing")
        with pytest.raises(ValueError, match="not found in langsmith"):
            ls_integration.evaluate("nope", llm_or_chain_factory=MagicMock())
        client.run_on_dataset.assert_not_called()

    def test_runs_with_custom_metrics(self, client):
        factory = MagicMock()
        result = ls_integration.evaluate(
            "ds",
            llm_or_chain_factory=factory,
            experiment_name="exp-1",
            metrics=[ContextCount()],
            verbose=True,
        )

        assert result is client.run_on_dataset.return_value
        kwargs = client.run_on_dataset.call_args.kwargs
        assert kwargs["dataset_name"] == "ds"
        assert kwargs["llm_or_chain_factory"] is factory
        assert kwargs["project_name"] == "exp-1"
        assert kwargs["verbose"] is True
        evaluators = kwargs["evaluation"].custom_evaluators
        assert len(evaluators) == 1
        assert isinstance(evaluators[0], EvaluatorChain)
        assert evaluators[0].metric.name == "context_count"

    def test_default_metrics(self, client):
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._faithfulness import faithfulness

        with (
            patch.object(ls_integration, "EvaluatorChain") as chain_cls,
            patch.object(ls_integration, "RunEvalConfig") as config_cls,
        ):
            ls_integration.evaluate("ds", llm_or_chain_factory=MagicMock())

        wrapped = [c.args[0] for c in chain_cls.call_args_list]
        assert wrapped == [
            answer_relevancy,
            context_precision,
            faithfulness,
            context_recall,
        ]
        config_cls.assert_called_once_with(
            custom_evaluators=[chain_cls.return_value] * 4
        )
