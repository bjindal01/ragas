from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ragas.dataset_schema import (
    EvaluationDataset,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

pytest.importorskip("llama_index.core")

from ragas.integrations import llama_index as li_integration


class _Node:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class _FakeQueryEngine:
    def __init__(self, answers, fail_on=()):
        self.answers = answers
        self.fail_on = set(fail_on)
        self.queries = []

    async def aquery(self, query):
        self.queries.append(query)
        if query in self.fail_on:
            raise RuntimeError("query engine exploded")
        answer, contexts = self.answers[query]
        return SimpleNamespace(
            response=answer, source_nodes=[_Node(c) for c in contexts]
        )


def _dataset(*questions):
    return EvaluationDataset(
        samples=[
            SingleTurnSample(user_input=q, reference=f"ref {q}") for q in questions
        ]
    )


class TestEvaluate:
    def test_populates_responses_and_contexts(self):
        engine = _FakeQueryEngine(
            {
                "q1": ("a1", ["c1", "c2"]),
                "q2": (None, ["c3"]),
            }
        )
        dataset = _dataset("q1", "q2")
        sentinel = object()

        with patch.object(
            li_integration, "ragas_evaluate", return_value=sentinel
        ) as mock_eval:
            result = li_integration.evaluate(
                engine, dataset, metrics=[], show_progress=False
            )

        assert result is sentinel
        assert sorted(engine.queries) == ["q1", "q2"]
        s1, s2 = dataset.samples
        assert s1.response == "a1"
        assert s1.retrieved_contexts == ["c1", "c2"]
        # a None response from the engine becomes an empty string
        assert s2.response == ""
        assert s2.retrieved_contexts == ["c3"]

        kwargs = mock_eval.call_args.kwargs
        assert kwargs["dataset"] is dataset
        assert kwargs["llm"] is None
        assert kwargs["embeddings"] is None
        assert kwargs["return_executor"] is False
        assert kwargs["run_config"] is not None

    def test_failed_queries_become_none(self, caplog):
        engine = _FakeQueryEngine({"ok": ("fine", ["ctx"])}, fail_on=["bad"])
        dataset = _dataset("ok", "bad")

        with patch.object(li_integration, "ragas_evaluate", return_value=None):
            li_integration.evaluate(
                engine,
                dataset,
                metrics=[],
                show_progress=False,
                raise_exceptions=False,
            )

        ok, bad = dataset.samples
        assert ok.response == "fine"
        assert bad.response is None
        assert bad.retrieved_contexts is None
        assert "Query engine failed for query 1" in caplog.text

    def test_wraps_llamaindex_llm_and_embeddings(self):
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.llms import MockLLM

        from ragas.embeddings.base import LlamaIndexEmbeddingsWrapper
        from ragas.llms.base import LlamaIndexLLMWrapper

        engine = _FakeQueryEngine({"q": ("a", [])})
        llm, embeddings = MockLLM(), MockEmbedding(embed_dim=4)
        with patch.object(li_integration, "ragas_evaluate") as mock_eval:
            li_integration.evaluate(
                engine,
                _dataset("q"),
                metrics=[],
                llm=llm,
                embeddings=embeddings,
                show_progress=False,
            )
        kwargs = mock_eval.call_args.kwargs
        assert isinstance(kwargs["llm"], LlamaIndexLLMWrapper)
        assert kwargs["llm"].llm is llm
        assert isinstance(kwargs["embeddings"], LlamaIndexEmbeddingsWrapper)
        assert kwargs["embeddings"].embeddings is embeddings

    def test_rejects_non_evaluation_dataset(self):
        with pytest.raises(ValueError, match="EvaluationDataset"):
            li_integration.evaluate(
                _FakeQueryEngine({}),
                [{"user_input": "q"}],
                metrics=[],  # type: ignore[arg-type]
            )

    def test_multi_turn_not_supported(self):
        dataset = EvaluationDataset(
            samples=[MultiTurnSample(user_input=[HumanMessage(content="hi")])]
        )
        with pytest.raises(NotImplementedError, match="Multi-turn"):
            li_integration.evaluate(
                _FakeQueryEngine({}), dataset, metrics=[], show_progress=False
            )


class TestConvertToRagasMessages:
    @pytest.fixture
    def llama_types(self):
        workflow = pytest.importorskip("llama_index.core.agent.workflow")
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        from llama_index.core.llms.llm import ToolSelection
        from llama_index.core.tools import ToolOutput

        return SimpleNamespace(
            AgentInput=workflow.AgentInput,
            AgentOutput=workflow.AgentOutput,
            ToolCallResult=workflow.ToolCallResult,
            ChatMessage=ChatMessage,
            MessageRole=MessageRole,
            ToolSelection=ToolSelection,
            ToolOutput=ToolOutput,
        )

    def _tool_result(self, lt, content, return_direct=False):
        return lt.ToolCallResult(
            tool_name="add",
            tool_kwargs={"a": 1, "b": 2},
            tool_id="call-1",
            tool_output=lt.ToolOutput(
                content=content, tool_name="add", raw_input={}, raw_output=content
            ),
            return_direct=return_direct,
        )

    def test_full_agent_trace(self, llama_types):
        lt = llama_types
        user = lt.ChatMessage(role=lt.MessageRole.USER, content="What is 1 + 2?")
        call = lt.ToolSelection(
            tool_id="call-1", tool_name="add", tool_kwargs={"a": 1, "b": 2}
        )
        events = [
            lt.AgentInput(input=[user], current_agent_name="agent"),
            lt.AgentOutput(
                response=lt.ChatMessage(role=lt.MessageRole.ASSISTANT, content=""),
                tool_calls=[call],
                raw=None,
                current_agent_name="agent",
            ),
            self._tool_result(lt, "3"),
            # the agent loop re-sends the input after a tool result; it must be skipped
            lt.AgentInput(input=[user], current_agent_name="agent"),
            lt.AgentOutput(
                response=lt.ChatMessage(
                    role=lt.MessageRole.ASSISTANT, content="The answer is 3."
                ),
                tool_calls=[call],  # duplicate tool id must not be repeated
                raw=None,
                current_agent_name="agent",
            ),
        ]

        messages = li_integration.convert_to_ragas_messages(events)

        assert [type(m) for m in messages] == [
            HumanMessage,
            AIMessage,
            ToolMessage,
            AIMessage,
        ]
        assert messages[0].content == "What is 1 + 2?"
        assert messages[1].tool_calls == [ToolCall(name="add", args={"a": 1, "b": 2})]
        assert messages[2].content == "3"
        assert messages[3].content == "The answer is 3."
        assert messages[3].tool_calls is None

    def test_return_direct_tool_output_is_ai_message(self, llama_types):
        messages = li_integration.convert_to_ragas_messages(
            [self._tool_result(llama_types, "direct", return_direct=True)]
        )
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "direct"

    def test_non_user_input_is_ignored(self, llama_types):
        lt = llama_types
        system = lt.ChatMessage(role=lt.MessageRole.SYSTEM, content="be nice")
        messages = li_integration.convert_to_ragas_messages(
            [lt.AgentInput(input=[system], current_agent_name="agent")]
        )
        assert messages == []
