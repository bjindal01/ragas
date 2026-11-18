import json

from ragas.integrations.amazon_bedrock import (
    convert_to_ragas_messages,
    extract_kb_trace,
    extract_messages_from_model_invocation,
    get_last_orchestration_value,
)
from ragas.messages import AIMessage, HumanMessage


def _orch(**kwargs):
    return {"trace": {"orchestrationTrace": kwargs}}


def _model_input(messages):
    return {"text": json.dumps({"messages": messages})}


def _kb_input(text):
    return _orch(
        invocationInput={
            "invocationType": "KNOWLEDGE_BASE",
            "knowledgeBaseLookupInput": {"text": text},
        }
    )


def _kb_output(*texts):
    return _orch(
        observation={
            "knowledgeBaseLookupOutput": {
                "retrievedReferences": [{"content": {"text": t}} for t in texts]
            }
        }
    )


def _final(text):
    return _orch(observation={"finalResponse": {"text": text}})


class TestGetLastOrchestrationValue:
    def test_returns_last_occurrence(self):
        traces = [
            _orch(modelInvocationInput="first"),
            _orch(observation={}),
            _orch(modelInvocationInput="second"),
            {"trace": {}},
        ]
        assert get_last_orchestration_value(traces, "modelInvocationInput") == (
            2,
            "second",
        )

    def test_missing_key(self):
        traces = [{"other": 1}, _orch(observation={})]
        assert get_last_orchestration_value(traces, "modelInvocationInput") == (
            -1,
            None,
        )

    def test_empty_traces(self):
        assert get_last_orchestration_value([], "observation") == (-1, None)


class TestExtractMessagesFromModelInvocation:
    def test_maps_roles_and_drops_last_message(self):
        model_inv = _model_input(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"text": "hi"}]},
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "pending question"},
            ]
        )
        messages = extract_messages_from_model_invocation(model_inv)

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "hello"
        assert isinstance(messages[1], AIMessage)
        # non-string content is stringified
        assert messages[1].content == str([{"text": "hi"}])

    def test_missing_text_returns_empty(self):
        assert extract_messages_from_model_invocation({}) == []


class TestConvertToRagasMessages:
    def test_appends_final_response_after_model_invocation(self):
        traces = [
            _orch(
                modelInvocationInput=_model_input(
                    [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "Let me check"},
                        {"role": "user", "content": "current turn"},
                    ]
                )
            ),
            _final("4"),
        ]
        messages = convert_to_ragas_messages(traces)

        assert [type(m) for m in messages] == [HumanMessage, AIMessage, AIMessage]
        assert [m.content for m in messages] == ["What is 2+2?", "Let me check", "4"]

    def test_ignores_observation_before_last_model_invocation(self):
        traces = [
            _final("stale"),
            _orch(
                modelInvocationInput=_model_input(
                    [
                        {"role": "user", "content": "q"},
                        {"role": "user", "content": "dropped"},
                    ]
                )
            ),
        ]
        messages = convert_to_ragas_messages(traces)
        assert [m.content for m in messages] == ["q"]

    def test_only_observation(self):
        messages = convert_to_ragas_messages([_final("answer")])
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "answer"

    def test_no_traces(self):
        assert convert_to_ragas_messages([]) == []


class TestExtractKbTrace:
    def test_single_group(self):
        traces = [
            _kb_input("capital of France?"),
            _kb_output("Paris is the capital.", "France is in Europe."),
            _final("Paris"),
        ]
        assert extract_kb_trace(traces) == [
            {
                "user_input": "capital of France?",
                "retrieved_contexts": [
                    "Paris is the capital.",
                    "France is in Europe.",
                ],
                "response": "Paris",
            }
        ]

    def test_multiple_groups_share_final_response(self):
        traces = [
            _kb_input("q1"),
            _kb_input("q2"),
            _kb_output("c1"),
            _kb_output("c2"),
            _final("combined answer"),
        ]
        result = extract_kb_trace(traces)
        assert result == [
            {
                "user_input": "q1",
                "retrieved_contexts": ["c1"],
                "response": "combined answer",
            },
            {
                "user_input": "q2",
                "retrieved_contexts": ["c2"],
                "response": "combined answer",
            },
        ]

    def test_sequential_groups_get_their_own_responses(self):
        traces = [
            _kb_input("q1"),
            _kb_output("c1"),
            _final("a1"),
            _kb_input("q2"),
            _kb_output("c2"),
            _final("a2"),
        ]
        result = extract_kb_trace(traces)
        assert [(g["user_input"], g["response"]) for g in result] == [
            ("q1", "a1"),
            ("q2", "a2"),
        ]

    def test_incomplete_group_is_not_returned(self):
        traces = [
            _kb_input("q1"),
            _final("answer without retrieval"),
        ]
        assert extract_kb_trace(traces) == []

    def test_non_kb_invocations_are_ignored(self):
        traces = [
            _orch(invocationInput={"invocationType": "ACTION_GROUP"}),
            _kb_output("orphan context"),
            _final("done"),
        ]
        assert extract_kb_trace(traces) == []
