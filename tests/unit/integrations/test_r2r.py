from types import SimpleNamespace

import pytest

from ragas.dataset_schema import EvaluationDataset
from ragas.integrations.r2r import _process_search_results, transform_to_ragas_dataset


class _SearchResults:
    def __init__(self, data):
        self._data = data

    def as_dict(self):
        return self._data


def _r2r_response(answer, search_results):
    return SimpleNamespace(
        results=SimpleNamespace(
            generated_answer=answer,
            search_results=_SearchResults(search_results),
        )
    )


class TestProcessSearchResults:
    def test_collects_chunk_and_web_results(self):
        results = {
            "chunk_search_results": [
                {"text": "chunk one"},
                {"text": ""},
                {"other": "no text"},
                {"text": "chunk two"},
            ],
            "web_search_results": [{"snippet": "web snippet"}, {"snippet": None}],
        }
        assert _process_search_results(results) == [
            "chunk one",
            "chunk two",
            "web snippet",
        ]

    @pytest.mark.parametrize(
        "key", ["graph_search_results", "context_document_results"]
    )
    def test_warns_on_unsupported_result_types(self, key):
        with pytest.warns(UserWarning, match=key):
            contexts = _process_search_results(
                {key: [{"text": "x"}], "chunk_search_results": [{"text": "kept"}]}
            )
        assert contexts == ["kept"]

    def test_empty_unsupported_results_do_not_warn(self, recwarn):
        assert _process_search_results({"graph_search_results": []}) == []
        assert len(recwarn) == 0


class TestTransformToRagasDataset:
    def test_builds_samples_from_all_inputs(self):
        responses = [
            _r2r_response("Paris", {"chunk_search_results": [{"text": "ctx1"}]}),
            _r2r_response("Berlin", {"web_search_results": [{"snippet": "ctx2"}]}),
        ]
        dataset = transform_to_ragas_dataset(
            user_inputs=["capital of France?", "capital of Germany?"],
            r2r_responses=responses,
            references=["Paris", "Berlin"],
        )

        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset) == 2
        first, second = dataset.samples
        assert first.user_input == "capital of France?"
        assert first.response == "Paris"
        assert first.retrieved_contexts == ["ctx1"]
        assert first.reference == "Paris"
        assert second.retrieved_contexts == ["ctx2"]
        assert second.response == "Berlin"

    def test_only_some_lists_provided(self):
        dataset = transform_to_ragas_dataset(user_inputs=["q1"], references=["r1"])
        sample = dataset.samples[0]
        assert sample.user_input == "q1"
        assert sample.reference == "r1"
        assert sample.response is None
        assert sample.retrieved_contexts is None

    def test_inconsistent_lengths_raise(self):
        with pytest.raises(ValueError, match="Inconsistent length for references"):
            transform_to_ragas_dataset(
                user_inputs=["q1", "q2"], references=["only one"]
            )
