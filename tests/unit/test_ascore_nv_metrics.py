"""Scripted-LLM tests for the legacy NVIDIA metrics in ``ragas.metrics._nv_metrics``."""

import math
import typing as t

import numpy as np
import pytest
from langchain_core.outputs import Generation, LLMResult

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.metrics._nv_metrics import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)


class ScriptedTextLLM(BaseRagasLLM):
    """Returns queued completions; a queued exception is raised instead."""

    def __init__(self, outputs: t.List[t.Any]):
        super().__init__()
        self.outputs = list(outputs)
        self.prompts: t.List[str] = []

    def generate_text(self, prompt, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def agenerate_text(self, prompt, *args, **kwargs):
        self.prompts.append(prompt.to_string())
        item = self.outputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return LLMResult(generations=[[Generation(text=item)]])

    def is_finished(self, response):  # pragma: no cover
        return True


def _sample(**kwargs) -> SingleTurnSample:
    return SingleTurnSample(**kwargs)


class TestNvAnswerAccuracy:
    SAMPLE = dict(user_input="When?", response="In 1879.", reference="1879")

    @pytest.mark.asyncio
    async def test_averages_two_judges(self):
        llm = ScriptedTextLLM(["4", "2"])
        metric = AnswerAccuracy(llm=llm)
        score = await metric._single_turn_ascore(_sample(**self.SAMPLE), None)
        assert score == pytest.approx(0.75)
        # Judge 1 sees the response as "User Answer"; judge 2 swaps roles.
        assert "### User Answer: In 1879." in llm.prompts[0]
        assert "Reference Answer: 1879" in llm.prompts[1]
        assert llm.prompts[1].index("Reference Answer: 1879") < llm.prompts[1].index(
            "User Answer: In 1879."
        )

    @pytest.mark.asyncio
    async def test_retries_until_a_rating_is_found(self):
        llm = ScriptedTextLLM(["no idea", "rating: 4", "?", "0"])
        metric = AnswerAccuracy(llm=llm)
        score = await metric._single_turn_ascore(_sample(**self.SAMPLE), None)
        assert score == pytest.approx(0.5)
        assert len(llm.prompts) == 4

    @pytest.mark.asyncio
    async def test_second_judge_failing_keeps_first_judge(self):
        llm = ScriptedTextLLM(["4"] + ["n/a"] * AnswerAccuracy.retry)
        metric = AnswerAccuracy(llm=llm)
        score = await metric._single_turn_ascore(_sample(**self.SAMPLE), None)
        assert score == pytest.approx(1.0)
        assert len(llm.prompts) == 1 + AnswerAccuracy.retry

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "average_scores uses max(score0, score1); max(nan, x) returns nan, so a "
            "failing first judge discards a valid second judge"
        ),
    )
    @pytest.mark.asyncio
    async def test_first_judge_failing_keeps_second_judge(self):
        llm = ScriptedTextLLM(["n/a"] * AnswerAccuracy.retry + ["4"])
        metric = AnswerAccuracy(llm=llm)
        score = await metric._single_turn_ascore(_sample(**self.SAMPLE), None)
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_llm_error_yields_nan(self):
        llm = ScriptedTextLLM([RuntimeError("api down")])
        metric = AnswerAccuracy(llm=llm)
        score = await metric._single_turn_ascore(_sample(**self.SAMPLE), None)
        assert math.isnan(score)

    @pytest.mark.parametrize(
        "text, expected",
        [("4", 1.0), ("2", 0.5), ("0", 0.0), ("3", 0.75), ("10", 0.0)],
    )
    def test_process_score(self, text, expected):
        assert AnswerAccuracy().process_score(text) == expected

    def test_process_score_without_digit_is_nan(self):
        assert np.isnan(AnswerAccuracy().process_score("excellent"))

    @pytest.mark.asyncio
    async def test_public_single_turn_ascore_filters_columns(self):
        llm = ScriptedTextLLM(["4", "4"])
        metric = AnswerAccuracy(llm=llm)
        sample = _sample(retrieved_contexts=["unused"], **self.SAMPLE)
        assert await metric.single_turn_ascore(sample) == pytest.approx(1.0)


class TestNvContextRelevance:
    @pytest.mark.asyncio
    async def test_averages_two_judges(self):
        llm = ScriptedTextLLM(["2", "1"])
        metric = ContextRelevance(llm=llm)
        sample = _sample(user_input="When?", retrieved_contexts=["a", "b"])
        score = await metric._single_turn_ascore(sample, None)
        assert score == pytest.approx(0.75)
        assert all("### Context: a\nb" in p for p in llm.prompts)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user_input, contexts",
        [
            ("  ", ["ctx"]),
            ("question", ["  "]),
            ("same", ["same"]),
            ("what about paris", ["paris"]),
        ],
    )
    async def test_degenerate_inputs_score_zero(self, user_input, contexts):
        llm = ScriptedTextLLM([])
        metric = ContextRelevance(llm=llm)
        sample = _sample(user_input=user_input, retrieved_contexts=contexts)
        assert await metric._single_turn_ascore(sample, None) == 0.0
        assert llm.prompts == []

    @pytest.mark.asyncio
    async def test_retry_and_single_judge_fallback(self):
        llm = ScriptedTextLLM(["unsure", "1"] + ["?"] * ContextRelevance.retry)
        metric = ContextRelevance(llm=llm)
        sample = _sample(user_input="q", retrieved_contexts=["ctx"])
        assert await metric._single_turn_ascore(sample, None) == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_llm_error_yields_nan(self):
        llm = ScriptedTextLLM([RuntimeError("x")])
        metric = ContextRelevance(llm=llm)
        sample = _sample(user_input="q", retrieved_contexts=["ctx"])
        assert math.isnan(await metric._single_turn_ascore(sample, None))

    def test_process_score_prefers_highest_digit(self):
        metric = ContextRelevance()
        assert metric.process_score("0 or 2") == 1.0
        assert metric.process_score("1") == 0.5
        assert metric.process_score("0") == 0.0
        assert np.isnan(metric.process_score("none"))


class TestNvResponseGroundedness:
    @pytest.mark.asyncio
    async def test_averages_two_judges(self):
        llm = ScriptedTextLLM(["1", "0"])
        metric = ResponseGroundedness(llm=llm)
        sample = _sample(response="Born in 1879", retrieved_contexts=["Born in Ulm"])
        assert await metric._single_turn_ascore(sample, None) == pytest.approx(0.25)
        assert "<Born in Ulm>" in llm.prompts[0]
        assert "[Born in 1879]" in llm.prompts[1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response, contexts, expected",
        [
            ("  ", ["ctx"], 0.0),
            ("resp", [" "], 0.0),
            ("Paris", ["Paris"], 1.0),  # identical
            ("Paris", ["Paris is in France"], 1.0),  # contained in context
        ],
    )
    async def test_shortcuts_skip_llm(self, response, contexts, expected):
        llm = ScriptedTextLLM([])
        metric = ResponseGroundedness(llm=llm)
        sample = _sample(response=response, retrieved_contexts=contexts)
        assert await metric._single_turn_ascore(sample, None) == expected
        assert llm.prompts == []

    @pytest.mark.asyncio
    async def test_retry(self):
        llm = ScriptedTextLLM(["?", "2", "-", "2"])
        metric = ResponseGroundedness(llm=llm)
        sample = _sample(response="resp", retrieved_contexts=["ctx"])
        assert await metric._single_turn_ascore(sample, None) == pytest.approx(1.0)
        assert len(llm.prompts) == 4

    @pytest.mark.asyncio
    async def test_both_judges_failing_is_nan(self):
        llm = ScriptedTextLLM(["?"] * (2 * ResponseGroundedness.retry))
        metric = ResponseGroundedness(llm=llm)
        sample = _sample(response="resp", retrieved_contexts=["ctx"])
        assert math.isnan(await metric._single_turn_ascore(sample, None))

    @pytest.mark.asyncio
    async def test_llm_error_yields_nan(self):
        llm = ScriptedTextLLM([ValueError("x")])
        metric = ResponseGroundedness(llm=llm)
        sample = _sample(response="resp", retrieved_contexts=["ctx"])
        assert math.isnan(await metric._single_turn_ascore(sample, None))
