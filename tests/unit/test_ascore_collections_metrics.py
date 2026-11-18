"""Scripted-LLM tests for the ``ascore`` paths of collections metrics.

Each test feeds pre-built Pydantic responses to the metric through a fake
``InstructorBaseRagasLLM`` and checks the score against a value computed by
hand from the metric's scoring rules.
"""

import json
import math
import typing as t

import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics.collections.agent_goal_accuracy.metric import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
)
from ragas.metrics.collections.agent_goal_accuracy.util import (
    CompareOutcomeInput,
    CompareOutcomeOutput,
    WorkflowOutput,
)
from ragas.metrics.collections.answer_accuracy.metric import AnswerAccuracy
from ragas.metrics.collections.answer_accuracy.util import (
    AnswerAccuracyInput,
    AnswerAccuracyOutput,
)
from ragas.metrics.collections.answer_correctness.metric import AnswerCorrectness
from ragas.metrics.collections.answer_correctness.util import (
    ClassificationWithReason,
    StatementGeneratorOutput as ACStatements,
    StatementsWithReason,
)
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevancy
from ragas.metrics.collections.answer_relevancy.util import AnswerRelevanceOutput
from ragas.metrics.collections.context_entity_recall.metric import (
    ContextEntityRecall,
)
from ragas.metrics.collections.context_entity_recall.util import EntitiesList
from ragas.metrics.collections.context_precision.metric import (
    ContextPrecision,
    ContextPrecisionWithoutReference,
    ContextPrecisionWithReference,
    ContextUtilization,
)
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
)
from ragas.metrics.collections.context_recall.metric import ContextRecall
from ragas.metrics.collections.context_recall.util import (
    ContextRecallClassification,
    ContextRecallOutput,
)
from ragas.metrics.collections.context_relevance.metric import ContextRelevance
from ragas.metrics.collections.context_relevance.util import ContextRelevanceOutput
from ragas.metrics.collections.factual_correctness.metric import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionOutput,
    NLIStatementOutput as FCVerdicts,
    StatementFaithfulnessAnswer as FCVerdict,
)
from ragas.metrics.collections.faithfulness.metric import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput as FVerdicts,
    StatementFaithfulnessAnswer as FVerdict,
    StatementGeneratorOutput as FStatements,
)
from ragas.metrics.collections.noise_sensitivity.metric import NoiseSensitivity
from ragas.metrics.collections.noise_sensitivity.util import (
    StatementFaithfulnessAnswer as NSVerdict,
    StatementFaithfulnessOutput as NSVerdicts,
    StatementGeneratorOutput as NSStatements,
)
from ragas.metrics.collections.response_groundedness.metric import (
    ResponseGroundedness,
)
from ragas.metrics.collections.response_groundedness.util import (
    ResponseGroundednessOutput,
)
from ragas.metrics.collections.summary_score.metric import SummaryScore
from ragas.metrics.collections.summary_score.util import (
    AnswersGenerated,
    ExtractedKeyphrases,
    QuestionsGenerated,
)
from ragas.metrics.collections.topic_adherence.metric import TopicAdherence
from ragas.metrics.collections.topic_adherence.util import (
    TopicClassificationOutput,
    TopicExtractionOutput,
    TopicRefusedOutput,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedLLM(InstructorBaseRagasLLM):
    """Returns queued responses keyed by the requested response model.

    A queued ``BaseException`` instance is raised instead of returned, which
    lets tests exercise the retry/except branches of the metrics.
    """

    def __init__(self, script: t.Optional[t.Dict[type, t.List[t.Any]]] = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: t.List[t.Tuple[str, type]] = []

    def generate(self, prompt, response_model):  # pragma: no cover - unused
        raise NotImplementedError

    async def agenerate(self, prompt, response_model):
        self.calls.append((prompt, response_model))
        queue = self.script.get(response_model)
        if not queue:
            raise LookupError(f"no scripted response for {response_model.__name__}")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def prompts_for(self, model: type) -> t.List[str]:
        return [p for p, m in self.calls if m is model]

    def remaining(self) -> int:
        return sum(len(v) for v in self.script.values())


class FakeEmbedding(BaseRagasEmbedding):
    def __init__(self, vectors: t.Dict[str, t.List[float]]):
        super().__init__()
        self.vectors = vectors
        self.calls: t.List[str] = []

    def embed_text(self, text, **kwargs):  # pragma: no cover - unused
        return list(self.vectors[text])

    async def aembed_text(self, text, **kwargs):
        self.calls.append(text)
        return list(self.vectors[text])


class NotAnInstructorLLM:
    async def agenerate(self, prompt, response_model):  # pragma: no cover
        raise AssertionError("should never be called")


def _stmts(*items: str) -> t.List[StatementsWithReason]:
    return [StatementsWithReason(statement=s, reason="r") for s in items]


def _fverdicts(*verdicts: int) -> FVerdicts:
    return FVerdicts(
        statements=[
            FVerdict(statement=f"s{i}", reason="r", verdict=v)
            for i, v in enumerate(verdicts)
        ]
    )


def _fcverdicts(*verdicts: int) -> FCVerdicts:
    return FCVerdicts(
        statements=[
            FCVerdict(statement=f"c{i}", reason="r", verdict=v)
            for i, v in enumerate(verdicts)
        ]
    )


def _nsverdicts(*verdicts: int) -> NSVerdicts:
    return NSVerdicts(
        statements=[
            NSVerdict(statement=f"s{i}", reason="r", verdict=v)
            for i, v in enumerate(verdicts)
        ]
    )


# ---------------------------------------------------------------------------
# Component validation shared by every LLM-backed collections metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda llm: AnswerAccuracy(llm=llm),
        lambda llm: ContextRelevance(llm=llm),
        lambda llm: ResponseGroundedness(llm=llm),
        lambda llm: Faithfulness(llm=llm),
        lambda llm: FactualCorrectness(llm=llm),
        lambda llm: AnswerCorrectness(llm=llm, weights=[1.0, 0.0]),
        lambda llm: NoiseSensitivity(llm=llm),
        lambda llm: ContextPrecision(llm=llm),
        lambda llm: ContextRecall(llm=llm),
        lambda llm: ContextEntityRecall(llm=llm),
        lambda llm: SummaryScore(llm=llm),
        lambda llm: TopicAdherence(llm=llm),
        lambda llm: AgentGoalAccuracyWithReference(llm=llm),
    ],
)
def test_legacy_or_foreign_llm_is_rejected(factory):
    with pytest.raises(ValueError, match="only support modern InstructorLLM"):
        factory(NotAnInstructorLLM())


def test_embeddings_must_be_modern():
    with pytest.raises(ValueError, match="only support modern embeddings"):
        AnswerRelevancy(llm=ScriptedLLM(), embeddings=object())


# ---------------------------------------------------------------------------
# Dual-judge metrics
# ---------------------------------------------------------------------------


class TestAnswerAccuracy:
    @pytest.mark.asyncio
    async def test_averages_both_judges_on_0_to_4_scale(self):
        llm = ScriptedLLM(
            {
                AnswerAccuracyOutput: [
                    AnswerAccuracyOutput(rating=4),
                    AnswerAccuracyOutput(rating=2),
                ]
            }
        )
        metric = AnswerAccuracy(llm=llm)
        result = await metric.ascore(user_input="Q?", response="resp", reference="ref")
        # (4/4 + 2/4) / 2
        assert result.value == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_second_judge_receives_swapped_answers(self):
        llm = ScriptedLLM(
            {
                AnswerAccuracyOutput: [
                    AnswerAccuracyOutput(rating=0),
                    AnswerAccuracyOutput(rating=0),
                ]
            }
        )
        metric = AnswerAccuracy(llm=llm)
        result = await metric.ascore(user_input="Q?", response="resp", reference="ref")
        assert result.value == 0.0
        prompts = llm.prompts_for(AnswerAccuracyOutput)
        assert prompts[0] == metric.judge1_prompt.to_string(
            AnswerAccuracyInput(query="Q?", user_answer="resp", reference_answer="ref")
        )
        assert prompts[1] == metric.judge2_prompt.to_string(
            AnswerAccuracyInput(query="Q?", user_answer="ref", reference_answer="resp")
        )

    @pytest.mark.asyncio
    async def test_invalid_rating_and_exception_trigger_retries(self):
        llm = ScriptedLLM(
            {
                AnswerAccuracyOutput: [
                    AnswerAccuracyOutput(rating=3),  # invalid -> retry
                    AnswerAccuracyOutput(rating=4),
                    RuntimeError("boom"),  # exception -> retry
                    AnswerAccuracyOutput(rating=0),
                ]
            }
        )
        metric = AnswerAccuracy(llm=llm)
        result = await metric.ascore(user_input="Q?", response="resp", reference="ref")
        assert result.value == pytest.approx(0.5)
        assert len(llm.calls) == 4

    @pytest.mark.asyncio
    async def test_judge_exhausting_retries_falls_back_to_other_judge(self):
        llm = ScriptedLLM(
            {
                AnswerAccuracyOutput: [
                    AnswerAccuracyOutput(rating=1),
                    AnswerAccuracyOutput(rating=5),
                    AnswerAccuracyOutput(rating=2),
                ]
            }
        )
        metric = AnswerAccuracy(llm=llm, max_retries=2)
        result = await metric.ascore(user_input="Q?", response="resp", reference="ref")
        assert result.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_both_judges_failing_returns_nan(self):
        llm = ScriptedLLM({AnswerAccuracyOutput: [RuntimeError("x")] * 2})
        metric = AnswerAccuracy(llm=llm, max_retries=1)
        result = await metric.ascore(user_input="Q?", response="resp", reference="ref")
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs, missing",
        [
            ({"user_input": "", "response": "r", "reference": "g"}, "user_input"),
            ({"user_input": "q", "response": "", "reference": "g"}, "response"),
            ({"user_input": "q", "response": "r", "reference": ""}, "reference"),
        ],
    )
    async def test_missing_inputs_raise(self, kwargs, missing):
        metric = AnswerAccuracy(llm=ScriptedLLM())
        with pytest.raises(ValueError, match=f"{missing} is missing"):
            await metric.ascore(**kwargs)


class TestContextRelevance:
    @pytest.mark.asyncio
    async def test_averages_both_judges_on_0_to_2_scale(self):
        llm = ScriptedLLM(
            {
                ContextRelevanceOutput: [
                    ContextRelevanceOutput(rating=2),
                    ContextRelevanceOutput(rating=1),
                ]
            }
        )
        metric = ContextRelevance(llm=llm)
        result = await metric.ascore(
            user_input="When was Einstein born?",
            retrieved_contexts=["Einstein was born", "in 1879."],
        )
        assert result.value == pytest.approx(0.75)
        # Contexts are joined with a newline before being judged.
        assert all(
            json.dumps("Einstein was born\nin 1879.") in p
            for p in llm.prompts_for(ContextRelevanceOutput)
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user_input, contexts",
        [
            ("   ", ["some context"]),  # blank question
            ("question?", ["  ", ""]),  # blank contexts
            ("same text", ["same text"]),  # context equals question
            ("what about paris today", ["paris"]),  # context inside question
        ],
    )
    async def test_degenerate_inputs_score_zero_without_llm_calls(
        self, user_input, contexts
    ):
        llm = ScriptedLLM()
        metric = ContextRelevance(llm=llm)
        result = await metric.ascore(user_input=user_input, retrieved_contexts=contexts)
        assert result.value == 0.0
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_retries_on_invalid_rating_and_exception(self):
        llm = ScriptedLLM(
            {
                ContextRelevanceOutput: [
                    ContextRelevanceOutput(rating=7),
                    ContextRelevanceOutput(rating=2),
                    ValueError("bad json"),
                    ContextRelevanceOutput(rating=0),
                ]
            }
        )
        metric = ContextRelevance(llm=llm)
        result = await metric.ascore(user_input="q", retrieved_contexts=["ctx"])
        assert result.value == pytest.approx(0.5)
        assert len(llm.calls) == 4

    @pytest.mark.asyncio
    async def test_nan_handling(self):
        llm = ScriptedLLM(
            {
                ContextRelevanceOutput: [
                    ContextRelevanceOutput(rating=9),
                    ContextRelevanceOutput(rating=1),
                ]
            }
        )
        metric = ContextRelevance(llm=llm, max_retries=1)
        result = await metric.ascore(user_input="q", retrieved_contexts=["ctx"])
        assert result.value == pytest.approx(0.5)

        llm = ScriptedLLM({ContextRelevanceOutput: [RuntimeError("x")] * 2})
        metric = ContextRelevance(llm=llm, max_retries=1)
        result = await metric.ascore(user_input="q", retrieved_contexts=["ctx"])
        assert math.isnan(result.value)

    def test_average_scores_prefers_any_valid_judge(self):
        metric = ContextRelevance(llm=ScriptedLLM())
        nan = float("nan")
        assert metric._average_scores(nan, 0.5) == 0.5
        assert metric._average_scores(1.0, nan) == 1.0
        assert math.isnan(metric._average_scores(nan, nan))

    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self):
        metric = ContextRelevance(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(user_input="", retrieved_contexts=["c"])
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(user_input="q", retrieved_contexts=[])


class TestResponseGroundedness:
    @pytest.mark.asyncio
    async def test_averages_both_judges(self):
        llm = ScriptedLLM(
            {
                ResponseGroundednessOutput: [
                    ResponseGroundednessOutput(rating=1),
                    ResponseGroundednessOutput(rating=0),
                ]
            }
        )
        metric = ResponseGroundedness(llm=llm)
        result = await metric.ascore(
            response="Einstein was born in 1879.",
            retrieved_contexts=["Einstein was born in Ulm."],
        )
        assert result.value == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_blank_inputs_score_zero(self):
        llm = ScriptedLLM()
        metric = ResponseGroundedness(llm=llm)
        assert (await metric.ascore(response="  ", retrieved_contexts=["c"])).value == 0
        assert (await metric.ascore(response="r", retrieved_contexts=[" "])).value == 0
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_retry_then_fallback_to_single_judge(self):
        llm = ScriptedLLM(
            {
                ResponseGroundednessOutput: [
                    ResponseGroundednessOutput(rating=3),
                    RuntimeError("timeout"),
                    ResponseGroundednessOutput(rating=2),
                ]
            }
        )
        metric = ResponseGroundedness(llm=llm, max_retries=2)
        result = await metric.ascore(response="r", retrieved_contexts=["c"])
        # Judge 1 never produced a valid rating; judge 2 rated 2/2.
        assert result.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_both_judges_nan(self):
        llm = ScriptedLLM(
            {ResponseGroundednessOutput: [ResponseGroundednessOutput(rating=-1)] * 2}
        )
        metric = ResponseGroundedness(llm=llm, max_retries=1)
        result = await metric.ascore(response="r", retrieved_contexts=["c"])
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self):
        metric = ResponseGroundedness(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(response="", retrieved_contexts=["c"])
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(response="r", retrieved_contexts=[])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_cls, kwargs",
    [
        (AnswerAccuracy, {"user_input": "q", "response": "r", "reference": "g"}),
        (ContextRelevance, {"user_input": "q", "retrieved_contexts": ["c"]}),
        (ResponseGroundedness, {"response": "r", "retrieved_contexts": ["c"]}),
    ],
)
async def test_dual_judge_with_zero_retries_never_calls_llm(metric_cls, kwargs):
    llm = ScriptedLLM()
    metric = metric_cls(llm=llm, max_retries=0)
    result = await metric.ascore(**kwargs)
    assert math.isnan(result.value)
    assert llm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_cls, output_cls, kwargs",
    [
        (
            AnswerAccuracy,
            AnswerAccuracyOutput,
            {"user_input": "q", "response": "r", "reference": "g"},
        ),
        (
            ResponseGroundedness,
            ResponseGroundednessOutput,
            {"response": "r", "retrieved_contexts": ["c"]},
        ),
    ],
)
async def test_dual_judge_last_attempt_exception_gives_nan_for_that_judge(
    metric_cls, output_cls, kwargs
):
    top = 4 if metric_cls is AnswerAccuracy else 2
    llm = ScriptedLLM(
        {
            output_cls: [
                RuntimeError("first"),
                output_cls(rating=top),
                RuntimeError("a"),
                RuntimeError("b"),
            ]
        }
    )
    metric = metric_cls(llm=llm, max_retries=2)
    result = await metric.ascore(**kwargs)
    # Judge 1 recovers on its retry; judge 2 fails twice and is ignored.
    assert result.value == pytest.approx(1.0)
    assert len(llm.calls) == 4


# ---------------------------------------------------------------------------
# Statement / claim based metrics
# ---------------------------------------------------------------------------


class TestFaithfulness:
    @pytest.mark.asyncio
    async def test_ratio_of_supported_statements(self):
        llm = ScriptedLLM(
            {
                FStatements: [FStatements(statements=["s1", "s2", "s3"])],
                FVerdicts: [_fverdicts(1, 0, 1)],
            }
        )
        metric = Faithfulness(llm=llm)
        result = await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["ctx A", "ctx B"]
        )
        assert result.value == pytest.approx(2 / 3)
        assert json.dumps("ctx A\nctx B") in llm.prompts_for(FVerdicts)[0]

    @pytest.mark.asyncio
    async def test_no_statements_returns_nan_without_nli_call(self):
        llm = ScriptedLLM({FStatements: [FStatements(statements=[])]})
        metric = Faithfulness(llm=llm)
        result = await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["c"]
        )
        assert math.isnan(result.value)
        assert llm.prompts_for(FVerdicts) == []

    @pytest.mark.asyncio
    async def test_empty_verdicts_returns_nan(self):
        llm = ScriptedLLM(
            {
                FStatements: [FStatements(statements=["s1"])],
                FVerdicts: [FVerdicts(statements=[])],
            }
        )
        metric = Faithfulness(llm=llm)
        result = await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["c"]
        )
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self):
        metric = Faithfulness(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(user_input="q", response="", retrieved_contexts=["c"])
        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(user_input="", response="r", retrieved_contexts=["c"])
        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(user_input="q", response="r", retrieved_contexts=[])


class TestFactualCorrectness:
    @pytest.mark.asyncio
    async def test_precision_mode_only_checks_response_claims(self):
        llm = ScriptedLLM(
            {
                ClaimDecompositionOutput: [
                    ClaimDecompositionOutput(claims=["c1", "c2", "c3"])
                ],
                FCVerdicts: [_fcverdicts(1, 1, 0)],
            }
        )
        metric = FactualCorrectness(llm=llm, mode="precision")
        result = await metric.ascore(response="resp", reference="ref")
        # tp=2, fp=1, rounded to two decimals
        assert result.value == 0.67
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_recall_mode(self):
        llm = ScriptedLLM(
            {
                ClaimDecompositionOutput: [
                    ClaimDecompositionOutput(claims=["c1", "c2", "c3"]),
                    ClaimDecompositionOutput(claims=["r1", "r2"]),
                ],
                FCVerdicts: [_fcverdicts(1, 1, 0), _fcverdicts(1, 0)],
            }
        )
        metric = FactualCorrectness(llm=llm, mode="recall")
        result = await metric.ascore(response="resp", reference="ref")
        # tp=2 (response side), fn=1 (reference side)
        assert result.value == 0.67
        assert len(llm.calls) == 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize("beta, expected", [(1.0, 0.57), (2.0, 0.53)])
    async def test_f1_mode_with_beta(self, beta, expected):
        llm = ScriptedLLM(
            {
                ClaimDecompositionOutput: [
                    ClaimDecompositionOutput(claims=["c1", "c2", "c3"]),
                    ClaimDecompositionOutput(claims=["r1", "r2"]),
                ],
                FCVerdicts: [_fcverdicts(1, 1, 0), _fcverdicts(0, 0)],
            }
        )
        metric = FactualCorrectness(llm=llm, mode="f1", beta=beta)
        result = await metric.ascore(response="resp", reference="ref")
        # tp=2, fp=1, fn=2 -> precision 2/3, recall 1/2
        # beta=1: 4/7 ~ 0.571 ; beta=2: 10/19 ~ 0.526
        assert result.value == expected

    @pytest.mark.asyncio
    async def test_decomposition_settings_reach_prompt(self):
        llm = ScriptedLLM(
            {
                ClaimDecompositionOutput: [ClaimDecompositionOutput(claims=[])],
            }
        )
        metric = FactualCorrectness(
            llm=llm, mode="precision", atomicity="high", coverage="high"
        )
        result = await metric.ascore(response="resp", reference="ref")
        # No claims -> no NLI call and tp=fp=0
        assert result.value == 0.0
        assert len(llm.calls) == 1
        assert '"atomicity": "high"' in llm.calls[0][0]
        assert '"coverage": "high"' in llm.calls[0][0]

    @pytest.mark.asyncio
    async def test_empty_nli_verdicts_count_as_no_claims(self):
        llm = ScriptedLLM(
            {
                ClaimDecompositionOutput: [
                    ClaimDecompositionOutput(claims=["c1"]),
                    ClaimDecompositionOutput(claims=["r1"]),
                ],
                FCVerdicts: [FCVerdicts(statements=[]), _fcverdicts(1)],
            }
        )
        metric = FactualCorrectness(llm=llm, mode="f1")
        result = await metric.ascore(response="resp", reference="ref")
        assert result.value == 0.0

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="Beta must be a float"):
            FactualCorrectness(llm=ScriptedLLM(), beta="high")  # type: ignore

    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self):
        metric = FactualCorrectness(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(response="", reference="ref")
        with pytest.raises(ValueError, match="reference is missing"):
            await metric.ascore(response="resp", reference="")


class TestAnswerCorrectness:
    @pytest.mark.asyncio
    async def test_pure_factuality_f1(self):
        llm = ScriptedLLM(
            {
                ACStatements: [
                    ACStatements(statements=["a", "b"]),
                    ACStatements(statements=["c"]),
                ],
                ClassificationWithReason: [
                    ClassificationWithReason(
                        TP=_stmts("a"), FP=_stmts("b"), FN=_stmts("c")
                    )
                ],
            }
        )
        metric = AnswerCorrectness(llm=llm, weights=[1.0, 0.0])
        result = await metric.ascore(user_input="q", response="r", reference="g")
        assert result.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_beta_favours_recall(self):
        llm = ScriptedLLM(
            {
                ACStatements: [
                    ACStatements(statements=["a"]),
                    ACStatements(statements=["a", "c"]),
                ],
                ClassificationWithReason: [
                    ClassificationWithReason(TP=_stmts("a"), FP=[], FN=_stmts("c"))
                ],
            }
        )
        metric = AnswerCorrectness(llm=llm, weights=[1.0, 0.0], beta=2.0)
        result = await metric.ascore(user_input="q", response="r", reference="g")
        # precision 1, recall 0.5 -> F2 = 5*0.5 / (4*1 + 0.5) = 5/9
        assert result.value == pytest.approx(5 / 9)

    @pytest.mark.asyncio
    async def test_weighted_combination_with_similarity(self):
        llm = ScriptedLLM(
            {
                ACStatements: [
                    ACStatements(statements=[]),
                    ACStatements(statements=["c"]),
                ],
            }
        )
        emb = FakeEmbedding({"resp": [1.0, 0.0], "ref": [1.0, 1.0]})
        metric = AnswerCorrectness(llm=llm, embeddings=emb)
        result = await metric.ascore(user_input="q", response="resp", reference="ref")
        # No response statements -> factuality 1.0 without classification call.
        # cosine([1,0],[1,1]) = 1/sqrt(2)
        expected = 0.75 * 1.0 + 0.25 * (1 / math.sqrt(2))
        assert result.value == pytest.approx(expected)
        assert llm.prompts_for(ClassificationWithReason) == []
        assert emb.calls == ["resp", "ref"]

    @pytest.mark.asyncio
    async def test_zero_norm_embedding_gives_zero_similarity(self):
        llm = ScriptedLLM(
            {
                ACStatements: [
                    ACStatements(statements=["a"]),
                    ACStatements(statements=["a"]),
                ],
                ClassificationWithReason: [
                    ClassificationWithReason(TP=_stmts("a"), FP=[], FN=[])
                ],
            }
        )
        emb = FakeEmbedding({"resp": [0.0, 0.0], "ref": [1.0, 0.0]})
        metric = AnswerCorrectness(llm=llm, embeddings=emb, weights=[0.5, 0.5])
        result = await metric.ascore(user_input="q", response="resp", reference="ref")
        assert result.value == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "tp, fp, fn, expected",
        [
            (0, 0, 0, 1.0),  # nothing to compare -> perfect
            (0, 2, 0, 0.0),  # only false positives
            (0, 0, 2, 0.0),  # only false negatives
            (3, 1, 0, 6 / 7),
        ],
    )
    def test_compute_f1_edge_cases(self, tp, fp, fn, expected):
        metric = AnswerCorrectness(llm=ScriptedLLM(), weights=[1.0, 0.0])
        cls = ClassificationWithReason(
            TP=_stmts(*["t"] * tp), FP=_stmts(*["f"] * fp), FN=_stmts(*["n"] * fn)
        )
        assert metric._compute_f1_score(cls) == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_similarity_without_embeddings_raises(self):
        metric = AnswerCorrectness(llm=ScriptedLLM(), weights=[1.0, 0.0])
        with pytest.raises(RuntimeError, match="Embeddings required"):
            await metric._calculate_similarity("a", "b")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"weights": [1.0]}, "two weights"),
            ({"weights": [0.0, 0.0]}, "non-zero"),
            ({"weights": [1.0, -0.5]}, "non-negative"),
            ({"weights": [0.5, 0.5]}, "Embeddings are required"),
            ({"weights": [1.0, 0.0], "beta": 1}, "Beta must be a float"),
        ],
    )
    def test_init_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            AnswerCorrectness(llm=ScriptedLLM(), **kwargs)


class TestNoiseSensitivity:
    @staticmethod
    def _llm(gt_to_answer: t.Tuple[int, int]) -> ScriptedLLM:
        # Two reference statements (g1, g2), two response statements (a1, a2)
        # and two contexts: c1 supports g1 (relevant) and a1; c2 supports no
        # reference statement (irrelevant) but supports a2.
        return ScriptedLLM(
            {
                NSStatements: [
                    NSStatements(statements=["g1", "g2"]),
                    NSStatements(statements=["a1", "a2"]),
                ],
                NSVerdicts: [
                    _nsverdicts(1, 0),  # c1 vs reference statements
                    _nsverdicts(1, 0),  # c1 vs response statements
                    _nsverdicts(0, 0),  # c2 vs reference statements
                    _nsverdicts(0, 1),  # c2 vs response statements
                    _nsverdicts(*gt_to_answer),  # reference vs response statements
                ],
            }
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode, gt_to_answer, expected",
        [
            ("relevant", (0, 0), 0.5),  # a1 incorrect & from relevant ctx
            ("irrelevant", (0, 0), 0.5),  # a2 incorrect & from irrelevant ctx
            ("relevant", (1, 1), 0.0),  # everything correct
            ("irrelevant", (1, 1), 0.0),
            ("relevant", (0, 1), 0.5),
            ("irrelevant", (0, 1), 0.0),  # a2 is correct
        ],
    )
    async def test_modes(self, mode, gt_to_answer, expected):
        llm = self._llm(gt_to_answer)
        metric = NoiseSensitivity(llm=llm, mode=mode)
        result = await metric.ascore(
            user_input="q",
            response="resp",
            reference="ref",
            retrieved_contexts=["c1", "c2"],
        )
        assert result.value == pytest.approx(expected)
        assert llm.remaining() == 0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid argument passed for 'mode'"):
            NoiseSensitivity(llm=ScriptedLLM(), mode="both")  # type: ignore

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field", ["reference", "user_input", "response", "retrieved_contexts"]
    )
    async def test_missing_inputs_raise(self, field):
        kwargs: t.Dict[str, t.Any] = {
            "user_input": "q",
            "response": "r",
            "reference": "g",
            "retrieved_contexts": ["c"],
        }
        kwargs[field] = [] if field == "retrieved_contexts" else ""
        metric = NoiseSensitivity(llm=ScriptedLLM())
        with pytest.raises(ValueError, match=f"{field} is missing"):
            await metric.ascore(**kwargs)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def _cp(verdict: int) -> ContextPrecisionOutput:
    return ContextPrecisionOutput(reason="r", verdict=verdict)


class TestContextPrecision:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "verdicts, expected",
        [
            ([1, 0, 1], (1.0 + 2 / 3) / 2),
            ([0, 1], 0.5),
            ([1, 1], 1.0),
            ([0, 0, 0], 0.0),
        ],
    )
    async def test_average_precision_with_reference(self, verdicts, expected):
        llm = ScriptedLLM({ContextPrecisionOutput: [_cp(v) for v in verdicts]})
        metric = ContextPrecisionWithReference(llm=llm)
        contexts = [f"ctx{i}" for i in range(len(verdicts))]
        result = await metric.ascore(
            user_input="q", reference="ref", retrieved_contexts=contexts
        )
        assert result.value == pytest.approx(expected)
        # One judgement per context, each against the reference answer.
        prompts = llm.prompts_for(ContextPrecisionOutput)
        assert prompts == [
            metric.prompt.to_string(
                ContextPrecisionInput(question="q", context=c, answer="ref")
            )
            for c in contexts
        ]

    @pytest.mark.asyncio
    async def test_without_reference_uses_response(self):
        llm = ScriptedLLM({ContextPrecisionOutput: [_cp(0), _cp(1), _cp(1)]})
        metric = ContextPrecisionWithoutReference(llm=llm)
        result = await metric.ascore(
            user_input="q", response="resp", retrieved_contexts=["a", "b", "c"]
        )
        # (1/2 + 2/3) / 2
        assert result.value == pytest.approx((0.5 + 2 / 3) / 2)
        assert llm.calls[0][0] == metric.prompt.to_string(
            ContextPrecisionInput(question="q", context="a", answer="resp")
        )

    def test_aliases_have_legacy_names(self):
        assert ContextPrecision(llm=ScriptedLLM()).name == "context_precision"
        assert ContextUtilization(llm=ScriptedLLM()).name == "context_utilization"

    @pytest.mark.asyncio
    async def test_alias_scores_like_parent(self):
        llm = ScriptedLLM({ContextPrecisionOutput: [_cp(1)]})
        result = await ContextUtilization(llm=llm).ascore(
            user_input="q", response="r", retrieved_contexts=["c"]
        )
        assert result.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self):
        with_ref = ContextPrecisionWithReference(llm=ScriptedLLM())
        without_ref = ContextPrecisionWithoutReference(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="user_input cannot be empty"):
            await with_ref.ascore(
                user_input="", reference="g", retrieved_contexts=["c"]
            )
        with pytest.raises(ValueError, match="reference cannot be empty"):
            await with_ref.ascore(
                user_input="q", reference="", retrieved_contexts=["c"]
            )
        with pytest.raises(ValueError, match="retrieved_contexts cannot be empty"):
            await with_ref.ascore(user_input="q", reference="g", retrieved_contexts=[])
        with pytest.raises(ValueError, match="user_input cannot be empty"):
            await without_ref.ascore(
                user_input="", response="r", retrieved_contexts=["c"]
            )
        with pytest.raises(ValueError, match="response cannot be empty"):
            await without_ref.ascore(
                user_input="q", response="", retrieved_contexts=["c"]
            )
        with pytest.raises(ValueError, match="retrieved_contexts cannot be empty"):
            await without_ref.ascore(
                user_input="q", response="r", retrieved_contexts=[]
            )


class TestContextRecall:
    @staticmethod
    def _output(*attributed: int) -> ContextRecallOutput:
        return ContextRecallOutput(
            classifications=[
                ContextRecallClassification(statement=f"s{i}", reason="r", attributed=a)
                for i, a in enumerate(attributed)
            ]
        )

    @pytest.mark.asyncio
    async def test_fraction_of_attributed_statements(self):
        llm = ScriptedLLM({ContextRecallOutput: [self._output(1, 0, 1, 1)]})
        metric = ContextRecall(llm=llm)
        result = await metric.ascore(
            user_input="q", retrieved_contexts=["c1", "c2"], reference="ref"
        )
        assert result.value == pytest.approx(0.75)
        assert json.dumps("c1\nc2") in llm.calls[0][0]

    @pytest.mark.asyncio
    async def test_no_classifications_is_nan(self):
        llm = ScriptedLLM({ContextRecallOutput: [self._output()]})
        metric = ContextRecall(llm=llm)
        result = await metric.ascore(
            user_input="q", retrieved_contexts=["c"], reference="ref"
        )
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self):
        metric = ContextRecall(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="user_input cannot be empty"):
            await metric.ascore(user_input="", retrieved_contexts=["c"], reference="g")
        with pytest.raises(ValueError, match="reference cannot be empty"):
            await metric.ascore(user_input="q", retrieved_contexts=["c"], reference="")
        with pytest.raises(ValueError, match="retrieved_contexts cannot be empty"):
            await metric.ascore(user_input="q", retrieved_contexts=[], reference="g")


class TestContextEntityRecall:
    @pytest.mark.asyncio
    async def test_recall_over_unique_reference_entities(self):
        llm = ScriptedLLM(
            {
                EntitiesList: [
                    EntitiesList(entities=["Paris", "France", "52 BC", "Paris"]),
                    EntitiesList(entities=["Paris", "France", "Europe"]),
                ]
            }
        )
        metric = ContextEntityRecall(llm=llm)
        result = await metric.ascore(
            reference="ref", retrieved_contexts=["ctx one", "ctx two"]
        )
        # Reference set has 3 unique entities, 2 of them appear in the contexts.
        assert result.value == pytest.approx(2 / 3)
        assert json.dumps("ctx one\nctx two") in llm.calls[1][0]

    @pytest.mark.asyncio
    async def test_no_reference_entities_scores_zero(self):
        llm = ScriptedLLM(
            {EntitiesList: [EntitiesList(entities=[]), EntitiesList(entities=["X"])]}
        )
        result = await ContextEntityRecall(llm=llm).ascore(
            reference="ref", retrieved_contexts=["ctx"]
        )
        assert result.value == 0.0


class TestAnswerRelevancy:
    @pytest.mark.asyncio
    async def test_mean_cosine_similarity_of_generated_questions(self):
        llm = ScriptedLLM(
            {
                AnswerRelevanceOutput: [
                    AnswerRelevanceOutput(question="q1", noncommittal=0),
                    AnswerRelevanceOutput(question="q2", noncommittal=1),
                    AnswerRelevanceOutput(question="", noncommittal=0),  # dropped
                ]
            }
        )
        emb = FakeEmbedding({"orig": [1.0, 0.0], "q1": [2.0, 0.0], "q2": [0.0, 3.0]})
        metric = AnswerRelevancy(llm=llm, embeddings=emb, strictness=3)
        result = await metric.ascore(user_input="orig", response="resp")
        # cos(q1)=1, cos(q2)=0 -> mean 0.5; not every answer is noncommittal.
        assert result.value == pytest.approx(0.5)
        assert len(llm.calls) == 3
        assert sorted(emb.calls) == ["orig", "q1", "q2"]

    @pytest.mark.asyncio
    async def test_all_noncommittal_zeroes_score(self):
        llm = ScriptedLLM(
            {
                AnswerRelevanceOutput: [
                    AnswerRelevanceOutput(question="q1", noncommittal=1),
                    AnswerRelevanceOutput(question="q1", noncommittal=1),
                ]
            }
        )
        emb = FakeEmbedding({"orig": [1.0, 0.0], "q1": [1.0, 0.0]})
        metric = AnswerRelevancy(llm=llm, embeddings=emb, strictness=2)
        result = await metric.ascore(user_input="orig", response="I don't know")
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_no_generated_questions_scores_zero_without_embedding(self):
        llm = ScriptedLLM(
            {
                AnswerRelevanceOutput: [
                    AnswerRelevanceOutput(question="", noncommittal=0)
                ]
            }
        )
        emb = FakeEmbedding({})
        metric = AnswerRelevancy(llm=llm, embeddings=emb, strictness=1)
        result = await metric.ascore(user_input="orig", response="resp")
        assert result.value == 0.0
        assert emb.calls == []

    @pytest.mark.asyncio
    async def test_missing_inputs_raise(self):
        metric = AnswerRelevancy(llm=ScriptedLLM(), embeddings=FakeEmbedding({}))
        with pytest.raises(ValueError, match="user_input cannot be empty"):
            await metric.ascore(user_input="", response="r")
        with pytest.raises(ValueError, match="response cannot be empty"):
            await metric.ascore(user_input="q", response="")


class TestSummaryScore:
    @staticmethod
    def _llm(answers: t.List[str], keyphrases: t.Optional[t.List[str]] = None):
        return ScriptedLLM(
            {
                ExtractedKeyphrases: [
                    ExtractedKeyphrases(
                        keyphrases=["k1"] if keyphrases is None else keyphrases
                    )
                ],
                QuestionsGenerated: [
                    QuestionsGenerated(questions=[f"q{i}" for i in range(len(answers))])
                ],
                AnswersGenerated: [AnswersGenerated(answers=answers)],
            }
        )

    @pytest.mark.asyncio
    async def test_qa_score_with_length_penalty(self):
        llm = self._llm(["1", "0", "1", "1"])
        metric = SummaryScore(llm=llm, coeff=0.5)
        # text "abcde\nfghi" has length 10, summary length 4
        result = await metric.ascore(
            reference_contexts=["abcde", "fghi"], response="abcd"
        )
        qa = 0.75
        conciseness = 1 - 4 / 10
        assert result.value == pytest.approx(qa * 0.5 + conciseness * 0.5)

    @pytest.mark.asyncio
    async def test_qa_score_without_length_penalty(self):
        llm = self._llm(["1", "0", "0", "1"])
        metric = SummaryScore(llm=llm, length_penalty=False)
        result = await metric.ascore(reference_contexts=["ctx"], response="summary")
        assert result.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_summary_longer_than_text_has_zero_conciseness(self):
        llm = self._llm(["1"])
        metric = SummaryScore(llm=llm, coeff=1.0)
        result = await metric.ascore(
            reference_contexts=["short"], response="a much longer summary"
        )
        assert result.value == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_empty_keyphrases_still_continue(self, caplog):
        llm = self._llm(["1", "1"], keyphrases=[])
        metric = SummaryScore(llm=llm, length_penalty=False)
        result = await metric.ascore(reference_contexts=["ctx"], response="s")
        assert result.value == pytest.approx(1.0)
        assert "No keyphrases generated" in caplog.text

    @pytest.mark.asyncio
    async def test_no_answers_raises_zero_division(self):
        llm = self._llm([])
        metric = SummaryScore(llm=llm)
        with pytest.raises(ZeroDivisionError):
            await metric.ascore(reference_contexts=["ctx"], response="s")

    @pytest.mark.asyncio
    async def test_input_validation(self):
        metric = SummaryScore(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="reference_contexts cannot be empty"):
            await metric.ascore(reference_contexts=[], response="s")
        with pytest.raises(ValueError, match="reference_contexts cannot be empty"):
            await metric.ascore(reference_contexts=["  ", ""], response="s")
        with pytest.raises(ValueError, match="response cannot be empty"):
            await metric.ascore(reference_contexts=["ctx"], response="   ")

    def test_invalid_coeff(self):
        with pytest.raises(ValueError, match="Coefficient must be between"):
            SummaryScore(llm=ScriptedLLM(), coeff=1.5)


# ---------------------------------------------------------------------------
# Agentic metrics
# ---------------------------------------------------------------------------


CONVERSATION = [
    HumanMessage(content="Tell me about physics, cooking and history."),
    AIMessage(content="Physics is ... I can't help with cooking. History is ..."),
]


class TestTopicAdherence:
    @staticmethod
    def _llm(classifications: t.List[bool]) -> ScriptedLLM:
        return ScriptedLLM(
            {
                TopicExtractionOutput: [
                    TopicExtractionOutput(topics=["physics", "cooking", "history"])
                ],
                TopicRefusedOutput: [
                    TopicRefusedOutput(refused_to_answer=False),
                    TopicRefusedOutput(refused_to_answer=True),
                    TopicRefusedOutput(refused_to_answer=False),
                ],
                TopicClassificationOutput: [
                    TopicClassificationOutput(classifications=classifications)
                ],
            }
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mode, expected",
        [
            # answered=[T, F, T], in-scope=[T, T, T] -> tp=2, fp=0, fn=1
            ("precision", 1.0),
            ("recall", 2 / 3),
            ("f1", 0.8),
        ],
    )
    async def test_modes(self, mode, expected):
        llm = self._llm([True, True, True])
        metric = TopicAdherence(llm=llm, mode=mode)
        result = await metric.ascore(
            user_input=CONVERSATION, reference_topics=["science", "history"]
        )
        assert result.value == pytest.approx(expected, abs=1e-6)
        refused_prompts = llm.prompts_for(TopicRefusedOutput)
        assert len(refused_prompts) == 3
        assert "cooking" in refused_prompts[1]

    @pytest.mark.asyncio
    async def test_short_classification_list_is_padded_with_false(self):
        # Classifications only cover "physics"; the rest count as off-topic.
        llm = self._llm([True])
        metric = TopicAdherence(llm=llm, mode="precision")
        result = await metric.ascore(user_input=CONVERSATION, reference_topics=["x"])
        # answered=[T, F, T], in-scope=[T, F, F] -> tp=1, fp=1
        assert result.value == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.asyncio
    async def test_long_classification_list_is_truncated(self):
        llm = self._llm([False, True, True, True, True])
        metric = TopicAdherence(llm=llm, mode="recall")
        result = await metric.ascore(user_input=CONVERSATION, reference_topics=["x"])
        # in-scope=[F, T, T] -> tp=1 (history), fn=1 (cooking)
        assert result.value == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.asyncio
    async def test_no_topics_returns_nan(self):
        llm = ScriptedLLM({TopicExtractionOutput: [TopicExtractionOutput(topics=[])]})
        metric = TopicAdherence(llm=llm)
        result = await metric.ascore(user_input=CONVERSATION, reference_topics=["x"])
        assert math.isnan(result.value)
        assert len(llm.calls) == 1

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ([True, False], [True, False]),
            ([1, 0, 2], [True, False, True]),
            (["true", "No", "1", "YES"], [True, False, True, True]),
            ([True, "yes", 0, None], [True, True, False, False]),
            ([1.0, 0.0], [True, False]),
        ],
    )
    def test_safe_bool_conversion(self, raw, expected):
        metric = TopicAdherence(llm=ScriptedLLM())
        assert metric._safe_bool_conversion(raw).tolist() == expected

    @pytest.mark.asyncio
    async def test_input_validation(self):
        metric = TopicAdherence(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="user_input must be a list"):
            await metric.ascore(user_input="hello", reference_topics=["x"])  # type: ignore
        with pytest.raises(ValueError, match="reference_topics must be a non-empty"):
            await metric.ascore(user_input=CONVERSATION, reference_topics=[])


class TestAgentGoalAccuracy:
    @pytest.mark.asyncio
    async def test_with_reference_compares_reference_to_end_state(self):
        llm = ScriptedLLM(
            {
                WorkflowOutput: [
                    WorkflowOutput(user_goal="book table", end_state="table booked")
                ],
                CompareOutcomeOutput: [CompareOutcomeOutput(reason="ok", verdict="1")],
            }
        )
        metric = AgentGoalAccuracyWithReference(llm=llm)
        result = await metric.ascore(
            user_input=CONVERSATION, reference="Table booked at a restaurant"
        )
        assert result.value == 1.0
        assert llm.prompts_for(CompareOutcomeOutput)[
            0
        ] == metric.compare_outcome_prompt.to_string(
            CompareOutcomeInput(
                desired_outcome="Table booked at a restaurant",
                arrived_outcome="table booked",
            )
        )
        # The formatted conversation is what the workflow prompt sees.
        assert CONVERSATION[0].content in llm.prompts_for(WorkflowOutput)[0]

    @pytest.mark.asyncio
    async def test_without_reference_compares_inferred_goal(self):
        llm = ScriptedLLM(
            {
                WorkflowOutput: [
                    WorkflowOutput(user_goal="book table", end_state="nothing booked")
                ],
                CompareOutcomeOutput: [
                    CompareOutcomeOutput(reason="not done", verdict="0")
                ],
            }
        )
        metric = AgentGoalAccuracyWithoutReference(llm=llm)
        result = await metric.ascore(user_input=CONVERSATION)
        assert result.value == 0.0
        assert llm.prompts_for(CompareOutcomeOutput)[
            0
        ] == metric.compare_outcome_prompt.to_string(
            CompareOutcomeInput(
                desired_outcome="book table", arrived_outcome="nothing booked"
            )
        )

    @pytest.mark.asyncio
    async def test_input_validation(self):
        with_ref = AgentGoalAccuracyWithReference(llm=ScriptedLLM())
        without_ref = AgentGoalAccuracyWithoutReference(llm=ScriptedLLM())
        with pytest.raises(ValueError, match="user_input must be a list"):
            await with_ref.ascore(user_input="x", reference="r")  # type: ignore
        with pytest.raises(ValueError, match="reference must be provided"):
            await with_ref.ascore(user_input=CONVERSATION, reference="")
        with pytest.raises(ValueError, match="user_input must be a list"):
            await without_ref.ascore(user_input="x")  # type: ignore
