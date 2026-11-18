import json
import typing as t
from dataclasses import dataclass, field

import pandas as pd
import pytest
from langchain_core.outputs import Generation, LLMResult
from pydantic import BaseModel

from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)
from ragas.llms.base import BaseRagasLLM
from ragas.losses import BinaryMetricLoss
from ragas.metrics._aspect_critic import AspectCritic
from ragas.metrics.base import MetricOutputType
from ragas.optimizers.genetic import FormattedExamples, GeneticOptimizer

PROMPT = "single_turn_aspect_critic_prompt"


@dataclass
class ScriptedLLM(BaseRagasLLM):
    """Answers each optimizer prompt with a fixed, schema-valid JSON payload."""

    prompts: t.List[str] = field(default_factory=list)
    feedbacks: t.List[str] = field(default_factory=lambda: ["Be stricter."])

    def _respond(self, prompt) -> LLMResult:
        text = prompt.to_string()
        self.prompts.append(text)
        if "expert reviewer" in text:
            payload = {"feedbacks": self.feedbacks}
        elif "familiar with the concept of cross-over" in text:
            payload = {"instruction": "crossed instruction"}
        elif "set of feedbacks" in text:
            payload = {"instruction": "improved instruction"}
        elif "guess and generate the instruction" in text:
            payload = {"instruction": "reverse engineered instruction"}
        else:
            raise AssertionError(f"unexpected prompt: {text[:200]}")
        return LLMResult(generations=[[Generation(text=json.dumps(payload))]])

    def generate_text(self, prompt, *args, **kwargs):  # type: ignore[override]
        return self._respond(prompt)

    async def agenerate_text(self, prompt, *args, **kwargs):  # type: ignore[override]
        return self._respond(prompt)

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def count(self, marker: str) -> int:
        return sum(marker in p for p in self.prompts)


class _Input(BaseModel):
    user_input: str


class _Output(BaseModel):
    verdict: int


class FakeResult:
    """Stands in for EvaluationResult, which only needs to_pandas() and traces."""

    def __init__(self, metric_name, predictions):
        self.metric_name = metric_name
        self.predictions = predictions
        self.traces = [
            {
                metric_name: {
                    PROMPT: {
                        "input": _Input(user_input=f"q{i}"),
                        "output": _Output(verdict=int(p)),
                    }
                }
            }
            for i, p in enumerate(predictions)
        ]

    def to_pandas(self):
        return pd.DataFrame({self.metric_name: self.predictions})


def _sample(i, output, accepted=True, edited=None):
    return SampleAnnotation(
        metric_input={"user_input": f"q{i}", "response": f"a{i}"},
        metric_output=output,
        prompts={
            PROMPT: PromptAnnotation(
                prompt_input={"user_input": f"q{i}", "response": f"a{i}", "x": None},
                prompt_output={"reason": "r", "verdict": int(output)},
                edited_output=edited,
            )
        },
        is_accepted=accepted,
    )


def _dataset(n=10):
    return SingleMetricAnnotation(
        name="crit", samples=[_sample(i, float(i % 2)) for i in range(n)]
    )


@pytest.fixture
def llm():
    return ScriptedLLM()


@pytest.fixture
def metric(llm):
    return AspectCritic(name="crit", definition="Is the answer correct?", llm=llm)


@pytest.fixture
def optimizer(metric, llm):
    return GeneticOptimizer(metric=metric, llm=llm)


def test_formatted_examples():
    formatted = FormattedExamples.from_examples(
        [{"input": {"q": "hi", "r": "there"}, "output": {"verdict": 1}}]
    )
    assert formatted.examples == [("\nq:\n\thi\n\nr:\n\tthere\n", {"verdict": 1})]


class TestOptimizeValidation:
    def test_requires_metric(self, llm):
        with pytest.raises(ValueError, match="No metric"):
            GeneticOptimizer(llm=llm).optimize(_dataset(), BinaryMetricLoss(), {})

    def test_requires_llm(self, metric):
        with pytest.raises(ValueError, match="No llm"):
            GeneticOptimizer(metric=metric).optimize(_dataset(), BinaryMetricLoss(), {})

    def test_requires_enough_annotations(self, optimizer):
        with pytest.raises(ValueError, match="annotate 7 more samples"):
            optimizer.optimize(_dataset(3), BinaryMetricLoss(), {})


@pytest.mark.asyncio
async def test_reverse_engineer_prefers_edited_output(optimizer, llm):
    batch = [
        _sample(0, 1.0),
        _sample(1, 0.0, edited={"reason": "fixed", "verdict": 1}),
    ]
    candidates = await optimizer._reverse_engineer_instruction(batch)

    assert candidates == {PROMPT: "reverse engineered instruction"}
    prompt = llm.prompts[-1]
    assert '"reason": "fixed"' in prompt
    # None-valued prompt inputs are dropped from the demonstrations
    assert "x:" not in prompt


@pytest.mark.asyncio
async def test_cross_over_chain(optimizer, llm):
    offspring = await optimizer._cross_over_chain(
        {"a": "p1", "b": "p2"}, {"a": "p3", "b": "p4"}, callbacks=None
    )
    assert offspring == {"a": "crossed instruction", "b": "crossed instruction"}
    assert llm.count("familiar with the concept of cross-over") == 2

    with pytest.raises(ValueError, match="same prompt names"):
        await optimizer._cross_over_chain({"a": "x"}, {"b": "y"}, callbacks=None)


@pytest.mark.asyncio
async def test_implement_feedbacks(optimizer, llm, caplog):
    improved = await optimizer._implement_feedbacks(
        {"a": "original a", "b": "original b"},
        {"a": ["add detail"], "b": []},
    )
    assert improved == {"a": "improved instruction", "b": "original b"}
    assert llm.count("set of feedbacks") == 1
    assert "No feedbacks found for the prompt b" in caplog.text


@pytest.mark.asyncio
async def test_get_feedbacks_only_for_mispredictions(optimizer, llm):
    dataset = _dataset(4)
    target = [0.0, 1.0, 0.0, 1.0]
    results = FakeResult("crit", [0.0, 0.0, 0.0, 1.0])

    feedbacks = await optimizer._get_feedbacks(
        {PROMPT: "instr"}, dataset, results, target
    )

    assert feedbacks == {PROMPT: ["Be stricter."]}
    prompt = llm.prompts[-1]
    # only the mispredicted sample (index 1) is sent for review
    assert "q1" in prompt
    assert "q0" not in prompt and "q2" not in prompt


@pytest.mark.asyncio
async def test_get_feedbacks_no_mispredictions(optimizer, llm, caplog):
    results = FakeResult("crit", [0.0, 1.0])
    feedbacks = await optimizer._get_feedbacks(
        {PROMPT: "instr"}, _dataset(2), results, [0.0, 1.0]
    )
    assert feedbacks == {PROMPT: []}
    assert llm.prompts == []
    assert "No samples found" in caplog.text


class TestGetEvaluationDataset:
    def test_binary_metric_flips_rejected_samples(self, optimizer):
        dataset = SingleMetricAnnotation(
            name="crit",
            samples=[
                _sample(0, 1.0),
                _sample(1, 1.0, accepted=False),
                _sample(2, 0.0),
            ],
        )
        eval_dataset, y_true = optimizer._get_evaluation_dataset(dataset)
        assert y_true == [1.0, 0, 0.0]
        assert [s.user_input for s in eval_dataset.samples] == ["q0", "q1", "q2"]

    def test_non_binary_metric_drops_rejected_samples(self, optimizer):
        optimizer.metric.output_type = MetricOutputType.CONTINUOUS
        dataset = SingleMetricAnnotation(
            name="crit",
            samples=[_sample(0, 0.7), _sample(1, 0.2, accepted=False)],
        )
        eval_dataset, y_true = optimizer._get_evaluation_dataset(dataset)
        assert y_true == [0.7]
        assert len(eval_dataset) == 1

    def test_requires_output_type(self, optimizer):
        optimizer.metric.output_type = None
        with pytest.raises(ValueError, match="No output type"):
            optimizer._get_evaluation_dataset(_dataset(2))


def test_evaluate_candidate_sets_instructions(optimizer, monkeypatch):
    calls = {}

    def fake_evaluate(dataset, **kwargs):
        calls["dataset"] = dataset
        calls.update(kwargs)
        return "result"

    monkeypatch.setattr("ragas.optimizers.genetic.evaluate", fake_evaluate)
    eval_dataset = _dataset(2).to_evaluation_dataset()

    result = optimizer.evaluate_candidate(
        candidate={PROMPT: "new instruction"}, eval_dataset=eval_dataset
    )

    assert result == "result"
    assert optimizer.metric.get_prompts()[PROMPT].instruction == "new instruction"
    assert calls["dataset"] is eval_dataset
    assert calls["metrics"] == [optimizer.metric]
    assert calls["llm"] is optimizer.llm
    assert calls["return_executor"] is False


def _patch_predictions(monkeypatch, optimizer, predict):
    """Replace evaluate_candidate with a deterministic per-candidate predictor."""
    seen = []

    def fake_evaluate_candidate(*, candidate, eval_dataset, **kwargs):
        seen.append(candidate)
        preds = [predict(candidate, s) for s in eval_dataset.samples]
        return FakeResult(optimizer.metric.name, preds)

    monkeypatch.setattr(optimizer, "evaluate_candidate", fake_evaluate_candidate)
    return seen


def test_evaluate_fitness(optimizer, monkeypatch):
    dataset = _dataset(4)  # targets: 0, 1, 0, 1
    _patch_predictions(
        monkeypatch,
        optimizer,
        lambda cand, s: (
            1.0 if cand[PROMPT] == "always yes" else float(int(s.user_input[1:]) % 2)
        ),
    )

    scores = optimizer.evaluate_fitness(
        candidates=[{PROMPT: "always yes"}, {PROMPT: "perfect"}],
        dataset=dataset,
        loss_fn=BinaryMetricLoss(),
        callbacks=[],
    )
    assert scores == [0.5, 1.0]


def test_cross_over_mutation(optimizer, llm, monkeypatch):
    seen = _patch_predictions(monkeypatch, optimizer, lambda cand, s: 1.0)
    candidates = [{PROMPT: "p1"}, {PROMPT: "p2"}, {PROMPT: "p3"}]

    offspring = optimizer.cross_over_mutation(
        candidates=candidates, dataset=_dataset(4), callbacks=[]
    )

    assert offspring == [{PROMPT: "crossed instruction"}] * 3
    assert seen == candidates
    assert llm.count("familiar with the concept of cross-over") == 3


def test_feedback_mutation(optimizer, llm, monkeypatch):
    # always predict 1 so every target-0 sample is a misprediction
    _patch_predictions(monkeypatch, optimizer, lambda cand, s: 1.0)

    improved = optimizer.feedback_mutation(
        [{PROMPT: "p1"}, {PROMPT: "p2"}], _dataset(10), sample_size=4, callbacks=[]
    )

    assert improved == [{PROMPT: "improved instruction"}] * 2
    assert llm.count("expert reviewer") == 2
    assert llm.count("set of feedbacks") == 2


def test_optimize_end_to_end(optimizer, llm, monkeypatch):
    seen = _patch_predictions(
        monkeypatch, optimizer, lambda cand, s: float(int(s.user_input[1:]) % 2 == 0)
    )

    best = optimizer.optimize(
        _dataset(10),
        BinaryMetricLoss(),
        config={"population_size": 2, "num_demonstrations": 2, "sample_size": 4},
    )

    assert best == {PROMPT: "crossed instruction"}
    # one reverse-engineered candidate plus the metric's seed prompt
    assert llm.count("guess and generate the instruction") == 1
    assert llm.count("expert reviewer") == 2
    assert llm.count("familiar with the concept of cross-over") == 2
    # feedback (2) + cross-over (2) + fitness (2) evaluations
    assert len(seen) == 6
