"""``DataCompyScore.ascore`` tests that run against any installed datacompy.

datacompy 1.x removed ``Compare`` (now ``PandasCompare``), which the metric
imports at construction time. When ``datacompy.core.Compare`` is missing, the
``compare_available`` fixture exposes ``PandasCompare`` under that name so the
scoring logic can still be exercised.
"""

import math
import sys
import types

import pytest

datacompy = pytest.importorskip("datacompy")

from ragas.metrics.collections.datacompy_score.metric import (  # noqa: E402
    DataCompyScore,
)


@pytest.fixture
def compare_available(monkeypatch):
    try:
        from datacompy.core import Compare  # noqa: F401
    except ImportError:
        compare_cls = getattr(datacompy, "Compare", None) or getattr(
            datacompy, "PandasCompare", None
        )
        if compare_cls is None:
            pytest.skip("installed datacompy exposes no pandas Compare class")
        shim = types.ModuleType("datacompy.core")
        shim.Compare = compare_cls  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "datacompy.core", shim)


REFERENCE = "id,name\n1,Alice\n2,Bob\n3,Cara\n4,Dan"
RESPONSE = "id,name\n1,Alice\n2,Bob\n3,Carol"


@pytest.mark.usefixtures("compare_available")
class TestDataCompyScoreAscore:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metric_name, expected",
        [
            # 2 matching rows; reference has 4 rows, response has 3
            ("precision", 2 / 3),
            ("recall", 2 / 4),
            ("f1", 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5)),
        ],
    )
    async def test_rows_mode(self, metric_name, expected):
        metric = DataCompyScore(mode="rows", metric=metric_name)
        result = await metric.ascore(reference=REFERENCE, response=RESPONSE)
        assert result.value == pytest.approx(expected)
        assert result.reason.startswith("Mode: rows")

    @pytest.mark.asyncio
    async def test_columns_mode_counts_fully_equal_columns(self):
        metric = DataCompyScore(mode="columns", metric="precision")
        response = "id,name,extra\n1,Alice,x\n2,Bob,y\n3,Carol,z\n4,Dan,w"
        result = await metric.ascore(reference=REFERENCE, response=response)
        # "id" matches, "name" differs on one row, "extra" is not shared:
        # 1 matching column of 3 response columns.
        assert result.value == pytest.approx(1 / 3)
        recall = await DataCompyScore(mode="columns", metric="recall").ascore(
            reference=REFERENCE, response=response
        )
        assert recall.value == pytest.approx(1 / 2)

    @pytest.mark.asyncio
    async def test_no_overlap_f1_is_zero(self):
        metric = DataCompyScore(mode="rows", metric="f1")
        result = await metric.ascore(
            reference="id,name\n1,Alice", response="id,name\n1,Zed"
        )
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_unparseable_csv_returns_nan(self):
        metric = DataCompyScore()
        result = await metric.ascore(reference=REFERENCE, response="")
        assert math.isnan(result.value)
        assert "CSV parsing error" in result.reason

    @pytest.mark.asyncio
    async def test_non_string_inputs_raise(self):
        metric = DataCompyScore()
        with pytest.raises(ValueError, match="reference must be a CSV string"):
            await metric.ascore(reference=1, response=RESPONSE)  # type: ignore
        with pytest.raises(ValueError, match="response must be a CSV string"):
            await metric.ascore(reference=REFERENCE, response=None)  # type: ignore

    def test_invalid_mode_and_metric(self):
        with pytest.raises(ValueError, match="mode must be"):
            DataCompyScore(mode="cells")  # type: ignore
        with pytest.raises(ValueError, match="metric must be"):
            DataCompyScore(metric="accuracy")  # type: ignore


def test_missing_compare_class_raises_import_error(monkeypatch):
    fake_pkg = types.ModuleType("datacompy")
    monkeypatch.setitem(sys.modules, "datacompy", fake_pkg)
    monkeypatch.setitem(sys.modules, "datacompy.core", None)
    with pytest.raises(ImportError, match="datacompy is required for DataCompyScore"):
        DataCompyScore()
