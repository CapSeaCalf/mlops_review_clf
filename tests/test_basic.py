import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import assign_classes, clean_text  # noqa: E402


def test_clean_restores_negation():
    assert clean_text("doesn't have any flaws") == "does not have any flaws"


def test_clean_drops_single_chars():
    assert clean_text("I love it!!") == "i love it"


def test_rule_bands():
    r = {"type": "asym", "lo": 0.15, "hi": 0.60}
    assert list(assign_classes([0.05, 0.3, 0.7], r)) == [0, 1, 2]


@pytest.mark.skipif(not (ROOT / "models/sentiment_rnn.pt").exists(), reason="немає моделі")
def test_predictor_runs():
    from predict import SentimentPredictor
    res = SentimentPredictor().predict(["great product", "the speaker broke after a week"])
    assert len(res) == 2 and res["p_positive"].between(0, 1).all()
    assert set(res["label"]) <= {"negative", "neutral", "positive"}
