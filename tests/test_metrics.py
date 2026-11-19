import pytest

from llm_opt_lab.metrics import (
    flatten,
    generic_path,
    normalized_edit_distance,
    per_field_stats,
    score_sample,
    summarize,
)

GOLD = {"store": {"name": "an"}, "items": [{"price": "1"}, {"price": "2"}], "note": None}


def score(pred):
    return score_sample("s1", "receipt", "image", GOLD, pred)


def test_flatten():
    assert flatten(GOLD) == {
        "store.name": "an",
        "items[0].price": "1",
        "items[1].price": "2",
        "note": None,
    }
    assert flatten({"items": [], "store": {}}) == {"items": None, "store": None}
    assert generic_path("items[12].sub[0].price") == "items[].sub[].price"


def test_exact_match():
    result = score(GOLD)
    assert result.exact_match and result.parsed
    assert all(f.correct and f.ned == 0 for f in result.fields)


def test_missing_null_field_is_correct():
    pred = {k: v for k, v in GOLD.items() if k != "note"}
    assert score(pred).exact_match


def test_wrong_field():
    result = score({**GOLD, "store": {"name": "am"}})
    assert not result.exact_match
    wrong = [f for f in result.fields if not f.correct]
    assert [f.path for f in wrong] == ["store.name"]
    assert wrong[0].ned == pytest.approx(0.5)


def test_extra_field_breaks_exact_match_only():
    result = score({**GOLD, "items": GOLD["items"] + [{"price": "3"}]})
    assert result.extra_fields == ["items[2].price"]
    assert not result.exact_match
    assert all(f.correct for f in result.fields)


def test_shorter_list_misses_fields():
    result = score({**GOLD, "items": GOLD["items"][:1]})
    assert [f.path for f in result.fields if not f.correct] == ["items[1].price"]


def test_unparsed_prediction_gets_every_field_wrong():
    result = score(None)
    assert not result.parsed and not result.exact_match
    assert all(not f.correct and f.ned == 1.0 for f in result.fields)


def test_ned():
    assert normalized_edit_distance(None, None) == 0.0
    assert normalized_edit_distance("abcd", None) == 1.0
    assert normalized_edit_distance("abcd", "abce") == pytest.approx(0.25)


def test_summarize_and_per_field():
    scores = [score(GOLD), score({**GOLD, "store": {"name": "x"}}), score(None)]
    stats = summarize(scores)
    assert stats.n == 3 and stats.n_fields == 12
    assert stats.exact_match == pytest.approx(1 / 3)
    assert stats.field_accuracy == pytest.approx(7 / 12)
    assert stats.parse_rate == pytest.approx(2 / 3)
    worst = per_field_stats(scores)[0]
    assert (worst.field, worst.accuracy) == ("store.name", pytest.approx(1 / 3))
