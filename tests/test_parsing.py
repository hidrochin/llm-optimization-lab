import pytest

from llm_opt_lab.parsing import parse_json_output


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '<think>\n\n</think>\n\n{"a": 1}',
        'Here is the result: {"a": 1} Hope this helps.',
    ],
)
def test_parse_success(text):
    assert parse_json_output(text) == ({"a": 1}, None)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (None, "no output"),
        ("   ", "empty output"),
        ('{"a": 1', "invalid JSON"),
        ("[1, 2]", "expected a JSON object"),
    ],
)
def test_parse_failure(text, reason):
    obj, error = parse_json_output(text)
    assert obj is None
    assert error.startswith(reason)
