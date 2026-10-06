import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex4.tokenizer import tokenize


@pytest.mark.parametrize(
    ("input_text", "expected_tokens"),
    [
        ("", []),
        ("Hello World PyThOn", ["hello", "world", "python"]),
        (
            "Тестовий корпус ТЕКСТІВ для Лабораторної",
            ["тестовий", "корпус", "текстів", "для", "лабораторної"],
        ),
        (
            f"caf{'e' + unicodedata.lookup('COMBINING ACUTE ACCENT')}",
            ["café"],
        ),
        (
            "Hello, world! Price is 100$ (test_ver 2.0).",
            ["hello", "world", "price", "is", "test_ver"],
        ),
        (
            "об'єм don't word-level будь-ласка",
            ["об'єм", "don't", "word-level", "будь-ласка"],
        ),
        (
            "Line1 \n\t Line2   \r\n Line3",
            ["line1", "line2", "line3"],
        ),
    ],
)
def test_tokenize_cases(input_text: str, expected_tokens: list[str]) -> None:
    tokens = list(tokenize(input_text))
    assert tokens == expected_tokens

    for token in tokens:
        assert unicodedata.is_normalized("NFC", token)




@given(st.text())
def test_tokenize_properties(text: str) -> None:
    tokens = list(tokenize(text))
    for token in tokens:
        assert token == token.casefold()
        assert unicodedata.is_normalized("NFC", token)
        assert not token.isdigit()
        assert not token.startswith("_")