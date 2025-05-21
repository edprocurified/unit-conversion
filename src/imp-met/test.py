import pytest
from tool import evaluate


@pytest.mark.parametrize(
    "expr, expected",
    [
        # single numbers, whitespace
        ("5", 5.0),
        ("   5   ", 5.0),
        ("+5", 5.0),
        ("-+5", -5.0),
        # decimals & sci-notation
        ("3.5 + 2.1", 5.6),
        ("1e3 + 2e2", 1200.0),
        # nested parens & precedence
        ("((2 + 3) * (4 - 1)) / 5", 3.0),
        ("(1 + 2) - (3 - 4)", 4.0),
        # unary with binops
        ("2 + -3", -1.0),
        ("2 + (-3)", -1.0),
        # powers—right assoc & negatives
        ("2 ** 3 ** 2", 512.0),
        ("(2 ** 3) ** 2", 64.0),
        ("2 ** -3", 0.125),
        # more complex
        ("3 + 4 * 2 / (1 - 5) ** 2", 3.5),
        ("((2+3)*4/(1-5))**2", 25.0),
    ],
)
def test_additional_valid(expr: str, expected: int | float):
    assert abs(evaluate(expr) - expected) < 1e-6


@pytest.mark.parametrize(
    "expr",
    [
        # bitwise & floor/mod
        "2 << 3",
        "2 & 1",
        "2 | 1",
        "~1",
        "10 // 3",
        "5 % 3",
        # names, calls, conditionals
        "x + 1",
        "round(5.1)",
        "3 if True else 4",
        # comprehensions, strings, collections
        "[x for x in [1,2]]",
        "'foo' + 'bar'",
        "{1,2,3}",
        # completely invalid
        "import os",
        "",
        "   ",
    ],
)
def test_additional_invalid(expr: str):
    with pytest.raises(ValueError):
        evaluate(expr)
