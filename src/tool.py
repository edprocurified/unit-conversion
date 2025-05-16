import ast
import operator as op
from typing import Callable, Dict, Final, Union

# Supported types
Number = Union[int, float]
Expression = str | int | float

# Mapping of supported binary operators
OPERATORS: Final[Dict[type[ast.AST], Callable[[Number, Number], Number]]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}

# Mapping of supported unary operators
UNARY_OPERATORS: Final[Dict[type[ast.AST], Callable[[Number], Number]]] = {
    ast.UAdd: lambda x: x,
    ast.USub: op.neg,
}


def evaluate(expr: Expression) -> float:
    """
    Safely evaluate a numeric expression supporting +, -, *, /, ** and unary +/-,
    using AST parsing. Raises ValueError on invalid input.
    Use it to compute the conversions.
    """
    # Direct numeric input
    if isinstance(expr, (int, float)):
        return float(expr)

    expr_str = expr.strip()
    if expr_str == "":
        raise ValueError("Empty expression")

    # Parse and catch syntax/indent errors
    try:
        parsed = ast.parse(expr_str, mode="eval")
    except (SyntaxError, IndentationError) as e:
        raise ValueError(f"Invalid expression: {expr!r}") from e

    def _walk(node: ast.AST) -> Number:
        # Expression wrapper
        if isinstance(node, ast.Expression):
            return _walk(node.body)

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            operator = OPERATORS.get(type(node.op))
            if operator is None:
                raise ValueError(
                    f"Unsupported binary operator: {type(node.op).__name__}"
                )
            return operator(left, right)

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = _walk(node.operand)
            operator = UNARY_OPERATORS.get(type(node.op))
            if operator is None:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
            return operator(operand)

        # Numeric literals
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

        # Anything else is disallowed
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    result = _walk(parsed)
    return float(result)
