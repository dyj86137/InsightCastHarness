"""内置计算器工具。"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict

from pydantic import BaseModel, Field

from core.tool import ToolCall, ToolResult
from infra.exception import ToolExecutionError
from tools.base import BaseTool, ToolExecutionContext


class CalculatorInput(BaseModel):
    """计算器输入参数。"""

    expression: str = Field(..., description="需要计算的数学表达式")


class CalculatorTool(BaseTool):
    """只支持安全数学表达式的计算器工具。"""

    name = "calculator"
    description = "Evaluate a safe mathematical expression."
    input_model = CalculatorInput
    is_read_only = True
    source = "builtin"
    version = "v1"
    tags = ("math", "calculator", "builtin")

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行数学表达式计算。"""

        del context
        calculator_input = cast_input(arguments)
        try:
            value = SafeExpressionEvaluator().evaluate(calculator_input.expression)
        except Exception as exc:
            raise ToolExecutionError(
                "Calculator expression evaluation failed.",
                details={
                    "tool_call_id": tool_call.id,
                    "expression": calculator_input.expression,
                },
                cause=exc,
            ) from exc

        output = format_number(value)
        return ToolResult.success(
            tool_call=tool_call,
            output=output,
            data={"value": value, "expression": calculator_input.expression},
        )


class SafeExpressionEvaluator:
    """基于 AST 白名单的安全数学表达式求值器。"""

    def __init__(self) -> None:
        self.binary_operators: Dict[type, Callable[[float, float], float]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        self.unary_operators: Dict[type, Callable[[float], float]] = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }
        self.functions: Dict[str, Callable[..., float]] = {
            "abs": abs,
            "ceil": math.ceil,
            "floor": math.floor,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
            "sqrt": math.sqrt,
        }
        self.constants: Dict[str, float] = {
            "e": math.e,
            "pi": math.pi,
        }

    def evaluate(self, expression: str) -> float:
        """计算表达式并返回数值。"""

        if not expression or not expression.strip():
            raise ValueError("Expression cannot be empty.")

        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Num):
            return node.n

        if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)
            operation = self.binary_operators.get(operator_type)
            if operation is None:
                raise ValueError(f"Unsupported binary operator: {operator_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)
            operation = self.unary_operators.get(operator_type)
            if operation is None:
                raise ValueError(f"Unsupported unary operator: {operator_type.__name__}")
            return operation(self._eval_node(node.operand))

        if isinstance(node, ast.Name):
            if node.id in self.constants:
                return self.constants[node.id]
            raise ValueError(f"Unknown name: {node.id}")

        if isinstance(node, ast.Call):
            return self._eval_call(node)

        raise ValueError(f"Unsupported expression node: {node.__class__.__name__}")

    def _eval_call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")

        function = self.functions.get(node.func.id)
        if function is None:
            raise ValueError(f"Unsupported function: {node.func.id}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")

        args = [self._eval_node(arg) for arg in node.args]
        return function(*args)


def cast_input(arguments: BaseModel) -> CalculatorInput:
    """把通用 BaseModel 参数收窄为 CalculatorInput。"""

    if not isinstance(arguments, CalculatorInput):
        if hasattr(arguments, "model_dump"):
            return CalculatorInput(**arguments.model_dump())
        return CalculatorInput(**arguments.dict())
    return arguments


def format_number(value: Any) -> str:
    """格式化计算结果，避免 2.0 这类结果污染模型上下文。"""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


__all__ = [
    "CalculatorInput",
    "CalculatorTool",
    "SafeExpressionEvaluator",
    "format_number",
]
