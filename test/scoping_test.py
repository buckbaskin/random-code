from random_code.impl import merge_unbundled_asts, BagOfConcepts, RandomizingTransformer

from ast import (
    arguments,
    ExceptHandler,
    Expr,
    FunctionDef,
    IfExp,
    Lambda,
    Module,
    Try,
    With,
)


def _strip_module(ast):
    if isinstance(ast, Module):
        body = ast.body
        for expr in body:
            return expr
    raise ValueError("No Module to strip")


def _strip_expr(ast):
    if isinstance(ast, Expr):
        return ast.value
    raise ValueError("No Expr to strip")


def str_to_ast(s):
    import ast

    return _strip_module(ast.parse(s))


def build_transformer(ast):
    ast_set = {"string": ast}
    raw_materials = merge_unbundled_asts(ast_set.values())
    gen = BagOfConcepts(raw_materials, seed=0)

    transformer = RandomizingTransformer(corpus=gen, log_level="DEBUG")
    return transformer


def test_FunctionDef_annotation():
    input_text = """
def main(i: i):
    return 1
"""
    ast = str_to_ast(input_text)
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, FunctionDef)
    args = result.args
    assert isinstance(args, arguments)
    assert "i" not in args._ending_scope


def test_Lambda_args():
    input_text = "lambda x: x + 1"
    ast = _strip_expr(str_to_ast(input_text))
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, Lambda)
    assert "x" in result.args._ending_scope
    assert "x" in result.body._ending_scope


def test_ExceptHandler():
    input_text = """
try:
    pass
except ValueError as ve:
    pass
"""
    ast = str_to_ast(input_text)
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, Try)
    handler = result.handlers[0]
    assert "ve" in handler.body[0]._ending_scope


def test_With():
    input_text = """
with open('f') as f:
    pass
"""
    ast = str_to_ast(input_text)
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, With)
    assert "f" in result.body[0]._ending_scope


# TODO(buck): GeneratorExpressions
# TODO(buck): ListComps
# TODO(buck): Assignment
# TODO(buck): DictComps
# TODO(buck): SetComps
# TODO(buck): ClassDef
