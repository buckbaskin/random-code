from random_code.impl import merge_unbundled_asts, BagOfConcepts, RandomizingTransformer

from ast import Module, Expr, IfExp, FunctionDef, arguments


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
