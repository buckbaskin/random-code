from random_code.impl import merge_unbundled_asts, BagOfConcepts, RandomizingTransformer

from ast import Module, Expr, IfExp


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

    transformer = RandomizingTransformer(corpus=gen)
    return transformer


def test_FunctionDef_annotation():
    input_text = """
def main(i: i):
    return 1
"""
    ast = str_to_ast(input_text)
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    out_of_scope = sorted(list(transformer.out_of_scope))

    assert out_of_scope == ["i"]
