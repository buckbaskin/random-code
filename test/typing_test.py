from random_code.impl import merge_unbundled_asts, BagOfConcepts, RandomizingTransformer

from ast import Module, arg


def _strip_module(ast):
    if isinstance(ast, Module):
        body = ast.body
        for expr in body:
            return expr
    raise ValueError("No Module to strip")


def str_to_ast(s):
    import ast

    return _strip_module(ast.parse(s))


def build_transformer():
    ast_set = {}
    raw_materials = merge_unbundled_asts(ast_set.values())
    gen = BagOfConcepts(raw_materials, seed=0)

    transformer = RandomizingTransformer(corpus=gen)
    return transformer


def test_arg_typing_not_matched():
    definition1 = """
def x(i: int):
    pass"""
    definition2 = """
def y(s: str):
    pass"""
    base = str_to_ast(definition1).args.args[0]
    swap = str_to_ast(definition2).args.args[0]

    assert isinstance(base, arg)
    assert isinstance(swap, arg)

    transformer = build_transformer()
    transformer.scope = {"int": "Type", "str": "Type"}

    assert not transformer.valid_swap(base, swap)


def test_arg_typing_matched():
    definition1 = """
def x(i: int):
    pass"""
    definition2 = """
def y(j: int):
    pass"""
    base = str_to_ast(definition1).args.args[0]
    swap = str_to_ast(definition2).args.args[0]

    assert isinstance(base, arg)
    assert isinstance(swap, arg)

    transformer = build_transformer()
    transformer.scope = {"int": "Type", "str": "Type"}

    assert transformer.valid_swap(base, swap)
