from random_code.impl import merge_unbundled_asts, BagOfConcepts, RandomizingTransformer

from ast import Module, arg, Name
from collections import ChainMap


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
    transformer.scope = ChainMap({"int": "Type", "str": "Type"})

    assert not transformer.valid_swap(base, swap)


def test_arg_typing_exact_matched():
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


def test_arg_typing_any_matched():
    definition1 = """
def x(i: Any):
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

    assert transformer.valid_swap(base, swap)


def test_arg_typing_no_annotation_matched():
    definition1 = """
def x(i):
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

    assert transformer.valid_swap(base, swap)


def test_name_typing_in_scope_matching():
    definition1 = """x = 0"""
    definition2 = """y = 1"""
    base = str_to_ast(definition1).targets[0]
    swap = str_to_ast(definition2).targets[0]

    print(base._fields)

    assert isinstance(base, Name)
    assert isinstance(swap, Name)

    transformer = build_transformer()
    transformer.scope = ChainMap({"x": "int", "y": "int"})

    assert transformer.valid_swap(base, swap)


def test_name_typing_in_scope_nonmatching():
    definition1 = """x = 0"""
    definition2 = """y = 1"""
    base = str_to_ast(definition1).targets[0]
    swap = str_to_ast(definition2).targets[0]

    print(base._fields)

    assert isinstance(base, Name)
    assert isinstance(swap, Name)

    transformer = build_transformer()
    transformer.scope = {"x": "int", "y": "FunctionDef"}

    assert not transformer.valid_swap(base, swap)
