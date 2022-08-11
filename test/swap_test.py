from random_code.impl import (
    merge_unbundled_asts,
    BagOfConcepts,
    RandomizingTransformer,
    loop_detection,
)

from ast import (
    arguments,
    ClassDef,
    Assign,
    DictComp,
    ExceptHandler,
    Expr,
    FunctionDef,
    GeneratorExp,
    IfExp,
    Lambda,
    ListComp,
    Module,
    SetComp,
    Try,
    With,
)
from ast import fix_missing_locations, NodeVisitor

try:
    # python3.9 and after
    from ast import unparse
except ImportError:
    # before python3.9's ast.unparse
    from astunparse import unparse


def ast_unparse(ast):
    return unparse(fix_missing_locations(ast))


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


def str_to_ast(s, keep_module=False):
    import ast

    if keep_module:
        return ast.parse(s)
    return _strip_module(ast.parse(s))


def build_transformer(ast, seed=0):
    ast_set = {"string": ast}
    raw_materials = merge_unbundled_asts(ast_set.values())
    gen = BagOfConcepts(raw_materials, seed=seed)

    transformer = RandomizingTransformer(corpus=gen, log_level="DEBUG")
    return transformer


def test_FunctionDef_recursion():
    input_text = """
def main():
    def alt():
        def alt2():
            return 3
        return 2
    return 1
"""
    main_func_def = str_to_ast(input_text)
    assert isinstance(main_func_def, FunctionDef)
    alt_func_def = main_func_def.body[0]
    assert isinstance(alt_func_def, FunctionDef)

    transformer = build_transformer(main_func_def, seed=0)
    transformer.scope["__random_code_return_ok"] = True

    result = transformer.visit(main_func_def)

    assert loop_detection(result)
