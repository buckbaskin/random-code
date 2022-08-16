from random_code.impl import (
    merge_unbundled_asts,
    BagOfConcepts,
    RandomizingTransformer,
    loop_detection,
)

from ast import (
    arguments,
    Assign,
    ClassDef,
    DictComp,
    ExceptHandler,
    Expr,
    FunctionDef,
    GeneratorExp,
    IfExp,
    Lambda,
    ListComp,
    ListComp,
    Module,
    Name,
    SetComp,
    Try,
    Tuple,
    With,
)
from ast import fix_missing_locations, NodeVisitor
from collections import ChainMap

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


def build_transformer(ast, seed=0, visit_only=False):
    ast_set = {"string": ast}
    raw_materials = merge_unbundled_asts(ast_set.values())
    gen = BagOfConcepts(raw_materials, seed=seed)

    transformer = RandomizingTransformer(
        corpus=gen, log_level="DEBUG", visit_only=visit_only
    )
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


def test_ListComp_elt_swaping():
    input_text = "[x for x, y in ((1, 'one'), (2, 'two'), (3, 'three'))]"

    list_comp = _strip_expr(str_to_ast(input_text))
    assert isinstance(list_comp, ListComp)

    transformer = build_transformer(list_comp, visit_only=True)

    result = transformer.visit(list_comp)

    assert "x" in result.elt._ending_scope
    assert "y" in result.elt._ending_scope

    transformer.scope = ChainMap(result.elt._ending_scope)
    assert transformer.valid_swap(result.elt, Name("y"))


def test_ListComp_generator_names_swaping():
    input_text = "[x for x, y in ((1, 'one'), (2, 'two'), (3, 'three'))]"

    list_comp = _strip_expr(str_to_ast(input_text))
    assert isinstance(list_comp, ListComp)

    transformer = build_transformer(list_comp, visit_only=True)

    result = transformer.visit(list_comp)

    print(result._fields)
    print(ast_unparse(result.elt))
    for idx, gen in enumerate(result.generators):
        print(idx, ast_unparse(gen), gen._fields)
        for f in gen._fields:
            if isinstance(getattr(gen, f), list):
                print(f, getattr(gen, f))
                for idx2, elem in enumerate(getattr(gen, f)):
                    print("   ", idx2, ast_unparse(elem))
            elif isinstance(getattr(gen, f), int):
                print(f, getattr(gen, f))
            else:
                print(f, getattr(gen, f), ast_unparse(getattr(gen, f)))

    assert transformer.valid_swap(result.generators[0].target, Tuple([Name("z")]))
