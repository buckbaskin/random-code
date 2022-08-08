from random_code.impl import merge_unbundled_asts, BagOfConcepts, RandomizingTransformer

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
from ast import fix_missing_locations

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
print(main)
"""
    ast = str_to_ast(input_text, keep_module=True)
    transformer = build_transformer(ast)
    transformer.scope["print"] = "builtin"
    result = transformer.visit(ast)

    funcdef = result.body[0]
    the_rest = result.body[1]
    assert isinstance(funcdef, FunctionDef)
    args = funcdef.args
    assert isinstance(args, arguments)
    # TODO(buck): Make sure the arg name isn't in scope for the annotation
    # assert "i" not in args.args[0].annotation._ending_scope
    assert "i" in args._ending_scope
    assert "i" in funcdef.body[0]._ending_scope
    assert "main" in funcdef.body[0]._ending_scope

    assert "main" in the_rest._ending_scope


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
with 'f' as f:
    pass
"""
    ast = str_to_ast(input_text)
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, With)
    assert "f" in result.body[0]._ending_scope
    assert "f" not in result._ending_scope


def test_ListComp():
    input_text = "[x[0] for x in [1, 2, 3] if x > 0]"
    ast = _strip_expr(str_to_ast(input_text))
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, ListComp)
    print(ast_unparse(result))
    print(ast_unparse(result.elt))
    for idx, gen in enumerate(result.generators):
        print(idx, ast_unparse(gen))
    assert "x" in result.elt._ending_scope


def test_SetComp():
    input_text = "{x for x in [1, 2, 3]}"
    ast = _strip_expr(str_to_ast(input_text))
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, SetComp)
    print(ast_unparse(result))
    print(ast_unparse(result.elt))
    for idx, gen in enumerate(result.generators):
        print(idx, ast_unparse(gen))
    assert "x" in result.elt._ending_scope


def test_DictComp():
    input_text = "{str(x): x for x in [1, 2, 3]}"
    ast = _strip_expr(str_to_ast(input_text))
    assert isinstance(ast, DictComp)

    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, DictComp)
    print(ast_unparse(result))
    print(ast_unparse(result.key))
    print(ast_unparse(result.value))
    for idx, gen in enumerate(result.generators):
        print(idx, ast_unparse(gen))
    assert "x" in result.key._ending_scope


def test_GeneratorExp():
    input_text = "(x for x in [1, 2, 3])"
    ast = _strip_expr(str_to_ast(input_text))
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result, GeneratorExp)
    print(ast_unparse(result))
    print(ast_unparse(result.elt))
    for idx, gen in enumerate(result.generators):
        print(idx, ast_unparse(gen))
    assert "x" in result.elt._ending_scope


def test_Assign():
    input_text = """
x = 5
print(x)"""
    ast = str_to_ast(input_text, keep_module=True)
    transformer = build_transformer(ast)
    result = transformer.visit(ast)

    assert isinstance(result.body[0], Assign)
    assert "x" in result.body[1]._ending_scope


def test_ClassDef():
    input_text = """
class InterestingName(object):
    pass
print(x)"""
    ast = str_to_ast(input_text, keep_module=True)
    transformer = build_transformer(ast)
    transformer.scope["object"] = "builtin"
    result = transformer.visit(ast)

    assert isinstance(result.body[0], ClassDef)
    assert "InterestingName" in result.body[0].body[0]._ending_scope
    assert "InterestingName" in result.body[1]._ending_scope
