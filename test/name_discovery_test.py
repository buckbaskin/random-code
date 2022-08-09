from random_code import nested_unpack

from ast import (
    AugAssign,
    BinOp,
    BoolOp,
    Break,
    Call,
    Compare,
    Dict,
    DictComp,
    Expr,
    GeneratorExp,
    IfExp,
    Index,
    JoinedStr,
    ListComp,
    Module,
    Pass,
    Raise,
    Return,
    Set,
    SetComp,
    Slice,
    Subscript,
    UnaryOp,
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


# Assert
def test_assert():
    ast = str_to_ast("""assert name == 0""")
    assert ["name"] == nested_unpack(ast)


# Break
def test_Break():
    ast = str_to_ast("break")
    assert isinstance(ast, Break)
    assert [] == nested_unpack(ast)


# withitem
def test_withitem():
    ast = str_to_ast(
        """
with name as f:
    pass"""
    )
    assert ["name"] == nested_unpack(ast)


# While
def test_While():
    ast = str_to_ast(
        """
while name > 0:
    pass"""
    )
    assert ["name"] == nested_unpack(ast)


# Assign
def test_Assign():
    ast = str_to_ast("""assign = name""")
    assert ["name"] == nested_unpack(ast)


# AugAssign
def test_AugAssign():
    ast = str_to_ast("""assign += name""")
    assert ["assign", "name"] == nested_unpack(ast)


# BinOp
def test_BinOp_none():
    ast = _strip_expr(str_to_ast("""1 | False"""))
    assert isinstance(ast, BinOp)
    assert [] == nested_unpack(ast)


def test_BinOp_one():
    ast = _strip_expr(str_to_ast("""name | False"""))
    assert isinstance(ast, BinOp)
    assert ["name"] == nested_unpack(ast)


def test_BinOp_two():
    ast = _strip_expr(str_to_ast("""name1 | name2"""))
    assert isinstance(ast, BinOp)
    assert ["name1", "name2"] == nested_unpack(ast)


# BoolOp
def test_BoolOp_none():
    ast = _strip_expr(str_to_ast("""True or False"""))
    assert isinstance(ast, BoolOp)
    assert [] == nested_unpack(ast)


def test_BoolOp_one():
    ast = _strip_expr(str_to_ast("""name or False"""))
    assert isinstance(ast, BoolOp)
    assert ["name"] == nested_unpack(ast)


def test_BoolOp_two():
    ast = _strip_expr(str_to_ast("""name1 or name2"""))
    assert isinstance(ast, BoolOp)
    assert ["name1", "name2"] == nested_unpack(ast)


# Call
def test_Call_direct():
    ast = _strip_expr(str_to_ast("""name()"""))
    assert isinstance(ast, Call)
    assert ["name"] == nested_unpack(ast)


def test_Call_member():
    ast = _strip_expr(str_to_ast("""name.x()"""))
    assert isinstance(ast, Call)
    assert ["name"] == nested_unpack(ast)


# ClassDef
def test_ClassDef_simple():
    ast = str_to_ast(
        """
class Major(name):
    pass"""
    )
    assert ["name"] == nested_unpack(ast)


# Compare
def test_Compare_none():
    ast = _strip_expr(str_to_ast("""True == False"""))
    assert isinstance(ast, Compare)
    assert [] == nested_unpack(ast)


def test_Compare_one():
    ast = _strip_expr(str_to_ast("""name == False"""))
    assert isinstance(ast, Compare)
    assert ["name"] == nested_unpack(ast)


def test_Compare_two():
    ast = _strip_expr(str_to_ast("""name1 == name2"""))
    assert isinstance(ast, Compare)
    assert ["name1", "name2"] == nested_unpack(ast)


# Constant
def test_Constant():
    ast = str_to_ast("""0""")
    assert [] == nested_unpack(ast)


# Delete
def test_Delete_simple():
    ast = str_to_ast("""del name""")
    assert ["name"] == nested_unpack(ast)


def test_Delete_subscript():
    ast = str_to_ast("""del name[0]""")
    assert ["name"] == nested_unpack(ast)


def test_Delete_member():
    ast = str_to_ast("""del name.other""")
    assert ["name"] == nested_unpack(ast)


# Dict
def test_Dict_empty():
    ast = _strip_expr(str_to_ast("""{}"""))
    assert isinstance(ast, Dict)
    assert [] == nested_unpack(ast)


def test_Dict_one_key():
    ast = _strip_expr(str_to_ast("""{name: 1.0}"""))
    assert isinstance(ast, Dict)
    assert ["name"] == nested_unpack(ast)


def test_Dict_one_value():
    ast = _strip_expr(str_to_ast("""{"x": name}"""))
    assert isinstance(ast, Dict)
    assert ["name"] == nested_unpack(ast)


def test_Dict_one():
    ast = _strip_expr(str_to_ast("""{name1: name2}"""))
    assert isinstance(ast, Dict)
    assert ["name1", "name2"] == nested_unpack(ast)


# DictComp
def test_DictComp():
    ast = _strip_expr(str_to_ast("{k: 1 for k in [1, name, 3]}"))
    assert isinstance(ast, DictComp)
    assert ["name"] == nested_unpack(ast)


# ExceptHandler
def test_ExceptHandler():
    ast = str_to_ast(
        """
try:
    pass
except NameError:
    pass
"""
    )
    assert ["NameError"] == nested_unpack(ast)


# For
def test_For_loop():
    ast = str_to_ast(
        """
for x in name:
    pass"""
    )
    assert ["name"] == nested_unpack(ast)


def test_For_body():
    ast = str_to_ast(
        """
for x in [1, 2, 3]:
    name(x)"""
    )
    assert ["name", "x"] == nested_unpack(ast)


# FormattedValue
# FunctionDef
def test_FunctionDef_typing():
    ast = str_to_ast(
        """
def simple(x: target_name):
    pass"""
    )
    assert ["target_name"] == nested_unpack(ast)


def test_FunctionDef_decorator():
    ast = str_to_ast(
        """
@name
def simple(x):
    pass"""
    )
    assert ["name"] == nested_unpack(ast)


# GeneratorExp
def test_GeneratorExp():
    ast = _strip_expr(str_to_ast("(x**2 for x in name)"))
    assert isinstance(ast, GeneratorExp)
    assert ["name"] == nested_unpack(ast)


# If
def test_If():
    ast = str_to_ast(
        """
if name():
    pass"""
    )
    assert ["name"] == nested_unpack(ast)


# IfExp
def test_IfExp():
    ast = _strip_expr(str_to_ast("""0 if name else 1"""))
    assert isinstance(ast, IfExp)
    assert ["name"] == nested_unpack(ast)


# Import
def test_Import():
    ast = str_to_ast("import x")
    assert [] == nested_unpack(ast)


# ImportFrom
def test_ImportFrom():
    ast = str_to_ast("from y import x")
    assert [] == nested_unpack(ast)


# Index
def test_Index():
    ast = _strip_expr(str_to_ast('"abc"[name]'))
    assert ["name"] == nested_unpack(ast)


# JoinedStr
def test_JoinedStr():
    ast = _strip_expr(str_to_ast("f'value{i}'"))
    assert isinstance(ast, JoinedStr)
    assert ["i"] == nested_unpack(ast)


# keyword
# Lambda
def test_Lambda():
    ast = str_to_ast(
        """
lambda x: name[0]
"""
    )
    assert ["name"] == nested_unpack(ast)


# List
def test_List_empty():
    ast = str_to_ast("[]")
    assert [] == nested_unpack(ast)


def test_List_one():
    ast = str_to_ast("[name]")
    assert ["name"] == nested_unpack(ast)


def test_List_many():
    ast = str_to_ast("[name1, name2, name3]")
    assert ["name1", "name2", "name3"] == nested_unpack(ast)


# Pass
def test_Pass():
    ast = str_to_ast("pass")
    assert isinstance(ast, Pass)
    assert [] == nested_unpack(ast)


# ListComp
def test_ListComp():
    ast = _strip_expr(str_to_ast("[k for k in [1, name, 3]]"))
    assert isinstance(ast, ListComp)
    assert ["name"] == nested_unpack(ast, "DEBUG")


# Module
def test_Module():
    import ast

    ast_ = ast.parse("pass")
    assert isinstance(ast_, Module)
    assert [] == nested_unpack(ast_)


# Raise
def test_Raise():
    ast = str_to_ast("raise ValueError(name)")
    assert isinstance(ast, Raise)
    assert ["ValueError", "name"] == nested_unpack(ast)


# Return
def test_Return():
    ast = str_to_ast("return name")
    assert isinstance(ast, Return)
    assert ["name"] == nested_unpack(ast)


# Set
def test_Set():
    ast = _strip_expr(str_to_ast("{1, name, 3}"))
    assert isinstance(ast, Set)
    assert ["name"] == nested_unpack(ast)


# SetComp
def test_SetComp():
    ast = _strip_expr(str_to_ast("{k for k in [1, name, 3]}"))
    assert isinstance(ast, SetComp)
    assert ["name"] == nested_unpack(ast)


# Slice
def test_Slice():
    ast = _strip_expr(str_to_ast("[1, 2, 3, 4, 5][name:]")).slice
    assert isinstance(ast, Slice)
    assert ["name"] == nested_unpack(ast)


# Starred
# Subscript
def test_Subscript_none():
    ast = _strip_expr(str_to_ast('"abc"[0]'))
    assert isinstance(ast, Subscript)
    assert [] == nested_unpack(ast)


def test_Subscript_one():
    ast = _strip_expr(str_to_ast("name1[0]"))
    assert isinstance(ast, Subscript)
    assert ["name1"] == nested_unpack(ast)


def test_Subscript_two():
    ast = _strip_expr(str_to_ast("name1[name2]"))
    assert isinstance(ast, Subscript)
    assert ["name1", "name2"] == nested_unpack(ast)


# Try
def test_Try():
    ast = str_to_ast(
        """
try:
    name()
except:
    pass
"""
    )
    assert ["name"] == nested_unpack(ast)


# Tuple
def test_Tuple_empty():
    ast = str_to_ast("()")
    assert [] == nested_unpack(ast)


def test_Tuple_one():
    ast = str_to_ast("(name,)")
    assert ["name"] == nested_unpack(ast)


def test_Tuple_many():
    ast = str_to_ast("(name1, name2, name3)")
    assert ["name1", "name2", "name3"] == nested_unpack(ast)


# UnaryOp
def test_UnaryOp():
    ast = _strip_expr(str_to_ast("-name"))
    assert isinstance(ast, UnaryOp)
    assert ["name"] == nested_unpack(ast)
