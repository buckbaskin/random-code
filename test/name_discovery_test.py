from random_code import nested_unpack

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


# Assert
def test_assert():
    ast = str_to_ast("""assert name == 0""")
    assert ["name"] == nested_unpack(ast)


# Break
# withitem
# While
def test_while():
    ast = str_to_ast(
        """
while name > 0:
    print(name)"""
    )
    assert ["name"] == nested_unpack(ast)


# Assign
def test_Assign():
    ast = str_to_ast("""assign = name""")
    assert ["name"] == nested_unpack(ast)


# Attribute
# BinOp
# BoolOp
def test_BoolOp():
    ast = str_to_ast("""name or False""")
    assert ["name"] == nested_unpack(ast)


# Call
# ClassDef
def test_ClassDef_simple():
    ast = str_to_ast(
        """
class Major(name):
    pass"""
    )
    assert ["name"] == nested_unpack(ast)


# Compare
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
# DictComp
# dump
# Expr
# ExceptHandler
# fix_missing_locations
# For
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
# If
# IfExp
def test_IfExp():
    ast = _strip_expr(str_to_ast("""0 if name else 1"""))
    assert isinstance(ast, IfExp)
    assert ["name"] == nested_unpack(ast)


# Import
# ImportFrom
# Index
# JoinedStr
# keyword
# Lambda
# List
# Pass
# ListComp
# Module
# Name
# NodeTransformer
# NodeVisitor
# parse
# Raise
# Return
# Set
# SetComp
# Starred
# Subscript
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
# UnaryOp
