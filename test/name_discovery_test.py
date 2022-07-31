from random_code import nested_unpack

from ast import Module


def _strip_module(ast):
    if isinstance(ast, Module):
        body = ast.body
        for expr in body:
            return expr
    return ast


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
# Compare
# Constant
# Delete
# Dict
# DictComp
# dump
# Expr
# ExceptHandler
# fix_missing_locations
# For
# FormattedValue
# FunctionDef
# GeneratorExp
# If
# IfExp
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
# Tuple
# UnaryOp
