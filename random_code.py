from ast import (
    Assert,
    withitem,
    While,
    Assign,
    AST,
    Attribute,
    BinOp,
    BoolOp,
    Call,
    ClassDef,
    Compare,
    Constant,
    Delete,
    Dict,
    DictComp,
    dump,
    Expr,
    fix_missing_locations,
    For,
    FormattedValue,
    FunctionDef,
    GeneratorExp,
    If,
    IfExp,
    Import,
    ImportFrom,
    Index,
    JoinedStr,
    keyword,
    Lambda,
    List,
    Pass,
    ListComp,
    Module,
    Name,
    NodeTransformer,
    NodeVisitor,
    parse,
    Raise,
    Return,
    Set,
    SetComp,
    Starred,
    Subscript,
    Try,
    Tuple,
    UnaryOp,
    With,
    Yield,
)

### Feature List
# "alias": {"name": [], "asname": []},
# "AnnAssign": {"target": [], "annotation": [], "value": [], "simple": []},
# "arg": {"arg": [], "annotation": [], "type_comment": []},
# "arguments": {
# "Assert": {"test": [], "msg": []},
# "Assign": {"targets": [], "value": [], "type_comment": []},
# "AsyncFunctionDef": {
# "Attribute": {"value": [], "attr": [], "ctx": []},
# "AugAssign": {"target": [], "op": [], "value": []},
# "Await": {"value": []},
# "BinOp": {"left": [], "right": [], "op": []},
# "BoolOp": {"op": [], "values": []},
# "Call": {"func": [], "args": [], "keywords": []},
# "ClassDef": {
# "Compare": {"left": [], "ops": [], "comparators": []},
# "comprehension": {"target": [], "iter": [], "ifs": [], "is_async": []},
# "Constant": {"value": [], "kind": []},
# "Delete": {"targets": []},
# "Dict": {"keys": [], "values": []},
# "DictComp": {"key": [], "value": [], "generators": []},
# "ExceptHandler": {"type": [], "name": [], "body": []},
# "Expr": {"value": []},
# "For": {
# "FormattedValue": {"value": [], "conversion": [], "format_spec": []},
# "FunctionDef": {
# "GeneratorExp": {"elt": [], "generators": []},
# "Global": {"names": []},
# "If": {"test": [], "body": [], "orelse": []},
# "IfExp": {"test": [], "body": [], "orelse": []},
# "Import": {"names": []},
# "ImportFrom": {"module": [], "names": [], "level": []},
# "Index": {"value": []},
# "JoinedStr": {"values": []},
# "keyword": {"arg": [], "value": []},
# "Lambda": {"args": [], "body": []},
# "List": {"elts": [], "ctx": []},
# "ListComp": {"elt": [], "generators": []},
# "Module": {"body": [], "type_ignores": []},
# "Name": {"id": [], "ctx": []},
# "Nonlocal": {"names": []},
# "Raise": {"exc": [], "cause": []},
# "Return": {"value": []},
# "Set": {"elts": []},
# "SetComp": {"elt": [], "generators": []},
# "Slice": {"lower": [], "upper": [], "step": []},
# "Starred": {"value": [], "ctx": []},
# "Subscript": {"value": [], "slice": [], "ctx": []},
# "Try": {"body": [], "handlers": [], "orelse": [], "finalbody": []},
# "Tuple": {"elts": [], "ctx": []},
# "TypeIgnore": {"lineno": [], "tag": []},
# "UnaryOp": {"op": [], "operand": []},
# "While": {"test": [], "body": [], "orelse": []},
# "With": {"items": [], "body": [], "type_comment": []},
# "withitem": {"context_expr": [], "optional_vars": []},
# "Yield": {"value": []},
# "YieldFrom": {"value": []},
### End Feature List (~60 elements)

try:
    # python3.9 and after
    from ast import unparse as ast_unparse
except ImportError:
    # before python3.9's ast.unparse
    from astunparse import unparse as ast_unparse

import code

from collections import defaultdict, ChainMap
from typing import List as tList, Dict as tDict
from frozendict import frozendict
from random import Random

from pprint import pprint

UnbundledElementsType = tDict[str, tDict[str, tList[AST]]]


class UnbundlingVisitor(NodeVisitor):
    def __init__(self, *, prettyprinter=False, max_depth=10000):
        self.prettyprinter = prettyprinter

        self.depth = 0
        self.max_depth = max_depth
        self.missed_parents = set()
        self.missed_children = set()

        # currently setting up as a collection of lists
        # easy to select and random body, type_ignores in the Module case
        # harder if you want to keep bodies and type_ignores paired
        self.visited = {
            "alias": {"name": [], "asname": []},
            "AnnAssign": {"target": [], "annotation": [], "value": [], "simple": []},
            "arg": {"arg": [], "annotation": [], "type_comment": []},
            "arguments": {
                "posonlyargs": [],
                "args": [],
                "vararg": [],
                "kwonlyargs": [],
                "kw_defaults": [],
                "kwarg": [],
                "defaults": [],
            },
            "Assert": {"test": [], "msg": []},
            "Assign": {"targets": [], "value": [], "type_comment": []},
            "AsyncFunctionDef": {
                "name": [],
                "args": [],
                "body": [],
                "decorator_list": [],
                "returns": [],
                "type_comment": [],
            },
            "Attribute": {"value": [], "attr": [], "ctx": []},
            "AugAssign": {"target": [], "op": [], "value": []},
            "Await": {"value": []},
            "BinOp": {"left": [], "right": [], "op": []},
            "BoolOp": {"op": [], "values": []},
            "Call": {"func": [], "args": [], "keywords": []},
            "ClassDef": {
                "name": [],
                "bases": [],
                "keywords": [],
                "body": [],
                "decorator_list": [],
            },
            "Compare": {"left": [], "ops": [], "comparators": []},
            "comprehension": {"target": [], "iter": [], "ifs": [], "is_async": []},
            "Constant": {"value": [], "kind": []},
            "Delete": {"targets": []},
            "Dict": {"keys": [], "values": []},
            "DictComp": {"key": [], "value": [], "generators": []},
            "ExceptHandler": {"type": [], "name": [], "body": []},
            "Expr": {"value": []},
            "For": {
                "target": [],
                "iter": [],
                "body": [],
                "orelse": [],
                "type_comment": [],
            },
            "FormattedValue": {"value": [], "conversion": [], "format_spec": []},
            "FunctionDef": {
                "name": [],
                "args": [],
                "body": [],
                "decorator_list": [],
                "returns": [],
                "type_comment": [],
            },
            "GeneratorExp": {"elt": [], "generators": []},
            "Global": {"names": []},
            "If": {"test": [], "body": [], "orelse": []},
            "IfExp": {"test": [], "body": [], "orelse": []},
            "Import": {"names": []},
            "ImportFrom": {"module": [], "names": [], "level": []},
            "Index": {"value": []},
            "JoinedStr": {"values": []},
            "keyword": {"arg": [], "value": []},
            "Lambda": {"args": [], "body": []},
            "List": {"elts": [], "ctx": []},
            "ListComp": {"elt": [], "generators": []},
            "Module": {"body": [], "type_ignores": []},
            "Name": {"id": [], "ctx": []},
            "Nonlocal": {"names": []},
            "Raise": {"exc": [], "cause": []},
            "Return": {"value": []},
            "Set": {"elts": []},
            "SetComp": {"elt": [], "generators": []},
            "Slice": {"lower": [], "upper": [], "step": []},
            "Starred": {"value": [], "ctx": []},
            "Subscript": {"value": [], "slice": [], "ctx": []},
            "Try": {"body": [], "handlers": [], "orelse": [], "finalbody": []},
            "Tuple": {"elts": [], "ctx": []},
            "TypeIgnore": {"lineno": [], "tag": []},
            "UnaryOp": {"op": [], "operand": []},
            "While": {"test": [], "body": [], "orelse": []},
            "With": {"items": [], "body": [], "type_comment": []},
            "withitem": {"context_expr": [], "optional_vars": []},
            "Yield": {"value": []},
            "YieldFrom": {"value": []},
        }
        self.ignore = [
            "Add",
            "Eq",
            "IsNot",
            "Load",
            "Lt",
            "LtE",
            "Mult",
            "Store",
            "Sub",
        ]
        self.explore = ["Expr"]

        for k in self.visited:
            name = "visit_%s" % (k,)
            setattr(self, name, self._helper_function_factory(k))

        for k in self.ignore:
            name = "visit_%s" % (k,)
            setattr(self, name, self._ignore_function_factory(k))

        for k in self.explore:
            if k in self.visited or k in self.ignore:
                continue
            name = "visit_%s" % (k,)
            setattr(self, name, self._explore_function_factory(k))

    def _helper_function_factory(self, node_name):
        def _visit_X(node_):
            self._known_visit(node_name, node_)
            self._post_visit(node_)

        _visit_X.name = node_name
        _visit_X.__name__ = node_name
        return _visit_X

    def _ignore_function_factory(self, node_name):
        def _visit_X_ignore(node_):
            self._post_visit(node_)

        _visit_X_ignore.name = node_name
        _visit_X_ignore.__name__ = node_name
        return _visit_X_ignore

    def _explore_function_factory(self, node_name):
        def _visit_X_explore(node_):
            self._explore_visit(node_name, node_)
            self.generic_visit(node_)

        _visit_X_explore.name = node_name
        _visit_X_explore.__name__ = node_name
        return _visit_X_explore

    def _post_visit(self, node):
        if self.prettyprinter:
            print(" " * self.depth + type(node).__name__)
        self.depth += 1
        NodeVisitor.generic_visit(self, node)
        self.depth -= 1

    def _explore_visit(self, name, node):
        print("Explore")
        print(name)
        print(dir(node))
        for k in sorted(list(dir(node))):
            if not k.startswith("__"):
                print(k, getattr(node, k))
        print("---")

    def _known_visit(self, name, nodex):
        for k in self.visited[name]:
            self.visited[name][k].append(getattr(nodex, k))

    def unbundled(self):
        return self.visited

    def generic_visit(self, node):
        if len(node._fields) > 0:
            if type(node).__name__ not in self.missed_parents:

                def field_str(node):
                    for f in node._fields:
                        yield '"%s": []' % (f,)

                print('"%s": {%s},' % (type(node).__name__, ",".join(field_str(node))))
            self.missed_parents.add(type(node).__name__)
        else:
            self.missed_children.add(type(node).__name__)

        self._post_visit(node)


def unbundle_ast(ast: AST):
    v = UnbundlingVisitor(prettyprinter=False)
    v.visit(ast)

    result = v.unbundled()

    try:
        assert len(v.missed_parents) == 0
    except AssertionError:
        print("Missed AST types to handle")
        print(sorted(list(v.missed_parents)))
        print("Optional AST types to handle")
        print(sorted(list(v.missed_children)))
        raise

    return result


def merge_unbundled_asts(asts: tList[UnbundledElementsType]):
    unbundled = {}

    for _map in asts:
        for ast_type, elements in unbundle_ast(_map).items():
            if ast_type not in unbundled:
                unbundled[ast_type] = defaultdict(list)
            for k, list_of_vals in elements.items():
                unbundled[ast_type][k].extend(list_of_vals)

    for k in list(unbundled.keys()):
        unbundled[k] = dict(unbundled[k])

    return unbundled


class BagOfConcepts(object):
    def __init__(self, corpus, seed=1):
        self.corpus = corpus

        self.rng = Random(seed)

        for ast_element in self.corpus:
            setattr(self, ast_element, self._strategy_strict_pairs(ast_element))

    def _strategy_strict_pairs(self, node_name):
        min_examples = min([len(v) for k, v in self.corpus[node_name].items()])
        reference = self.corpus[node_name]

        batch = sorted(list(reference.items()))
        identifiers = [k for k, v in batch]
        data_lists = [v for k, v in batch]

        common_data_pairs = list(zip(*data_lists))

        def _visit_strict_pairs():
            # TODO(buck) infinite recursion blockers
            self.rng.shuffle(common_data_pairs)
            for data_pair in common_data_pairs:
                kwargs = {k: v for k, v in zip(identifiers, data_pair)}

                import ast

                yield getattr(ast, node_name)(**kwargs)

        _visit_strict_pairs.name = node_name
        _visit_strict_pairs.__name__ = node_name
        return _visit_strict_pairs


class RandomizingTransformer(NodeTransformer):
    def __init__(self, corpus, *, prettyprinter=False):
        self.corpus = corpus
        self.prettyprinter = prettyprinter

        self.depth = 0
        self.max_depth = 10
        self.missed_parents = set()
        self.missed_children = set()

        # This is where the craziness begins
        self.scope = ChainMap(
            {
                "tuple": "Type",
                "dict": "Type",
                "object": "Type",
                "IndexError": "Error",
                "ValueError": "Error",
                "AttributeError": "Error",
            }
        )
        # TODO(buck): Check to make sure we're not rejecting builtins
        self.out_of_scope = set()

        self.visited = set(corpus.corpus.keys())
        self.ignore = ["Module"]

        for k in self.visited:
            name = "visit_%s" % (k,)
            if not hasattr(self, name):
                setattr(self, name, self._helper_function_factory(k))

        for k in self.ignore:
            name = "visit_%s" % (k,)
            setattr(self, name, self._ignore_function_factory(k))

    def nested_unpack(self, element, top_level=None):
        # TODO(buck): Make this a while loop with controlled depth
        if element is None:
            return []

        if isinstance(element, Name):
            return [element.id]
        elif isinstance(element, Constant) or isinstance(element, Pass):
            return []
        elif isinstance(element, ImportFrom) or isinstance(element, Import):
            # TODO(buck): Maybe re-evaluate unpacking imports?
            return []
        elif isinstance(element, Attribute) or isinstance(element, Index):
            return self.nested_unpack(element.value, top_level)
        elif hasattr(element, "func"):
            return self.nested_unpack(element.func, top_level)
        elif isinstance(element, Lambda):
            # TODO(buck): check underlying
            return []
        elif isinstance(element, List):
            # TODO(buck): check underlying
            return []
        elif isinstance(element, Tuple):
            # TODO(buck): check underlying
            return []
        elif isinstance(element, Starred):
            return self.nested_unpack(element.value, top_level)
        elif isinstance(element, Subscript):
            return self.nested_unpack(element.value, top_level)
        elif (
            isinstance(element, If)
            or isinstance(element, IfExp)
            or isinstance(element, While)
        ):
            # Note: the body, orelse can be undefined depending on the result of the test, so taking the less strict approach here
            return self.nested_unpack(element.test, top_level)
        elif isinstance(element, UnaryOp):
            return self.nested_unpack(element.operand, top_level)
        elif isinstance(element, BinOp):
            return [
                *self.nested_unpack(element.left, top_level),
                *self.nested_unpack(element.right, top_level),
            ]
        elif isinstance(element, BoolOp):

            def flattened_BoolOp():
                for v in element.values:
                    for vid in self.nested_unpack(v, top_level):
                        yield v

            return list(flattened_BoolOp())
        elif isinstance(element, Compare):

            def flattened_Compare():
                for lid in self.nested_unpack(element.left, top_level):
                    yield lid
                for comparator in element.comparators:
                    for cid in self.nested_unpack(comparator, top_level):
                        yield cid

            return list(flattened_Compare())

        elif isinstance(element, Dict):

            def flattened_Dict():
                for k in element.keys:
                    for kid in self.nested_unpack(k, top_level):
                        yield kid
                for v in element.values:
                    for vid in self.nested_unpack(v, top_level):
                        yield vid

            return list(flattened_Dict())
        elif isinstance(element, Set):

            def flattened_Set():
                for k in element.elts:
                    for kid in self.nested_unpack(k, top_level):
                        yield kid

            return list(flattened_Set())
        elif isinstance(element, JoinedStr):
            return []
        elif isinstance(element, GeneratorExp):

            def flattened_GeneratorExp():
                for elt_id in self.nested_unpack(element.elt, top_level):
                    yield elt_id

                for gen in element.generators:
                    for if_ in gen.ifs:
                        for ifid in self.nested_unpack(if_, top_level):
                            yield ifid
                    for iid in self.nested_unpack(gen.iter, top_level):
                        yield iid

            return list(flattened_GeneratorExp())
        elif isinstance(element, ListComp):

            # TODO(buck): Combine GeneratorExp, ListComp impl
            def flattened_ListComp():
                for elt_id in self.nested_unpack(element.elt, top_level):
                    yield elt_id

                for gen in element.generators:
                    for if_ in gen.ifs:
                        for ifid in self.nested_unpack(if_, top_level):
                            yield ifid
                    for iid in self.nested_unpack(gen.iter, top_level):
                        yield iid

            return list(flattened_ListComp())
        elif isinstance(element, DictComp):

            def flattened_DictComp():
                # TODO(buck): check key, value
                for gen in element.generators:
                    for if_ in gen.ifs:
                        for ifid in self.nested_unpack(if_, top_level):
                            yield ifid
                    for iid in self.nested_unpack(gen.iter, top_level):
                        yield iid

            return list(flattened_DictComp())
        elif isinstance(element, SetComp):

            # TODO(buck): Combine GeneratorExp, SetComp impl
            def flattened_SetComp():
                for elt_id in self.nested_unpack(element.elt, top_level):
                    yield elt_id

                for gen in element.generators:
                    for if_ in gen.ifs:
                        for ifid in self.nested_unpack(if_, top_level):
                            yield ifid
                    for iid in self.nested_unpack(gen.iter, top_level):
                        yield iid

            return list(flattened_SetComp())
        elif isinstance(element, Yield) or isinstance(element, Return):
            return self.nested_unpack(element.value, top_level)
        elif isinstance(element, Expr):
            return self.nested_unpack(element.value, top_level)
        elif isinstance(element, With):

            def flattened_With():
                for withitem in element.items:
                    for cid in self.nested_unpack(withitem.context_expr, top_level):
                        yield cid

                for expr in element.body:
                    for eid in self.nested_unpack(expr, top_level):
                        yield eid

            return list(flattened_With())
        elif isinstance(element, withitem):
            return self.nested_unpack(element.context_expr, top_level)
        elif isinstance(element, ClassDef):

            def flattened_ClassDef():
                for base in element.bases:
                    for eid in self.nested_unpack(base, top_level):
                        yield eid

                for decorator in element.decorator_list:
                    for did in self.nested_unpack(decorator, top_level):
                        yield did

                if len(element.keywords) > 0:
                    print(element)
                    print(ast_unparse(element))
                    print(element.keywords)
                    print(ast_unparse(element.keywords))
                    1 / 0

            return list(flattened_ClassDef())
        elif isinstance(element, FunctionDef):

            def flattened_FunctionDef():
                for decorator in element.decorator_list:
                    for did in self.nested_unpack(decorator, top_level):
                        yield did

            return list(flattened_FunctionDef())
        elif isinstance(element, keyword):
            return self.nested_unpack(element.value, top_level)
        elif isinstance(element, Assign):
            return self.nested_unpack(element.value, top_level)
        elif isinstance(element, Try):
            # Note: handlers, orelse, finalbody conditionally executed and ignored

            def flattened_Try():
                for expr in element.body:
                    for eid in self.nested_unpack(expr, top_level):
                        yield eid

            return list(flattened_Try())
        elif isinstance(element, Assert):
            # Note: structurally like if
            return self.nested_unpack(element.test, top_level)
        elif isinstance(element, For):
            return self.nested_unpack(element.iter, top_level)
        elif isinstance(element, List):

            def flattened_List():
                for elem in element.elts:
                    for eid in self.nested_unpack(elem, top_level):
                        yield eid

            return list(flattened_List())
        elif isinstance(element, Raise):
            if element.cause is not None:
                print(element)
                print(ast_unparse(element))
                print(element.cause)
                print(ast_unparse(element.cause))
                1 / 0
            return self.nested_unpack(element.exc)
        elif isinstance(element, Delete):

            def flattened_Delete():
                for elem in element.targets:
                    for eid in self.nested_unpack(elem, top_level):
                        yield eid

            return list(flattened_Delete())
        elif isinstance(element, FormattedValue):
            # TODO(buck): Revisit FormattedValue expansion
            return []

        else:
            print("args unpacking?")
            if top_level is not None:
                print("Top Level")
                print(top_level)
                print(ast_unparse(top_level))
                print("Element")
            print(element)
            print(ast_unparse(element))
            print(element._fields)
            code.interact(local=dict(ChainMap({"ast_unparse": ast_unparse}, locals())))
            1 / 0

    def valid_swap(self, node_, proposed_swap):
        # TODO(buck): check for mixed usage of node_, proposed_swap
        node_type = type(node_).__name__
        new_definitions = ["Module", "arguments"]
        if node_type in new_definitions:
            return True

        i_know_its_wrong = ["alias", "ImportFrom"]
        # TODO(buck) Revisit alias when I'm ready to inspect modules
        if node_type in i_know_its_wrong:
            return True

        if node_type == "arg":
            if node_.arg == "self" or proposed_swap.arg == "self":
                # Heuristic because self has special usage
                return False
            return True

        if node_type == "Name":
            if proposed_swap.id not in self.scope:
                self.out_of_scope.add(proposed_swap.id)

            if node_.id not in self.scope:
                self.out_of_scope.add(node_.id)
                # If the original is out of scope, only expect the new one to be in scope
                # No need to type match
                return proposed_swap.id in self.scope

            type_to_match = self.scope[node_.id]

            return (
                proposed_swap.id in self.scope
                and self.scope[proposed_swap.id] == type_to_match
            )

        # TODO(buck): un-ignore op types, swap op types
        # TODO(buck): Allow swapping import, import from

        if node_type == "Call":
            names_to_check = []

            names_to_check.extend(self.nested_unpack(proposed_swap.func, proposed_swap))
            if len(proposed_swap.args) > 0:
                for a in proposed_swap.args:
                    names_to_check.extend(self.nested_unpack(a, proposed_swap))
            if len(proposed_swap.keywords) > 0:
                for k in proposed_swap.keywords:
                    arg_value = k.value
                    names_to_check.extend(self.nested_unpack(arg_value, proposed_swap))

            try:
                pass
                # assert len(names_to_check) > 0
                # Counter Example: 'a b c'.split()
            except AssertionError:
                print("Failed to find names to check")
                print(proposed_swap)
                print(ast_unparse(proposed_swap))
                code.interact(
                    local=dict(ChainMap({"ast_unparse": ast_unparse}, locals()))
                )
                raise

            for name in names_to_check:
                if name not in self.scope:
                    self.out_of_scope.add(name)
                    return False

            return True

            # TODO(buck): Either overwrite Load/etc context ctx
            # TODO(buck): Or match on Load/etc context ctx
            1 / 0

        names_to_check = self.nested_unpack(proposed_swap, proposed_swap)
        for name in names_to_check:
            if name not in self.scope:
                self.out_of_scope.add(name)
                return False
        return True

    def args_to_names(self, arguments):
        args = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        if arguments.vararg is not None:
            print("arguments.vararg")
            print(arguments.vararg)
            print(ast_unparse(arguments.vararg))
            # TODO(add to list of args)
            1 / 0
        if arguments.kwarg is not None:
            print("arguments.kwarg")
            print(arguments.kwarg)
            print(ast_unparse(arguments.kwarg))
            # TODO(add to list of args)
            1 / 0
        return args

    def _helper_function_factory(self, node_name):
        def _visit_X(node_):
            if self.depth > self.max_depth:
                return node_

            # TODO(buck): Scan for a valid swap
            for swapout in getattr(self.corpus, node_name)():
                if self.valid_swap(node_, swapout):
                    # Let python scoping drop this variable
                    break

            if node_name == "arguments":
                print(
                    "Swapped arguments in Scope %s for %s"
                    % (
                        [a.arg for a in self.args_to_names(node_)],
                        [a.arg for a in self.args_to_names(swapout)],
                    )
                )
                for start_arg in self.args_to_names(node_):
                    del self.scope[start_arg.arg]
                for next_arg in self.args_to_names(swapout):
                    # TODO type annotation available
                    self.scope[next_arg.arg] = "Arg"

            # TODO(buck): Lambda
            # TODO(buck): GeneratorExpressions
            # TODO(buck): ListComps
            # TODO(buck): Assignment
            # TODO(buck): DictComps
            # TODO(buck): SetComps
            # TODO(buck): With
            # TODO(buck): ClassDef
            # TODO(buck): check/sync with list of new def functionality
            if node_name == "FunctionDef":
                # TODO(buck): Typing a FunctionDef would enable swapping a function call for a value
                self.scope[swapout.name] = "FunctionDef"
                self.scope = self.scope.new_child()
                args = self.args_to_names(swapout.args)

                for arg in args:
                    # TODO(buck) add Typing info
                    type_ = "Any"
                    if arg.annotation is not None:
                        type_ = arg.annotation.id
                    elif arg.type_comment is not None:
                        # TODO(buck): check this code path
                        type_ = arg.type_comment.id
                        1 / 0
                    self.scope[arg.arg] = type_

            result = self._post_visit(swapout)

            if node_name == "FunctionDef":
                self.scope = self.scope.parents

            if self.prettyprinter:
                print(
                    " " * self.depth + "Swapped " + str(node_) + " for " + str(result)
                )
            return result

        _visit_X.name = node_name
        _visit_X.__name__ = node_name
        return _visit_X

    def _ignore_function_factory(self, node_name):
        def _visit_X_ignore(node_):
            if self.depth > self.max_depth:
                return node_

            result = self._post_visit(node_)
            return result

        _visit_X_ignore.name = node_name
        _visit_X_ignore.__name__ = node_name
        return _visit_X_ignore

    def _post_visit(self, node):
        if self.prettyprinter:
            print(" " * self.depth + type(node).__name__ + " " + str(self.scope))

        self.depth += 1
        result = NodeTransformer.generic_visit(self, node)
        assert result is not None
        self.depth -= 1
        return result

    def generic_visit(self, node):
        node_type_str = type(node).__name__
        name = "visit_%s" % (node_type_str,)
        # print("generic_visit: Providing default ignore case for %s" % (node_type_str,))
        setattr(self, name, self._ignore_function_factory(node_type_str))
        return self._post_visit(node)


def the_sauce(gen: BagOfConcepts, start: Module):
    transformer = RandomizingTransformer(gen, prettyprinter=True)
    result = transformer.visit(start)
    assert result is not None
    # result = fix_missing_locations(result)
    return result


def make_asts(corpus: tList[str]):
    ast_set = {}

    syntax_errors = []

    for corpus_file_path in corpus:
        with open(corpus_file_path) as f:
            file_contents = []
            for line in f:
                file_contents.append(line)
            try:
                ast_set[corpus_file_path] = parse(
                    "\n".join(file_contents), corpus_file_path, type_comments=True
                )
            except SyntaxError:
                syntax_errors.append(corpus_file_path)

    if len(syntax_errors) > 0:
        print("Syntax Mishaps")
        print(syntax_errors[:5])
        print("...")

    return ast_set


def find_files(directory: str):
    import os

    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


# todo: accept str or path
def give_me_random_code(corpus: tList[str]):
    assert len(corpus) > 0

    ast_set = make_asts(corpus)

    raw_materials = merge_unbundled_asts(ast_set.values())

    gen = BagOfConcepts(raw_materials, seed=1)

    starter_home = next(gen.Module())

    print("Module as Generated Source")
    print(starter_home)
    print(ast_unparse(starter_home))

    result = the_sauce(gen, starter_home)

    print("Modifed version as Generated Source")
    print(result)

    text_result = ast_unparse(result)
    print(text_result)

    return text_result


def main():
    corpus_paths = list(find_files("corpus"))
    print(corpus_paths)
    random_source = give_me_random_code(["corpus/int_functions.py", "corpus/main.py"])


if __name__ == "__main__":
    main()
