from ast import (
    AST,
    dump,
    fix_missing_locations,
    FunctionDef,
    Module,
    NodeTransformer,
    NodeVisitor,
    parse,
)

try:
    # python3.9 and after
    from ast import unparse as ast_unparse
except ImportError:
    # before python3.9's ast.unparse
    from astunparse import unparse as ast_unparse

from collections import defaultdict, ChainMap
from typing import List, Dict
from frozendict import frozendict
from random import Random

from pprint import pprint

UnbundledElementsType = Dict[str, Dict[str, List[AST]]]


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


def merge_unbundled_asts(asts: List[UnbundledElementsType]):
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
        self.scope = ChainMap()

        self.visited = set(corpus.corpus.keys())
        self.ignore = ["Module"]

        for k in self.visited:
            name = "visit_%s" % (k,)
            if not hasattr(self, name):
                setattr(self, name, self._helper_function_factory(k))

        for k in self.ignore:
            name = "visit_%s" % (k,)
            setattr(self, name, self._ignore_function_factory(k))

    def valid_swap(self, node_, proposed_swap):
        node_type = type(node_).__name__
        new_definitions = ["Module", "FunctionDef", "arguments"]
        if node_type in new_definitions:
            return True

        i_know_its_wrong = ["alias"]
        # TODO(buck) Revisit alias when I'm ready to inspect modules
        if node_type in i_know_its_wrong:
            return True

        builtins = ["If", "Compare", "Return", "BinOp", "Expr", "ImportFrom"]
        if node_type in builtins:
            return True

        if node_type == "arg":
            if node_.arg == "self" or proposed_swap.arg == "self":
                # Heuristic because self has special usage
                return False
            return True

        if node_type == "Name":
            if node_.id not in self.scope:
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
            names_to_check = [node_.func.id]
            if len(node_.args) > 0:
                # TODO(buck): Implement this case
                1 / 0
            if len(node_.keywords) > 0:
                # TODO(buck): Implement this case
                1 / 0

            for name in names_to_check:
                if name not in self.scope:
                    return False

            return True

            # TODO(buck): Either overwrite Load/etc context ctx
            # TODO(buck): Or match on Load/etc context ctx
            1 / 0

        if node_type == "Assign":
            1 / 0  # TODO(buck): Start here

        # TODO(buck): Start with name resolution
        # TODO(buck): Move to type-aware?

        print("ERR I don't know how to swap %s with scope" % (node_type,))
        print(node_._fields)
        print("Trying")
        print(ast_unparse(node_))
        print("Swapping For")
        print(ast_unparse(proposed_swap))
        print("Scope")
        pprint(self.scope)
        1 / 0

    def args_to_names(self, arguments):
        args = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        if arguments.vararg is not None:
            print(node_.args.vararg)
            # TODO(add to list of args)
            1 / 0
        if arguments.kwarg is not None:
            print(node_.args.kwarg)
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


def make_asts(corpus: List[str]):
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
def give_me_random_code(corpus: List[str]):
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
