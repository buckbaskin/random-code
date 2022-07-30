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

from collections import defaultdict
from typing import List, Dict
from frozendict import frozendict
from random import Random

from pprint import pprint

UnbundledElementsType = Dict[str, Dict[str, List[AST]]]


class UnbundlingVisitor(NodeVisitor):
    def __init__(self, *, prettyprinter=False):
        self.prettyprinter = prettyprinter

        self.depth = 0
        self.missed_parents = set()
        self.missed_children = set()

        # currently setting up as a collection of lists
        # easy to select and random body, type_ignores in the Module case
        # harder if you want to keep bodies and type_ignores paired
        self.visited = {
            "Module": {"body": [], "type_ignores": []},
            "BinOp": {"left": [], "right": [], "op": []},
            "Constant": {"value": [], "kind": []},
            "Compare": {"left": [], "ops": [], "comparators": []},
            "Name": {"id": [], "ctx": []},
            "Call": {"func": [], "args": [], "keywords": []},
            "Return": {"value": []},
            "If": {"test": [], "body": [], "orelse": []},
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
            "FunctionDef": {
                "name": [],
                "args": [],
                "body": [],
                "decorator_list": [],
                "returns": [],
                "type_comment": [],
            },
            "Expr": {"value": []},
        }
        self.ignore = ["Add", "Mult", "Eq", "LtE", "Sub", "Load"]
        self.explore = ["Expr"]

        for k in self.visited:
            name = "visit_%s" % (k,)
            setattr(self, name, self._helper_function_definer(k))

        for k in self.ignore:
            name = "visit_%s" % (k,)
            setattr(self, name, self._ignore_function_definer(k))

        for k in self.explore:
            if k in self.visited or k in self.ignore:
                continue
            name = "visit_%s" % (k,)
            setattr(self, name, self._explore_function_definer(k))

    def _helper_function_definer(self, node_name):
        def _visit_X(node_):
            self._known_visit(node_name, node_)
            self._post_visit(node_)

        _visit_X.name = node_name
        _visit_X.__name__ = node_name
        return _visit_X

    def _ignore_function_definer(self, node_name):
        def _visit_X_ignore(node_):
            self._post_visit(node_)

        _visit_X_ignore.name = node_name
        _visit_X_ignore.__name__ = node_name
        return _visit_X_ignore

    def _explore_function_definer(self, node_name):
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
        for ast_type, elements in _map.items():
            if ast_type not in unbundled:
                unbundled[ast_type] = defaultdict(list)
            for k, list_of_vals in elements.items():
                unbundled[ast_type][k].extend(list_of_vals)

    for k in list(unbundled.keys()):
        unbundled[k] = dict(unbundled[k])

    return unbundled


class SourceSoup(object):
    def __init__(self, corpus, seed=1):
        self.corpus = corpus

        self.rng = Random(seed)

        for ast_element in self.corpus:
            print("SourceSoup", "Dynamically generating %s" % (ast_element,))
            setattr(self, ast_element, self._strategy_strict_pairs(ast_element))

    def _strategy_strict_pairs(self, node_name):
        min_examples = min([len(v) for k, v in self.corpus[node_name].items()])
        if min_examples <= 1:
            print("Boring Case %s" % (node_name,))

        def _visit_strict_pairs():
            reference = self.corpus[node_name]
            index = self.rng.randrange(0, min_examples)
            print("index", index, "/", "min_examples", min_examples)

            kwargs = {}
            for k, list_of_val in reference.items():
                kwargs[k] = list_of_val[index]

            import ast

            return getattr(ast, node_name)(**kwargs)

        _visit_strict_pairs.name = node_name
        _visit_strict_pairs.__name__ = node_name
        return _visit_strict_pairs


class RandomizingTransformer(NodeTransformer):
    def __init__(self, corpus, *, prettyprinter=False):
        self.corpus = corpus
        self.prettyprinter = prettyprinter

        self.depth = 0
        self.missed_parents = set()
        self.missed_children = set()

        # currently setting up as a collection of lists
        # easy to select and random body, type_ignores in the Module case
        # harder if you want to keep bodies and type_ignores paired
        # TODO generate based on corpus instead of copied code
        self.visited = {
            "BinOp": {"left": [], "right": [], "op": []},
            "Constant": {"value": [], "kind": []},
            "Compare": {"left": [], "ops": [], "comparators": []},
            "Name": {"id": [], "ctx": []},
            "Call": {"func": [], "args": [], "keywords": []},
            "Return": {"value": []},
            "If": {"test": [], "body": [], "orelse": []},
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
            "FunctionDef": {
                "name": [],
                "args": [],
                "body": [],
                "decorator_list": [],
                "returns": [],
                "type_comment": [],
            },
            "Expr": {"value": []},
        }
        self.ignore = ["Add", "Mult", "Eq", "LtE", "Sub", "Load"]
        self.explore = []

        for k in self.visited:
            name = "visit_%s" % (k,)
            setattr(self, name, self._helper_function_definer(k))

        for k in self.ignore:
            name = "visit_%s" % (k,)
            setattr(self, name, self._ignore_function_definer(k))

        for k in self.explore:
            if k in self.visited or k in self.ignore:
                continue
            name = "visit_%s" % (k,)
            setattr(self, name, self._explore_function_definer(k))

    def _helper_function_definer(self, node_name):
        def _visit_X(node_):
            print("Removing %s %s" % (node_name, type(node_)))
            print(ast_unparse(node_))
            swapout = getattr(self.corpus, node_name)()
            print("Inserting %s" % (node_name,))
            print(ast_unparse(swapout))

            result = self._post_visit(swapout)

            if self.prettyprinter:
                print(
                    " " * self.depth + "Swapped " + str(node_) + " for " + str(result)
                )
            return result

        _visit_X.name = node_name
        _visit_X.__name__ = node_name
        return _visit_X

    def _ignore_function_definer(self, node_name):
        def _visit_X_ignore(node_):
            result = self._post_visit(node_)
            print("Ignoring/passing through " + str(node_) + " as " + node_name)
            assert result is not None
            return result

        _visit_X_ignore.name = node_name
        _visit_X_ignore.__name__ = node_name
        return _visit_X_ignore

    def _explore_function_definer(self, node_name):
        def _visit_X_explore(node_):
            self._explore_visit(node_name, node_)
            self.generic_visit(node_)
            return node_

        _visit_X_explore.name = node_name
        _visit_X_explore.__name__ = node_name
        return _visit_X_explore

    def _post_visit(self, node):
        if self.prettyprinter:
            print(" " * self.depth + type(node).__name__)

        self.depth += 1
        result = NodeTransformer.generic_visit(self, node)
        assert result is not None
        self.depth -= 1
        return result

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

    def visit_Module(self, module):
        return self._post_visit(module)

    def unbundled(self):
        return self.visited

    def generic_visit(self, node):
        if len(node._fields) > 0:
            self.missed_parents.add(type(node).__name__)
        else:
            self.missed_children.add(type(node).__name__)

        return self._post_visit(node)


class AutoFixer(NodeVisitor):
    def generic_visit(self, node):
        try:
            fix_missing_locations(node)
        except AttributeError:
            print("Broken node %s %s" % (type(node), node))
            NodeVisitor.generic_visit(self, node)
            return

        print("Healthy node %s %s" % (type(node), node))
        return


def the_sauce(gen: SourceSoup, start: Module):
    transformer = RandomizingTransformer(gen, prettyprinter=True)
    result = transformer.visit(start)
    assert result is not None
    result = fix_missing_locations(result)
    return result


# todo: accept str or path
def give_me_random_code(corpus: List[str]):
    assert len(corpus) > 0

    ast_set = {}
    source_generated = {}
    for corpus_file_path in corpus:
        with open(corpus_file_path) as f:
            file_contents = []
            for line in f:
                file_contents.append(line)
            ast_set[corpus_file_path] = parse(
                "\n".join(file_contents), corpus_file_path, type_comments=True
            )

    for corpus_file_path in ast_set:
        source_generated[corpus_file_path] = ast_unparse(ast_set[corpus_file_path])

    unbundled_separates = []
    for file_path, tree in ast_set.items():
        unbundled_separates.append(unbundle_ast(tree))

    raw_materials = merge_unbundled_asts(unbundled_separates)

    pprint("Source Corpus")
    pprint(raw_materials)

    gen = SourceSoup(raw_materials, seed=2)

    starter_home = gen.Module()

    result = the_sauce(gen, starter_home)

    print("Module as Generated Source")
    print(starter_home)
    print(ast_unparse(starter_home))
    print("Modifed version as Generated Source")
    print(result)
    print(ast_unparse(result))

    return ""


def main():
    random_source = give_me_random_code(["corpus/int_functions.py", "corpus/main.py"])


if __name__ == "__main__":
    main()
